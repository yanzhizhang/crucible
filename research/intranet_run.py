"""One command for the intranet trip (10.11.1.97): preflight, build, compare, bundle.

Runs every reproduction step whose inputs exist, keeps going when one fails, and packs the
reports into one tarball to bring back. Each step is a subprocess with its own log, so a crash
in one family never costs the others::

    python research/intranet_run.py --preflight-only          # check first, change nothing
    python research/intranet_run.py                           # everything it can
    python research/intranet_run.py --feitu-root '/data/feitu/{date}'   # decode missing days too

Steps (``--only`` / ``--skip`` take these names):

``preflight``  imports, PM tree, free disk / memory, raw data present per date
``inventory``  ``research/pm_inventory.py`` over the PM tree (schema manifest)
``models``     ``research/pm_models_probe.py`` over ``models/`` (stage 5: objective, trees, gain per
               feature and family, s1..s4 / fit1..3 structure) -- needs no market data
``decode``     ``research/decode_feitu_day.py`` + quality gate for dates without raw data
               (only with ``--feitu-root``; the template gets ``{date}``)
``catalog``    ``research/catalog.py`` (the DuckDB views every builder reads)
``export``     every PM ``samplerS.nc`` / ``sampler*.nc`` / ``samplerR.parquet`` /
               ``1min_src`` file -> long parquet (``compare_pm.py --export``); prints dims
``bars``       ``research/build_bars_from_l2.py`` for each date
``sampler_r``  our samplerR, all variants, for each date and the session before it
``candidates`` ``research/pm_factors/candidates.py`` for each date
``compare``    samplerR vs PM (per variant), 1-minute bars vs ``1min_src`` (root and p1..p10),
               ``rank_candidates.py`` per family with candidates
``bundle``     ``data/intranet_run/<ts>.tar.gz``: logs, reports, the step table -- **no PM
               data** (the exports stay on the intranet; reports hold match statistics and at
               most five sample rows per column)

Dates default to what the PM tree holds (``hstats/sod/HS300/*/<date>``); ``sod/<D>`` is built
from session ``D-1``, so both are covered.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/work/crucible_data")
PY = sys.executable
STEPS = ("preflight", "inventory", "models", "decode", "catalog", "export", "bars", "sampler_r",
         "candidates", "compare", "bundle")
CANDIDATE_FAMILIES = ("ZZUG", "ZZDS", "ZZQI", "ZZVC", "ZZHL", "ZZMS", "ZZGW", "ZZXC", "ZZTS", "ZZQO",
                      "ZZWA", "ZZAL", "ZZSQ", "ZZCR")  # families research/pm_factors/candidates.py builds
SAMPLER_R_VARIANTS = ("order_all", "order_cont", "trade_all")


class Runner:
    """Runs steps as subprocesses, logs each, never stops on a failure."""

    def __init__(self, out: Path, dry: bool) -> None:
        self.out = out
        self.logs = out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.dry = dry
        self.table: list[dict] = []

    def run(self, step: str, name: str, args: list[str], timeout: int = 4 * 3600) -> bool:
        """One subprocess; stdout+stderr to ``logs/<step>__<name>.log``."""
        log = self.logs / f"{step}__{name}.log"
        cmd = [PY, *args]
        t0 = time.time()
        if self.dry:
            print(f"  [dry] {' '.join(cmd)}")
            self.table.append({"step": step, "name": name, "ok": None, "seconds": 0.0, "log": log.name})
            return True
        print(f"  {step}/{name} ... ", end="", flush=True)
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
        with log.open("w") as fh:
            fh.write(f"$ {' '.join(cmd)}\n\n")
            fh.flush()
            try:
                rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    timeout=timeout, check=False).returncode
            except subprocess.TimeoutExpired:
                rc = -9
        ok = rc == 0
        secs = time.time() - t0
        print(f"{'ok' if ok else f'FAILED (rc {rc})'} {secs:.0f}s")
        self.table.append({"step": step, "name": name, "ok": ok, "rc": rc, "seconds": round(secs, 1),
                           "log": log.name})
        return ok

    def note(self, step: str, name: str, ok: bool, detail: str) -> None:
        """Record an in-process check."""
        print(f"  {step}/{name}: {'ok' if ok else 'PROBLEM'} -- {detail}")
        self.table.append({"step": step, "name": name, "ok": ok, "detail": detail})


def pm_dates(prod: Path) -> list[str]:
    """Dates the PM tree has samplerS / samplerR for."""
    days = {p.name for p in (prod / "hstats").glob("**/HS300/*/[0-9]" + "?" * 7) if p.is_dir()}
    return sorted(days)


def prev_session(day: str) -> str:
    """Previous SSE session (``almanac`` calendar)."""
    sys.path.insert(0, str(REPO / "src"))
    from almanac.calendar import EQUITY, TradingCalendar

    d = TradingCalendar("XSHG", EQUITY).shift(dt.date(int(day[:4]), int(day[4:6]), int(day[6:])), -1)
    return d.strftime("%Y%m%d")


def raw_present(day: str) -> dict[str, bool]:
    """Which decoded raw streams exist for ``day``."""
    root = DATA / "feitu_raw"
    return {k: any((root / f"kind={k}" / f"date={day}").glob("*.parquet"))
            for k in ("order", "transaction", "quotation")}


def preflight(r: Runner, prod: Path, dates: list[str]) -> bool:
    """Everything the later steps need; returns False when nothing can run."""
    good = True
    missing = []
    for mod in ("polars", "duckdb", "pyarrow", "numpy", "xarray", "h5netcdf", "psutil"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    r.note("preflight", "imports", not missing, f"missing {missing}" if missing else "all present")
    good &= not missing
    r.note("preflight", "pm_tree", prod.is_dir(), str(prod))
    good &= prod.is_dir()
    DATA.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(DATA).free / 2**30
    r.note("preflight", "disk", free > 60, f"{free:.0f} GiB free under {DATA} (want > 60 per raw day)")
    try:
        import psutil

        mem = psutil.virtual_memory().available / 2**30
        r.note("preflight", "memory", mem > 8, f"{mem:.0f} GiB available (builders peak ~7)")
    except ImportError:
        pass
    r.note("preflight", "dates", bool(dates), f"{dates}")
    for d in dates:
        have = raw_present(d)
        r.note("preflight", f"raw_{d}", all(have.values()),
               f"{have} under {DATA / 'feitu_raw'}" + ("" if all(have.values())
                                                        else " -- use --feitu-root to decode it"))
    return good


def exports(r: Runner, prod: Path, out: Path) -> list[Path]:
    """PM files -> long parquet under ``out`` (stays on the intranet)."""
    files = sorted({*prod.glob("hstats/**/sampler*.nc"), *prod.glob("hstats/**/samplerR.parquet"),
                    *prod.glob("1min_src/*.nc"), *prod.glob("1min_src/p*/*.nc")})
    done = []
    for f in files:
        rel = f.relative_to(prod)
        dst = out / (str(rel.with_suffix("")).replace("/", "__") + ".parquet")
        if r.run("export", rel.as_posix().replace("/", "_"),
                 ["research/compare_pm.py", "--pm", str(f), "--export", str(dst)], timeout=1800):
            done.append(dst)
    return done


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prod", type=Path, default=Path("/work/prod"))
    ap.add_argument("--dates", help="comma-separated; default: dates found in the PM tree")
    ap.add_argument("--feitu-root", help="raw dump dir template with {date}, enables decoding")
    ap.add_argument("--only", help=f"comma-separated subset of {', '.join(STEPS)}")
    ap.add_argument("--skip", default="", help="comma-separated steps to skip")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args()

    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    out = REPO / "data" / "intranet_run" / stamp
    r = Runner(out, a.dry_run)
    steps = [s for s in (a.only.split(",") if a.only else STEPS) if s not in a.skip.split(",")]
    if a.preflight_only:
        steps = ["preflight"]
    pm_days = pm_dates(a.prod) if a.prod.is_dir() else []
    dates = sorted(a.dates.split(",")) if a.dates else pm_days
    # sod/<D> is built from D-1: our trade days are the PM days and the sessions before them
    trade_days = sorted({*dates, *(prev_session(d) for d in dates)}) if dates else []
    print(f"run {stamp}: PM dates {pm_days}, trade days {trade_days}, steps {steps}")
    export_dir = REPO / "data" / "pm_export"

    if "preflight" in steps and not preflight(r, a.prod, trade_days) and a.preflight_only:
        print("preflight found problems (see above)")
    if "inventory" in steps:
        r.run("inventory", "pm_tree", ["research/pm_inventory.py", "--root", str(a.prod),
                                       "--out", str(out / "reports" / "pm_manifest")])
    if "models" in steps and (a.prod / "models").is_dir():
        r.run("models", "probe", ["research/pm_models_probe.py", "--models", str(a.prod / "models"),
                                  "--hstats", str(a.prod / "hstats" / "sod" / "HS300"),
                                  "--out", str(out / "reports" / "stage5")])
    if "decode" in steps and a.feitu_root:
        for d in trade_days:
            if all(raw_present(d).values()):
                continue
            if r.run("decode", d, ["research/decode_feitu_day.py", "--root", a.feitu_root.format(date=d),
                                   "--date", d]):
                r.run("decode", f"quality_{d}", ["research/md_quality/run.py", "--date", d])
    if "catalog" in steps:
        r.run("catalog", "views", ["research/catalog.py"])
    exported: list[Path] = sorted(export_dir.glob("*.parquet"))
    if "export" in steps:
        exported = exports(r, a.prod, export_dir)
    have_raw = [d for d in trade_days if all(raw_present(d).values())]
    if "bars" in steps:
        for d in have_raw:
            r.run("bars", d, ["research/build_bars_from_l2.py", "--date", d])
    if "sampler_r" in steps and have_raw:
        r.run("sampler_r", "all", ["research/pm_factors/sampler_r.py", "--dates", ",".join(have_raw)])
    if "candidates" in steps and have_raw:
        r.run("candidates", "all", ["research/pm_factors/candidates.py", "--dates", ",".join(have_raw)])
    if "compare" in steps:
        rep = out / "reports"
        sr = DATA / "store" / "pm_factors" / "sampler_r"
        for f in sorted(a.prod.glob("hstats/**/samplerR.parquet")):
            day = f.parent.name
            ours_day = prev_session(day) if "/sod/" in f.as_posix() else day
            fam = f.parent.parent.name
            tag = "sod" if "/sod/" in f.as_posix() else "plain"
            for v in SAMPLER_R_VARIANTS:
                ours = sr / f"variant={v}" / f"date={ours_day}" / "samplerR.parquet"
                if ours.exists():
                    r.run("compare", f"samplerR_{fam}_{tag}_{day}_{v}",
                          ["research/compare_pm.py", "--pm", str(f), "--ours", str(ours), "--keys", "sid",
                           "--atol", "0.5", "--out", str(rep / "stage3" / f"{fam}_{tag}_{day}_{v}.parquet")])
        for f in sorted({*a.prod.glob("1min_src/*.nc"), *a.prod.glob("1min_src/p*/*.nc")}):
            day = f.stem
            ours = DATA / "store" / "bars" / "1min" / f"date={day}" / "part.parquet"
            if ours.exists():
                tag = f.parent.name if f.parent.name.startswith("p") else "root"
                # best effort: PM key names are only known once the export has been read
                r.run("compare", f"bars_{tag}_{day}",
                      ["research/compare_pm.py", "--pm", str(f), "--ours", str(ours), "--keys", "symbol,ts",
                       "--out", str(rep / "stage2" / f"1min_src_{tag}_{day}.parquet")])
        for f in exported:
            fam = next((p for p in f.stem.split("__") if p in CANDIDATE_FAMILIES), None)
            if fam and "samplerS" in f.stem:
                r.run("compare", f"rank_{f.stem}",
                      ["research/pm_factors/rank_candidates.py", "--family", fam, "--pm", str(f),
                       "--out", str(rep / "stage4" / f"{f.stem}.parquet")])
    (out / "steps.json").write_text(json.dumps(r.table, indent=1, default=str))
    bad = [t for t in r.table if t.get("ok") is False]
    print(f"\n{len(r.table)} steps, {len(bad)} with problems:")
    for t in bad:
        print(f"  {t['step']}/{t['name']}: {t.get('detail') or 'see logs/' + str(t.get('log'))}")
    if "bundle" in steps and not a.dry_run:
        tgz = out.parent / f"{stamp}.tar.gz"
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(out, arcname=stamp)
        print(f"\nbring back: {tgz} ({tgz.stat().st_size / 2**20:.1f} MiB) -- reports and logs only")


if __name__ == "__main__":
    main()

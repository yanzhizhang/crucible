r"""Where PM's chain of day-end state files starts, and which days were not produced by a run.

Reads every ``<dir>/YYYYMMDD.jz`` (gzip JSON {sid: {key: value}}) in date order and prints one row per file:

* link   - symlink target, if the file is one (a seed linked forward to a later date);
* md5    - first 8 hex of the content md5; ``dup:<date>`` when an earlier file has the same content;
* nsid / nkey - stocks, and the median number of keys per stock (a cold start has fewer or zeros);
* zero   - share of values that are 0 or null (high on a cold start, before EMA sums and weights fill);
* same   - share of (sid, key) values equal to the previous file (1.0 = a copy, not a new trading day);
* open   - share of stocks whose ``zzpk.open`` changed from the previous file (a new trading day ~ 1.0);
* stop   - latest ``zzba.p.t`` (trading clock: seconds since 09:30, lunch removed) as local HH:MM:SS; no date in it.

The guess printed at the end: the first file, the first one with a full key set, and every day that looks
copied rather than run. Clocks are seconds of day, so a copy is told only by content, not by its timestamps.

    python research/pm_factors/seed_probe.py --dir /data/prod/h/HS300/p1_intrading_dcp
    python research/pm_factors/seed_probe.py --dir /work/crucible_data/seed_probe/HS300 --out data/reports/seed_probe_23.parquet
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import polars as pl


def load(path: Path) -> tuple[str, dict]:
    raw = path.read_bytes()
    return hashlib.md5(raw).hexdigest(), json.loads(gzip.decompress(raw))


def hms(t: float | None) -> str:
    if t is None:
        return "-"
    t = int(t)
    t = t + 9 * 3600 + 1800 if t < 7200 else t - 7200 + 13 * 3600  # trading clock (s since 09:30, lunch removed) -> local
    return f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--from", dest="start", default="00000000", help="first date to read, YYYYMMDD")
    ap.add_argument("--to", dest="end", default="99999999", help="last date to read, YYYYMMDD")
    ap.add_argument("--out", type=Path, help="parquet of the table")
    a = ap.parse_args()

    files = sorted(p for p in a.dir.iterdir() if re.fullmatch(r"\d{8}\.jz", p.name) and a.start <= p.stem <= a.end)
    others = sorted(p.name for p in a.dir.iterdir() if not re.fullmatch(r"\d{8}\.jz", p.name))
    print(f"{a.dir}: {len(files)} state files {files[0].stem if files else '-'} .. {files[-1].stem if files else '-'}"
          f"; {len(others)} other entries {others[:10]}")
    seen: dict[str, str] = {}
    prev: dict | None = None
    rows = []
    for p in files:
        link = os.readlink(p) if p.is_symlink() else ""
        try:
            md5, d = load(p)
        except Exception as e:  # noqa: BLE001 - a broken file is itself a finding
            print(f"{p.stem}  unreadable: {e}")
            rows.append({"date": p.stem, "link": link, "md5": "", "dup": "", "bad": str(e)[:80]})
            prev = None
            continue
        dup = seen.get(md5, "")
        seen.setdefault(md5, p.stem)
        vals = [(s, k, v) for s, kv in d.items() for k, v in kv.items()]
        nkeys = sorted(len(kv) for kv in d.values())
        zero = sum(1 for _, _, v in vals if v is None or v == 0) / max(len(vals), 1)
        same = opened = None
        if prev is not None:
            common = [(s, k, v) for s, k, v in vals if s in prev and k in prev[s]]
            same = sum(1 for s, k, v in common if prev[s][k] == v) / max(len(common), 1)
            op = [(kv.get("zzpk.open"), prev[s].get("zzpk.open")) for s, kv in d.items() if s in prev]
            op = [(x, y) for x, y in op if x is not None and y is not None]
            opened = sum(1 for x, y in op if x != y) / max(len(op), 1) if op else None
        stops = [kv.get("zzba.p.t") for kv in d.values() if isinstance(kv.get("zzba.p.t"), (int, float))]
        rows.append({"date": p.stem, "link": link, "md5": md5[:8], "dup": dup, "bad": "", "nsid": len(d),
                     "nkey": nkeys[len(nkeys) // 2] if nkeys else 0, "nkey_max": nkeys[-1] if nkeys else 0,
                     "zero": zero, "same": same, "open": opened, "stop": hms(max(stops) if stops else None),
                     "bytes": p.stat().st_size})
        prev = d

    df = pl.DataFrame(rows, infer_schema_length=None)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=60, float_precision=3):
        print(df)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(a.out)

    ok = df.filter(pl.col("bad") == "")
    if ok.is_empty():
        return
    full = ok["nkey"].max()
    first_full = ok.filter(pl.col("nkey") >= 0.95 * full)["date"].head(1).to_list()
    copied = ok.filter((pl.col("dup") != "") | (pl.col("link") != "") | (pl.col("same") > 0.99) | (pl.col("open") < 0.5))
    print(f"guess: chain starts {ok['date'][0]} (nkey {ok['nkey'][0]}, zero {ok['zero'][0]:.3f});"
          f" first with a full key set ({full}) {first_full[0] if first_full else '-'}")
    print(f"guess: {copied.height} days look linked / copied / not a new trading day:"
          f" {copied.select('date', 'link', 'dup', 'same', 'open').rows()[:40]}")


if __name__ == "__main__":
    main()

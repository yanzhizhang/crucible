r"""Stage 5e: calibrate the model fingerprint on the features PM shipped real values for.

``feature_match`` / ``feature_intraday`` compare a candidate's distribution with the split
thresholds the trees chose on a feature. On 97 every candidate came out 2-16x narrower than the
thresholds: LightGBM splits more often in the tails, so threshold quantiles are not value quantiles.
For the families with a shipped ``samplerS.nc`` (real 5-minute values: ZZWA, ZZAL, ZZQO, ZZXC,
ZZTS, ...) both sides are known, so this measures the bias:

* per feature, where the threshold p10 / median / p90 fall in the true value distribution
  (``F(t_p10)`` = share of values at or below the threshold p10, ...), and the range vs the values';
* the median of those levels over all calibrated features goes to ``fingerprint_calib.json``.

With ``--calib`` the matchers then score a candidate by where the thresholds fall in **its**
distribution against those levels -- a correct formula lands on them whatever the tail bias.

Intranet only (reads samplerS through ``compare_pm.load_pm``).

    python research/pm_factors/fingerprint_calib.py --hstats /data/prod/hstats/sod/HS300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_pm import load_pm  # the netCDF reader (xarray stays in compare_pm)

KEYS = {"S", "D", "I", "V"}


def family_values(fam_dir: Path, dates: set[str] | None) -> pl.DataFrame:
    """All samplerS values of one family as (feature, v); feature = '<family>.<V>' unless V already has a dot."""
    out = []
    for f in sorted(fam_dir.glob("*/samplerS.*")):  # .nc on 97 (a .parquet export reads too)
        if dates and f.parent.name not in dates:
            continue
        df = load_pm(f)
        val = [c for c, t in df.schema.items() if c not in KEYS and t.is_numeric()]
        if "V" not in df.columns or not val:
            print(f"skip {f}: columns {df.columns}")
            continue
        fam = fam_dir.name.lower()
        vname = pl.col("V").cast(pl.String).str.strip_chars(" \x00").str.to_lowercase()  # NC_CHAR names are padded
        out.append(df.select(pl.when(vname.str.contains(r"\.")).then(vname).otherwise(pl.lit(fam + ".") + vname).alias("feature"),
                             pl.col(val[0]).cast(pl.Float64).alias("v")))
    return pl.concat(out).filter(pl.col("v").is_finite()) if out else pl.DataFrame(schema={"feature": pl.String, "v": pl.Float64})


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    ap.add_argument("--hstats", type=Path, default=Path("/data/prod/hstats/sod/HS300"))
    ap.add_argument("--dates", default="", help="comma list of day folders (default: all)")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage5"))
    args = ap.parse_args()
    fp = pl.read_parquet(args.fingerprint)
    dates = {d.strip() for d in args.dates.split(",") if d.strip()} or None
    fams = sorted(p for p in args.hstats.iterdir() if p.is_dir() and p.name.upper() in set(fp["family"].to_list()))
    print(f"families with samplerS and a fingerprint: {[p.name for p in fams]}")
    rows = []
    for fd in fams:
        vals = family_values(fd, dates)
        have = sorted(vals["feature"].unique().to_list())
        want = sorted(fp.filter(pl.col("family") == fd.name.upper())["feature"].to_list())
        print(f"{fd.name}: samplerS {have[:12]}{' ...' if len(have) > 12 else ''} | model {want[:12]}{' ...' if len(want) > 12 else ''}"
              f" | shared {len(set(have) & set(want))}")
        for g in vals.partition_by("feature"):
            feat = g["feature"][0]
            f = fp.filter(pl.col("feature") == feat)
            if f.is_empty() or f["t_med"][0] is None:
                continue
            f = f.row(0, named=True)
            v = g["v"].to_numpy()
            rows.append({"feature": feat, "gain_share": f["gain_share"], "n": v.size,
                         "F_p10": float(np.mean(v <= f["t_p10"])), "F_med": float(np.mean(v <= f["t_med"])),
                         "F_p90": float(np.mean(v <= f["t_p90"])),
                         "t_p10": f["t_p10"], "v_p10": float(np.quantile(v, 0.1)), "t_med": f["t_med"], "v_med": float(np.median(v)),
                         "t_p90": f["t_p90"], "v_p90": float(np.quantile(v, 0.9)),
                         "lo": f["lo"], "v_min": float(v.min()), "hi": f["hi"], "v_max": float(v.max())})
        print(f"done {fd.name}")
    if not rows:
        print("no feature had both samplerS values and a fingerprint row")
        return
    res = pl.DataFrame(rows).sort("gain_share", descending=True)
    levels = {q: float(res[f"F_{q}"].median()) for q in ("p10", "med", "p90")}
    args.out.mkdir(parents=True, exist_ok=True)
    res.write_csv(args.out / "fingerprint_calib.csv")
    (args.out / "fingerprint_calib.json").write_text(json.dumps({"levels": levels, "features": res.height}, indent=1))
    with pl.Config(tbl_rows=80, tbl_cols=17, tbl_width_chars=250, float_precision=5):
        print("\n== where the threshold quantiles fall in the true values (F = share of values at or below), per feature")
        print(res)
        print("\n== spread of those levels over the features")
        print(res.select([pl.col(f"F_{q}").quantile(x).alias(f"F_{q}_{n}") for q in ("p10", "med", "p90")
                          for x, n in ((0.25, "q25"), (0.5, "med"), (0.75, "q75"))]))
    print(f"calibration levels {levels} -> {args.out / 'fingerprint_calib.json'}")


if __name__ == "__main__":
    main()

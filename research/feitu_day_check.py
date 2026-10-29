r"""Which source files of a copied feitu tree lost data, from the decoded days: rows and sequence holes.

Run day by day after ``decode_feitu_day.py --keep-going`` (``feitu_day_round.sh`` does both and deletes
the decoded day afterwards), then once with ``--report`` over all days:

* **rows**: per stream and minute file, the decoded rows against the median of the same stream and
  minute on the other days; a decode error, an empty file and a minute other days have are listed too;
* **sequence**: orders and trades of one exchange share one id sequence per channel, so every hole
  in it is a lost record. A hole is charged to the minute file of the record after it; per exchange
  and minute the holes of one day are compared with the same minute on the other days (SSE ids run
  across securities feitu does not carry, so SSE has a steady background and only the excess counts).
  A flagged minute names its order and transaction files with their row ratios, so the one that lost
  rows is the short one.

    python research/feitu_day_check.py --raw /work/crucible_data/feitu_check_raw --date 20260901 --out data/reports/feitu_day_check
    python research/feitu_day_check.py --out data/reports/feitu_day_check --report
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

KINDS = ("order", "transaction", "quotation", "index")


def check_day(raw: Path, d: str, out: Path) -> None:
    """Write man_<d>.parquet (one row per source file) and seq_<d>.parquet (holes per exchange, channel and minute)."""
    t0 = time.perf_counter()
    man = []
    for k in KINDS:
        f = raw / f"kind={k}" / f"date={d}" / "_manifest.parquet"
        if not f.exists():
            print(f"{d}: no {k} manifest (stream absent or decode failed)")
            continue
        m = pl.read_parquet(f)
        if "error" not in m.columns:
            m = m.with_columns(pl.lit(None, pl.String).alias("error"))
        man.append(m.select(pl.lit(int(d)).alias("date"), pl.lit(k).alias("kind"), "file", "minute", "bytes", "rows",
                            pl.col("error").cast(pl.String)))
    if man:
        man = pl.concat(man)
        man.write_parquet(out / f"man_{d}.parquet")
        print(f"{d}: {man.height} files, {man['rows'].sum():,} rows, {man['error'].is_not_null().sum()} decode errors")
    parts = [raw / f"kind={k}" / f"date={d}" for k in ("order", "transaction")]
    parts = [str(p / "part-*.parquet") for p in parts if any(p.glob("part-*.parquet"))]
    if not parts:
        print(f"{d}: no order / transaction parts, sequence skipped")
        return
    minute = pl.col("part").str.extract(r"part-(\w+)\.parquet$")
    lf = pl.concat([pl.scan_parquet(p, include_file_paths="part").select("market_id", "channel_id", "seq_id", minute.alias("minute"))
                    for p in parts])
    chans = lf.group_by("market_id", "channel_id").agg(pl.len().alias("n")).collect(engine="streaming").sort("market_id", "channel_id")
    rows, holes = [], []
    for mkt, ch, n in chans.iter_rows():
        s = (lf.filter((pl.col("market_id") == mkt) & (pl.col("channel_id") == ch)).select("seq_id", "minute")
             .collect(engine="streaming").sort("seq_id"))
        s = s.with_columns((pl.col("seq_id").shift(-1) - pl.col("seq_id") - 1).alias("missing"), pl.col("minute").shift(-1).alias("next_minute"))
        h = s.filter(pl.col("missing") > 0)
        dups = int((s["missing"] == -1).sum())
        rows.append(h.group_by("next_minute").agg(pl.col("missing").sum(), pl.len().alias("holes"), pl.col("missing").max().alias("max_hole"))
                    .with_columns(pl.lit(mkt).alias("market_id"), pl.lit(ch).alias("channel_id")))
        holes.append(h.sort("missing", descending=True).head(5).select(
            pl.lit(int(d)).alias("date"), pl.lit(mkt).alias("market_id"), pl.lit(ch).alias("channel_id"), "seq_id", "missing", "minute", "next_minute"))
        print(f"{d}: market {mkt} channel {ch}: {n:,} records, ids {s['seq_id'][0]}..{s['seq_id'][-1]}, "
              f"{int(h['missing'].sum()):,} missing in {h.height:,} holes, {dups:,} duplicates")
    seq = pl.concat(rows).rename({"next_minute": "minute"}).with_columns(pl.lit(int(d)).alias("date"))
    seq.write_parquet(out / f"seq_{d}.parquet")
    pl.concat(holes).write_parquet(out / f"holes_{d}.parquet")
    print(f"{d}: done ({time.perf_counter() - t0:.0f}s)")


def report(out: Path, low: float, excess: float) -> None:
    """Flag source files over all checked days."""
    mans = sorted(out.glob("man_*.parquet"))
    if not mans:
        raise SystemExit(f"no man_*.parquet under {out}")
    man = pl.concat([pl.read_parquet(f) for f in mans])
    days = man["date"].n_unique()
    print(f"== {days} days checked: {man['date'].min()} .. {man['date'].max()}")
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_fmt_str_lengths(90)
    print("-- files and rows per day and stream (rows in millions)")
    print(man.group_by("date", "kind").agg(pl.len().alias("files"), (pl.col("rows").sum() / 1e6).round(2).alias("mrows"))
          .pivot(on="kind", index="date", values=["files", "mrows"]).sort("date"))

    med = man.group_by("kind", "minute").agg(pl.col("rows").median().alias("med_rows"), pl.len().alias("days_with"))
    m = man.join(med, on=["kind", "minute"]).with_columns((pl.col("rows") / pl.col("med_rows")).alias("ratio"))
    bad = m.filter(pl.col("error").is_not_null() | ((pl.col("rows") == 0) & (pl.col("med_rows") > 0))
                   | ((pl.col("ratio") < low) & (pl.col("med_rows") > 1000) & (pl.col("days_with") >= 3)))
    print(f"== files that failed to decode, decoded empty, or hold < {low:g} x the rows of the same minute on the other days: {bad.height}")
    if bad.height:
        print(bad.sort("date", "kind", "minute").select("date", "kind", "file", "bytes", "rows", "med_rows", "ratio", "error"))
    allmin = man.select("kind", "minute").unique()
    gone = allmin.join(man.select("date").unique(), how="cross").join(man.select("date", "kind", "minute"), on=["date", "kind", "minute"], how="anti")
    gone = gone.join(man.select("date", "kind").unique(), on=["date", "kind"])  # only streams that day has at all
    print(f"== minute files other days have and this day lacks: {gone.height}")
    if gone.height:
        print(gone.group_by("date", "kind").agg(pl.len().alias("n"), pl.col("minute").sort().str.join(" ").alias("minutes")).sort("date", "kind"))

    seqs = sorted(out.glob("seq_*.parquet"))
    if not seqs:
        return
    seq = pl.concat([pl.read_parquet(f) for f in seqs])
    tot = seq.group_by("date", "market_id").agg(pl.col("missing").sum(), pl.col("holes").sum(), pl.col("max_hole").max()).sort("date", "market_id")
    print("-- sequence holes per day and exchange (order + trade ids per channel)")
    print(tot.pivot(on="market_id", index="date", values=["missing", "holes", "max_hole"]).sort("date"))
    per = seq.group_by("date", "market_id", "minute").agg(pl.col("missing").sum(), pl.col("max_hole").max())
    alld = per.select("market_id", "minute").unique().join(seq.select("date").unique(), how="cross")
    per = alld.join(per, on=["date", "market_id", "minute"], how="left").fill_null(0)
    base = per.group_by("market_id", "minute").agg(pl.col("missing").median().alias("med_missing"))
    hot = (per.join(base, on=["market_id", "minute"])
           .filter(pl.col("missing") > excess * pl.col("med_missing") + 200).sort("date", "market_id", "minute"))
    print(f"== minutes with sequence holes > {excess:g} x the same minute's median over days + 200 ids: {hot.height}")
    if hot.height:
        files = (m.filter(pl.col("kind").is_in(["order", "transaction"]))
                 .group_by("date", "minute").agg(pl.format("{} ({}x)", "file", pl.col("ratio").round(2)).sort().str.join("  ").alias("source_files")))
        print(hot.join(files, on=["date", "minute"], how="left").head(60))
        hot.write_csv(out / "hot_minutes.csv")
    holes = pl.concat([pl.read_parquet(f) for f in sorted(out.glob("holes_*.parquet"))])
    print("-- the 15 largest single holes (minute = file before the hole, next_minute = file after it)")
    print(holes.sort("missing", descending=True).head(15))
    bad.write_csv(out / "bad_files.csv")
    print(f"tables: {out}/bad_files.csv, hot_minutes.csv")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, help="decoded tree (decode_feitu_day.py --out)")
    ap.add_argument("--date", help="day to check, YYYYMMDD")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", action="store_true", help="flag files over every checked day under --out")
    ap.add_argument("--low", type=float, default=0.5, help="row ratio under which a file is flagged")
    ap.add_argument("--excess", type=float, default=5.0, help="hole multiple of the minute's median that flags it")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.report:
        report(a.out, a.low, a.excess)
    elif a.raw and a.date:
        check_day(a.raw, a.date, a.out)
    else:
        ap.error("give --raw and --date, or --report")


if __name__ == "__main__":
    main()

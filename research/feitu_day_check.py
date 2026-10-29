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
  rows is the short one;
* **quotes against trades**: snapshots carry no sequence, but the exchange sends one on its ~3 s grid
  whenever a stock changed, and a trade changes it. So every second with a trade (the trade stream is
  complete by its sequence) needs a snapshot of that stock stamped at or after the trade within
  ``--cover`` seconds of exchange time. Slots are keyed by exchange time and repeated exchange times
  are collapsed: a late snapshot re-sent with the same exchange time but an updated price is not a
  loss, and is counted on its own (share, arrival delay, price changed);
* **index cadence**: indices are published on a fixed timer; per index the usual step between
  exchange times is measured from the day itself, and every step of k > 1.5 usual steps is k - 1
  missing points, charged to the minute it starts in;
* **arrival tiling**: feitu cuts minute files by arrival, and in session something arrives every
  fraction of a second, so the first / last arrival of each file must sit at its minute's edges; a
  file whose arrivals stop early lost its tail.

Flags compare a day with the same minute on the reference days: ``--base`` (days known good, e.g. the
feitu days already in the catalog, checked straight from ``feitu_raw``) or, without it, all days.

    python research/feitu_day_check.py --raw /work/crucible_data/feitu_check_raw --date 20260901 --out data/reports/feitu_day_check
    python research/feitu_day_check.py --out data/reports/feitu_day_check --report --base 20260922,20260923
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import polars as pl

KINDS = ("order", "transaction", "quotation", "index")
NS = 1_000_000_000


def sod_ms(col: str) -> pl.Expr:
    """CST milliseconds of day from an epoch-ns UTC column."""
    return ((pl.col(col) // 1_000_000 + 8 * 3_600_000) % 86_400_000).cast(pl.Int64)


def in_session(ms: pl.Expr) -> pl.Expr:
    """Continuous session: 09:30-11:30 and 13:00-14:57 CST."""
    return ms.is_between(34_200_000, 41_399_999) | ms.is_between(46_800_000, 53_819_999)


def hhmm(ms: pl.Expr) -> pl.Expr:
    """'HHMM' of a CST ms-of-day expression, the minute-file naming."""
    s = ms // 1000
    return pl.format("{}{}", (s // 3600).cast(pl.String).str.zfill(2), (s % 3600 // 60).cast(pl.String).str.zfill(2))


def _scan(raw: Path, kind: str, d: str) -> pl.LazyFrame | None:
    p = raw / f"kind={kind}" / f"date={d}"
    return pl.scan_parquet(str(p / "part-*.parquet")) if any(p.glob("part-*.parquet")) else None


def quote_cover(raw: Path, d: str, out: Path, cover_s: float) -> None:
    """Seconds with a trade but no snapshot of that stock within cover_s after it, per exchange and minute."""
    q, t = _scan(raw, "quotation", d), _scan(raw, "transaction", d)
    if q is None or t is None:
        print(f"{d}: quotation or transaction missing, quote cover skipped")
        return
    snaps = q.select("market_id", "symbol_id", sod_ms("time").alias("qms"), "spider_ts", "price").collect(engine="streaming")
    rep = (snaps.group_by("market_id", "symbol_id", "qms").agg(pl.len().alias("n"), (pl.col("spider_ts").max() - pl.col("spider_ts").min()).alias("spread"),
                                                               pl.col("price").n_unique().alias("prices"))
           .filter(pl.col("n") > 1))
    stat = (snaps.group_by("market_id").agg(pl.len().alias("snaps"))
            .join(rep.group_by("market_id").agg(pl.len().alias("dup_times"), (pl.col("spread").median() / 1e6).alias("dup_delay_ms_med"),
                                                (pl.col("prices") > 1).mean().alias("dup_price_changed")), on="market_id", how="left"))
    qs = snaps.select("market_id", "symbol_id", "qms").unique().sort("qms")
    tr = (t.filter(pl.col("trade_type") == 1).select("market_id", "symbol_id", sod_ms("time").alias("tms"))
          .filter(in_session(pl.col("tms"))).group_by("market_id", "symbol_id", (pl.col("tms") // 1000).alias("sec"))
          .agg(pl.col("tms").max().alias("t_last"))  # the last trade of the second: the snapshot must come after it
          .collect(engine="streaming").join(qs.select("market_id", "symbol_id").unique(), on=["market_id", "symbol_id"], how="semi")
          .sort("t_last"))
    with warnings.catch_warnings():  # sorted globally by time, which is what the by-groups need; polars only cannot verify it
        warnings.simplefilter("ignore")
        j = tr.join_asof(qs, left_on="t_last", right_on="qms", by=["market_id", "symbol_id"], strategy="forward")
    j = j.with_columns(((pl.col("qms") - pl.col("t_last")) / 1000).alias("gap_s"))
    j = j.with_columns((pl.col("gap_s").is_null() | (pl.col("gap_s") > cover_s)).alias("lost"), hhmm((pl.col("t_last") // 3000 + 1) * 3000).alias("minute"))  # the minute the snapshot was due in
    stat = stat.join(j.group_by("market_id").agg(pl.len().alias("trade_secs"), pl.col("lost").sum().alias("uncovered"),
                                                 pl.col("gap_s").quantile(0.5).alias("gap_p50"), pl.col("gap_s").quantile(0.99).alias("gap_p99"),
                                                 pl.col("gap_s").quantile(0.999).alias("gap_p999")), on="market_id", how="left")
    stat.with_columns(pl.lit(int(d)).alias("date")).write_parquet(out / f"qstat_{d}.parquet")
    (j.group_by("market_id", "minute").agg(pl.len().alias("trade_secs"), pl.col("lost").sum().alias("uncovered"))
     .with_columns(pl.lit(int(d)).alias("date")).write_parquet(out / f"qcov_{d}.parquet"))
    for r in stat.sort("market_id").iter_rows(named=True):
        print(f"{d}: quotes market {r['market_id']}: {r['snaps']:,} snapshots, {r['trade_secs'] or 0:,} trade-seconds, "
              f"{r['uncovered'] or 0:,} without a snapshot within {cover_s:g} s, gap p50/p99 {r['gap_p50']}/{r['gap_p99']} s, "
              f"repeated exchange times {r['dup_times'] or 0:,}")


def index_cadence(raw: Path, d: str, out: Path) -> None:
    """Missing points on each index's own publishing step, per minute."""
    x = _scan(raw, "index", d)
    if x is None:
        print(f"{d}: no index parts, cadence skipped")
        return
    ix = (x.select("market_id", "symbol_id", sod_ms("time").alias("ms")).filter(in_session(pl.col("ms"))).unique()
          .collect(engine="streaming").sort("market_id", "symbol_id", "ms"))
    ix = ix.with_columns(pl.col("ms").diff().over("market_id", "symbol_id").alias("step"),
                         (pl.col("ms") >= 46_800_000).alias("pm"))
    ix = ix.with_columns(pl.col("pm").shift(1).over("market_id", "symbol_id").alias("pm_prev"),
                         pl.col("ms").shift(1).over("market_id", "symbol_id").alias("ms_prev"))
    ix = ix.filter(pl.col("step").is_not_null() & (pl.col("pm") == pl.col("pm_prev")))  # no step across lunch
    mode = (ix.group_by("market_id", "symbol_id", ((pl.col("step") + 500) // 1000).alias("step_s")).agg(pl.len().alias("n"))
            .sort("n", descending=True).group_by("market_id", "symbol_id").first().select("market_id", "symbol_id", "step_s"))
    g = ix.join(mode, on=["market_id", "symbol_id"]).filter(pl.col("step_s") > 0)
    g = g.with_columns(pl.when(pl.col("step") > 1500 * pl.col("step_s")).then((pl.col("step") / (1000 * pl.col("step_s"))).round() - 1)
                       .otherwise(0).cast(pl.Int64).alias("missing"), hhmm(pl.col("ms_prev") + 1000 * pl.col("step_s")).alias("minute"))  # first missing point
    per = g.group_by("market_id", "minute").agg(pl.col("missing").sum(), pl.len().alias("points")).with_columns(pl.lit(int(d)).alias("date"))
    per.write_parquet(out / f"icad_{d}.parquet")
    modes = mode.group_by("market_id", "step_s").agg(pl.len().alias("indices")).sort("market_id", "step_s")
    print(f"{d}: index cadence: {mode.height} indices, step s x count {[tuple(r) for r in modes.iter_rows()]}, "
          f"{int(g['missing'].sum()):,} missing points")


def check_day(raw: Path, d: str, out: Path, cover_s: float) -> None:
    """Write man_<d> (one row per source file), seq_<d> / holes_<d> (sequence), qstat_<d> / qcov_<d> (quotes) and icad_<d> (index)."""
    t0 = time.perf_counter()
    man = []
    for k in KINDS:
        f = raw / f"kind={k}" / f"date={d}" / "_manifest.parquet"
        if not f.exists():
            print(f"{d}: no {k} manifest (stream absent or decode failed)")
            continue
        m = pl.read_parquet(f)
        for c, typ in (("error", pl.String), ("arrival_ts_min", pl.Int64), ("arrival_ts_max", pl.Int64)):
            if c not in m.columns:
                m = m.with_columns(pl.lit(None, typ).alias(c))
        man.append(m.select(pl.lit(int(d)).alias("date"), pl.lit(k).alias("kind"), "file", "minute", "bytes", "rows",
                            pl.col("error").cast(pl.String), pl.col("arrival_ts_min").cast(pl.Int64), pl.col("arrival_ts_max").cast(pl.Int64)))
    if man:
        man = pl.concat(man)
        man.write_parquet(out / f"man_{d}.parquet")
        print(f"{d}: {man.height} files, {man['rows'].sum():,} rows, {man['error'].is_not_null().sum()} decode errors")
    parts = [raw / f"kind={k}" / f"date={d}" for k in ("order", "transaction")]
    parts = [str(p / "part-*.parquet") for p in parts if any(p.glob("part-*.parquet"))]
    if not parts:
        print(f"{d}: no order / transaction parts, sequence skipped")
    else:
        sequence(parts, d, out)
    quote_cover(raw, d, out, cover_s)
    index_cadence(raw, d, out)
    print(f"{d}: done ({time.perf_counter() - t0:.0f}s)")


def sequence(parts: list[str], d: str, out: Path) -> None:
    """Holes in the order + trade id sequence of each channel, charged to the minute file after each hole."""
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


def _load(out: Path, prefix: str) -> pl.DataFrame | None:
    fs = sorted(out.glob(f"{prefix}_*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed") if fs else None


def flag(per: pl.DataFrame, keys: list[str], val: str, ref: list[int], excess: float, add: float) -> pl.DataFrame:
    """Rows of per (date x keys, absent = 0) whose val exceeds excess x its median over the reference days + add."""
    full = (per.select(keys).unique().join(per.select("date").unique(), how="cross")
            .join(per, on=["date", *keys], how="left").fill_null(0))
    r = full.filter(pl.col("date").is_in(ref)) if ref else full
    med = r.group_by(keys).agg(pl.col(val).median().alias("ref_med"))
    return full.join(med, on=keys).filter(pl.col(val) > excess * pl.col("ref_med") + add).sort("date", *keys)


def files_of(m: pl.DataFrame, kinds: list[str]) -> pl.DataFrame:
    """Per (date, minute): the source files of kinds with their row ratio to the reference median."""
    return (m.filter(pl.col("kind").is_in(kinds)).group_by("date", "minute")
            .agg(pl.format("{} ({}x)", "file", pl.col("ratio").round(2)).sort().str.join("  ").alias("source_files")))


def report(out: Path, low: float, excess: float, edge: float, ref: list[int]) -> None:
    """Flag source files over all checked days."""
    man = _load(out, "man")
    if man is None:
        raise SystemExit(f"no man_*.parquet under {out}")
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(230)
    pl.Config.set_fmt_str_lengths(110)
    isref = pl.col("date").is_in(ref) if ref else pl.lit(True)
    print(f"== {man['date'].n_unique()} days checked: {man['date'].min()} .. {man['date'].max()}; reference: {ref or 'all days'}")
    print("-- files and rows per day and stream (rows in millions)")
    print(man.group_by("date", "kind").agg(pl.len().alias("files"), (pl.col("rows").sum() / 1e6).round(2).alias("mrows"))
          .pivot(on="kind", index="date", values=["files", "mrows"]).sort("date"))

    # rows per file
    med = man.filter(isref).group_by("kind", "minute").agg(pl.col("rows").median().alias("med_rows"), pl.len().alias("ref_days"))
    m = man.join(med, on=["kind", "minute"], how="left").with_columns((pl.col("rows") / pl.col("med_rows")).alias("ratio"))
    bad = m.filter(pl.col("error").is_not_null() | ((pl.col("rows") == 0) & (pl.col("med_rows") > 0))
                   | ((pl.col("ratio") < low) & (pl.col("med_rows") > 1000) & (pl.col("ref_days") >= 3)))
    print(f"== 1 files that failed to decode, decoded empty, or hold < {low:g} x the reference rows of that minute: {bad.height}")
    if bad.height:
        print(bad.sort("date", "kind", "minute").select("date", "kind", "file", "bytes", "rows", "med_rows", "ratio", "error"))
    bad.write_csv(out / "bad_files.csv")
    gone = (man.filter(isref).select("kind", "minute").unique().join(man.select("date", "kind").unique(), on="kind")
            .join(man.select("date", "kind", "minute"), on=["date", "kind", "minute"], how="anti"))
    print(f"== 2 minute files the reference days have and a day lacks: {gone.height}")
    if gone.height:
        print(gone.group_by("date", "kind").agg(pl.len().alias("n"), pl.col("minute").sort().str.join(" ").alias("minutes")).sort("date", "kind"))

    # arrival tiling: first / last arrival against the minute's edges, in session
    t = (m.filter(pl.col("minute").str.contains(r"^\d{4}$") & pl.col("arrival_ts_min").is_not_null())
         .with_columns((pl.col("minute").str.slice(0, 2).cast(pl.Int64) * 3600 + pl.col("minute").str.slice(2, 2).cast(pl.Int64) * 60).alias("start"))
         .filter(pl.col("start").is_between(34200, 41340) | pl.col("start").is_between(46800, 53760))
         .with_columns((sod_ms("arrival_ts_min") / 1000 - pl.col("start")).alias("head_s"),
                       (pl.col("start") + 60 - sod_ms("arrival_ts_max") / 1000).alias("tail_s")))
    if t.height:
        print("-- 3 arrival tiling on the reference days: seconds from a minute's start to its first arrival (head), from its last arrival to its end (tail)")
        print(t.filter(isref).group_by("kind").agg(*[pl.col(c).quantile(q).round(3).alias(f"{c}_{n}") for c in ("head_s", "tail_s")
                                                     for q, n in ((0.5, "p50"), (0.99, "p99"), (1.0, "max"))]).sort("kind"))
        tm = t.filter(isref).group_by("kind", "minute").agg(pl.col("head_s").median().alias("head_ref"), pl.col("tail_s").median().alias("tail_ref"))
        cut = (t.join(tm, on=["kind", "minute"], how="left")
               .filter((pl.col("head_s") > pl.col("head_ref").fill_null(0) + edge) | (pl.col("tail_s") > pl.col("tail_ref").fill_null(0) + edge)))
        print(f"== 3 files starting late or ending early by > {edge:g} s beyond the reference: {cut.height}")
        if cut.height:
            print(cut.sort("date", "kind", "minute").select("date", "kind", "file", "rows", "ratio", "head_s", "tail_s").head(60))
        cut.write_csv(out / "cut_files.csv")

    # order / trade sequence
    seq = _load(out, "seq")
    if seq is not None:
        tot = seq.group_by("date", "market_id").agg(pl.col("missing").sum(), pl.col("holes").sum(), pl.col("max_hole").max())
        print("-- 4 sequence holes per day and exchange (order + trade ids per channel)")
        print(tot.pivot(on="market_id", index="date", values=["missing", "holes", "max_hole"]).sort("date"))
        hot = flag(seq.group_by("date", "market_id", "minute").agg(pl.col("missing").sum(), pl.col("max_hole").max()),
                   ["market_id", "minute"], "missing", ref, excess, 200)
        print(f"== 4 minutes with sequence holes > {excess:g} x the reference median + 200 ids: {hot.height}")
        if hot.height:
            print(hot.join(files_of(m, ["order", "transaction"]), on=["date", "minute"], how="left").head(60))
        hot.write_csv(out / "hot_minutes.csv")
        holes = _load(out, "holes")
        print("-- the 10 largest single holes (minute = file before the hole, next_minute = file after it)")
        print(holes.sort("missing", descending=True).head(10))

    # quotes against trades
    qs = _load(out, "qstat")
    if qs is not None:
        print("-- 5 quotes vs trades per day: trade-seconds without a snapshot within the cover window; repeated exchange times (late re-sends)")
        print(qs.with_columns((pl.col("uncovered") / pl.col("trade_secs")).alias("unc_share"), (pl.col("dup_times") / pl.col("snaps")).alias("dup_share"))
              .select("date", "market_id", "snaps", "trade_secs", "uncovered", "unc_share", "gap_p50", "gap_p99", "gap_p999",
                      "dup_share", "dup_delay_ms_med", "dup_price_changed").sort("date", "market_id"))
        hq = flag(_load(out, "qcov"), ["market_id", "minute"], "uncovered", ref, excess, 20)
        print(f"== 5 minutes with trade-seconds lacking a snapshot > {excess:g} x the reference median + 20: {hq.height}")
        if hq.height:
            print(hq.join(files_of(m, ["quotation"]), on=["date", "minute"], how="left").head(60))
        hq.write_csv(out / "hot_quote_minutes.csv")

    # index cadence
    ic = _load(out, "icad")
    if ic is not None:
        print("-- 6 index points missing from each index's own step, per day")
        print(ic.group_by("date", "market_id").agg(pl.col("missing").sum(), pl.col("points").sum())
              .pivot(on="market_id", index="date", values=["missing", "points"]).sort("date"))
        hi = flag(ic, ["market_id", "minute"], "missing", ref, excess, 10)
        print(f"== 6 minutes with missing index points > {excess:g} x the reference median + 10: {hi.height}")
        if hi.height:
            print(hi.join(files_of(m, ["index"]), on=["date", "minute"], how="left").head(60))
        hi.write_csv(out / "hot_index_minutes.csv")
    print(f"tables: {out}/bad_files.csv, cut_files.csv, hot_minutes.csv, hot_quote_minutes.csv, hot_index_minutes.csv")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, help="decoded tree (decode_feitu_day.py --out)")
    ap.add_argument("--date", help="day to check, YYYYMMDD")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", action="store_true", help="flag files over every checked day under --out")
    ap.add_argument("--low", type=float, default=0.5, help="row ratio under which a file is flagged")
    ap.add_argument("--excess", type=float, default=5.0, help="hole multiple of the minute's median that flags it")
    ap.add_argument("--cover", type=float, default=6.0, help="seconds after a trade within which a snapshot of the stock must be stamped")
    ap.add_argument("--edge", type=float, default=5.0, help="arrival tiling: seconds a file may start late / end early in session")
    ap.add_argument("--base", default="", help="comma list of known-good days the others are compared with (default: all days)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.report:
        report(a.out, a.low, a.excess, a.edge, [int(x) for x in a.base.split(",") if x.strip()])
    elif a.raw and a.date:
        check_day(a.raw, a.date, a.out, a.cover)
    else:
        ap.error("give --raw and --date, or --report")


if __name__ == "__main__":
    main()

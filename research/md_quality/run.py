"""Run the market-data quality standard on one decoded day of one source.

Reads ``<raw>/kind=<k>/date=<d>/`` (written by ``research/decode_feitu_day.py``), runs every
check in :mod:`quarry.quality`, prints the report worst-first, and writes
``<out>/date=<d>/source=<s>/checks.parquet`` plus a one-line ``verdict.json``.

Usage (WSL)::

    python research/md_quality/run.py --date 20260615
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.resmon import ResourceMonitor, append_history

from quarry.feitu import normalize_columns
from quarry.quality import (
    CheckResult,
    Level,
    check_book_sanity,
    check_daily_totals,
    check_latency,
    check_minute_coverage,
    check_sequence,
    check_trade_prices,
    check_trade_symbol_coverage,
    check_volume_identity,
    load_thresholds,
    results_frame,
    skipped,
    verdict,
)
from quarry.raw import A_SHARE_PREFIXES

REPO = Path(__file__).resolve().parents[2]


def run(raw: Path, date: str, th_path: Path) -> list[CheckResult]:
    """All checks for one day; streams absent from the dump become SKIP / FAIL, never silence."""
    th = load_thresholds(th_path)
    res: list[CheckResult] = []
    frames: dict[str, pl.LazyFrame] = {}
    for kind in ("order", "transaction", "quotation", "index"):
        d = raw / f"kind={kind}" / f"date={date}"
        man = d / "_manifest.parquet"
        if not man.exists() or not any(d.glob("part-*.parquet")):
            res.append(
                CheckResult(
                    "completeness.stream_present",
                    kind,
                    Level.FAIL,
                    0.0,
                    None,
                    1.0,
                    0,
                    "no decoded files for this stream",
                )
            )
            continue
        res.extend(check_minute_coverage(pl.read_parquet(man), kind, th))
        frames[kind] = normalize_columns(pl.scan_parquet(d / "part-*.parquet"), kind)

    for kind, lf in frames.items():
        res.extend(check_latency(lf, kind, th))

    o, t, q = frames.get("order"), frames.get("transaction"), frames.get("quotation")
    if o is not None and t is not None:
        res.extend(check_sequence(o, t, th))
    else:
        res.append(skipped("sequence.missing_frac", "all", "needs both order and transaction"))
    if q is not None and t is not None:
        res.extend(check_trade_symbol_coverage(q, t, th))
        res.extend(check_volume_identity(q, t, th))
        res.extend(check_trade_prices(q, t, th))
    else:
        res.append(
            skipped(
                "consistency.volume_mismatch_frac", "all", "needs both quotation and transaction"
            )
        )
    if q is not None:
        qdir = raw / "kind=quotation" / f"date={date}"
        per_file = (
            normalize_columns(pl.scan_parquet(f), "quotation")
            for f in sorted(qdir.glob("part-*.parquet"))
        )
        res.extend(check_book_sanity(per_file, th))
    res.append(
        skipped(
            "consistency.book_vs_mbo_match",
            "all",
            "needs the MBO order-book replay (stage 7 kernel)",
        )
    )
    tdx = PUBLIC_CACHE / "tdx_daily" / f"date={date}" / "part.parquet"
    if t is not None and tdx.exists():
        res.extend(check_daily_totals(t, load_tdx_reference(tdx), "tdx_eod", th))
    else:
        why = "no transaction stream" if t is None else f"no TDX package cached at {tdx}"
        res.append(skipped("multisource.daily_volume_mismatch_frac", "tdx_eod", why))
    res.append(
        skipped(
            "multisource.daily_totals_vs_wind",
            "all",
            "needs WindPy cache (terminal not logged in yet)",
        )
    )
    return res


PUBLIC_CACHE = REPO / "data" / "public_cache"


def load_tdx_reference(path: Path) -> pl.DataFrame:
    """TDX official end-of-day package -> reference frame for :func:`check_daily_totals`."""
    df = pl.read_parquet(path).filter(pl.col("market").is_in(["sh", "sz"]))
    exch = pl.when(pl.col("market") == "sh").then(pl.lit("XSHG")).otherwise(pl.lit("XSHE"))
    stock = pl.lit(False)
    for venue, prefixes in A_SHARE_PREFIXES.items():
        stock = stock | ((exch == venue) & pl.col("code").str.slice(0, 3).is_in(prefixes))
    return df.select(
        exch.alias("exchange"),
        pl.col("code").alias("symbol"),
        stock.alias("is_a_share"),
        pl.col("volume").cast(pl.Int64),
        pl.col("amount").cast(pl.Float64),
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--source", default="feitu_v3")
    ap.add_argument("--raw", type=Path, default=Path("/work/crucible_data/feitu_raw"))
    ap.add_argument("--out", type=Path, default=Path("/work/crucible_data/quality"))
    ap.add_argument("--thresholds", type=Path, default=REPO / "cfg" / "md_quality.toml")
    a = ap.parse_args()

    with ResourceMonitor("W0.quality", params={"date": a.date, "source": a.source}) as mon:
        results = run(a.raw, a.date, a.thresholds)
        mon.rows = len(results)
    assert mon.stats is not None
    append_history(mon.stats, a.out.parent / "bench" / "history.parquet")

    df = results_frame(results)
    v = verdict(results)
    dst = a.out / f"date={a.date}" / f"source={a.source}"
    dst.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dst / "checks.parquet")
    (dst / "verdict.json").write_text(
        json.dumps({"date": a.date, "source": a.source, "verdict": v.value})
    )
    with pl.Config(
        tbl_rows=200, tbl_width_chars=220, fmt_str_lengths=90, tbl_hide_dataframe_shape=True
    ):
        print(df.select("level", "check", "scope", "value", "warn_at", "fail_at", "n", "detail"))
    print(f"VERDICT {a.date} {a.source}: {v.value}")
    print(mon.stats.line())


if __name__ == "__main__":
    main()

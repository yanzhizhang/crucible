"""Step 1 -- load a prism dump and align it.

Demonstrates the two things almanac exists to prevent: a stale suspension price
reaching a factor, and today's index constituents being used historically.

Run: ``uv run python examples/01_load_and_align.py``
"""

from __future__ import annotations

import polars as pl

from _bootstrap import open_store, rule
from almanac.calendar import TradingCalendar
from almanac.masks import apply_masks, build_masks
from almanac.universe import Membership, Universe
from quarry.loaders import dump_fingerprint, load_daily, load_factor_frame

conn, truth = open_store()

rule("Calendar: sessions and the 3-second slot grid")
cal = TradingCalendar()
day = cal.sessions("2024-01-02", "2024-01-31")[0]
print(f"first session          : {day}")
print(f"1-minute slots         : {len(cal.slot_grid(day, '1min'))}")
print(f"3-second slots         : {len(cal.slot_grid(day, '3s'))}")
print(f"first / last label     : {cal.slot_grid(day, '1min')[0]} .. {cal.slot_grid(day, '1min')[-1]}")
print("labels are right-edged: 09:31 covers (09:30, 09:31]")

rule("Producer contract")
fp = dump_fingerprint(conn, "factor_frame")
print(f"factors in dump        : {list(fp.names)}")
print(f"digest                 : {fp.digest()[:16]}...")
factors = load_factor_frame(conn, expected=fp)
print(f"loaded                 : {factors.height} rows (fingerprint matched)")

rule("Masks: the stale-price trap")
daily = load_daily(conn)
masks = build_masks(daily)

sym = next(s for s, days in truth.suspensions.items() if days)
susp = truth.suspensions[sym][0]
raw = daily.filter((pl.col("symbol") == sym) & (pl.col("ts").dt.date() == susp))
print(f"{sym} suspended {susp}")
print(f"  raw close / prev_close : {raw['close'][0]:.2f} / {raw['prev_close'][0]:.2f}  <- identical")
print(f"  raw volume             : {raw['volume'][0]:.0f}")

masked = apply_masks(daily, masks, columns=["close", "vwap", "volume"])
out = masked.filter((pl.col("symbol") == sym) & (pl.col("ts").dt.date() == susp))
print(f"  masked close           : {out['close'][0]}  <- null, not a fake 0% return")
print(f"\ngrid preserved: {masked.height} rows in, {daily.height} out")

print("\nmask coverage by flag (first 3 sessions):")
print(masks.coverage().head(3))

rule("Point-in-time universe")
members = Membership.from_frame(conn.execute("SELECT * FROM membership").pl())
listings = conn.execute("SELECT * FROM listings").pl()
u = Universe(membership=members, listings=listings)
print(f"CSI 300 members on {truth.dates[0]}: {len(u.at(truth.dates[0], '000300.SH'))}")
print(f"whole listed universe        : {len(u.at(truth.dates[0], 'all'))}")
print("\nA membership table without effective dates is refused at construction,")
print("so today's constituents can never be applied to a historical date.")

conn.close()

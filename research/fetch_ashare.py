"""Fetch real A-share daily bars into a raw CSV cache. Stdlib only.

Deliberately dependency-free so it can run before the project venv exists, and
so a network fetch never drags research code into the import graph.

Sources and why each
--------------------
* **THS** ``d.10jqka.com.cn/v6/line/hs_{code}/01/{year}.js`` -- unadjusted OHLC
  plus **amount** and **turnover%**, which Tencent's kline does not carry and
  which Amihud illiquidity and turnover factors need. Full history per year file.
* **Tencent** ``fqkline ... qfq`` -- forward-adjusted closes. Used only to
  *derive* a backward adjustment factor (see below), never as the price itself.
* **Eastmoney** ``clist`` -- one request for the universe with market cap and
  industry.

Field orders were confirmed empirically (``_probe_api.py``), not assumed:

    THS     : date, open, high,  low, close, volume, amount, turnover%
    Tencent : date, open, close, high, low,  volume          <- close is 3rd

Getting these backwards transposes high/low and corrupts every volatility
factor without raising anything.

Adjustment
----------
Tencent's qfq series is anchored at *today*, so ``qfq_t / raw_t`` is the
cumulative forward-adjustment ratio. Dividing by that ratio on the first date
converts it to a **backward** factor, which is the point-in-time-safe direction
(see :mod:`almanac.adjust`). Raw prices are kept alongside, because masks,
price limits and costs are all defined on the traded price.

Known bias
----------
The universe is "today's largest N by market cap", which is **selection bias**:
these are the companies that grew into the top by the end date. crucible's
:class:`almanac.Universe` exists to prevent exactly this, but point-in-time
index membership is not available from these free endpoints. Results are
therefore optimistic in level; relative comparisons between factors on the same
universe remain informative, which is what this study uses them for.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode Chinese company names
# and kills the run mid-fetch with a UnicodeEncodeError -- after the network
# work is already done. Force UTF-8 on the streams before printing anything.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/117.0.0.0 Safari/537.36"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE = RAW / "cache"

START_YEAR = 2024
"""First calendar year of history to pull. ~2.5 years through today."""


def _get(url: str, referer: str = "", encoding: str = "utf-8", tries: int = 3) -> str:
    """GET with retries and a polite delay. Returns "" on persistent failure."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", UA)
            if referer:
                req.add_header("Referer", referer)
            with urllib.request.urlopen(req, timeout=25) as r:
                return r.read().decode(encoding, "ignore")
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                print(f"    ! {type(exc).__name__} {str(exc)[:70]}", file=sys.stderr)
                return ""
            time.sleep(1.5 * (attempt + 1))
    return ""


def _prefix(code: str) -> str:
    """Market prefix. 92x is Beijing and must be tested before the 9x branch."""
    if code.startswith("92"):
        return "bj"
    if code.startswith(("5", "6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


# ---------------------------------------------------------------- universe


def fetch_universe(n: int = 80) -> list[dict[str, str]]:
    """Top ``n`` mainland stocks by total market cap, with industry.

    One request. ``fid=f20`` sorts by total market cap descending; ``f100`` is
    the Eastmoney industry name.
    """
    url = (
        "https://push2.eastmoney.com/api/qt/clist/get"
        f"?pn=1&pz={n}&po=1&np=1&fltt=2&invt=2&fid=f20"
        "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
        "&fields=f12,f14,f20,f100,f3"
    )
    txt = _get(url, "https://quote.eastmoney.com/")
    if not txt:
        return []
    items = (json.loads(txt).get("data") or {}).get("diff") or []
    if isinstance(items, dict):
        items = list(items.values())
    out = []
    for it in items:
        code = str(it.get("f12", ""))
        # ST names and Beijing exchange are excluded: 5% limit bands and thin
        # liquidity make them a different statistical population, and mixing
        # them in muddies every cross-sectional estimate.
        if not code or code.startswith(("4", "8", "92")):
            continue
        out.append(
            {
                "symbol": code,
                "name": str(it.get("f14", "")),
                "industry": str(it.get("f100", "") or "unknown"),
                "mcap": str(it.get("f20", 0) or 0),
            }
        )
    return out


# ---------------------------------------------------------------- bars


def fetch_ths_year(code: str, year: str) -> list[list[str]]:
    """One THS year file. ``year`` is "2024" etc., or "last" for the current partial.

    Row layout (confirmed): ``date, open, high, low, close, volume, amount, turnover%``
    """
    url = f"https://d.10jqka.com.cn/v6/line/hs_{code}/01/{year}.js"
    txt = _get(url, "https://stockpage.10jqka.com.cn/")
    if not txt or "(" not in txt:
        return []
    try:
        obj = json.loads(txt[txt.index("(") + 1 : txt.rindex(")")])
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in (obj.get("data") or "").split(";"):
        parts = line.split(",")
        if len(parts) >= 8 and len(parts[0]) == 8:
            rows.append(parts[:8])
    return rows


def fetch_tencent_qfq(code: str, start: str, end: str) -> dict[str, float]:
    """Forward-adjusted closes keyed by ``YYYYMMDD``.

    Row layout (confirmed): ``date, open, close, high, low, volume`` -- note
    close sits at index 2, unlike THS.
    """
    sym = _prefix(code) + code
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={sym},day,{start},{end},900,qfq"
    )
    txt = _get(url, "https://gu.qq.com/")
    if not txt:
        return {}
    try:
        node = json.loads(txt)["data"][sym]
    except Exception:  # noqa: BLE001
        return {}
    rows = node.get("qfqday") or node.get("day") or []
    out = {}
    for r in rows:
        if len(r) >= 5:
            out[r[0].replace("-", "")] = float(r[2])  # index 2 == close
    return out


def fetch_symbol(code: str, years: list[str]) -> list[dict[str, object]]:
    """All bars for one symbol, with a derived backward adjustment factor."""
    cache_file = CACHE / f"{code}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass

    raw: list[list[str]] = []
    for y in years:
        raw.extend(fetch_ths_year(code, y))
        time.sleep(0.25 + random.uniform(0, 0.2))
    if not raw:
        return []

    raw.sort(key=lambda r: r[0])
    seen: set[str] = set()
    deduped = []
    for r in raw:
        if r[0] not in seen:
            seen.add(r[0])
            deduped.append(r)
    raw = deduped

    lo, hi = raw[0][0], raw[-1][0]
    qfq = fetch_tencent_qfq(code, f"{lo[:4]}-{lo[4:6]}-{lo[6:]}", f"{hi[:4]}-{hi[4:6]}-{hi[6:]}")
    time.sleep(0.25 + random.uniform(0, 0.2))

    # Backward adjustment: qfq/raw is the cumulative FORWARD ratio (anchored at
    # today). Normalising by its value on the first date re-anchors it at the
    # start, which is the point-in-time-safe direction.
    base = None
    for r in raw:
        q = qfq.get(r[0])
        if q and float(r[4]) > 0:
            base = q / float(r[4])
            break

    out = []
    for r in raw:
        date, o, h, low, c, vol, amt, turn = r[:8]
        try:
            c_f = float(c)
            if c_f <= 0:
                continue
            q = qfq.get(date)
            adj = (q / c_f) / base if (q and base) else 1.0
            out.append(
                {
                    "date": date,
                    "symbol": code,
                    "open": float(o),
                    "high": float(h),
                    "low": float(low),
                    "close": c_f,
                    "volume": float(vol or 0),
                    "amount": float(amt or 0),
                    "turnover": float(turn or 0),
                    "adj_factor": round(adj, 8),
                }
            )
        except ValueError:
            continue

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(out), encoding="utf-8")
    return out


def main(n_symbols: int = 80) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    # Reuse a previously fetched universe. Eastmoney's push2 applies IP-level
    # throttling and refuses repeat calls (RemoteDisconnected); re-requesting a
    # list we already have is both rude and a good way to get the whole session
    # blocked for hours.
    uni_path = RAW / "universe.csv"
    universe: list[dict[str, str]] = []
    if uni_path.exists():
        with uni_path.open(encoding="utf-8") as f:
            universe = list(csv.DictReader(f))[:n_symbols]
        print(f"reusing cached universe: {len(universe)} symbols")
    else:
        print(f"fetching universe (top {n_symbols} by market cap) ...")
        universe = fetch_universe(n_symbols)
        if not universe:
            print("FAILED: could not fetch universe", file=sys.stderr)
            raise SystemExit(1)
        print(f"  got {len(universe)} symbols")
        with uni_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["symbol", "name", "industry", "mcap"])
            w.writeheader()
            w.writerows(universe)

    this_year = time.strftime("%Y")
    years = [str(y) for y in range(START_YEAR, int(this_year))] + ["last"]
    print(f"year files per symbol: {years}")

    all_rows: list[dict[str, object]] = []
    ok = 0
    for i, u in enumerate(universe, 1):
        code = u["symbol"]
        rows = fetch_symbol(code, years)
        if rows:
            ok += 1
            all_rows.extend(rows)
        status = f"{len(rows):>4} bars" if rows else "  NO DATA"
        print(f"  [{i:>3}/{len(universe)}] {code} {u['name'][:8]:<10} {status}")

    if not all_rows:
        print("FAILED: no bars fetched", file=sys.stderr)
        raise SystemExit(1)

    out = RAW / "bars.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "date", "symbol", "open", "high", "low", "close",
                "volume", "amount", "turnover", "adj_factor",
            ],
        )
        w.writeheader()
        w.writerows(all_rows)

    dates = sorted({r["date"] for r in all_rows})
    print()
    print(f"wrote {out}")
    print(f"  {len(all_rows):,} rows | {ok} symbols | {len(dates)} sessions")
    print(f"  {dates[0]} .. {dates[-1]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)

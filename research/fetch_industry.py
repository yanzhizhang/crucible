"""Fetch an A-share industry classification snapshot into the raw cache. Stdlib only.

Why this file exists
--------------------
The classifications a desk actually runs on -- **Wind** (``wind_ind``) and
**CITIC** (``中信一/二/三级``) -- are licensed products. They are on the
intranet and there is no external endpoint for them, point-in-time or
otherwise. Nothing here can or should pretend to reproduce them.

What is reachable from outside is Eastmoney's own industry board
classification (~500 boards, one level, exhaustive over listed A-shares). That
is enough to build and validate the *pipeline* -- neutralisation, industry
caps, exposure attribution -- so that when the real Wind/CITIC table arrives it
drops into the same loader with a different ``scheme`` and nothing downstream
changes.

Point-in-time
-------------
This endpoint returns **current** membership only. A single snapshot applied
across history is exactly the survivorship/reclassification lie
:mod:`almanac.universe` refuses for index membership, and it is just as wrong
here: an issuer moved from ``机械设备`` to ``汽车`` in 2023 would look like it
was always ``汽车``.

So a snapshot is never written as history. Each run writes one dated file:

    data/raw/industry/scheme=<scheme>/asof=YYYYMMDD/snapshot.csv

and :func:`almanac.industry.classification_from_snapshots` turns an accumulated
series of them into effective-dated intervals. Run it on a schedule; the
history becomes usable as it accrues. Until two snapshots exist, every interval
is open-ended from the first ``asof`` and that is stated, not hidden.

Snapshot schema (long, one row per symbol x level)
--------------------------------------------------
    asof            date     the observation date -- when this was *seen*
    scheme          str      classification system id, e.g. "em"
    symbol          str      6-digit code, VARCHAR always
    level           int      1 = coarsest; Wind/CITIC will use 1..3 or 1..4
    industry_code   str      stable code within the scheme (Eastmoney: BKxxxx)
    industry_name   str      display name at that level
"""

from __future__ import annotations

import csv
import datetime as dt
import http.client
import json
import random
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/117.0.0.0 Safari/537.36"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "industry"

CLIST = "https://push2.eastmoney.com/api/qt/clist/get"
SHARDS = 99
"""push2 is served from numbered shards ``1..99.push2.eastmoney.com``. A given
request is dropped mid-connection often enough (measured ~2 in 3 from here,
independent of pacing) that a retry has to land on a different shard to be
worth making."""
MARKET_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
"""Whole-market selector: SSE main, SSE STAR, SZSE main, SZSE ChiNext, BSE."""
REFERER = "https://quote.eastmoney.com/"

SCHEME = "em"
"""Scheme id written into the snapshot. The real dumps will use "citic"/"wind"."""

LEVEL = 1
"""Eastmoney exposes a single flat board level; Wind/CITIC will populate 1..N."""

PAGE = 100
"""Rows per clist page. The server caps ``pz`` here whatever is asked for."""

COOL_OFF = 120.0
"""Seconds to wait out a blackout window before retrying a page."""

MIN_INTERVAL = 1.0
"""Seconds between Eastmoney requests, retries included. Their WAF bans an IP
for hours past roughly 200 requests/minute; one snapshot is ~65 pages, so even
with retries this stays well inside that."""

_last_call = 0.0


def _request(url: str, host: str, timeout: float = 20.0) -> str:
    """Single GET against one push2 shard."""
    parts = urlsplit(url)
    path = parts.path + ("?" + parts.query if parts.query else "")
    conn = http.client.HTTPSConnection(host, timeout=timeout)
    try:
        conn.request(
            "GET",
            path,
            headers={"User-Agent": UA, "Referer": REFERER, "Accept": "*/*"},
        )
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", "replace")
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} from {host}{path}")
        return body
    finally:
        conn.close()


def _get(url: str, tries: int = 20, rounds: int = 6) -> str:
    """Throttled GET, retried across shards. Raises once the retries run out.

    Deliberately not "returns empty on failure". A dropped connection and an
    empty body are indistinguishable from a genuinely empty page -- and an
    empty page here means "this board has no members" or "the market has 100
    industries". A transport failure has to reach the caller as a failure.

    Two tiers, because the failures come in two shapes. Individually dropped
    connections are frequent and independent -- roughly two in three from here,
    unaffected by pacing -- and ``tries`` rapid shard-rotating attempts get
    through those. Blackout windows of a minute or more also happen, and no
    number of immediate retries survives one; ``rounds`` waits them out.
    """
    global _last_call
    last: Exception | None = None
    for rnd in range(rounds):
        if rnd:
            print(f"    ...{tries} attempts failed, cooling off {COOL_OFF:.0f}s", flush=True)
            time.sleep(COOL_OFF)
        for _ in range(tries):
            wait = MIN_INTERVAL - (time.time() - _last_call) + random.uniform(0.1, 0.4)
            if wait > 0:
                time.sleep(wait)
            host = f"{random.randint(1, SHARDS)}.push2.eastmoney.com"
            try:
                body = _request(url, host)
                if body.strip():
                    return body
                last = RuntimeError("empty body")
            except Exception as exc:
                last = exc
            finally:
                _last_call = time.time()
    raise RuntimeError(
        f"GET failed after {rounds} rounds x {tries} tries: {url}\n  last error: {last!r}"
    )


def _page(fs: str, pn: int, fields: str = "f12,f14") -> tuple[list[dict[str, object]], int]:
    """One clist page: its rows and the server's declared total.

    ``diff`` comes back as a list on most pages and as a ``{"0": {...}}`` dict
    on some. Iterating the dict directly yields its *keys* -- strings -- and
    every field read then returns empty without raising.
    """
    url = f"{CLIST}?pn={pn}&pz={PAGE}&po=0&np=1&fltt=2&invt=2&fid=f12&fs={fs}&fields={fields}"
    body = _get(url)
    try:
        data = json.loads(body).get("data")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"clist {fs!r} page {pn} returned non-JSON: {body[:200]!r}") from exc
    if not data:
        return [], 0
    diff = data.get("diff") or []
    if isinstance(diff, dict):
        diff = list(diff.values())
    return [d for d in diff if isinstance(d, dict)], int(data.get("total") or 0)


def _clist_all(fs: str, fields: str = "f12,f14") -> list[dict[str, object]]:
    """Every row for a clist selector, paged.

    ``pz`` is capped server-side at 100 regardless of what is asked for: a
    request for 500 returns 100 rows and a ``total`` of 496. Trusting the
    single page is how a 496-board classification silently becomes 100 boards
    and four fifths of the market loses its industry.

    Paging is ordered by ``f12`` (the code) ascending, not by the default
    ``f3`` (change %). Sorting by a *live* quantity means the ranking moves
    between page requests -- minutes apart at this pacing -- so some rows come
    back twice and others never come back at all. That is how the 496-board
    list lost ``风电设备`` on an earlier run while still looking complete.
    The unique-count check below is the backstop.
    """
    rows, total = _page(fs, 1, fields)
    pn = 2
    print(f"  {fs}: {len(rows)}/{total}", flush=True)
    while total and len(rows) < total:
        more, _ = _page(fs, pn, fields)
        if not more:
            raise RuntimeError(
                f"clist {fs!r} stopped at {len(rows)}/{total} rows on page {pn}. "
                "A partial fetch written as a snapshot becomes permanent history."
            )
        rows += more
        pn += 1
        print(f"  {fs}: {len(rows)}/{total}", flush=True)

    seen: dict[str, dict[str, object]] = {}
    for r in rows:
        code = str(r.get("f12", ""))
        if code:
            seen[code] = r
    if total and len(seen) != total:
        raise RuntimeError(
            f"clist {fs!r} returned {len(rows)} rows covering {len(seen)} distinct codes, "
            f"but the server declares {total}. Pagination lost or duplicated rows."
        )
    return list(seen.values())


def fetch_boards() -> list[dict[str, str]]:
    """Return the industry board list: ``[{"code": "BK0475", "name": "银行"}, ...]``."""
    boards = [
        {"code": str(d.get("f12", "")), "name": str(d.get("f14", ""))}
        for d in _clist_all("m:90+t:2")
    ]
    boards = [b for b in boards if b["code"] and b["name"]]
    if not boards:
        raise RuntimeError(
            "Eastmoney returned no industry boards. Either the endpoint changed or this "
            "IP is rate-limited -- both are failures, not an empty classification."
        )
    return boards


def fetch_stocks() -> list[dict[str, str]]:
    """Every listed A-share with its industry name, from the whole-market list.

    ``f100`` carries the industry board name. Reading it off the market list is
    one page per 100 symbols -- ~60 requests -- against ~500 for walking each
    board's constituents. Same classification, an eighth of the exposure to a
    WAF ban, and the ban is what actually stops this script.

    The selector covers SSE main + STAR, SZSE main + ChiNext, and BSE. A
    missing market segment would silently produce an incomplete snapshot, so
    the caller checks the count against the server's ``total``.

    Eastmoney writes ``"-"`` for a symbol it has not classified -- observed on
    a handful of names, typically just-listed or long-suspended. Those are
    returned with an empty ``industry_name`` for the caller to drop, rather
    than silently mapped to a board named ``"-"``, which would become a
    real-looking industry with real-looking members.
    """
    rows = _clist_all(MARKET_FS, "f12,f14,f100")
    return [
        {
            "symbol": str(d.get("f12", "")).zfill(6),
            "name": str(d.get("f14", "")),
            "industry_name": _clean(d.get("f100")),
        }
        for d in rows
        if str(d.get("f12", ""))
    ]


def _clean(value: object) -> str:
    """Normalise Eastmoney's placeholders for a missing value to an empty string."""
    text = str(value or "").strip()
    return "" if text in {"-", "--"} else text


def snapshot_path(asof: dt.date, scheme: str = SCHEME) -> Path:
    """Path this run's snapshot is written to."""
    return RAW / f"scheme={scheme}" / f"asof={asof:%Y%m%d}" / "snapshot.csv"


def main(asof: dt.date | None = None) -> None:
    """Fetch one snapshot and write it under its ``asof`` date."""
    asof = asof or dt.date.today()  # noqa: DTZ011 -- exchange-local date, not an instant

    # Board list first: it is the only place the stable BKxxxx code lives. The
    # market list gives the industry *name*, and names get renamed -- keying
    # history on a display string would split one industry into two on a rename.
    boards = fetch_boards()
    code_of = {b["name"]: b["code"] for b in boards}
    print(f"{len(boards)} industry boards")

    stocks = fetch_stocks()
    print(f"{len(stocks)} listed symbols")

    # Unclassified names are left out of the snapshot entirely rather than
    # written with a placeholder industry. Absence is the honest encoding: the
    # panel join then gives them a null label, which is what "we do not know
    # this symbol's industry" means downstream.
    unclassified = [(s["symbol"], s["name"]) for s in stocks if not s["industry_name"]]
    if len(unclassified) > len(stocks) // 5:
        # Measured 2026-09: ~6% unclassified, dominated by delisting-board and
        # suspended shells (names ending 退, ST). That tail is real and stable.
        # A fifth of the market would instead mean f100 stopped being populated,
        # which must not be mistaken for "most issuers have no industry".
        raise RuntimeError(
            f"{len(unclassified)} of {len(stocks)} symbols have no industry. The tail of "
            "delisted and suspended names is a few percent; this many means the field moved."
        )
    stocks = [s for s in stocks if s["industry_name"]]
    if unclassified:
        print(f"{len(unclassified)} symbols left out as unclassified, e.g. {unclassified[:8]}")

    unmapped = sorted({s["industry_name"] for s in stocks if s["industry_name"] not in code_of})
    if unmapped:
        raise RuntimeError(
            f"{len(unmapped)} industry names on the market list have no board code "
            f"(e.g. {unmapped[:5]}). The two endpoints have drifted; mapping them by "
            "name alone would key history on a display string."
        )

    if len(stocks) < 4000:
        raise RuntimeError(
            f"only {len(stocks)} symbols; the A-share market has >5000 listed names, so "
            "this snapshot is truncated. Writing it would put a partial classification "
            "into the history permanently."
        )

    rows = [
        {
            "asof": asof.isoformat(),
            "scheme": SCHEME,
            "symbol": s["symbol"],
            "level": LEVEL,
            "industry_code": code_of[s["industry_name"]],
            "industry_name": s["industry_name"],
        }
        for s in stocks
    ]

    out = snapshot_path(asof)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["asof", "scheme", "symbol", "level", "industry_code", "industry_name"]
        )
        w.writeheader()
        w.writerows(rows)  # type: ignore[arg-type]
    n_ind = len({r["industry_code"] for r in rows})
    print(f"wrote {len(rows)} rows across {n_ind} industries -> {out}")


if __name__ == "__main__":
    main()

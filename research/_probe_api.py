"""Probe the A-share daily-bar endpoints to confirm field order before relying on it.

Stdlib only, so it runs before the project venv exists. Assuming a field order
here is exactly the kind of silent error the rest of this repo is built to
prevent -- a transposed high/low would corrupt every volatility factor without
raising anything.
"""

from __future__ import annotations

import json
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/117.0.0.0 Safari/537.36"


def get(url: str, referer: str = "") -> str:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", UA)
    if referer:
        req.add_header("Referer", referer)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", "ignore")


print("=" * 70)
print("1) Tencent fqkline (qfq daily)")
print("=" * 70)
try:
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        "?param=sh600519,day,2024-01-01,2024-01-15,20,qfq"
    )
    txt = get(url, "https://gu.qq.com/")
    d = json.loads(txt)
    node = d["data"]["sh600519"]
    print("keys under data.sh600519:", list(node.keys()))
    key = "qfqday" if "qfqday" in node else "day"
    rows = node[key]
    print(f"using key={key!r}, {len(rows)} rows")
    for r in rows[:3]:
        print("  ", r)
except Exception as exc:  # noqa: BLE001
    print("FAILED:", type(exc).__name__, exc)

print()
print("=" * 70)
print("2) THS full-history daily line")
print("=" * 70)
try:
    txt = get(
        "https://d.10jqka.com.cn/v6/line/hs_600519/01/last.js",
        "https://stockpage.10jqka.com.cn/",
    )
    print("raw head:", txt[:120])
    inner = txt[txt.index("(") + 1 : txt.rindex(")")]
    obj = json.loads(inner)
    print("keys:", list(obj.keys()))
    print("total:", obj.get("total"), "start:", obj.get("start"))
    data = obj.get("data", "")
    rows = data.split(";")
    print(f"{len(rows)} rows; first 3:")
    for r in rows[:3]:
        print("  ", r)
    print("last row:", rows[-1])
except Exception as exc:  # noqa: BLE001
    print("FAILED:", type(exc).__name__, exc)

print()
print("=" * 70)
print("3) Tencent realtime quote (mcap / industry-free fields)")
print("=" * 70)
try:
    req = urllib.request.Request("https://qt.gtimg.cn/q=sh600519,sz000001")
    req.add_header("User-Agent", UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        txt = r.read().decode("gbk", "ignore")
    for line in txt.strip().split(";"):
        if '"' not in line:
            continue
        vals = line.split('"')[1].split("~")
        print(f"  {vals[2]} {vals[1]}: n_fields={len(vals)}")
        print(
            f"    price={vals[3]} float_mcap={vals[44]} total_mcap={vals[45]} "
            f"pb={vals[46]} pe_ttm={vals[39]} turnover%={vals[38]}"
        )
except Exception as exc:  # noqa: BLE001
    print("FAILED:", type(exc).__name__, exc)

r"""Check a copied feitu v3 tree file by file: layout, zstd integrity, missing minutes, short files.

Meant for a fresh copy that may have lost data on the way. Stdlib only (plus the ``zstd`` command, or
the ``zstandard`` module when the command is missing), so it runs before anything is decoded:

* **layout**: per top-level directory the file count, size, name patterns and a few sample paths;
* **integrity**: every ``.zst`` is decompressed to nowhere (``zstd -t``); a cut-off copy fails as
  truncated, a zero-filled one as corrupt; empty files are flagged on their own;
* **coverage**: per stream and day the minute files present, and the minutes some other day of the
  same stream has but this one does not; days that one stream has and another lacks;
* **short files**: a file under ``--short`` times the median size of the same stream and minute on
  the other days (an intact but partial copy would not show as corrupt);
* **source** (optional): ``--source-list`` = ``find <src> -type f -printf '%P %s\\n'`` run on the
  machine copied from; every path missing here or with a different size is listed.

Writes ``<out>/files.csv`` (one row per file) and prints a summary short enough to photograph.

    python research/feitu_check.py /data/feitu_0901_0919 --out data/reports/feitu_check
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MINUTE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.zst$")  # v3_order_2026_09_01_09_30.zst
DAYFILE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(SH|SZ)\.zst$")  # v3_order_2026_09_01_SH.zst
STREAM = re.compile(r"v3_[a-z]+")


def classify(rel: str) -> tuple[str, str, str, str]:
    """(stream, date, slot, pattern) from a path relative to the root; slot = HHMM or SH/SZ."""
    name = rel.rsplit("/", 1)[-1]
    m = STREAM.search(name) or STREAM.search(rel)
    stream = m.group(0) if m else rel.split("/", 1)[0]
    if (x := MINUTE.search(name)) is not None:
        return stream, x[1] + x[2] + x[3], x[4] + x[5], "minute"
    if (x := DAYFILE.search(name)) is not None:
        return stream, x[1] + x[2] + x[3], x[4], "day"
    return stream, "", "", "other:" + (Path(name).suffix or "(none)")


def tool() -> str:
    """Which decompressor test_zst uses: the zstd command, else Python 3.14's compression.zstd, else zstandard."""
    if shutil.which("zstd"):
        return "zstd"
    try:
        import compression.zstd  # noqa: F401
    except ImportError:
        try:
            import zstandard  # noqa: F401
        except ImportError:
            raise SystemExit("no zstd decoder: install the zstd command (apt/yum install zstd) or pip install zstandard") from None
        return "zstandard"
    return "compression.zstd"


def test_zst(path: Path, how: str) -> tuple[str, str]:
    """('ok' | 'EMPTY' | 'BAD' | 'skip', message) for one file."""
    if path.stat().st_size == 0:
        return "EMPTY", "0 bytes"
    if path.suffix != ".zst":
        return "skip", "not .zst"
    if how == "zstd":
        r = subprocess.run(["zstd", "-t", "-q", str(path)], capture_output=True, text=True, check=False)
        return ("ok", "") if r.returncode == 0 else ("BAD", (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1][-160:])
    try:
        if how == "compression.zstd":
            from compression import zstd

            with zstd.ZstdFile(path) as f:  # reads across frames; a cut-off file raises EOFError
                while f.read(1 << 24):
                    pass
        else:
            import zstandard

            with path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True) as f:
                while f.read(1 << 24):
                    pass
    except Exception as e:  # any decoder error (EOFError when cut off, ZstdError when garbled) is a bad file
        return "BAD", f"{type(e).__name__}: {e}"[-160:]
    return "ok", ""


def _job(args: tuple[str, str, str]) -> dict:
    root, rel, how = args
    p = Path(root) / rel
    stream, date, slot, pattern = classify(rel)
    t0 = time.perf_counter()
    try:
        status, msg = test_zst(p, how)
    except OSError as e:  # unreadable: permission, I/O error from a bad disk
        status, msg = "BAD", f"{type(e).__name__}: {e}"[-160:]
    return {"rel": rel, "stream": stream, "date": date, "slot": slot, "pattern": pattern, "bytes": p.stat().st_size,
            "status": status, "msg": msg, "secs": round(time.perf_counter() - t0, 2)}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/reports/feitu_check"))
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 4))
    ap.add_argument("--short", type=float, default=0.3, help="flag files under this share of the same stream-minute median")
    ap.add_argument("--source-list", type=Path, help="'relpath size' lines from find -printf '%%P %%s\\n' on the source")
    ap.add_argument("--layout-only", action="store_true", help="only walk and print the layout, no decompression")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = str(a.root)
    t0 = time.perf_counter()
    rels = sorted(str(p.relative_to(a.root)).replace(os.sep, "/") for p in a.root.rglob("*") if p.is_file())
    print(f"== {root}: {len(rels)} files ({time.perf_counter() - t0:.0f}s to walk)")

    # layout: per top-level entry
    top: dict[str, list[str]] = defaultdict(list)
    for r in rels:
        top[r.split("/", 1)[0] if "/" in r else "(root files)"].append(r)
    for k, rs in sorted(top.items()):
        size = sum((a.root / r).stat().st_size for r in rs)
        pats = Counter(classify(r)[3] for r in rs)
        depth = Counter(r.count("/") for r in rs)
        print(f"-- {k}: {len(rs)} files, {size / 1e9:.1f} GB, patterns {dict(pats)}, depth {dict(depth)}")
        for r in rs[:2] + rs[-1:]:
            print(f"     {r}")
    if a.layout_only:
        return

    if not rels:
        return
    how = tool()
    print(f"== testing with {how}, {a.workers} workers")
    with ProcessPoolExecutor(a.workers) as ex:
        rows = []
        for i, row in enumerate(ex.map(_job, [(root, r, how) for r in rels], chunksize=4), 1):
            rows.append(row)
            if i % 500 == 0 or i == len(rels):
                print(f"   tested {i}/{len(rels)} ({time.perf_counter() - t0:.0f}s), bad so far {sum(x['status'] in ('BAD', 'EMPTY') for x in rows)}")

    # short files: against the same stream and slot on the other days
    by_slot: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok" and r["slot"]:
            by_slot[(r["stream"], r["slot"])].append(r["bytes"])
    for r in rows:
        sizes = by_slot.get((r["stream"], r["slot"]), [])
        med = statistics.median(sizes) if len(sizes) >= 3 else 0
        r["slot_median"] = int(med)
        r["short"] = bool(med and r["bytes"] < a.short * med)
    with (a.out / "files.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n########## SUMMARY (photograph from here) ##########")
    bad = [r for r in rows if r["status"] in ("BAD", "EMPTY")]
    print(f"== integrity: {len(rows)} files, {sum(r['status'] == 'ok' for r in rows)} ok, {len(bad)} BAD/EMPTY, "
          f"{sum(r['status'] == 'skip' for r in rows)} not .zst  (tool: {how})")
    for r in bad:
        print(f"   {r['status']:5} {r['rel']}  {r['bytes']:,} B  {r['msg']}")

    # coverage per stream x day
    have: dict[tuple[str, str], set[str]] = defaultdict(set)
    gb: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if r["date"]:
            have[(r["stream"], r["date"])].add(r["slot"])
            gb[(r["stream"], r["date"])] += r["bytes"]
    streams = sorted({s for s, _ in have})
    dates = sorted({d for _, d in have})
    union = {s: set().union(*(have[(s, d)] for d in dates if (s, d) in have)) for s in streams}
    print("== files per day (n / minutes missing vs the stream's other days / GB); '-' = no files that day")
    print("   date     " + "".join(f"{s:>26}" for s in streams))
    missing_lines = []
    for d in dates:
        cells = []
        for s in streams:
            if (s, d) not in have:
                cells.append(f"{'-':>26}")
                continue
            miss = sorted(union[s] - have[(s, d)])
            cells.append(f"{len(have[(s, d)]):>8} / {len(miss):>4} / {gb[(s, d)] / 1e9:7.2f}")
            if miss:
                missing_lines.append(f"   {s} {d}: {len(miss)} missing: {' '.join(miss[:30])}{' ...' if len(miss) > 30 else ''}")
        print(f"   {d} " + "".join(f"{c:>26}" for c in cells))
    print("== missing slots (present on another day of the same stream)")
    print("\n".join(missing_lines) if missing_lines else "   none")

    short = sorted((r for r in rows if r["short"] and r["status"] == "ok"), key=lambda r: r["bytes"] / max(r["slot_median"], 1))
    print(f"== short files (< {a.short:g} x median of the same stream and slot on other days): {len(short)}")
    for r in short[:25]:
        print(f"   {r['rel']}  {r['bytes']:,} B vs median {r['slot_median']:,} B")
    other = [r for r in rows if r["pattern"].startswith("other")]
    if other:
        print(f"== {len(other)} files with an unexpected name, first 10:")
        for r in other[:10]:
            print(f"   {r['rel']}  {r['bytes']:,} B")

    if a.source_list:
        src = {}
        for line in a.source_list.read_text().splitlines():
            if line.strip():
                p, _, s = line.rpartition(" ")
                src[p] = int(s)
        here = {r["rel"]: r["bytes"] for r in rows}
        gone = sorted(set(src) - set(here))
        diff = sorted(p for p in set(src) & set(here) if src[p] != here[p])
        extra = sorted(set(here) - set(src))
        print(f"== against the source list: {len(src)} source files, {len(gone)} missing here, {len(diff)} size differs, {len(extra)} only here")
        for p in gone[:30]:
            print(f"   MISSING  {p}  {src[p]:,} B")
        for p in diff[:30]:
            print(f"   SIZE     {p}  here {here[p]:,} B, source {src[p]:,} B")
    print(f"table: {a.out / 'files.csv'} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()

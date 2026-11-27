r"""Read PM's .mp market-data files (zstd-compressed stream of msgpack arrays, one array per record) with our own decoder.

No msgpack package needed: the files use only ints, doubles, nil, bools, short strings and arrays. ``records(path)`` yields
each record as a list; ``frame(path)`` returns a numpy float64 table (records x fields; non-numbers -> NaN). The CLI is for
working out the layout: ``--show`` prints records, ``--stats`` prints per-field type / min / median / max over a sample.
Read-only.

    python research/pm_factors/mp_read.py /work/crucible_data/pm_mp/20261119/SH_HS300_snap.mp --skip 200000 --show 3 --stats 50000
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np

CHUNK = 1 << 22


class Stream:
    """Byte reader over the decompressed stream, refilled in chunks."""

    def __init__(self, path: Path) -> None:
        import zstandard
        self.r = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        self.buf, self.pos = b"", 0

    def need(self, n: int) -> bool:
        if self.pos + n <= len(self.buf):
            return True
        rest = self.buf[self.pos:]
        more = self.r.read(max(CHUNK, n))
        self.buf, self.pos = rest + more, 0
        return len(self.buf) >= n

    def take(self, n: int) -> bytes:
        if not self.need(n):
            raise EOFError
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def byte(self) -> int:
        if not self.need(1):
            raise EOFError
        b = self.buf[self.pos]
        self.pos += 1
        return b


def value(s: Stream):
    """One msgpack value."""
    b = s.byte()
    if b <= 0x7F:
        return b
    if b >= 0xE0:
        return b - 0x100
    if 0x90 <= b <= 0x9F:
        return [value(s) for _ in range(b & 0x0F)]
    if 0xA0 <= b <= 0xBF:
        return s.take(b & 0x1F).decode("latin-1")
    if 0x80 <= b <= 0x8F:
        return {value(s): value(s) for _ in range(b & 0x0F)}
    if b == 0xCB:
        return struct.unpack(">d", s.take(8))[0]
    if b == 0xCA:
        return struct.unpack(">f", s.take(4))[0]
    if b in (0xCC, 0xCD, 0xCE, 0xCF):
        n = 1 << (b - 0xCC)
        return int.from_bytes(s.take(n), "big")
    if b in (0xD0, 0xD1, 0xD2, 0xD3):
        n = 1 << (b - 0xD0)
        return int.from_bytes(s.take(n), "big", signed=True)
    if b == 0xDC:
        return [value(s) for _ in range(int.from_bytes(s.take(2), "big"))]
    if b == 0xDD:
        return [value(s) for _ in range(int.from_bytes(s.take(4), "big"))]
    if b == 0xD9:
        return s.take(s.byte()).decode("latin-1")
    if b == 0xC0:
        return None
    if b in (0xC2, 0xC3):
        return b == 0xC3
    raise ValueError(f"msgpack type 0x{b:02x} not handled")


def records(path: Path) -> Iterator[list]:
    s = Stream(path)
    while True:
        try:
            yield value(s)
        except EOFError:
            return


def frame(path: Path, limit: int | None = None) -> np.ndarray:
    """Records x fields as float64 (NaN for non-numbers / missing); width = the widest record seen."""
    rows = []
    for i, r in enumerate(records(path)):
        if limit is not None and i >= limit:
            break
        rows.append([float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan for v in r])
    w = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), w), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--stats", type=int, default=0, help="records sampled for per-field stats (after --skip)")
    a = ap.parse_args()
    it = records(a.path)
    for _ in range(a.skip):
        next(it, None)
    sample, widths = [], Counter()
    for i, r in enumerate(it):
        if i < a.show:
            print(f"   rec {a.skip + i}: {r}")
        if i < a.stats:
            sample.append(r)
            widths[len(r)] += 1
        if i >= max(a.show, a.stats):
            break
    if sample:
        print(f"   {len(sample)} records sampled; widths {dict(widths)}")
        w = max(widths)
        for j in range(w):
            col = [r[j] for r in sample if len(r) > j]
            nums = np.array([float(v) for v in col if isinstance(v, (int, float)) and not isinstance(v, bool)])
            types = Counter(type(v).__name__ for v in col)
            if nums.size:
                print(f"   f{j:<2} {dict(types)}  min {nums.min():.10g}  med {np.median(nums):.10g}  max {nums.max():.10g}  distinct {len(np.unique(nums))}")
            else:
                print(f"   f{j:<2} {dict(types)}  e.g. {col[:3]}")
    print("[done]")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()

r"""Dump a class's vtable slots from a shared object without running it: which function sits in each slot.

The vtable symbol (readelf -Ws) gives address and size; its slots are filled by R_X86_64_RELATIVE / R_X86_64_64
relocations (readelf -rW), whose addend or symbol is the function. Mapped to names with nm -C. Read-only.

    python3 research/vtbl_probe.py /data/sht-stg-lib/plugins/libcatchRun.so _ZTVN3cts5qlive4runSE
"""

from __future__ import annotations

import re
import subprocess
import sys


def run(cmd: list[str]) -> str:
    """Run a read-only tool, return stdout."""
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout


def main() -> None:
    """CLI entry point."""
    so, sym = sys.argv[1], sys.argv[2]
    start = size = None
    for ln in run(["readelf", "-Ws", so]).splitlines():
        p = ln.split()
        if len(p) >= 8 and p[7] == sym:
            start, size = int(p[1], 16), int(p[2])
            break
    if start is None:
        sys.exit(f"{sym} not found")
    names = {}
    for ln in run(["nm", "-C", "--defined-only", so]).splitlines():
        m = re.match(r"([0-9a-f]+) \S (.+)$", ln)
        if m:
            names.setdefault(int(m.group(1), 16), m.group(2))
    slots = {}
    for ln in run(["readelf", "-rW", so]).splitlines():
        m = re.match(r"\s*([0-9a-f]+)\s+[0-9a-f]+\s+(R_X86_64_\w+)\s+([0-9a-f]+)?\s*(.*)$", ln)
        if not m:
            continue
        off = int(m.group(1), 16)
        if not (start <= off < start + size):
            continue
        rest = m.group(4).strip()
        if m.group(2) == "R_X86_64_RELATIVE":
            tgt = int(rest, 16) if rest else 0
            slots[off] = names.get(tgt, f"{tgt:#x}")
        else:
            slots[off] = run(["c++filt", rest.split("+")[0].strip()]).strip() or rest
    print(f"{sym}: addr {start:#x} size {size} ({size // 8} words)")
    for i in range(size // 8):
        print(f"  [{i:2d}] {slots.get(start + 8 * i, '(0 / offset-to-top / typeinfo)')}")


if __name__ == "__main__":
    main()

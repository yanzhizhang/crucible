r"""Find the function(s) that install a class's vtable (its constructor) in a stripped .so, and print them with every
.rodata load annotated (double / int64 / string), so the config keys and file variable names a term reads become visible.

Given one known virtual slot (address and index, from vt_asm.py's .s files), the vtable is the RELATIVE relocation whose
addend is that slot, minus 8 x index; the vptr a constructor stores is that address. Every instruction whose rip-relative
target equals the vptr is listed with its enclosing function (from .eh_frame), and each such function is disassembled.
binutils + stdlib only; nothing is run. Read-only.

    python3 find_ctor.py /work/exec_x/exec/libCtePolicyBTE.so 0xda930 7 --max-lines 260
"""

from __future__ import annotations

import argparse
import bisect
import re
import struct
import subprocess
from pathlib import Path


def run(*a: str) -> str:
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("so", type=Path)
    ap.add_argument("slot", help="address of one virtual slot of the class, hex")
    ap.add_argument("index", type=int, help="that slot's index in the vtable")
    ap.add_argument("--max-lines", type=int, default=260)
    a = ap.parse_args()
    so, slot = a.so, int(a.slot, 16)
    data = so.read_bytes()
    secs = [(m[1], int(m[2], 16), int(m[3], 16), int(m[4], 16)) for m in
            re.finditer(r"\]\s+(\S+)\s+\S+\s+([0-9a-f]{16})\s+([0-9a-f]{6,})\s+([0-9a-f]{6,})", run("readelf", "-SW", str(so)))]

    def v2o(v: int) -> int | None:
        for n, ad, o, s in secs:
            if ad <= v < ad + s and n not in (".bss", ".tbss"):
                return o + v - ad
        return None

    rel = [(int(m[1], 16), int(m[2], 16)) for m in
           re.finditer(r"^([0-9a-f]{12,16})\s+\S+\s+R_X86_64_RELATIVE\s+([0-9a-f]+)", run("readelf", "-rW", str(so)), re.M)]
    hits = [o for o, ad in rel if ad == slot]
    vptrs = sorted({o - 8 * a.index for o in hits})
    print(f"slot {slot:#x} is referenced by {len(hits)} relocation(s); candidate vptrs {[hex(v) for v in vptrs]}")
    fde = sorted((int(m[1], 16), int(m[2], 16)) for m in re.finditer(r"FDE cie=\S+ pc=([0-9a-f]+)\.\.([0-9a-f]+)", run("readelf", "--debug-dump=frames", str(so))))
    starts = [s for s, _ in fde]

    def enclosing(x: int) -> tuple[int, int] | None:
        i = bisect.bisect_right(starts, x) - 1
        return fde[i] if i >= 0 and fde[i][0] <= x < fde[i][1] else None

    text = [s for s in secs if s[0] == ".text"][0]
    asm = run("objdump", "-d", "--no-show-raw-insn", "-M", "intel", f"--start-address={text[1]:#x}", f"--stop-address={text[1] + text[3]:#x}", str(so))
    funcs = set()
    for ln in asm.splitlines():
        m = re.match(r"\s+([0-9a-f]+):.*#\s*([0-9a-f]+)", ln)
        if m and int(m[2], 16) in vptrs:
            f = enclosing(int(m[1], 16))
            print(f"   store site {ln.strip()[:120]}  in function {f[0]:#x}..{f[1]:#x}" if f else f"   {ln.strip()[:120]} (no FDE)")
            if f:
                funcs.add(f)

    def imm_text(line: str) -> str:
        """An immediate (mov / movabs) whose little-endian bytes are printable ASCII: short std::string literals built in place."""
        m = re.search(r",0x([0-9a-f]{6,16})\b", line)
        if not m:
            return ""
        b = int(m[1], 16).to_bytes(8, "little").rstrip(b"\0")
        return f'    ; imm "{b.decode()}"' if len(b) >= 3 and all(32 <= c < 127 for c in b) else ""

    def note(line: str) -> str:
        line = re.sub(r"\s*<[^>]*>", "", line)  # drop objdump's symbol labels so the annotation fits
        line += imm_text(line)
        m = re.search(r"#\s*([0-9a-f]+)", line)
        if not m:
            return line
        v = int(m[1], 16)
        o = v2o(v)
        if o is None or not any(n.startswith(".rodata") and ad <= v < ad + s for n, ad, _, s in secs) or o + 8 > len(data):
            return line
        d, = struct.unpack("<d", data[o:o + 8])
        q, = struct.unpack("<q", data[o:o + 8])
        s = data[o:o + 48].split(b"\0")[0]
        txt = f' str "{s.decode("ascii", "replace")}"' if len(s) >= 2 and all(32 <= c < 127 for c in s) else ""
        return f"{line}    ; [{d:.10g} | {q}]{txt}"

    for lo, hi in sorted(funcs):
        body = run("objdump", "-d", "-C", "--no-show-raw-insn", "-M", "intel", f"--start-address={lo:#x}", f"--stop-address={hi:#x}", str(so))
        rows = [note(r) for r in body.splitlines() if re.match(r"\s+[0-9a-f]+:", r)]
        strs = [r for r in rows if ' str "' in r or ' imm "' in r]
        print(f"\n== function {lo:#x}..{hi:#x}: {len(rows)} instructions, {len(strs)} string loads")
        for r in strs:
            print("   S " + r.strip()[:170])
        for r in rows[:a.max_lines]:
            print("   " + r.strip()[:170])
    print("[done]")


if __name__ == "__main__":
    main()

r"""so_probe stage 2: read the order rule out of PM's strategy plugin (libcatchRun.so, DWARF present) without running it.

Read-only: nm / objdump / gdb -batch on the file, nothing from the library is executed (gdb never starts the process).
For each target function (sizing, threshold, vol penalty, market-volume EMA, price, cancel timer, config load):
address, size, source lines it spans, its local variable names (gdb info scope), and every .rodata value it loads
rip-relative -- printed as a double, or as a string when it is text (config keys, log formats). Plus gdb ptype of the
key classes (field names = state and config). Full output goes to OUT on 97; the digest printed at the end is what
gets relayed (user approved, 2026-11-16).

    python research/so_probe2.py /data/sht-stg-lib/plugins/libcatchRun.so --out /work/crucible_data/so_probe/stage2
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

TARGETS = [
    r"cts::sol::Alg::", r"cts::sol::AlgW::(loadCfgs|loadDaily|setPosLimit|AddFa|onInnerMsg|OnMsg)",
    r"cts::sol::Estitp::", r"cts::sol::MktVolPool::", r"cts::sol::MktVs::", r"cts::sol::(roundBy|roundPrc|sAbs)",
    r"cts::ct::(EMA|EMS|PosClzer)::", r"cts::qlive::LiveEs::(autoCxl|PlaceExOrd|cancelExOrd|dOrd|OnFOM|OnEOM|OnECM|OnFCM|Po|GAsq|Gwfq|Gwfn|GPpp)",
    r"cts::qlive::Engine::(TimerLoop|StartTimer|Manage|toAlg)", r"cts::qlive::runS::(on_signal|signal2alpha|place_order|initialize|on_lv2|book2msg)",
    r"cts::qlive::qTrader::(loadCfgs|setupAlg|setupEs)", r"cts::es::(round100|configOrder)", r"cts::es::eSys::(GAsq|GWoq|initPos|aAsq|Po)",
    r"cts::GlobalConfig::(Lmb|Lmsr|Lvr|loadEx)",
]
CLASSES = ["cts::sol::Alg", "cts::sol::AlgW", "cts::sol::Estitp", "cts::sol::MktVolPool", "cts::sol::MktVs", "cts::ct::EMA", "cts::ct::EMS",
           "cts::ct::PosClzer", "cts::qlive::LiveEs", "cts::qlive::exOrdInfo", "cts::qlive::AlpInfo", "cts::es::eSys", "cts::es::Order", "cts::GlobalConfig"]


def run(cmd: list[str]) -> str:
    """Run a read-only tool, return stdout (stderr folded in)."""
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("so", type=Path)
    ap.add_argument("--out", type=Path, default=Path("/work/crucible_data/so_probe/stage2"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    so = str(a.so)
    syms = []
    for ln in run(["nm", "-C", "-S", "--defined-only", so]).splitlines():
        m = re.match(r"([0-9a-f]+) ([0-9a-f]+) [TtWw] (.+)$", ln)
        if m and any(re.search(t, m.group(3)) for t in TARGETS) and "::{lambda" not in m.group(3):
            syms.append((int(m.group(1), 16), int(m.group(2), 16), m.group(3)))
    syms = sorted(set(syms))
    print(f"{len(syms)} target functions")
    digest = []
    for addr, size, name in syms:
        asm = run(["objdump", "-d", "-l", "-C", "--no-show-raw-insn", f"--start-address={addr:#x}", f"--stop-address={addr + size:#x}", so])
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", name)[:120]
        (a.out / f"{safe}.asm").write_text(asm)
        lines = sorted({int(x) for x in re.findall(r"^/\S+:(\d+)", asm, re.M)})
        srcs = sorted(set(re.findall(r"^(/\S+?):\d+", asm, re.M)))
        refs = []
        cur = None
        for ln in asm.splitlines():
            m = re.match(r"^(/\S+):(\d+)", ln)
            if m:
                cur = f"{Path(m.group(1)).name}:{m.group(2)}"
            m = re.search(r"#\s*([0-9a-f]+)\s*<", ln)
            if m and "(%rip)" in ln:
                refs.append((cur, int(m.group(1), 16), ln.split("\t")[-1].strip()[:60]))
        vals = {}
        if refs:
            cmds = []
            for _, r, _ in refs:
                cmds += ["-ex", f"x/gf {r:#x}", "-ex", f"x/s {r:#x}"]
            out = run(["gdb", "-batch", "-nx", *cmds, so]).splitlines()
            for i, (_, r, _) in enumerate(refs):
                d = out[2 * i] if 2 * i < len(out) else ""
                s = out[2 * i + 1] if 2 * i + 1 < len(out) else ""
                sm = re.search(r':\s*"(.*)"', s)
                dm = re.search(r":\s*(\S+)$", d)
                if sm and len(sm.group(1)) >= 3 and re.fullmatch(r"[ -~]+", sm.group(1)):
                    vals[r] = '"' + sm.group(1)[:80] + '"'
                elif dm:
                    vals[r] = dm.group(1)
        scope = run(["gdb", "-batch", "-nx", "-ex", f"info scope *{addr:#x}", so])
        local = re.findall(r"Symbol (\S+) is", scope)
        digest.append(f"## {name}  addr {addr:#x} size {size}  src {','.join(Path(s).name for s in srcs)} lines {lines[0] if lines else '-'}..{lines[-1] if lines else '-'}")
        if local:
            digest.append(f"   locals: {' '.join(local[:60])}")
        seen = set()
        for where, r, ins in refs:
            v = vals.get(r, "?")
            if (where, v) in seen or v in ("0", "?"):
                continue
            seen.add((where, v))
            digest.append(f"   {where or '-':<18} {v:<26} {ins}")
    for c in CLASSES:
        pt = run(["gdb", "-batch", "-nx", "-ex", f"ptype /o {c}", so])
        (a.out / f"ptype_{c.replace('::', '_')}.txt").write_text(pt)
        digest.append(f"## ptype {c}")
        digest += ["   " + x.rstrip() for x in pt.splitlines() if x.strip()][:80]
    (a.out / "digest.txt").write_text("\n".join(digest) + "\n")
    print("\n".join(digest))
    print(f"[done] {len(syms)} functions, {len(CLASSES)} classes; full asm + ptype in {a.out}")


if __name__ == "__main__":
    main()

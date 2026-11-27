r"""One compact signature per class from vt_asm.py's per-class .s files, plus code-shape groups, to read 120 terms fast.

For every class file (vt_asm output dir): its OWN virtual slots (an address shared by several classes is an inherited one
and only counted) and, per own slot, the rodata doubles it loads (the clock / session constants are dropped), the object
members it reads and writes ([reg+0x..] with the object register), the calls it makes (plt names, and the known helpers by
address), op counts and whether it loops. Then every own slot's instruction skeleton (mnemonics, operands normalised,
constants -> K, offsets -> M) is hashed, and classes whose slot has the same skeleton are listed together with the
constants that differ: one decoded member of a group decodes the group. stdlib only; reads text files; nothing is run.

    python3 term_sig.py /work/exec_x_asm --cfg /work/exec_x/exec/cfg.json --out /work/exec_x_asm/term_sig.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CLOCK = {86400.0, 5400.0, 7200.0, 10800.0, 12600.0, 18000.0, 34199.99999, 41550.0, 46799.99999, 54150.0, 0.0, 1.0}
HELPERS = {0x132aa0: "clock", 0x1329d0: "ema"}  # decoded earlier: TermCFH trading clock, the shared average update
SLOT = re.compile(r"^// ---- slot (\d+) @ (0x[0-9a-f]+)")
INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
CONST = re.compile(r";\s*\[([^|\]]+)\|")
MEM = re.compile(r"\[(r[a-z0-9]+)\+0x([0-9a-f]+)\]")
CALL = re.compile(r"\bcall\s+([0-9a-f]+)(?:\s*<([^>]+)>)?")


def parse(fn: Path) -> tuple[str, dict[int, tuple[int, list[tuple[str, str, int]]]]]:
    """class name, {slot: (address, [(mnemonic, operands + note)])}."""
    name, slots, cur = fn.stem, {}, None
    for ln in fn.read_text(errors="replace").splitlines():
        if ln.startswith("// class "):
            name = ln[9:].split(":")[0]
        elif m := SLOT.match(ln):
            cur = int(m[1])
            slots[cur] = (int(m[2], 16), [])
        elif cur is not None and (m := INSN.match(ln)):
            slots[cur][1].append((m[2], m[3], int(m[1], 16)))
    return name, slots


def short(name: str) -> str:
    m = re.match(r"N3cte\d+Term(\w+?)E$", name)
    return m[1] if m else name


def sig(body: list[tuple[str, str, int]]) -> dict:
    consts, rd, wr, calls, ops = [], set(), set(), [], Counter()
    loop = False
    addrs = []
    for mn, op, at in body:
        ops[mn] += 1
        for c in CONST.findall(op):
            try:
                v = float(c)
            except ValueError:
                continue
            if v not in CLOCK and v not in consts:
                consts.append(v)
        args = op.split(";")[0]
        mems = MEM.findall(args)
        if mems and mn not in ("lea", "nop") and not mn.startswith("call"):
            first = args.split(",")[0]
            for reg, off in mems:
                if reg in ("rsp", "rip"):
                    continue
                (wr if f"[{reg}+0x{off}]" in first and mn.startswith(("mov", "add", "sub", "inc", "dec")) and "," in args else rd).add(int(off, 16))
        if mn.startswith("call") and (m := CALL.search(f"{mn} {op}")):
            a = int(m[1], 16)
            calls.append(HELPERS.get(a) or (m[2].split("@")[0] if m[2] and "@plt" in m[2] else f"{a:#x}"))
        if mn.startswith("j") and (m := re.match(r"([0-9a-f]+)", args.strip())):
            addrs.append((at, int(m[1], 16)))
    return {"consts": consts, "rd": sorted(rd), "wr": sorted(wr), "calls": calls, "ops": ops, "jumps": addrs}


def skeleton(body: list[tuple[str, str, int]]) -> str:
    out = []
    for mn, op, _ in body:
        a = op.split(";")[0]
        a = re.sub(r"<[^>]*>", "", a)
        a = re.sub(r"\[rip\+0x[0-9a-f]+\]", "K", a)
        a = re.sub(r"0x[0-9a-f]+", "M", a)
        a = re.sub(r"\b[0-9a-f]{4,}\b", "A", a)
        out.append(f"{mn} {a.strip()}")
    return hashlib.md5("\n".join(out).encode()).hexdigest()[:8]


def cfg_terms(path: Path | None) -> dict[str, list[str]]:
    """type -> ['name blen=..', ...] from any dicts with 'name' and 'type' in the cfg json."""
    if path is None or not path.exists():
        return {}
    by = defaultdict(list)

    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("name"), str) and isinstance(x.get("type"), str):
                extra = ",".join(f"{k}={x[k]}" for k in sorted(x) if k not in ("name", "type") and isinstance(x[k], (int, float, str)) and len(str(x[k])) < 24)
                by[x["type"]].append(f"{x['name']}[{extra}]" if extra else x["name"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(json.loads(path.read_text()))
    return by


def fmt(v: float) -> str:
    return f"{v:.6g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("asm_dir", type=Path)
    ap.add_argument("--cfg", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    classes = {}
    for fn in sorted(a.asm_dir.glob("*.s")):
        name, slots = parse(fn)
        classes[name] = slots
    use = Counter(addr for s in classes.values() for addr, _ in s.values())
    terms = cfg_terms(a.cfg)
    lines, groups = [], defaultdict(list)
    lines.append(f"# {len(classes)} classes; own slot = address used by this class only; consts exclude clock/session values")
    lines.append("# per own slot: s<i> n=<insns> c[consts] r[member reads] w[member writes] f[calls] op[div/mul/sqrt/round/cmp] J=backward jumps (loops)")
    for name, slots in classes.items():
        own = {i: v for i, v in slots.items() if use[v[0]] == 1}
        t = short(name)
        head = f"== {t} ({len(slots)} slots, own {sorted(own)})" + (f" cfg: {' '.join(terms[t])}" if t in terms else "")
        lines.append(head)
        for i, (addr, body) in sorted(own.items()):
            s = sig(body)
            o = s["ops"]
            cmp_ = o["comisd"] + o["ucomisd"] + o["cmp"]
            ops = f"div{o['divsd']} mul{o['mulsd']} add{o['addsd']} sub{o['subsd']} sqrt{o['sqrtsd']} rnd{o['roundsd']} cmp{cmp_}"
            back = sum(1 for at, j in s["jumps"] if j <= at)  # backward jumps = loops
            sk = skeleton(body)
            groups[(i, sk, len(body))].append((t, s["consts"]))
            lines.append(f"  s{i}@{addr:#x} n={len(body)} sk={sk} c[{','.join(fmt(v) for v in s['consts'][:16])}] "
                         f"r[{','.join(f'{x:x}' for x in s['rd'][:20])}] w[{','.join(f'{x:x}' for x in s['wr'][:12])}] "
                         f"f[{','.join(dict.fromkeys(s['calls']))}] op[{ops}] J={back}")
    lines.append("\n# code-shape groups: the same own-slot skeleton in several classes (slot, skeleton, insns): classes and their consts")
    for (i, sk, n), mem in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(mem) < 2:
            continue
        lines.append(f"  s{i} sk={sk} n={n} x{len(mem)}: " + " | ".join(f"{t}[{','.join(fmt(v) for v in c[:8])}]" for t, c in mem))
    single = sorted({t for (i, sk, n), mem in groups.items() if len(mem) == 1 for t, _ in mem})
    lines.append(f"# classes with at least one unique-shape slot ({len(single)}): {' '.join(single)}")
    a.out.write_text("\n".join(lines) + "\n")
    print(f"{len(classes)} classes, {sum(1 for g in groups.values() if len(g) > 1)} shared shapes -> {a.out}")


if __name__ == "__main__":
    main()

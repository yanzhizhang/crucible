r"""Where the runtime writes the alpha and where the strategy parameters are read: string / immediate cross-references.

For every ELF file under the root (public libraries and libcommon.a / libfatwriter.a skipped by name and listed), finds each
keyword as an exact NUL-terminated string (or as a substring for the --sub keywords) in .rodata, then the instructions that
load that address (objdump's "# addr" note) and, for keywords of 8 bytes or less, the mov immediates that build it in
place (short std::string literals). Each hit is mapped to its function (.eh_frame FDE). Output: per keyword the files and
functions that touch it, then per function every keyword it touches (a function touching many parameter names is the
config reader; its member stores lead to the code that uses them), and the file's own exported symbols matching
--sym. binutils + stdlib only; reads files, runs nothing from the archive.

    python3 exec_locate.py /work/exec_x --out /work/exec_x_asm/exec_locate.txt
"""

from __future__ import annotations

import argparse
import bisect
import re
import subprocess
from collections import defaultdict
from pathlib import Path

KEYS = ("alpha_name alpha_dur pqAlpha faw fac fan ahubs tqub nmvcst tpub adur vpen rvts asf neg_nmvcst sim_st sim_et "
        "depthLevel params_path alg_cfg NumLevels dataPath streams trader rec_epoch src_epoch livev1 predict alpha "
        "cancel Cancel order_dur cr").split()
SUBS = ("parquet", "alpha.", "Alpha", "ahub", "hub", "executor", "Executor", "fit1", "regression")
PUBLIC = ("arrow", "parquet", "lightgbm", "LightGBM", "_lightgbm", "boost", "zstd", "ssl", "crypto", "protobuf", "quill", "spdlog",
          "fmt", "tbb", "snappy", "lz4", "brotli", "gtest", "python", "numpy", "stdc++", "libgcc", "libm.", "pthread", "xtp", "qtapi",
          "thost", "ctp", "libcommon.a", "libfatwriter.a", "libz.", "gomp", "omp", "jemalloc", "tcmalloc", "curl", "thrift", "re2",
          "utf8proc", "glog", "gflags", "absl", "grpc", "aws", "orc", "bz2", "lzma", "event")


def run(*a: str) -> str:
    return subprocess.run(a, check=True, capture_output=True, text=True, errors="replace").stdout


def is_elf(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def scan(so: Path, sym_re: re.Pattern, out: list[str]) -> None:
    data = so.read_bytes()
    secs = [(m[1], int(m[2], 16), int(m[3], 16), int(m[4], 16)) for m in
            re.finditer(r"\]\s+(\S+)\s+\S+\s+([0-9a-f]{16})\s+([0-9a-f]{6,})\s+([0-9a-f]{6,})", run("readelf", "-SW", str(so)))]
    ro = [s for s in secs if s[0].startswith(".rodata")]
    text = [s for s in secs if s[0] == ".text"]
    if not text:
        out.append(f"## {so}: no .text")
        return
    targets: dict[int, str] = {}  # string vaddr -> keyword
    for n, ad, off, sz in ro:
        blob = data[off:off + sz]
        for k in KEYS:
            for m in re.finditer(b"\0" + re.escape(k.encode()) + b"\0", blob):
                targets[ad + m.start() + 1] = k
        for k in SUBS:
            for m in re.finditer(re.escape(k.encode()), blob):
                st = blob.rfind(b"\0", 0, m.start()) + 1  # loads point at the string's start
                s = blob[st:blob.find(b"\0", m.end())]
                if len(s) < 80 and all(32 <= c < 127 for c in s):
                    targets.setdefault(ad + st, f"~{s.decode()}")
    imms = {int.from_bytes(k.encode(), "little"): k for k in KEYS if 3 <= len(k) <= 8}
    fde = sorted((int(m[1], 16), int(m[2], 16)) for m in re.finditer(r"FDE cie=\S+ pc=([0-9a-f]+)\.\.([0-9a-f]+)", run("readelf", "--debug-dump=frames", str(so))))
    starts = [s for s, _ in fde]

    def func(x: int) -> str:
        i = bisect.bisect_right(starts, x) - 1
        return f"{fde[i][0]:#x}..{fde[i][1]:#x}" if i >= 0 and fde[i][0] <= x < fde[i][1] else "nofde"

    t = text[0]
    asm = run("objdump", "-d", "--no-show-raw-insn", "-M", "intel", f"--start-address={t[1]:#x}", f"--stop-address={t[1] + t[3]:#x}", str(so))
    by_key, by_fn = defaultdict(set), defaultdict(set)
    for ln in asm.splitlines():
        m = re.match(r"\s+([0-9a-f]+):", ln)
        if not m:
            continue
        hit = None
        if (c := re.search(r"#\s*([0-9a-f]+)", ln)) and int(c[1], 16) in targets:
            hit = targets[int(c[1], 16)]
        elif (c := re.search(r",0x([0-9a-f]{6,16})\b", ln)) and int(c[1], 16) in imms:
            hit = "imm:" + imms[int(c[1], 16)]
        if hit:
            f = func(int(m[1], 16))
            by_key[hit].add(f)
            by_fn[f].add(hit)
    syms = []
    try:
        syms = [ln.split(" ", 2)[-1] for ln in run("nm", "-DC", "--defined-only", str(so)).splitlines() if sym_re.search(ln)]
    except subprocess.CalledProcessError:
        pass
    out.append(f"## {so} ({so.stat().st_size:,} B): {len(targets)} keyword strings, {len(by_fn)} functions touch them, {len(syms)} matching exported symbols")
    for k in sorted(by_key):
        out.append(f"   key {k}: {' '.join(sorted(by_key[k])[:8])}{' ...' if len(by_key[k]) > 8 else ''}")
    for f in sorted(by_fn, key=lambda f: -len(by_fn[f]))[:25]:
        out.append(f"   fn {f}: {len(by_fn[f])} keys: {' '.join(sorted(by_fn[f]))[:300]}")
    for s in syms[:40]:
        out.append(f"   sym {s[:160]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sym", default=r"(?i)alpha|exec|order|cancel|trade|parquet|ahub|predict|regress|policy")
    a = ap.parse_args()
    out, skipped = [], []
    for p in sorted(x for x in a.root.rglob("*") if x.is_file() and not x.is_symlink() and is_elf(x)):
        if any(k in p.name for k in PUBLIC):
            skipped.append(p.name)
            continue
        try:
            scan(p, re.compile(a.sym), out)
        except subprocess.CalledProcessError as e:
            out.append(f"## {p}: {e.cmd[0]} failed")
    out.append(f"# skipped as public / helper by name ({len(skipped)}): {' '.join(skipped)}")
    a.out.write_text("\n".join(out) + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

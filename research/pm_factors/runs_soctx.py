r"""What precedes each of PM's SO lines in the log? (Why do only half of the orders have a decision flagged "nullptr: false"?)

The log is read in order. For every es.cpp "nowTs:...SO," line (one per order sent) the plugin lines since the last
catchRun "srcEpoch" line (one per snapshot handled) are classified:
  - was there an alg.cpp evaluation line (mp fa vpen ...), a decision line (executp ...), a "nullptr" line and its value;
  - how many SO lines came out of this one snapshot block, and for which stocks (the block's own sid or another one);
  - the block's stock against the SO line's stock.
Counts per class, then --show examples of each class: the block's lines up to the SO line (cut to 220 chars; the account
field of the SO line is masked). Read-only.

    python research/pm_factors/runs_soctx.py --logs /work/crucible_data/pm_224/20261118/chrono_logs --date 20261118
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def mask(line: str) -> str:
    """Hide the account field of an SO line (the third comma field after nowTs) and cut the line."""
    if "nowTs:" in line:
        head, _, tail = line.partition("nowTs:")
        f = tail.split(",")
        if len(f) > 2:
            f[2] = "<ac>"
        line = head + "nowTs:" + ",".join(f)
    return line.rstrip()[:220]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()
    from pm_orders import GLUE, PLUGIN, SRC
    from pm_signals import NUM, log_files

    cls, ex, files_seen = Counter(), {}, Counter()
    per_block = Counter()
    own, own_ex = Counter(), {}
    n_so = 0
    for f in log_files(a.logs, a.date):
        block: list[str] = []
        last_block: dict[str, tuple[list[str], dict]] = {}  # stock -> its latest finished snapshot block
        st = {"sid": None, "ev": 0, "dec": 0, "null": [], "so": 0, "other": Counter()}
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in (part for raw in fh for part in GLUE.split(raw) if part.strip()):
                if " trk " in line:
                    continue
                s = SRC.search(line)
                if not s or (line.startswith("[20") and not PLUGIN.search(line)):
                    continue
                where, rest = s.group(1), s.group(2)
                fname = where.rsplit(":", 1)[0].rsplit("/", 1)[-1]
                kv = {k.replace("this->", "").replace("book->", "book_").rstrip("_"): v for k, v in NUM.findall(rest)}
                if fname == "catchRun.cpp" and "srcEpoch" in kv:
                    if st["so"]:
                        per_block[st["so"]] += 1
                    if st["sid"] is not None and block:
                        last_block[str(int(float(st["sid"])))] = (block, st)
                    block = [line]
                    st = {"sid": kv.get("sid"), "ev": 0, "dec": 0, "null": [], "so": 0, "other": Counter(), "epoch": float(kv.get("srcEpoch", "nan"))}
                    continue
                block.append(line)
                if fname == "alg.cpp" and "vpen" in kv and "executp" not in kv:
                    st["ev"] += 1
                elif fname == "alg.cpp" and "executp" in kv:
                    st["dec"] += 1
                elif fname == "alg.cpp" and "nullptr" in rest:
                    m = re.search(r"nullptr\s*:\s*(\d+|true|false)", rest)
                    st["null"].append(m.group(1) if m else "?")
                elif fname == "es.cpp" and "nowTs:" in rest:
                    n_so += 1
                    st["so"] += 1
                    files_seen[where] += 1
                    fld = rest.split("nowTs:", 1)[1].split(",")
                    so_sid = fld[5].strip() if len(fld) > 5 else "?"
                    try:
                        same = st["sid"] is not None and int(float(so_sid)) == int(float(st["sid"]))
                    except ValueError:
                        same = False
                    key = (f"evaluation lines {min(st['ev'], 2)}{'+' if st['ev'] > 2 else ''}", f"decision lines {min(st['dec'], 2)}{'+' if st['dec'] > 2 else ''}",
                           f"nullptr values {','.join(st['null'][-2:]) or 'none'}", f"SO no. {min(st['so'], 3)} of the block", "same stock" if same else "OTHER stock than the block's snapshot",
                           f"lines in block {'<=6' if len(block) <= 6 else '<=20' if len(block) <= 20 else '>20'}")
                    cls[key] += 1
                    if not same:  # the order's own stock: what did its latest block decide, and how long ago
                        ob = last_block.get(so_sid.lstrip("0") or "0") or last_block.get(str(int(float(so_sid))) if so_sid.replace(".", "").isdigit() else "")
                        if ob is None:
                            k2 = ("no earlier block of the order's stock",)
                        else:
                            try:
                                age = float(fld[1]) - ob[1]["epoch"]
                            except (ValueError, IndexError):
                                age = float("nan")
                            k2 = (f"own block: decision lines {min(ob[1]['dec'], 2)}", f"nullptr {','.join(ob[1]['null'][-2:]) or 'none'}", f"SO lines in it {min(ob[1]['so'], 2)}",
                                  "age " + ("<0.5 s" if age < 0.5 else "<1.5 s" if age < 1.5 else "<3.5 s" if age < 3.5 else "<10 s" if age < 10 else ">=10 s or unknown"))
                        own[k2] += 1
                        if ob is not None and len(own_ex.setdefault(k2, [])) < a.show:
                            own_ex[k2].append([mask(b) for b in ob[0][-8:]] + ["      ... then, inside another stock's block:", mask(line)])
                    if len(ex.setdefault(key, [])) < a.show:
                        ex[key].append([mask(b) for b in block[-12:]])
                else:
                    st["other"][where] += 1
    print(f"== {a.date}: {n_so:,} SO lines; source lines that print them: {dict(files_seen)}; SO lines per snapshot block: {dict(sorted(per_block.items()))}")
    for key, k in cls.most_common(12):
        print(f"-- {k:>5}  " + " | ".join(key))
        for i, lines in enumerate(ex[key]):
            print(f"   example {i + 1}:")
            for b in lines:
                print("      " + b)
    print(f"== SO lines printed inside ANOTHER stock's block ({sum(own.values()):,}): the latest block of the order's own stock (age = SO field 2 - that block's srcEpoch)")
    for key, k in own.most_common(10):
        print(f"-- {k:>5}  " + " | ".join(key))
        for i, lines in enumerate(own_ex.get(key, [])):
            print(f"   example {i + 1}:")
            for b in lines:
                print("      " + b)
    print("[done]")


if __name__ == "__main__":
    main()

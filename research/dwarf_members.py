r"""Structures of a library's debug info that hold given member names: which class owns "tqub" / "vpen" / ... and at which offset.

Reads ``readelf --debug-dump=info <file>`` on stdin (text), nothing else: no code from the library is executed. For every
structure / class with at least one member whose name is in --names: the structure's name, byte size, and every member
with its offset. Short output, meant to be relayed.

    readelf --debug-dump=info libshtst.a | python research/dwarf_members.py --names tqub,nmvcst,tpub,adur,vpen,rvts,asf,neg_nmvcst
"""

from __future__ import annotations

import argparse
import re
import sys

TAG = re.compile(r"^\s*<(\d+)><[0-9a-f]+>: Abbrev Number: \d+ \((DW_TAG_\w+)\)")
ATTR = re.compile(r"^\s*<[0-9a-f]+>\s+(DW_AT_\w+)\s*:\s*(.*)$")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True)
    ap.add_argument("--max", type=int, default=40, help="most structures printed")
    a = ap.parse_args()
    want = set(a.names.split(","))
    structs: list[dict] = []
    stack: list[tuple[int, dict | None]] = []  # (depth, struct or member being filled)
    cur_struct: dict | None = None
    cur: dict | None = None
    for ln in sys.stdin:
        m = TAG.match(ln)
        if m:
            depth, tag = int(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            cur_struct = next((s for _, s in reversed(stack) if s is not None), None)
            if tag in ("DW_TAG_structure_type", "DW_TAG_class_type"):
                cur = {"name": "?", "size": "?", "members": []}
                structs.append(cur)
                stack.append((depth, cur))
            elif tag == "DW_TAG_member" and cur_struct is not None and stack and stack[-1][0] == depth - 1:
                cur = {"name": "?", "off": "?"}
                cur_struct["members"].append(cur)
                stack.append((depth, None))
            else:
                cur = None
                stack.append((depth, None))
            continue
        m = ATTR.match(ln)
        if m and cur is not None:
            k, v = m.group(1), m.group(2).strip()
            if k == "DW_AT_name":
                cur["name"] = v.split("): ")[-1].strip()
            elif k == "DW_AT_byte_size" and "members" in cur:
                cur["size"] = v
            elif k == "DW_AT_data_member_location" and "off" in cur:
                cur["off"] = v
    seen = set()
    n = 0
    for s in structs:
        names = [mm["name"] for mm in s["members"]]
        hit = want & set(names)
        key = (s["name"], tuple(names))
        if not hit or key in seen:
            continue
        seen.add(key)
        n += 1
        if n > a.max:
            break
        print(f"struct {s['name']} size {s['size']}: {len(names)} members, matched {sorted(hit)}")
        for mm in s["members"][:80]:
            print(f"   +{mm['off']:<6} {mm['name']}")
    print(f"[dwarf_members] {len(structs)} structures read, {n} with a wanted member")


if __name__ == "__main__":
    main()

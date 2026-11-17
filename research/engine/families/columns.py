r"""Model columns per family, from the baseline formula table (engine/baseline_r41c.txt stays the one source).

``attach`` gives every family class its ``columns``: column -> (status, formula). Exact columns are ``pinned``; columns
proven wrong by their units are ``wrong``; every other formula is a ``guessed`` stand-in, SUBJECT TO CHANGE. Families
with columns but no engine state get a generated class (``column_family``).
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import GUESSED, PINNED, WRONG, Family

BASELINE = Path(__file__).resolve().parent.parent / "baseline_r41c.txt"
EXACT = {"zzup", "zzvx.a", "zzxu.a", "zzxu.b"}
PROVEN_WRONG = {"zzqf.q.c3", "zzrg", "zzwa.g", "zzag2.e", "zzag2.a", "zzro2.a", "zzop2.b", "zzqf7.q.c2"}  # wrong units (R132)


def stem_of(col: str) -> str:
    """zzqf7.q.c2 -> zzqf."""
    return re.sub(r"\d+$", "", col.split(".")[0])


def baseline(path: Path = BASELINE) -> dict[str, str]:
    """Column -> formula."""
    out = {}
    for ln in path.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            k, _, rest = ln.partition(" ")
            out[k] = rest.strip()
    return out


def attach(families: list[type[Family]]) -> list[type[Family]]:
    """Fill ``columns`` on the given classes and return generated classes for the stems that have none."""
    by_stem = {f.stem: f for f in families}
    cols: dict[str, dict[str, tuple[str, str]]] = {}
    for col, formula in baseline().items():
        st = PINNED if col in EXACT else WRONG if col in PROVEN_WRONG else GUESSED
        cols.setdefault(stem_of(col), {})[col] = (st, formula)
    extra = []
    for stem, c in sorted(cols.items()):
        if stem in by_stem:
            by_stem[stem].columns = c
        else:
            extra.append(type(stem.capitalize(), (Family,), {"stem": stem, "status": GUESSED, "columns": c, "__doc__": f"{stem}: columns only, stand-in formulas.",
                                                              "evidence": "guess: stand-in formulas from the gated IC assignment (r41c); no PM state for this family"}))
    return extra

r"""The family registry: engine families in a fixed order (it is the rule table's order), then the column-only ones.

    from families import ENGINE, ALL, EMAS, describe
"""

from __future__ import annotations

from .base import GUESSED, OPEN, PINNED, WRONG, Family
from .cdc import Cdc
from .columns import attach
from .unbuilt import Zzcb, Zztx
from .zzab import Zzab
from .zzba import Zzba
from .zzbd import Zzbd
from .zzeb import Zzeb
from .zzeg import Zzeg
from .zzfb import Zzfb
from .zzgh import Zzgh
from .zzhj import Zzhj
from .zzjb import Zzjb
from .zzmh import Zzmh
from .zzpk import Zzpk
from .zzqj import Zzqj
from .zzui import Zzui
from .zzup import Zzup
from .zzvm import Zzvm
from .zzvx import Zzvx
from .zzxo import Zzxo
from .zzxu import Zzxu

ENGINE: list[type[Family]] = [Zzpk, Zzab, Zzup, Zzxu, Zzvx, Zzbd, Zzeb, Zzba, Zzqj, Zzeg, Zzmh, Zzvm, Zzxo, Zzui, Zzhj, Zzfb, Zzjb, Zzgh, Cdc, Zzcb, Zztx]
COLUMN_ONLY: list[type[Family]] = attach(ENGINE)
ALL: list[type[Family]] = ENGINE + COLUMN_ONLY
EMAS: dict[str, tuple[str, float, str]] = {name: rule for f in ENGINE for name, rule in f.emas.items()}


def describe() -> str:
    """One line per family: status, averages, columns by status."""
    out = []
    for f in ALL:
        c = {s: sum(1 for st, _ in f.columns.values() if st == s) for s in (PINNED, GUESSED, WRONG)}
        out.append(f"{f.stem:<6} {f.status:<8} emas {len(f.emas):>2}  columns pinned {c[PINNED]} guessed {c[GUESSED]} wrong {c[WRONG]}  {f.evidence}")
    return "\n".join(out)


if __name__ == "__main__":
    print(describe())

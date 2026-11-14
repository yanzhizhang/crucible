r"""How much of PM's model rests on the columns we already reproduce: split gain per feature, from the model text itself.

Sums ``split_gain`` over every split of every tree in each model file, by ``split_feature`` (index into the model's
feature list), and splits the total between the columns of --formulas (the settled ones, e.g. baseline_r41c.txt's 144)
and the rest. Prints the shares, the split counts, and the unsettled columns ranked by gain with their cumulative share.
Read-only.

    python research/featgen/gain_share.py --models $DATA/models_gen/B --date 20261112
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pred_day import feature_names, model_files  # noqa: E402

BASELINE = HERE.parent / "engine" / "baseline_r41c.txt"


def gains(path: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(gain, split count) per feature index over all trees of one model file."""
    g, k = np.zeros(n), np.zeros(n)
    feat = None
    with path.open() as f:
        for line in f:
            if line.startswith("end of trees"):
                break
            if line.startswith("split_feature="):
                feat = np.array(line.split("=", 1)[1].split(), int)
            elif line.startswith("split_gain=") and feat is not None:
                v = np.array(line.split("=", 1)[1].split(), float)
                np.add.at(g, feat, v)
                np.add.at(k, feat, 1)
                feat = None
    return g, k


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--date", required=True, help="picks the model month as ic_score does")
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    names = feature_names(a.feature_csv)
    have = {ln.split(" ", 1)[0] for ln in a.formulas.read_text().splitlines() if ln and not ln.startswith("#")}
    files = model_files(a.models, a.date, a.fits.split(","), a.subs.split(","))
    g, k = np.zeros(len(names)), np.zeros(len(names))
    for f in files:
        gi, ki = gains(f, len(names))
        g, k = g + gi, k + ki
    tot = g.sum()
    ok = np.array([n in have for n in names])
    print(f"models {[str(f.relative_to(a.models)) for f in files]}; {len(names)} model columns, {int(ok.sum())} settled ({a.formulas.name}), "
          f"{int((~ok).sum())} not; {int(k.sum()):,} splits")
    print(f"gain share: settled {g[ok].sum() / tot:.4f}, not settled {g[~ok].sum() / tot:.4f}; split share: settled {k[ok].sum() / k.sum():.4f}")
    print(f"columns never split on: settled {int((k[ok] == 0).sum())}, not settled {int((k[~ok] == 0).sum())}")
    order = np.argsort(-g)
    rank = {int(i): r + 1 for r, i in enumerate(order)}
    miss = [i for i in order if not ok[i]]
    cum = np.cumsum(g[miss]) / tot
    print(f"not-settled columns by gain (rank among all {len(names)}, gain share, cumulative share of the whole model, splits):")
    for j, i in enumerate(miss[: a.top]):
        print(f"  {rank[int(i)]:>4}  {names[i]:<40} {g[i] / tot:.4f}  cum {cum[j]:.4f}  splits {int(k[i]):,}")
    print("top 15 columns overall: " + ", ".join(f"{names[i]}{'' if ok[i] else '*'} {g[i] / tot:.3f}" for i in order[:15]) + "   (* = not settled)")


if __name__ == "__main__":
    main()

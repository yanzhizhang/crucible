r"""featvec (the formula layer) against ic_score's own transforms, on random base values and the real baseline formulas.

    python research/featgen/selftest_vec.py                         # build, run, compare (Linux)
    python research/featgen/selftest_vec.py --stage make|check --work W
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import featio as IO  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "engine" / "baseline_r41c.txt"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["all", "make", "check"], default="all")
    ap.add_argument("--work", type=Path, default=Path("data/featvec_selftest"))
    ap.add_argument("--stocks", type=int, default=300)
    ap.add_argument("--times", type=int, default=60)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    forms = IO.read_formulas(BASELINE)
    if a.stage in ("all", "make"):
        import re

        rng = np.random.default_rng(3)
        names = sorted({k for _, e in forms for k in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", e.removeprefix("cal:"))
                        if "." in k or k == "px"} - {"1e4"})
        names = [k for k in names if not k.endswith(":")]
        ns, nt = a.stocks, a.times
        val = rng.normal(0, 1, (ns, len(names), nt)) * rng.choice([1e-4, 1.0, 50.0], (1, len(names), 1))
        val[:, [i for i, k in enumerate(names) if not k.startswith("gen.")], :] = rng.uniform(5, 50, (ns, 1, nt))  # prices
        val = np.where(rng.random(val.shape) < 0.03, np.nan, val)
        val[:, 0, :] = np.round(val[:, 0, :], 1)  # ties for the ranks
        val[:, :, 0] = np.nan  # a time with nothing at all
        industry = rng.integers(0, 28, ns)
        industry[rng.random(ns) < 0.05] = -1
        dummies = np.zeros((ns, 28))
        dummies[np.arange(ns)[industry >= 0], industry[industry >= 0]] = 1.0
        expo = np.column_stack([dummies, rng.normal(0, 1, (ns, 10))])
        expo[industry < 0] = np.nan  # no CED row
        cal = {}
        for col, e in forms:
            if e.startswith("cal:") and rng.random() < 0.9:  # a few calibrated columns without training points
                t = np.sort(rng.normal(0, 1, 40))
                cal[col] = (rng.permutation(t), np.clip(rng.uniform(0, 1, 40), 0, 1))
        IO.write_vec_inputs(a.work, np.arange(ns) + 1, industry, expo, names, val, cal)
        ref = IO.vec_reference(forms, names, val, industry, expo, cal)
        np.save(a.work / "vec_ref.npy", ref)
        print(f"{len(forms)} formulas over {len(names)} names, {ns} stocks x {nt} times -> {a.work}")
    if a.stage == "all":
        IO.run_vec(a.work, BASELINE)
    if a.stage in ("all", "check"):
        ref, got = np.load(a.work / "vec_ref.npy"), IO.read_vec_output(a.work)
        bad = 0
        for c, (col, e) in enumerate(forms):
            fa, fb = np.isfinite(got[c]), np.isfinite(ref[c])
            both = fa & fb
            d = float(np.max(np.abs(got[c][both] - ref[c][both]) / np.maximum(np.abs(ref[c][both]), 1e-9))) if both.any() else 0.0
            off = (fa != fb).any() or d > 1e-7 or not fb.any()
            bad += off
            if off:
                print(f"  OFF {col:<12} {e:<60} finite ref {int(fb.sum())} got {int(fa.sum())} NaN mismatches {int((fa != fb).sum())} max rel diff {d:.3g}")
        print(f"{len(forms)} columns: {bad} off")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

r"""What PM's model files say about their own training: months per generation, training-row counts, training parameters.

Per model root (generation dirs and the production tk5m), per month, per fit / sub-model file:
  - trees, features, the root internal_count of the first trees (= rows the tree was grown on, after bagging),
    and the largest root count over all trees (bagging_fraction < 1 shrinks it; the max is closest to the full set);
  - the first tree's leaf values mean (with boost_from_average, the label mean of the training set);
  - the 'parameters:' block lines that can locate the data: data / valid paths, bagging, bin construction, seeds,
    max_bin, objective, num_iterations;
  - the file's own mtime.
Then, per root, the four PM-state columns' training [min:max] by month (shows when the window moves).
Read-only.

    python research/featgen/model_meta.py --roots /work/crucible_data/models_gen/A,/work/crucible_data/models_gen/B,/work/crucible_data/models_gen/C,/data/prod/models/tk5m
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pred_day as P  # noqa: E402

KEEP = re.compile(r"^\[(data|valid|input_model|bagging_fraction|bagging_freq|bagging_seed|feature_fraction|bin_construct_sample_cnt|data_random_seed|"
                  r"seed|max_bin|min_data_in_bin|objective|num_iterations|learning_rate|num_leaves|boost_from_average|header|label_column|"
                  r"weight_column|group_column|categorical_feature|forcedbins_filename|linear_tree|use_missing|zero_as_missing)[: ]")
PM_STATE = ["zzup", "zzvx.a", "zzxu.a", "zzxu.b"]


def describe(f: Path, names: list[str]) -> dict:
    """One model file's training facts."""
    text = f.read_text(errors="replace")
    head = text.split("\nTree=0", 1)[0]
    hd = dict(ln.split("=", 1) for ln in head.splitlines() if "=" in ln)
    roots, lv0 = [], None
    for blk in re.split(r"\nTree=\d+\n", text)[1:]:
        kv = dict(ln.split("=", 1) for ln in blk.splitlines() if "=" in ln)
        if "internal_count" in kv:
            roots.append(float(kv["internal_count"].split()[0]))
        elif "leaf_count" in kv:
            roots.append(float(kv["leaf_count"].split()[0]))
        if lv0 is None and "leaf_value" in kv:
            lv0 = float(np.mean([float(x) for x in kv["leaf_value"].split()]))
    params = []
    if "\nparameters:" in text:
        for ln in text.split("\nparameters:", 1)[1].splitlines():
            if ln.startswith("end of parameters"):
                break
            if KEEP.match(ln.strip()):
                params.append(ln.strip())
    infos = hd.get("feature_infos", "").split()
    st = {}
    for c in PM_STATE:
        if c in names and names.index(c) < len(infos):
            st[c] = infos[names.index(c)]
    return {"trees": len(roots), "features": len((hd.get("feature_names") or "").split()), "root0": roots[:3], "root_max": max(roots) if roots else 0,
            "lv0": lv0, "params": params, "mtime": dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"), "state": st}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--months", type=int, default=0, help="per root only the last N months (0 = all)")
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    for root in [Path(r) for r in a.roots.split(",") if r]:
        if not root.exists():
            print(f"######## {root}: missing")
            continue
        months = sorted(p.name for p in root.iterdir() if re.fullmatch(r"\d{6}", p.name))
        print(f"######## {root}: {len(months)} months {months[:3]}..{months[-3:]}")
        sel = months[-a.months:] if a.months else months
        shown_params = False
        for m in sel:
            for f in sorted((root / m / "lgb").glob("*/model_*.txt")):
                d = describe(f, names)
                print(f"   {m} {f.parent.name}/{f.name}: trees {d['trees']} feats {d['features']} root rows {[int(x) for x in d['root0']]} max {int(d['root_max'])}"
                      f"  tree0 leaf mean {d['lv0']:.6g}  mtime {d['mtime']}" if d["lv0"] is not None else f"   {m} {f.name}: no trees")
                if not shown_params and d["params"]:
                    print("      parameters: " + " | ".join(d["params"]))
                    shown_params = True
            fs = sorted((root / m / "lgb").glob("*/model_s1.txt"))
            if fs:
                d = describe(fs[0], names)
                print(f"      {m} PM-state ranges ({fs[0].parent.name}/s1): " + "  ".join(f"{k} {v}" for k, v in d["state"].items()))
    print("[done]")


if __name__ == "__main__":
    main()

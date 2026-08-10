"""Stage 0d — metadata EDA. Runs in seconds, needs zero audio.

Establishes the three facts that constrain the whole design:
  1. how many labelled (aviary, species) points actually exist   -> n=8
  2. which aviaries are duplicates of each other                 -> CV grouping
  3. the count range per species                                 -> saturation risk

Run this FIRST, before any download finishes.

Usage:
    python -m src.eda_metadata
"""
from __future__ import annotations

import io

import pandas as pd
from huggingface_hub import hf_hub_download

from .common import duplicate_group_map, load_config, resolve_dir

DEV_GT = "metadata/ground_truth.csv"
EVAL_GT = "metadata/eval_ground_truth.csv"
DEV_INFO = "metadata/recording_info.csv"
EVAL_INFO = "metadata/eval_recording_info.csv"


def grab(repo_id: str, fname: str, cache_dir) -> pd.DataFrame:
    p = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fname,
                        local_dir=cache_dir)
    return pd.read_csv(p)


def main() -> None:
    cfg = load_config()
    cache_dir = resolve_dir(cfg, "cache")
    results = resolve_dir(cfg, "results")
    rid = cfg["repo_id"]

    dev_gt = grab(rid, DEV_GT, cache_dir)
    eval_gt = grab(rid, EVAL_GT, cache_dir)
    dev_info = grab(rid, DEV_INFO, cache_dir)
    eval_info = grab(rid, EVAL_INFO, cache_dir)

    print("=" * 70)
    print("FACT 1 — how much supervision do we actually have?")
    print("=" * 70)
    dev_t = dev_gt[dev_gt.is_target == 1]
    print(f"\nDev files:            {dev_info.n_files.sum():,}")
    print(f"Dev LABELLED points:  {len(dev_t)}   <-- this is the training set size")
    print("\nPer species:")
    for sp, g in dev_t.groupby("common_name"):
        print(f"  {sp:<20} n={len(g)}  counts={sorted(g['count'].tolist())}")
    print("\n-> Any model with more free parameters than this will not generalise.")

    print("\n" + "=" * 70)
    print("FACT 2 — duplicate aviaries (CV leakage risk)")
    print("=" * 70)
    allgt = pd.concat([dev_gt.assign(split="dev"), eval_gt.assign(split="eval")])
    sig = (
        allgt.sort_values(["aviary_id", "scientific_name"])
        .groupby("aviary_id")
        .apply(lambda d: "|".join(f"{r.scientific_name}:{r['count']}" for _, r in d.iterrows()),
               include_groups=False)
    )
    dupes = sig[sig.duplicated(keep=False)].reset_index(name="sig")
    if len(dupes):
        print("\nAviaries sharing a byte-identical species inventory:")
        for s, g in dupes.groupby("sig"):
            print(f"  {sorted(g.aviary_id.tolist())}")
    print("\nCanonical grouping used for CV folds:")
    canon = duplicate_group_map(cfg)
    for a, c in sorted(canon.items()):
        if a != c:
            print(f"  {a} -> {c}")

    print("\n" + "=" * 70)
    print("FACT 3 — count range per target species")
    print("=" * 70)
    both = pd.concat([
        dev_t.assign(split="dev"),
        eval_gt[eval_gt.is_target.isin([1, 2])].assign(split="eval"),
    ])
    rng = both.groupby("common_name")["count"].agg(["min", "max", "count"])
    print("\n" + rng.to_string())
    print("\n-> Two orders of magnitude. A single linear model cannot span this.")

    zero_shot = set(eval_gt[eval_gt.is_target == 2].common_name) - set(dev_t.common_name)
    if zero_shot:
        print(f"\n!! ZERO-SHOT species (no dev calibration at all): {sorted(zero_shot)}")

    out = results / "eda_summary.csv"
    both.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

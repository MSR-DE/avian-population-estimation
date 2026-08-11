## Stage 1a — build a per-aviary species whitelist for the detector.
##
## WHY THIS EXISTS
## ARIA's ZooCustom model knows 87 species. Any given aviary contains ~2-22 of
## them. Every one of the other ~70 is a potential false positive.
##
## The damage is not evenly spread. The model vocabulary contains:
##     Phoenicopterus roseus_Greater Flamingo     <- our target
##     Phoenicopterus chilensis_Chilean Flamingo  <- congener, similar calls
##     Phoenicopterus ruber_American Flamingo     <- congener, similar calls
## None of the dev aviaries hold Chilean or American flamingos, so every such
## detection is noise being subtracted from the signal we care about.
##
## It matters most where counts are small. Hadada ibis is N=4; a handful of
## false detections moves the estimate by 25%.
##
## ground_truth.csv gives the EXACT species inventory of every aviary, target
## and non-target alike. Restricting the detector to that list is free accuracy.
##
## FORMAT
## We do not guess the whitelist format. We read the model's own label file
## (ZooCustom_v1_Labels.txt, lines like "Quelea quelea_Red-billed Quelea") and
## emit the matching lines verbatim, joined on scientific name. That guarantees
## the format is right and, as a side effect, tells us which species in an
## aviary the model cannot detect at all.
##
## Usage:
##     python -m src.species_lists --labels ~/aria-models/ZooCustom_v1_Labels.txt

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import load_config, resolve_dir


def load_labels(labels_path: Path) -> dict[str, str]:
    """Read the model's label file into {scientific_name: full_label_line}.

    Lines look like:  Quelea quelea_Red-billed Quelea
                      ^scientific    ^common
    We split on the FIRST underscore only — common names contain hyphens and
    spaces but the scientific binomial never contains an underscore.
    """
    mapping: dict[str, str] = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        sci = line.split("_", 1)[0].strip()
        mapping[sci.lower()] = line
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path,
                    help="path to ZooCustom_v1_Labels.txt")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = resolve_dir(cfg, "cache")
    out_dir = cache_dir / "species_lists"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) the model's vocabulary
    labels = load_labels(args.labels)
    print(f"Model vocabulary: {len(labels)} species\n")

    # 2) per-aviary inventories, dev and eval together
    frames = []
    for name in ("ground_truth.csv", "eval_ground_truth.csv"):
        p = cache_dir / "metadata" / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit(
            "No ground-truth CSVs in cache/metadata/. Run: python -m src.eda_metadata"
        )
    gt = pd.concat(frames, ignore_index=True)

    # 3) one whitelist file per aviary
    print(f"{'aviary':<16}{'species':>8}{'in model':>10}{'MISSING':>9}")
    print("-" * 45)
    summary = []
    for aviary, g in gt.groupby("aviary_id"):
        present, missing = [], []
        for _, r in g.iterrows():
            key = str(r.scientific_name).strip().lower()
            if key in labels:
                present.append(labels[key])
            else:
                ## The model simply cannot detect this species. If it is a
                ## TARGET that is fatal for this aviary; if it is a non-target
                ## it is harmless (we were going to ignore it anyway).
                missing.append((r.common_name, r.scientific_name, int(r.is_target)))

        out = out_dir / f"{aviary}.txt"
        out.write_text("\n".join(sorted(set(present))) + "\n", encoding="utf-8")

        flag = ""
        if any(t > 0 for _, _, t in missing):
            flag = "  <-- TARGET MISSING"
        print(f"{aviary:<16}{len(g):>8}{len(present):>10}{len(missing):>9}{flag}")

        for common, sci, is_t in missing:
            if is_t > 0:
                print(f"    !! TARGET not in model vocabulary: {common} ({sci})")
        summary.append({"aviary_id": aviary, "n_species": len(g),
                        "n_in_model": len(present), "n_missing": len(missing)})

    pd.DataFrame(summary).to_csv(out_dir / "_summary.csv", index=False)
    print(f"\nWrote {len(summary)} whitelists to {out_dir}")
    print("Pass one to the detector with --allowed-species-file")


if __name__ == "__main__":
    main()

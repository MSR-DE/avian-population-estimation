## Stage 4 — how much does the answer depend on an arbitrary choice?
##
## THIS IS NOT A TUNING SCRIPT. Read that twice.
##
## It sweeps the detector's confidence threshold and records CV MAE at each
## value. The tempting move is to pick the threshold with the lowest MAE and
## report that. Doing so would be overfitting — the exact failure this whole
## project is designed around.
##
## Why it would be overfitting: with 2-3 labelled aviaries per species, the
## sweep has more effective freedom than the data can support. Red-billed
## quelea illustrates it precisely. Its CV MAE across thresholds runs:
##     0.3 -> 7.43   0.5 -> 14.95   0.7 -> 0.32   0.8 -> 9.36   0.9 -> 25.45
## The 0.32 is not a discovery. With n=2, leave-one-out fits a single scale
## parameter on one point and predicts the other; some threshold will make that
## land almost exactly by chance. Selecting it would be reporting luck.
##
## So the headline number in the report uses the UNTUNED default (0.3, which is
## aria-inference's own `--min-confidence` default and what detection actually
## ran at). This sweep is reported as a SENSITIVITY ANALYSIS: it quantifies how
## unstable the estimate is, which is a limitation worth stating plainly rather
## than an opportunity to be exploited.
##
## Usage:
##     python -m src.sensitivity

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import duplicate_group_map, load_config, resolve_dir
from .estimate import MODELS, loao_cv

THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]
FEATURES = ["occ_per_sec", "rate"]
UNTUNED = 0.3  ## aria-inference's --min-confidence default; chosen a priori


def features_at(det, files, gt, canon, thresh: float) -> pd.DataFrame:
    rows = []
    for _, t in gt.iterrows():
        f = files[files.aviary == t.aviary_id]
        if f.empty:
            continue
        n, mean_clip = len(f), f.clip_seconds.mean()
        d = det[(det.aviary == t.aviary_id) & (det.sp == t.sp)
                & (det.confidence >= thresh)]
        p = d.filename.nunique() / n
        rows.append({
            "aviary": t.aviary_id, "species": t.common_name,
            "count": int(t["count"]), "p_active": p,
            "occ_per_sec": -np.log(1 - min(p, 1 - 1e-4)) / mean_clip,
            "rate": len(d) / f.clip_seconds.sum(),
            "cv_group": canon.get(t.aviary_id, t.aviary_id),
        })
    return pd.DataFrame(rows)


def score(F: pd.DataFrame) -> tuple[float, dict]:
    """Best (feature, model) per species by grouped LOAO-CV; combined MAE."""
    preds, per_sp = [], {}
    for sp, g in F.groupby("species"):
        best = (np.inf, None, None)
        for c in FEATURES:
            for m in MODELS:
                mae, pr = loao_cv(g, c, m)
                if mae == mae and mae < best[0]:
                    best = (mae, pr, c)
        if best[1]:
            preds += best[1]
            per_sp[sp] = (best[0], best[2], g.p_active.mean())
    combined = float(np.mean([abs(t - p) for _, t, p in preds])) if preds else np.nan
    return combined, per_sp


def main() -> None:
    cfg = load_config()
    cache, results = resolve_dir(cfg, "cache"), resolve_dir(cfg, "results")
    canon = duplicate_group_map(cfg)

    det = pd.read_parquet(cache / "detections_dev.parquet")
    files = pd.read_parquet(cache / "files_dev.parquet")
    gt = pd.read_csv(cache / "metadata" / "ground_truth.csv")
    gt = gt[(gt.is_target == 1) & (gt.aviary_id.isin(files.aviary.unique()))].copy()
    det["sp"] = det.species_common.str.lower()
    gt["sp"] = gt.common_name.str.lower()

    rows = []
    for th in THRESHOLDS:
        F = features_at(det, files, gt, canon, th)
        combined, per_sp = score(F)
        for sp, (mae, feat, p) in per_sp.items():
            rows.append({"threshold": th, "species": sp, "mae": round(mae, 2),
                         "feature": feat, "mean_p_active": round(p, 3),
                         "combined_mae": round(combined, 2)})
    out = pd.DataFrame(rows)
    out.to_csv(results / "threshold_sensitivity.csv", index=False)

    piv = out.pivot_table(index="threshold", columns="species", values="mae")
    piv["COMBINED"] = out.groupby("threshold").combined_mae.first()
    print("CV MAE vs detector confidence threshold")
    print(piv.round(2).to_string())

    base = out[out.threshold == UNTUNED].combined_mae.iloc[0]
    print(f"\nREPORTED (untuned, threshold={UNTUNED}): MAE {base:.2f}")
    print(f"range across thresholds: {out.combined_mae.min():.2f} - "
          f"{out.combined_mae.max():.2f}")
    print("\nThe spread IS the result: with 2-3 labelled aviaries per species,")
    print("an arbitrary threshold choice moves combined MAE by ~2x. Selecting")
    print("the minimum would report luck, not skill.")
    print(f"\nWrote {results / 'threshold_sensitivity.csv'}")


if __name__ == "__main__":
    main()

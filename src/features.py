## Stage 2 — collapse ~50,000 detections into ONE ROW per (aviary, species).
##
## This is where 18,237 files become 8 training rows. Everything upstream was
## measurement; everything downstream is a 1-2 parameter fit. So the features
## have to carry the signal — the regressor cannot rescue a bad one.
##
## THE CORE PROBLEM THEY MUST SOLVE
## Detection rate SATURATES. 195 flamingos honking synchronously trigger the
## detector about as often as 100 do, because the detector fires once per
## overlapping chorus. Rate alone therefore cannot span 4 -> 195 birds.
##
## So we compute three families and let the ablation decide which survives:
##   A. RATE       detections per second of audio. Simple, saturates.
##   B. OCCUPANCY  p_active = fraction of clips containing >=1 call.
##                 If N birds each call independently at rate L, then
##                     p = 1 - exp(-N*L*t)   =>   N  ~  -ln(1 - p) / (L*t)
##                 so -ln(1-p) is a DERIVED de-saturating transform with one
##                 free parameter. Unlike a fitted curve it keeps a sensible
##                 shape outside the calibration range, which matters because
##                 eval asks for flamingo=11 when dev only saw 52-161.
##   C. CONFIDENCE with more birds, calls overlap and get messier; the shape of
##                 the confidence distribution should shift.
##
## NORMALISATION — the thing that is easy to get wrong
## Every rate is per SECOND OF AUDIO ANALYSED, never per file. dev_aviary_2's
## clips are 5.25 s vs 2.75 s elsewhere, so identical birds would yield ~1.9x
## the per-file detections there. The denominator comes from files_*.parquet,
## which includes clips where NOTHING was detected — silence is evidence, and
## it is invisible in the detections table.
##
## Usage:
##     python -m src.features --split dev

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common import duplicate_group_map, load_config, resolve_dir

## Gaps longer than this are treated as "between bouts, or across a sampling
## window boundary" and excluded. We sampled contiguous 10-minute windows, so a
## gap of 600 s+ is an artifact of the sampler, not bird behaviour.
MAX_GAP_SECONDS = 600.0


def norm(s: str) -> str:
    """Species names differ in case between sources.

    ground_truth.csv : "Red-billed quelea"   (sentence case)
    ARIA output      : "Red-billed Quelea"   (title case)
    Join on lowercase or you silently match nothing and every count reads zero.
    """
    return str(s).strip().lower()


def build(det: pd.DataFrame, files: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    det = det.copy()
    det["sp"] = det["species_common"].map(norm)
    gt = gt.copy()
    gt["sp"] = gt["common_name"].map(norm)

    rows = []
    # one row per (aviary, TARGET species) — non-targets are context, not labels
    for _, t in gt[gt.is_target > 0].iterrows():
        aviary, sp = t.aviary_id, t.sp

        f = files[files.aviary == aviary]
        if f.empty:
            continue
        d = det[(det.aviary == aviary) & (det.sp == sp)]

        n_files = len(f)
        audio_sec = float(f.clip_seconds.sum())
        mean_clip = audio_sec / n_files if n_files else 0.0

        # ---- A. RATE ------------------------------------------------------
        rate = len(d) / audio_sec if audio_sec else 0.0
        ## high-confidence only: strips marginal detections that inflate rate
        ## in noisy aviaries without indicating more birds
        hi = d[d.confidence >= 0.9]
        rate_hi = len(hi) / audio_sec if audio_sec else 0.0

        # ---- B. OCCUPANCY -------------------------------------------------
        ## p_active needs ALL analysed files in the denominator, silent ones
        ## included. That is why files_*.parquet exists.
        active = d.filename.nunique()
        p_active = active / n_files if n_files else 0.0
        ## clip p just below 1: -ln(0) is infinite, and p==1 means the signal
        ## has fully saturated (every clip has a call) so we cannot resolve
        ## further. Record it rather than crashing.
        p_clip = min(p_active, 1 - 1e-4)
        occupancy = -np.log(1 - p_clip)

        p_hi = hi.filename.nunique() / n_files if n_files else 0.0
        occupancy_hi = -np.log(1 - min(p_hi, 1 - 1e-4))

        # ---- C. CONFIDENCE ------------------------------------------------
        conf = d.confidence.values if len(d) else np.array([0.0])
        # ---- multiplicity: detections per ACTIVE clip ----------------------
        ## a clip with several detections suggests several birds calling at
        ## once; this keeps rising after p_active has hit its ceiling
        per_active = len(d) / active if active else 0.0

        # ---- bout structure ------------------------------------------------
        ## gaps between consecutive detections within a day, censored at the
        ## sampling-window length (see MAX_GAP_SECONDS)
        if len(d) > 1:
            g = []
            for _, day_df in d.groupby("day"):
                ts = np.sort(day_df.t_sec.dropna().astype(float).values)
                if len(ts) > 1:
                    diffs = np.diff(ts)
                    g.append(diffs[diffs <= MAX_GAP_SECONDS])
            gaps = np.concatenate(g) if g else np.array([])
        else:
            gaps = np.array([])

        rows.append({
            "aviary": aviary,
            "species": t.common_name,
            "count": int(t["count"]),
            "is_target": int(t.is_target),
            "n_files": n_files,
            "audio_hours": round(audio_sec / 3600, 3),
            "n_detections": len(d),
            # A
            "rate": rate,
            "rate_hi": rate_hi,
            "log_rate": np.log1p(rate),
            # B
            "p_active": p_active,
            "occupancy": occupancy,
            "p_active_hi": p_hi,
            "occupancy_hi": occupancy_hi,
            ## occupancy is STILL clip-length dependent: a longer clip is more
            ## likely to contain a call regardless of how many birds there are.
            ## From p = 1 - exp(-N*L*t), the quantity -ln(1-p) equals N*L*t, so
            ## dividing by t leaves N*L — proportional to population, and
            ## finally comparable across aviaries with different clip lengths.
            ## Verified on synthetic data: without this, aviary_2 (N=107,
            ## 5.25 s) outranks aviary_4 (N=161, 2.75 s), i.e. backwards.
            "occ_per_sec": occupancy / mean_clip if mean_clip else 0.0,
            "occ_per_sec_hi": occupancy_hi / mean_clip if mean_clip else 0.0,
            # C
            "conf_mean": float(conf.mean()),
            "conf_p90": float(np.percentile(conf, 90)),
            "per_active_clip": per_active,
            # bout
            "gap_median": float(np.median(gaps)) if len(gaps) else np.nan,
            "gap_p10": float(np.percentile(gaps, 10)) if len(gaps) else np.nan,
            "n_gaps": len(gaps),
        })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache = resolve_dir(cfg, "cache")
    results = resolve_dir(cfg, "results")

    det = pd.read_parquet(cache / f"detections_{args.split}.parquet")
    files = pd.read_parquet(cache / f"files_{args.split}.parquet")

    gt_name = "ground_truth.csv" if args.split == "dev" else "eval_ground_truth.csv"
    gt = pd.read_csv(cache / "metadata" / gt_name)
    gt = gt[gt.aviary_id.isin(files.aviary.unique())]

    feats = build(det, files, gt)

    ## group duplicated aviaries so cross-validation never validates on a
    ## physical enclosure it was calibrated on
    canon = duplicate_group_map(cfg)
    feats["cv_group"] = feats.aviary.map(canon)

    out = results / f"features_{args.split}.csv"
    feats.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(feats[["aviary", "species", "count", "n_files", "n_detections",
                 "rate", "p_active", "occupancy", "per_active_clip"]]
          .to_string(index=False))
    print(f"\nWrote {out}  ({len(feats)} rows — this is the whole training set)")

    # --- does anything actually correlate with count? -----------------------
    ## With 2-4 points per species a correlation is indicative, not conclusive.
    ## Printed to steer the modelling, not to justify it.
    print("\nSpearman correlation with count, per species:")
    cand = ["rate", "rate_hi", "p_active", "occupancy", "occupancy_hi",
            "per_active_clip", "conf_mean", "gap_median"]
    for sp, g in feats.groupby("species"):
        if len(g) < 2:
            continue
        cors = {c: g[c].corr(g["count"], method="spearman") for c in cand
                if g[c].notna().all()}
        best = sorted(cors.items(), key=lambda kv: -abs(kv[1] if kv[1] == kv[1] else 0))
        s = ", ".join(f"{k}={v:+.2f}" for k, v in best[:4])
        print(f"  {sp:<20} (n={len(g)})  {s}")


if __name__ == "__main__":
    main()

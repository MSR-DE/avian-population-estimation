## Stage 3 — fit the count model and validate it honestly.
##
## THE CONSTRAINT THAT DEFINES THIS FILE
## Eight labelled points. Fitted PER SPECIES, so the real budget is:
##     Greater flamingo   4 rows (2 are the same physical aviary) -> 3
##     Red-billed quelea  2
##     Hadada ibis        2
## A two-parameter model on two points has zero degrees of freedom. So every
## candidate here has ONE free parameter, except the 2-parameter log-log which
## is included specifically to show it overfits.
##
## THE MODELS
##   mean       predict the species' mean count. Not a joke — the honest
##              baseline. If nothing beats this, the acoustic features carry
##              no usable signal and the write-up should say so.
##   scale      N = a * x                (1 param, through the origin)
##   loglog     N = a * x^b              (2 params — expected to overfit)
##   occupancy  N = a * -ln(1 - p)       (1 param, DERIVED not fitted)
##
## The occupancy link comes from a generative story: if N birds each call
## independently at rate L, the chance a clip of length t contains >=1 call is
## p = 1 - exp(-N*L*t). Invert it and N is proportional to -ln(1-p). Its SHAPE
## is therefore fixed by the assumption, and only its scale is fitted. That
## matters because the evaluation set asks for flamingo=11 while dev only ever
## saw 52-161 — a fitted curve is unconstrained below its training range and
## can return anything; a derived one stays sane.
##
## VALIDATION
## Leave-one-aviary-out, GROUPED BY PHYSICAL AVIARY. dev_aviary_5 and
## dev_aviary_6 have byte-identical ground truth, so training on one and
## validating on the other would be validating on training data.
##
## Usage:
##     python -m src.estimate
##     python -m src.estimate --feature rate

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from .common import load_config, resolve_dir

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Model family. Each returns a predict() closure fitted on (x, y).
# ---------------------------------------------------------------------------
def fit_mean(x, y):
    m = float(np.mean(y))
    return lambda xn: np.full(np.shape(xn), m)


def fit_scale(x, y):
    """N = a*x through the origin. Least-squares a = <x,y>/<x,x>."""
    x = np.asarray(x, float)
    denom = float((x * x).sum())
    a = float((x * y).sum() / denom) if denom > 0 else 0.0
    return lambda xn: a * np.asarray(xn, float)


def fit_loglog(x, y):
    """N = a*x^b, fitted linearly in log space. TWO parameters — included to
    demonstrate overfitting on 2-3 points, not because we expect it to win."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    ## A 2-parameter model needs >=2 distinct x values. For a species with only
    ## 2 labelled aviaries, leave-one-out leaves ONE training point, so the fit
    ## is not merely unstable — it is undefined (numpy raises LinAlgError).
    ## This is the sample-size constraint made concrete: with 2-3 points per
    ## species there is no room for a second parameter. We degrade to the mean
    ## baseline and let the CV table record how that performs.
    if ok.sum() < 2 or len(np.unique(x[ok])) < 2:
        return fit_mean(x, y)
    b, loga = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    return lambda xn: np.exp(loga) * np.power(np.maximum(np.asarray(xn, float), 1e-12), b)


MODELS = {"mean": fit_mean, "scale": fit_scale, "loglog": fit_loglog}


def loao_cv(df: pd.DataFrame, feature: str, model: str) -> tuple[float, list]:
    """Leave-one-aviary-out CV within one species.

    Folds are held out by cv_group (canonical physical aviary), so duplicated
    aviaries never appear on both sides of a split.
    """
    preds = []
    for grp in df.cv_group.unique():
        tr = df[df.cv_group != grp]
        te = df[df.cv_group == grp]
        if len(tr) < 1:
            continue
        f = MODELS[model](tr[feature].values, tr["count"].values)
        for _, r in te.iterrows():
            p = float(np.ravel(f(r[feature]))[0])
            ## a negative or absurd count is meaningless; clip to the plausible
            ## range rather than letting an unconstrained fit report nonsense
            p = float(np.clip(p, 1, 1000))
            preds.append((r.aviary, r["count"], p))
    if not preds:
        return np.nan, []
    mae = float(np.mean([abs(c - p) for _, c, p in preds]))
    return mae, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = resolve_dir(cfg, "results")
    feats = pd.read_csv(results / f"features_{args.split}.csv")
    feats = feats[feats.is_target == 1]  ## main leaderboard species only

    ## occ_per_sec first: it is the derived, clip-length-corrected occupancy
    ## link and the one with a generative justification behind it.
    candidates = ["occ_per_sec", "occ_per_sec_hi", "rate", "rate_hi",
                  "occupancy", "p_active", "per_active_clip"]
    candidates = [c for c in candidates if c in feats.columns and feats[c].notna().all()]

    # 1) every (feature, model) pair, scored by grouped LOAO-CV per species
    rows = []
    for sp, g in feats.groupby("species"):
        n_groups = g.cv_group.nunique()
        for feat in candidates:
            for mdl in MODELS:
                mae, _ = loao_cv(g, feat, mdl)
                rows.append({"species": sp, "n": len(g), "groups": n_groups,
                             "feature": feat, "model": mdl, "mae": mae})
    res = pd.DataFrame(rows).dropna(subset=["mae"])

    print("=" * 78)
    print("LEAVE-ONE-AVIARY-OUT CV  (folds grouped by physical aviary)")
    print("=" * 78)
    for sp, g in res.groupby("species"):
        print(f"\n{sp}  (n={g.n.iloc[0]} rows, {g.groups.iloc[0]} independent aviaries)")
        top = g.sort_values("mae").head(6)
        for _, r in top.iterrows():
            print(f"    {r.feature:<18}{r.model:<10}MAE {r.mae:8.2f}")
        base = g[g.model == "mean"].mae.min()
        best = g.mae.min()
        verdict = "beats" if best < base else "DOES NOT BEAT"
        print(f"    -> best {best:.2f} vs mean-baseline {base:.2f}  ({verdict} baseline)")

    # 2) pick ONE (feature, model) per species by CV, then report combined MAE
    print("\n" + "=" * 78)
    print("SELECTED MODEL PER SPECIES, AND COMBINED MAE")
    print("=" * 78)
    all_preds, chosen = [], []
    for sp, g in res.groupby("species"):
        b = g.sort_values("mae").iloc[0]
        _, preds = loao_cv(feats[feats.species == sp], b.feature, b.model)
        all_preds += preds
        chosen.append({"species": sp, "feature": b.feature, "model": b.model,
                       "cv_mae": round(b.mae, 2)})
        print(f"\n{sp}: {b.feature} + {b.model}  (CV MAE {b.mae:.2f})")
        for av, true, pred in preds:
            print(f"    {av:<16} true {true:>5}   pred {pred:8.1f}   err {pred-true:+8.1f}")

    if all_preds:
        errs = [abs(c - p) for _, c, p in all_preds]
        mae = float(np.mean(errs))
        rmse = float(np.sqrt(np.mean([(c - p) ** 2 for _, c, p in all_preds])))
        mape = float(np.mean([abs(c - p) / c * 100 for _, c, p in all_preds]))
        print("\n" + "-" * 78)
        print(f"COMBINED  MAE {mae:.2f}   RMSE {rmse:.2f}   MAPE {mape:.1f}%")
        print(f"Official baseline (dev): MAE 11.50, MAPE 10.6%")
        print("-" * 78)

        pd.DataFrame(all_preds, columns=["aviary", "true", "pred"]).to_csv(
            results / "cv_predictions.csv", index=False)
        pd.DataFrame(chosen).to_csv(results / "selected_models.csv", index=False)
        res.sort_values(["species", "mae"]).to_csv(
            results / "ablation.csv", index=False)
        print(f"\nWrote cv_predictions.csv, selected_models.csv, ablation.csv to {results}")


if __name__ == "__main__":
    main()

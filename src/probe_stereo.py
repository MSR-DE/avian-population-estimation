## Stage 0d — are the two channels actually DIFFERENT?
##
## WHY THIS MATTERS
## probe_audio.py showed the clips are stereo, which the challenge brief never
## mentions. But "stereo" can mean two very different things:
##
##   (a) DUAL MONO — one microphone, copied into both channels. Common when a
##       recorder writes stereo files regardless of input. Carries NO extra
##       information. We mix to mono and move on.
##
##   (b) TWO SPACED MICROPHONES — genuinely different signals. Then the small
##       time difference between when a sound reaches each mic tells you the
##       DIRECTION it came from.
##
## If (b) holds, it is potentially a big deal for this task. The known failure
## mode of acoustic counting is saturation: 195 flamingos honking together
## produce roughly the same detection RATE as 100, because the detector fires
## once per overlapping chorus. But 195 birds are spread over more physical
## positions than 100 are. The SPREAD of inter-channel delays should keep
## growing after the rate has flattened.
##
## That is a testable hypothesis the official baseline (which mixes to mono)
## cannot express. Which is why it's worth ten minutes to check.
##
## WHAT WE MEASURE
##   max|L-R|      -> exactly 0 means dual mono, no information
##   correlation   -> ~1.0 means near-identical; lower means genuinely different
##   RMS ratio     -> persistent level difference suggests mic gain/placement
##   peak xcorr lag-> nonzero, VARYING lag means spatial separation
##
## Sound travels ~343 m/s. At 48 kHz one sample is ~7 mm of path difference,
## so mics spaced 20 cm give delays up to ~28 samples. We search +/- 200.
##
## Usage:
##     python -m src.probe_stereo

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf

from .common import load_config, resolve_dir

MAX_LAG = 200  ## samples; ~1.4 m of path difference at 48 kHz — generous


def best_lag(left: np.ndarray, right: np.ndarray, max_lag: int = MAX_LAG) -> tuple[int, float]:
    """Find the sample delay between the channels via cross-correlation.

    Returns (lag_in_samples, normalised_peak_height).

    A positive lag means the sound reached the LEFT mic first.
    """
    # 1) remove DC offset — otherwise a constant bias dominates the correlation
    l = left - left.mean()
    r = right - right.mean()

    # 2) guard against silent clips (all zeros -> divide by zero later)
    denom = np.sqrt((l**2).sum() * (r**2).sum())
    if denom == 0:
        return 0, 0.0

    # 3) 'full' correlation gives every possible overlap of the two signals.
    ##   The middle element is zero-lag; index 0 is maximum negative lag.
    corr = np.correlate(l, r, mode="full") / denom
    mid = len(l) - 1

    # 4) only look at physically plausible lags — a bird cannot produce a
    ##   half-second inter-mic delay, so a peak out there is noise.
    lo, hi = mid - max_lag, mid + max_lag + 1
    window = corr[lo:hi]
    k = int(np.argmax(np.abs(window)))
    return k - max_lag, float(window[k])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = resolve_dir(cfg, "data")

    wavs = sorted(data_dir.rglob("*.wav"))[: args.n]
    if not wavs:
        raise SystemExit(f"No .wav files under {data_dir}. Download some first.")

    print(f"{'file':<34}{'corr':>8}{'max|L-R|':>10}{'dB(L/R)':>9}{'lag':>6}{'peak':>7}")
    print("-" * 76)

    identical = 0
    lags, corrs = [], []

    for w in wavs:
        x, sr = sf.read(str(w), always_2d=True)
        if x.shape[1] < 2:
            print(f"{w.name:<34}  mono file — nothing to compare")
            continue

        L, R = x[:, 0], x[:, 1]

        # 1) the decisive test: are the channels bit-identical?
        maxdiff = float(np.max(np.abs(L - R)))
        if maxdiff == 0.0:
            identical += 1

        # 2) how similar are they overall?
        ##   np.corrcoef returns a 2x2 matrix; [0,1] is the cross term.
        c = float(np.corrcoef(L, R)[0, 1]) if L.std() and R.std() else 1.0

        # 3) level difference in dB (+1e-12 avoids log of zero on silence)
        rms_l, rms_r = np.sqrt((L**2).mean()), np.sqrt((R**2).mean())
        db = 20 * np.log10((rms_l + 1e-12) / (rms_r + 1e-12))

        # 4) spatial delay
        lag, peak = best_lag(L, R)

        lags.append(lag)
        corrs.append(c)
        print(f"{w.name:<34}{c:>8.4f}{maxdiff:>10.2e}{db:>9.2f}{lag:>6d}{peak:>7.3f}")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 76)
    n = len(lags)
    if identical == n:
        print("VERDICT: DUAL MONO — both channels are bit-identical.")
        print("  The second channel carries no information. Mix to mono")
        print("  (or just take channel 0) and drop the spatial idea.")
    else:
        print(f"VERDICT: channels DIFFER on {n - identical}/{n} files.")
        print(f"  mean correlation : {np.mean(corrs):.4f}")
        print(f"  lag range        : {min(lags)} to {max(lags)} samples")
        print(f"  distinct lags    : {len(set(lags))}")
        ## A lag that is always the same value is just a fixed wiring/clock
        ## offset. Lags that VARY across clips are what indicate that different
        ## sounds arrive from different directions — i.e. real spatial info.
        if len(set(lags)) > max(2, n // 4):
            spread_m = (max(lags) - min(lags)) * 343.0 / 48000.0
            print(f"  -> Lags VARY across clips (spread ~{spread_m:.2f} m of path difference).")
            print("     This is consistent with two spaced microphones picking up")
            print("     sources from different directions. Worth building a")
            print("     spatial-spread feature: it should keep increasing with")
            print("     flock size after detection rate has saturated.")
        else:
            print("  -> Lag is essentially constant: likely a fixed offset, not")
            print("     direction. Spatial features unlikely to help.")


if __name__ == "__main__":
    main()

# Avian Population Estimation from Passive Acoustic Monitoring

**BioDCASE 2026 Task 6 — Technical Report**

Estimating the number of individuals of a target bird species in a zoo aviary
from short passive-acoustic recordings.

**Headline result: leave-one-aviary-out CV MAE 19.89 on the development set,
against an official baseline of 11.50.** The system does not beat the baseline.
Sections 5 and 6 explain why, and what I would do about it.

---

## 1. Reframing the problem

The dataset advertises ~141,000 development audio clips. The number of
*supervised targets* is **eight**:

| Target species | Labelled dev aviaries | Counts |
|---|---|---|
| Greater flamingo | 4 (2 are the same physical aviary → 3) | 52, 52, 107, 161 |
| Red-billed quelea | 2 | 61, 153 |
| Hadada ibis | 2 | 4, 6 |
| Pied avocet (optional) | 0 | — |

Models are fitted per species, so the effective budget is **3 / 2 / 2 points**.

This single fact determined the entire architecture. A model with more than one
or two free parameters cannot be justified on two data points, so no neural
regressor was trained. Instead the effort went into designing a *measurement* —
a quantity computed from audio that scales with flock size — with a
near-parameter-free calibration on top. The pipeline reduces 18,237 files and
28,986 detections to eight rows; everything before that step is measurement,
and only the last step is learning.

## 2. What the data actually is

Three properties were verified against the files rather than taken from the
brief. Each contradicted the documentation and each changed the code.

**Clip duration and channel count.** The brief states *"3-second WAV recordings
sampled at 48 kHz."* Every file is 528,044 bytes, but 3 s of 48 kHz mono 16-bit
PCM is ~288,044. Measurement (`src/probe_audio.py`) gives **2.75 s, 48 kHz,
2 channels**, which reconstructs the byte count exactly
(`2.75 × 48000 × 2 × 2 + 44 = 528,044`). Using the documented 3 s would
introduce a systematic 9% error in every rate.

**Duration is not constant across aviaries.** `dev_aviary_2` uses **5.25 s**
clips while the other five use 2.75 s. This matters more than it appears:
aviary_2 holds Greater flamingo = 107 and Hadada ibis = 6, i.e. two of the
eight training points. A 1.9× longer clip has a proportionally higher chance of
containing a call from identical birds, so uncorrected per-file rates would
read that as *more birds* and bias 25% of the training data in one direction.
Every feature in this system is therefore normalised **per second of audio
analysed**, never per file. `src/detect.py` measures duration per file rather
than trusting a constant.

**The stereo is dual mono.** The undocumented second channel raised a
hypothesis worth testing: if the two channels came from spaced microphones,
inter-channel delay would encode direction, and the *spread* of those delays
should keep growing with flock size after detection rate saturates — attacking
the known failure mode of this task directly. `src/probe_stereo.py` tested it
and found the channels bit-identical (`max|L−R| = 0` on every file). The
approach was rejected and downstream stages read channel 0 only.

Method note: each of these was found by widening the check after the previous
one proved too narrow — 20 files suggested a constant, per-aviary sampling
revealed a difference, and a full scan of all 18,237 files confirmed it was
uniform *within* aviaries. Verifying on a sample and generalising to the whole
is the error this sequence kept catching.

## 3. Sampling, and why it is valid

At 528,044–1,008,044 bytes per file the full dataset is ~74 GB (dev) plus
~200 GB (eval). Downloading it was not feasible inside the challenge window.

The justification for sampling is statistical, not logistical: **the estimand is
a rate**, and a rate is consistent under random sampling. Had the task required
a total (e.g. total calls over the recording period) sampling would cost far
more.

However, several intended features are *temporal* — bout duration, bouts per
hour, inter-call gaps — and those require contiguous time. Sampling 3,000 files
uniformly at random would place "consecutive" sampled files up to 40 minutes
apart, making every measured gap an artifact of the sampler. So
`src/build_manifest.py` samples **contiguous 10-minute windows**, stratified
across recording days and time-of-day buckets, and takes every file inside them:
18,237 files (13% of dev, 11.9 h of audio) at the same byte cost, with rate
features unaffected and temporal features still meaningful.

Two implementation details mattered:

- **Window length.** At 30 minutes, densely-recorded aviaries reached the file
  budget after only ~12 windows — too few to cover 18 day × time-of-day cells.
  At 10 minutes they need ~33 and reach full coverage. The trade-off is that
  shorter windows truncate long inter-*bout* intervals, so gap statistics are
  computed within a window only.
- **A sampler bug.** Walking cells in sorted `(day, bucket)` order meant a dense
  aviary exhausted its budget partway through day 2 and **never sampled day 3**
  — observed on `dev_aviary_4`, which has three recording days but appeared in
  the sample with two. Shuffling the cell order makes truncation land uniformly.
  Regression test: `tests/test_sampling.py`.

## 4. Method

**Detection.** ARIA (`aria-inference` 0.1.2) — an ensemble of a zoo-fine-tuned
BirdNET (`ZooCustom_v1`), PERCH v2, and a fusion head. Run once per aviary and
cached; all downstream work reads the cached parquet.

Detection is run **one aviary at a time** because ARIA's CSV records only the
file *basename*, and filenames repeat across aviaries — a merged run would be
unrecoverable.

Each aviary is restricted to a **species whitelist** built from its
ground-truth inventory (`src/species_lists.py`). The model's vocabulary contains
87 species; a given aviary holds 2–22 of them. It also contains *Chilean* and
*American* Flamingo, neither of which occurs in any dev aviary, so any such
detection would be a false positive stealing from the Greater Flamingo target.
Whitelisting reduced `dev_aviary_3` from 87 candidate species to 3.

**Features**, one row per (aviary, species), all normalised per audio-second:

- *Rate* — detections per second; the obvious feature, and the one that saturates.
- *Occupancy* — `p_active`, the fraction of clips containing ≥1 call. If N birds
  each call independently at rate λ, then `p = 1 − exp(−Nλt)`, so
  `−ln(1−p) = Nλt` and dividing by clip length t leaves a quantity proportional
  to N. This is a **derived** de-saturating transform: its shape comes from a
  generative assumption and only its scale is fitted.
- *Confidence and multiplicity* statistics.

Computing `p_active` requires knowing how many clips contained *nothing* —
silence is evidence, and it leaves no row in a detections table. `detect.py`
therefore emits a second table listing every file analysed.

**Estimation.** Per species, one free parameter: `N = a·x`. A two-parameter
`N = a·x^b` was included specifically to demonstrate the sample-size limit — for
a 2-row species, leave-one-out leaves a single training point and the
two-parameter fit is not merely unstable but *undefined*. Predicting the species
mean is included as an honest floor.

**Validation.** Leave-one-aviary-out, with folds **grouped by physical aviary**.
`dev_aviary_5` and `dev_aviary_6` have byte-identical ground truth and are
almost certainly the same enclosure recorded twice; training on one and
validating on the other would be validating on training data. Grouping costs
1.65 MAE — measured, not assumed.

## 5. Results

Leave-one-aviary-out CV, untuned detector threshold (0.3, ARIA's default):

| Species | n (independent aviaries) | Feature | CV MAE | Mean-baseline MAE |
|---|---|---|---|---|
| Greater flamingo | 4 (3) | occupancy/s | **35.07** | 68.33 |
| Red-billed quelea | 2 (2) | occupancy/s | **7.43** | 92.00 |
| Hadada ibis | 2 (2) | occupancy/s | **2.00** | 2.00 |
| **Combined** | | | **19.89** | |

Official baseline: **MAE 11.50, MAPE 10.6%**. This system is ~1.7× worse.

Two species behave as intended. Quelea drops from a 92.0 mean-baseline to 7.43
— the acoustic signal is doing real work. Hadada ibis ties its baseline at 2.00,
which is uninformative: with n=2, leave-one-out trains on one point and predicts
the other, so every model returns |6−4| = 2 by construction.

Greater flamingo is the failure, and it is the diagnostic one.

**Sensitivity analysis.** Sweeping the detector confidence threshold
(`src/sensitivity.py`):

| threshold | flamingo | quelea | hadada | combined |
|---|---|---|---|---|
| 0.30 | 35.07 | 7.43 | 2.00 | **19.89** |
| 0.50 | 31.53 | 14.95 | 2.00 | 20.00 |
| 0.70 | 33.49 | **0.32** | 2.00 | 17.32 |
| 0.90 | 33.77 | 25.45 | 2.00 | 23.75 |
| 0.99 | 38.51 | 69.12 | 2.00 | 37.03 |
| 0.999 | 49.79 | 92.00 | 2.00 | 48.40 |

Combined MAE ranges **17.32 to 48.40** depending on one arbitrary choice.

The 0.32 for quelea at threshold 0.7 is **not** a result. With n=2, some
threshold will make a single fitted scale parameter land almost exactly on the
held-out point by chance; the quelea column swings 7.43 → 14.95 → 0.32 → 9.36 →
25.45 with no structure. Selecting that threshold and reporting 17.32 would be
reporting luck, and would commit precisely the overfitting this design exists to
avoid. The reported figure therefore uses the *a priori* default.

**The spread is itself the finding**: at 2–3 labelled aviaries per species, an
incidental hyperparameter moves the headline metric by a factor of ~2.5. Any
single number from this dataset — including 11.50 — should be read with that in
mind.

## 6. Failure analysis

**Greater flamingo: acoustic saturation, confirmed.** `p_active` is 0.976,
0.980, 0.875, 0.920 for counts of 107, 161, 52, 52. Nearly every clip contains
a flamingo call regardless of whether there are 52 birds or 161. The occupancy
transform `−ln(1−p)` is numerically unstable that close to 1, and worse, the
ranking is wrong: the two N=52 aviaries show *higher* detection rates (0.318,
0.335 /s) than the N=107 aviary (0.186 /s). Raising the threshold to 0.999 only
brings mean `p_active` down to 0.54 and makes MAE worse (49.79). The signal is
not merely compressed; at these flock sizes it is gone. This matches the
dataset card's own note that *"raw detection rates saturate as flock size
grows."*

**Hadada ibis: no measurable validation.** N = 4 and 6. MAE 2.00 is a 40%
relative error and every candidate model produces the same value. Nothing about
this species is being validated; it should be reported as untested rather than
as a success.

**The benchmark's own structure limits what any result means.** Analysing the
public evaluation metadata (no compute required): of the 6 scored
main-leaderboard aviaries, `eval_aviary_1` and `eval_aviary_2` have
byte-identical inventories to `dev_aviary_1` and `dev_aviary_3` — scoring them
measures memorisation. `eval_aviary_6` and `eval_aviary_7` duplicate each other.
Only **three physically independent aviaries** remain. Worse, both independent
flamingo points are 11 and 195, while dev spans 52–161 — so the two
informative flamingo tests both require *extrapolation outside the calibration
range*, one nearly 5× below the minimum. This is the strongest argument for the
derived occupancy link over a fitted curve, which is unconstrained off-support.

**Rejected hypothesis: spatial separation.** Documented in §2. The undocumented
stereo channel is dual mono, so inter-channel delay carries no directional
information. Cost: ten minutes. Had it been genuine two-microphone stereo, the
spread of arrival-time differences would have been the natural attack on
flamingo saturation, since 195 birds occupy more positions than 52 even when
call rate has flattened.

## 7. Limitations

- **13% sample.** 18,237 of 140,899 dev files. Rates are consistent under
  sampling but noisier; the sampling error was not bootstrapped.
- **Evaluation split not run.** ~4.5 h of download and inference beyond the time
  budget. The reported metric is dev LOAO-CV only; the eval analysis in §6 is
  structural, from public metadata.
- **Detection ran on CPU.** PyTorch could not initialise CUDA (driver 12.4 vs a
  newer build requirement), so PERCH ran on CPU — 35 min/aviary, 3.5 h total.
  A deliberate scheduling choice over an uncertain environment fix on a 3-day
  deadline.
- **Bout and spectral-texture features were designed but not evaluated.** The
  contiguous-window sampling exists to make them possible; they were cut for
  time. They are the most promising unexplored direction (§8).
- **`per_active_clip` is degenerate.** For 2.75 s clips ARIA emits at most one
  row per species per file, so the feature is identically 1.0 and carries no
  information. It should be computed per segment, not per file.

## 8. What I would do next

1. **Attack flamingo saturation on a finer time base.** `p_active` saturates
   because a 2.75 s clip is long enough that *someone* in a large flock calls.
   Computing occupancy over sub-second frames instead of whole clips would move
   p back into its informative range while keeping the same generative model.
   This is the single highest-value change and it needs no new labels.
2. **Measure within-bout overlap density** — spectral flatness during calling
   bouts, amplitude-envelope dynamic range, simultaneous-onset polyphony. The
   hypothesis is that as N grows, calls overlap more even after the detector has
   stopped firing more often. This is the feature family the sampling design was
   built to support.
3. **Pool species with a shared shape.** With 3/2/2 points, per-species fitting
   is the binding constraint. A single shared exponent with per-species scale
   offsets would use all 8 points to constrain the shape while allowing species
   to differ in loudness.
4. **Bootstrap the sampling error** to state a confidence interval, and report
   it alongside MAE. Given §5, any point estimate from 8 labels is close to
   meaningless without one.
5. **Ablate the ensemble.** ARIA runs BirdNET + PERCH + fusion at ~3× the cost
   of BirdNET alone. Whether that buys accuracy *for counting* — as opposed to
   for classification, which is what it was built for — is unmeasured and cheap
   to test on one aviary.

## 9. Reproducing

```bash
pip install -r requirements.txt && pip install aria-inference
python -m src.eda_metadata                                    # n=8, duplicates
python -m src.build_manifest --split dev                      # block sampling
python -m src.download --split dev                            # ~11 GB
python -m src.probe_audio                                     # verify audio
python -m src.probe_stereo                                    # dual-mono check
python -m src.species_lists --labels ~/aria-models/ZooCustom_v1_Labels.txt
python -m src.detect --split dev                               # ~3.5 h CPU
python -m src.features --split dev
python -m src.estimate
python -m src.sensitivity
python tests/test_sampling.py && python tests/test_download_backoff.py
```

Sampling is seeded (`sampling.seed` in `config.yaml`), so the 13% subset is
reproducible exactly.

---

## Sources

- [BioDCASE 2026 Bird Counting dataset](https://huggingface.co/datasets/Emreargin/BioDCASE2026_Bird_Counting)
- [Task page — ML4Biodiversity](https://www.ml4biodiversity.org/biodcase26_birdcounts/)
- [Official baseline repository](https://github.com/ml4biodiversity/biodcase-population-estimation)

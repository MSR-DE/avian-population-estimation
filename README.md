# Avian Population Estimation from Passive Acoustic Monitoring

BioDCASE 2026 Task 6 — estimating how many individuals of a target bird species
are present in a zoo aviary from short passive acoustic recordings.

See [`PLAN.md`](PLAN.md) for the full strategy and the reasoning behind the
design decisions summarised below.

---

## The constraint that shapes everything

The dataset ships ~141,000 development audio clips. It ships **8 labels**.

| Target species | Labelled dev aviaries | Counts |
|---|---|---|
| Greater flamingo | 4 (2 are duplicates → 3 effective) | 107, 161, 52, 52 |
| Red-billed quelea | 2 | 153, 61 |
| Hadada ibis | 2 | 6, 4 |
| Pied avocet (optional) | **0** — zero-shot | — |

Models are fitted **per species**, so the real budget is 3 / 2 / 2 points.
This is therefore not a "train a network" problem: it is a measurement and
calibration problem. Effort goes into designing features that scale with flock
size, not into the regressor, which must stay near-parameter-free.

Two further facts drive the engineering:

- **Volume.** Files are 528,044 bytes each → ~74 GB dev + ~200 GB eval. We
  sample rather than download everything (see below).
- **Duplicate aviaries.** `dev_aviary_5` ≡ `dev_aviary_6`, `dev_aviary_1` ≡
  `eval_aviary_1`, `dev_aviary_3` ≡ `eval_aviary_2`, `eval_aviary_6` ≡
  `eval_aviary_7` have byte-identical ground truth. CV folds are grouped by
  *physical* aviary to avoid an optimistically biased validation score.

## Sampling: why it's valid, not just convenient

The estimand is a **rate** (detections per file), and a rate is recoverable
from a random sample — that is the justification, not the file size.

But several planned features are *temporal* (bouts/hour, bout duration,
inter-call gaps) and need contiguous time. Uniform random file sampling would
destroy them: two consecutive sampled files could be 40 minutes apart and every
computed "gap" would be a sampler artifact.

So we sample **contiguous 30-minute windows**, stratified across days and
hour-of-day buckets, and take every file inside them. Same byte budget; rate
features unaffected; bout features stay valid.

## Evaluation discipline

`metadata/eval_ground_truth.csv` is publicly downloadable because the challenge
round has concluded. **It is not used for model selection.** All
hyperparameters are chosen by leave-one-aviary-out CV on the development set;
the model is then frozen and scored on eval exactly once.

Official baseline to beat: **MAE 11.50 / MAPE 10.6%** (dev).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install aria-inference          # or: pip install aria-inference-birdnet
```

## Run order

```bash
# 0. Metadata EDA — seconds, no audio needed. Run this first.
python -m src.eda_metadata

# 1. Decide which files to download (writes cache/manifest_dev.csv)
python -m src.build_manifest --split dev

# 2. Smoke test the downloader on 20 files BEFORE committing to the full pull
python -m src.download --split dev --limit 20

# 3. Resolve the open audio-format question (see note below)
python -m src.probe_audio

# 4. Full sampled download (~25 GB, resumable — safe to interrupt)
python -m src.download --split dev
```

Then detection → features → estimation (next stage).

Run the tests any time:

```bash
python tests/test_sampling.py
```

## The brief is wrong about the audio (measured, not assumed)

The challenge document says *"3-second WAV recordings sampled at 48 kHz"*.
Every file is 528,044 bytes, but 3 s of 48 kHz mono 16-bit PCM is ~288,044
bytes — so something in that description had to be wrong.

`src/probe_audio.py` on real files:

```
2.7500 s, 48 kHz, 2 channels, PCM_16
2.75 x 48000 x 2ch x 2 bytes + 44 header = 528,044   <- matches exactly
```

Two corrections, both consequential:

1. **Clips are 2.75 s, not 3 s.** Per-file rates normalise by 2.75; using the
   documented 3 s introduces a systematic 9% error. Recorded in `config.yaml`
   under `audio.clip_seconds`.

2. **The recordings are stereo.** The brief never mentions channels, and the
   official baseline pipeline mixes to mono. If the two channels come from
   spaced microphones, inter-channel delay encodes direction — and the *spread*
   of those delays should keep growing with flock size after detection rate has
   saturated, which is the documented failure mode of this task.
   `src/probe_stereo.py` distinguishes genuine two-mic stereo from dual mono.

## Layout

```
config.yaml            all paths, sample sizes, species, duplicate groups
src/common.py          config loading, filename parsing, CV grouping
src/eda_metadata.py    establishes n=8, duplicates, count ranges
src/build_manifest.py  block sampling -> manifest of files to fetch
src/download.py        resumable parallel download of the manifest
src/probe_audio.py     verifies actual sample rate / channels / duration
tests/test_sampling.py parsing, window contiguity, diel spread, determinism
```

## Tuning for your machine

In `config.yaml`, `sampling.max_files_per_aviary` is the single knob:

| Value | Dev download | Eval download |
|---|---|---|
| 3000 | ~9.5 GB | ~16 GB |
| 1500 | ~4.8 GB | ~8 GB |

Lower it if disk or bandwidth is tight, then report the added variance rather
than hiding it.

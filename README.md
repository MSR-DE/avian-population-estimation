# Avian Population Estimation from Passive Acoustic Monitoring

BioDCASE 2026 Task 6 — estimating how many individuals of a target bird species
are present in a zoo aviary from short passive-acoustic recordings.

**Full methodology, results and failure analysis: [REPORT.md](REPORT.md).**

---

## Result

Leave-one-aviary-out CV on the development set, untuned detector threshold:

| Species | independent aviaries | CV MAE | mean-baseline |
|---|---|---|---|
| Greater flamingo | 3 | 35.07 | 68.33 |
| Red-billed quelea | 2 | 7.43 | 92.00 |
| Hadada ibis | 2 | 2.00 | 2.00 |
| **Combined** | | **19.89** | |

Official baseline: **MAE 11.50**. This system does not beat it. The gap is
almost entirely Greater flamingo, whose acoustic signal saturates
(`p_active` = 0.88–0.98 regardless of whether there are 52 birds or 161).
Quelea works well; Hadada ibis at n=2 is untested rather than successful.

## The constraint that shapes everything

The dataset ships ~141,000 development clips. It ships **8 labels**:

| Target species | Labelled dev aviaries | Counts |
|---|---|---|
| Greater flamingo | 4 (2 are the same physical aviary → 3) | 52, 52, 107, 161 |
| Red-billed quelea | 2 | 61, 153 |
| Hadada ibis | 2 | 4, 6 |
| Pied avocet (optional) | 0 | — |

Fitted per species, that's a budget of **3 / 2 / 2 points**. So this is not a
"train a network" problem — it's a measurement and calibration problem. The
pipeline reduces 18,237 files and 28,986 detections to eight rows; everything
before that is measurement, only the last step is learning.

## Three things the brief gets wrong

All three were found by checking the files rather than trusting the document,
and each changed the code:

1. **Clips are 2.75 s, not 3 s.** `2.75 × 48000 × 2ch × 2 bytes + 44 = 528,044`,
   the exact file size. Using 3 s introduces a systematic 9% rate error.
2. **Clip length varies by aviary.** `dev_aviary_2` uses 5.25 s clips; it holds
   two of the eight training points. All rates are therefore per *audio-second*,
   never per file.
3. **The undocumented second channel is dual mono** (`max|L−R| = 0`). Tested as
   a possible source of spatial information; hypothesis rejected.

## Sampling

~74 GB dev + ~200 GB eval is not downloadable inside the window. We sample 13%
(18,237 files, 11.9 h audio).

Sampling is valid because **the estimand is a rate**, and rates are consistent
under random sampling — that's the justification, not the file size. But
temporal features (bouts, inter-call gaps) need contiguous time, so we sample
**contiguous 10-minute windows** stratified across days and time-of-day, not
individual files. Seeded, so the subset is exactly reproducible.

## Setup

```bash
python3 -m venv ~/venvs/biodcase && source ~/venvs/biodcase/bin/activate
pip install -r requirements.txt
pip install aria-inference
aria-inference download-models --dir ~/aria-models
```

## Run order

```bash
python -m src.eda_metadata                     # n=8, duplicate aviaries, ranges
python -m src.build_manifest --split dev       # block-stratified sampling
python -m src.download --split dev             # ~11 GB, resumable
python -m src.probe_audio                      # verify duration/channels
python -m src.probe_stereo                     # dual-mono check
python -m src.species_lists --labels ~/aria-models/ZooCustom_v1_Labels.txt
python -m src.detect --split dev               # ~3.5 h CPU, cached per aviary
python -m src.features --split dev             # 28,986 detections -> 8 rows
python -m src.estimate                         # grouped LOAO-CV
python -m src.sensitivity                      # threshold stability
```

Tests:

```bash
python tests/test_sampling.py
python tests/test_download_backoff.py
python tests/test_manifest_e2e.py
```

## Layout

```
config.yaml              paths, sample size, seed, species, duplicate groups
src/common.py            Clip type, filename parsing, CV grouping
src/eda_metadata.py      establishes n=8, duplicates, count ranges
src/build_manifest.py    contiguous-window sampling -> manifest
src/download.py          resumable parallel download, backoff on HTTP 429
src/probe_audio.py       measures duration/channels per aviary and per file
src/probe_stereo.py      dual-mono vs two-microphone test
src/species_lists.py     per-aviary detector whitelists from ground truth
src/detect.py            runs ARIA per aviary; emits detections + file inventory
src/features.py          per (aviary, species) features, normalised per second
src/estimate.py          1-parameter models, leave-one-aviary-out CV
src/sensitivity.py       threshold sweep — stability, NOT tuning
tests/                   parsing, window contiguity, day coverage, retry backoff
```

## Notes on validation

CV folds are grouped by **physical aviary**: `dev_aviary_5` and `dev_aviary_6`
have byte-identical ground truth and are almost certainly the same enclosure
recorded twice. Grouping costs 1.65 MAE — measured, not assumed.

The evaluation split was **not run** (~4.5 h beyond budget). Its structure is
analysed in [REPORT.md](REPORT.md) §6 from public metadata: only 3 of the 6
scored aviaries are physically independent, and both independent flamingo points
(11 and 195) fall outside the dev calibration range of 52–161.

`src/sensitivity.py` is deliberately not a tuning script. Combined MAE ranges
17.32–48.40 across detector thresholds; the minimum is luck at n=2, so the
reported figure uses the a priori default. The spread is itself a result.

## Tools used

Built with AI assistance (Claude) for code scaffolding, review and drafting.
All design decisions, dataset findings and result interpretation are documented
and defended in [REPORT.md](REPORT.md).

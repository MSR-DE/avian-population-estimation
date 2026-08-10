"""Verify parsing + block sampling on a synthetic aviary before touching real data."""
import random, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from src.common import parse_repo_path, duplicate_group_map, load_config
from src.build_manifest import sample_aviary, window_key

# --- 1. filename parsing, including the real observed variants ---
cases = [
    ("dev_aviary_1/chunk_000/rec_d1_00_01_49.wav",        1, 109.0),
    ("dev_aviary_1/chunk_000/rec_d1_00_00_45.750000.wav", 1, 45.75),
    ("eval_aviary_10/chunk_012/rec_d7_23_59_59.250000.wav",7, 86399.25),
    ("metadata/ground_truth.csv",                          None, None),
    ("README.md",                                          None, None),
    (".gitattributes",                                     None, None),
]
for path, day, t in cases:
    c = parse_repo_path(path)
    if day is None:
        assert c is None, f"should have rejected {path}"
    else:
        assert c is not None, f"failed to parse {path}"
        assert c.day == day and abs(c.t_sec - t) < 1e-6, (path, c)
print("parsing: 6/6 OK (incl. non-audio rejection)")

# --- 2. block sampling on a synthetic aviary ---
# 3 days, a file every ~20s across 24h => ~4300/day
clips = []
for d in range(1, 4):
    t = 0.0
    while t < 86400:
        clips.append(parse_repo_path(
            f"dev_aviary_9/chunk_000/rec_d{d}_{int(t//3600):02d}_{int(t%3600//60):02d}_{t%60:.6f}.wav"))
        t += 20.0
clips = [c for c in clips if c]
print(f"synthetic aviary: {len(clips)} files over 3 days")

picked = sample_aviary(clips, window_seconds=1800, max_files=3000,
                       hour_buckets=6, stratify=True, rng=random.Random(17))

assert len(picked) >= 3000, len(picked)
# contiguity: every selected window must be COMPLETE (all its files present)
from collections import defaultdict
sel = defaultdict(int); tot = defaultdict(int)
for c in clips: tot[window_key(c,1800)] += 1
for c in picked: sel[window_key(c,1800)] += 1
assert all(sel[k] == tot[k] for k in sel), "windows must be downloaded whole"
print(f"contiguity: all {len(sel)} selected windows are complete")

# diel spread: selected files should cover many hours, not clump at one time
hours = sorted({int(c.hour) for c in picked})
assert len(hours) >= 15, f"poor diel coverage: only hours {hours}"
print(f"diel coverage: {len(hours)}/24 hours represented")
days = sorted({c.day for c in picked})
assert days == [1,2,3], days
print(f"day coverage: {days}")

# determinism
p2 = sample_aviary(clips, 1800, 3000, 6, True, random.Random(17))
assert [c.repo_path for c in p2] == [c.repo_path for c in picked]
print("determinism: same seed -> identical selection")

# --- 3. CV grouping ---
cfg = load_config()
canon = duplicate_group_map(cfg)
assert canon['dev_aviary_6'] == canon['dev_aviary_5']
assert canon['eval_aviary_1'] == canon['dev_aviary_1']
assert canon['dev_aviary_2'] != canon['dev_aviary_4']
print("CV grouping: duplicates collapse, distinct aviaries stay separate")

# --- 4. REGRESSION: dense aviary must not lose its later days ---------------
## The bug this guards against (found on real data, dev_aviary_4):
## a densely-recorded aviary hits the 3000-file budget after only ~12 windows.
## If the round-robin walks cells in sorted (day, bucket) order, it burns
## through day 1's six buckets, then day 2's, and STOPS before ever reaching
## day 3. The sample silently covers 2 of 3 recording days.
## Fix was to shuffle the cell order so truncation lands uniformly.
dense = []
for d in range(1, 4):              # 3 days
    t = 0.0
    while t < 86400:
        dense.append(parse_repo_path(
            f"dev_aviary_4/chunk_000/rec_d{d}_{int(t//3600):02d}_{int(t%3600//60):02d}_{t%60:.6f}.wav"))
        t += 7.0                   # very dense: ~12300 files/day
dense = [c for c in dense if c]

# budget is small relative to density, so only a handful of windows are needed
picked_dense = sample_aviary(dense, window_seconds=1800, max_files=3000,
                             hour_buckets=6, stratify=True, rng=random.Random(17))
n_windows = len({window_key(c, 1800) for c in picked_dense})
days_covered = sorted({c.day for c in picked_dense})
assert days_covered == [1, 2, 3], (
    f"dense aviary lost days: got {days_covered} from only {n_windows} windows")
print(f"regression: dense aviary used {n_windows} windows and still covered days {days_covered}")

print("\nALL CHECKS PASSED")

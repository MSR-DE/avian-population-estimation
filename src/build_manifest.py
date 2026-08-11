## Stage 0 — decide WHICH files to download, before downloading anything.
##
## THE PROBLEM
## The full dataset is ~270 GB (528 KB/file x ~520,000 files).

## WHY SAMPLING IS ALLOWED (this is the bit to understand)
## The thing we're measuring is a RATE — "detections per file". A rate is
## recoverable from a random sample: if 4% of a sample of files contain a
## flamingo call, roughly 4% of ALL files do too. If instead we needed a TOTAL
## (e.g. total number of calls across the whole recording), sampling would
## cost us much more.
## So the justification is statistical, not "the files were too big". Say it
## that way in the report — it's the difference between an excuse and an
## engineering decision.
##
## WHY WE SAMPLE BLOCKS, NOT INDIVIDUAL FILES (the non-obvious bit)
## Some features we want are TEMPORAL: bouts per hour, how long a calling bout
## lasts, the gap between consecutive calls. Those need CONTIGUOUS time.
## If we picked 3,000 files at random from an aviary, two "consecutive" files
## in our sample might be 40 minutes apart in reality — and every gap we
## measured would be an artifact of our own sampler, not the birds.
## So: pick whole 30-minute WINDOWS and take every file inside them.
## Same number of bytes. Rate features unaffected. Bout features survive.
##
## Usage:
##     python -m src.build_manifest --split dev
##     python -m src.build_manifest --split eval

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from .common import Clip, load_config, parse_repo_path, resolve_dir


def list_repo_files(repo_id: str, cache_dir: Path) -> list[str]:
    """Get every file path in the HF dataset repo. Cached to disk.

    This is ~520,000 strings. It takes a minute or two the first time and is
    instant afterwards, because we save the result to cache/repo_files.json.

    NOTE: this only downloads the LIST of filenames, not the audio. It's a few
    MB of text. Nothing expensive happens in this function.
    """
    cache = cache_dir / "repo_files.json"

    # 1) already fetched? just read it back
    if cache.exists():
        return json.loads(cache.read_text())

    # 2) otherwise ask the Hugging Face API for the full listing
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    # 3) save so we never pay this cost again
    cache.write_text(json.dumps(files))
    return files


def window_key(clip: Clip, window_seconds: int) -> tuple[int, int]:
    """Which time-window does this clip belong to?

    Integer division is doing the work: with window_seconds=1800 (30 min),
    a clip at t=1750s -> 1750//1800 = 0  (first window)
    a clip at t=1850s -> 1850//1800 = 1  (second window)

    Returns (day, window_index) because window 0 of day 1 and window 0 of
    day 2 are different windows — the day has to be part of the key.
    """
    return (clip.day, int(clip.t_sec // window_seconds))


def sample_aviary(
    clips: list[Clip],
    window_seconds: int,
    max_files: int,
    hour_buckets: int,
    stratify: bool,
    rng: random.Random,
) -> list[Clip]:
    """Choose whole time-windows from one aviary until we hit the file budget.

    Returns the list of clips to download.
    """

    # 1) BUCKET every clip into its 30-minute window.
    #    defaultdict(list) means we don't have to check "does this key exist
    #    yet" before appending — it creates the empty list automatically.
    windows: dict[tuple[int, int], list[Clip]] = defaultdict(list)
    for c in clips:
        windows[window_key(c, window_seconds)].append(c)

    keys = list(windows.keys())

    # 2) DECIDE THE ORDER we'll consider windows in.
    if stratify:
        ## Why stratify at all?
        ## Birds call far more at dawn than at 2pm. If our random pick happened
        ## to land mostly on dawn windows, our measured call rate would be
        ## inflated — and inflated by an amount that has nothing to do with how
        ## many birds are in the aviary. That's a bias we can just design out.
        ##
        ## So: group windows by (day, time-of-day bucket), then take turns
        ## picking one from each bucket in rotation. That guarantees the sample
        ## is spread across the daily cycle instead of clumped.

        # 2a) how many hours does one bucket span? 6 buckets -> 4 hours each
        bucket_span = 24 / hour_buckets

        # 2b) file each window into its (day, bucket) pigeonhole
        by_bucket: dict[tuple[int, int], list] = defaultdict(list)
        for k in keys:
            day, widx = k
            hour = (widx * window_seconds) / 3600.0
            by_bucket[(day, int(hour // bucket_span))].append(k)

        # 2c) shuffle within each bucket so we don't always take the earliest
        for v in by_bucket.values():
            rng.shuffle(v)

        # 2d) ROUND-ROBIN: walk the buckets in a loop, taking one window from
        #     each pass. This interleaves them: bucket0, bucket1, bucket2, ...
        #     then back to bucket0. Result is a spread-out ordering.
        ordered: list[tuple[int, int]] = []
        bucket_names = sorted(by_bucket.keys())
        i = 0
        while any(by_bucket[b] for b in bucket_names):
            b = bucket_names[i % len(bucket_names)]   ## % wraps around the list
            if by_bucket[b]:
                ordered.append(by_bucket[b].pop())
            i += 1
        keys = ordered
    else:
        # plain random ordering — kept as a comparison baseline
        rng.shuffle(keys)

    # 3) TAKE WHOLE WINDOWS until we reach the budget.
    ##  Note we check the budget BEFORE extending, never in the middle of a
    ##  window. Taking half a window would break the contiguity we went to all
    ##  this trouble to preserve. So we slightly overshoot max_files instead —
    ##  that's deliberate, not a bug.
    picked: list[Clip] = []
    used_windows = 0
    for k in keys:
        if len(picked) >= max_files:
            break
        picked.extend(windows[k])
        used_windows += 1

    print(
        f"    {len(windows)} windows available, used {used_windows}, "
        f"{len(picked)} files selected"
    )
    return picked


def main() -> None:
    # 1) parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval", "all"], default="dev")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    # 2) load settings and set up the random seed.
    ##   Seeding matters: it makes the file selection REPRODUCIBLE. Anyone can
    ##   regenerate the identical 25 GB subset from config.yaml. That turns
    ##   "I sampled the data" into a checkable claim.
    cfg = load_config(args.config)
    cache_dir = resolve_dir(cfg, "cache")
    s = cfg["sampling"]
    rng = random.Random(s["seed"])

    # 3) get the full file listing (names only — no audio yet)
    print(f"Listing files in {cfg['repo_id']} (cached after first run)...")
    files = list_repo_files(cfg["repo_id"], cache_dir)
    print(f"  {len(files)} paths in repo")

    # 4) parse each path into a Clip and group by aviary.
    ##   parse_repo_path returns None for README/metadata files, so the
    ##   `if c is not None` quietly filters those out.
    by_aviary: dict[str, list[Clip]] = defaultdict(list)
    for p in files:
        c = parse_repo_path(p)
        if c is not None:
            by_aviary[c.aviary].append(c)

    # 5) sample each aviary in the requested split(s)
    splits = ["dev", "eval"] if args.split == "all" else [args.split]
    rows = []
    for split in splits:
        for aviary in cfg["splits"][split]:
            clips = by_aviary.get(aviary, [])
            if not clips:
                print(f"  !! {aviary}: no files found — check the listing")
                continue

            print(f"  {aviary}: {len(clips)} files total")
            picked = sample_aviary(
                clips,
                window_seconds=s["window_seconds"],
                max_files=s["max_files_per_aviary"],
                hour_buckets=s["hour_buckets"],
                stratify=s["stratify_by_hour_bucket"],
                rng=rng,
            )

            # 5a) flatten to rows for the CSV
            for c in picked:
                rows.append(
                    {
                        "split": split,
                        "aviary": aviary,
                        "repo_path": c.repo_path,
                        "day": c.day,
                        "t_sec": c.t_sec,
                        "abs_sec": c.abs_sec,          ## for ordering/gaps later
                        "window": window_key(c, s["window_seconds"])[1],
                        "n_files_total_in_aviary": len(clips),  ## for sampling fraction
                    }
                )

    # 6) write the manifest — this is the ONLY output. Still no audio downloaded.
    df = pd.DataFrame(rows).sort_values(["aviary", "day", "t_sec"])
    out = cache_dir / f"manifest_{args.split}.csv"
    df.to_csv(out, index=False)

    # 7) report what fraction of each aviary we're taking.
    ##   Needed for two reasons: (a) to sanity-check we didn't accidentally
    ##   grab 2 files from one aviary and 3000 from another, and (b) to quote
    ##   honestly in the write-up.
    frac = (
        df.groupby("aviary")
        .agg(sampled=("repo_path", "size"), total=("n_files_total_in_aviary", "max"))
        .assign(fraction=lambda d: (d.sampled / d.total).round(4))
    )
    print("\nSampling fractions:")
    print(frac.to_string())

    ## 528_044 is the observed bytes-per-file. Underscores are just digit
    ## separators for readability — Python ignores them.
    est_gb = len(df) * 528_044 / 1e9
    print(f"\nWrote {out}  ({len(df)} files, ~{est_gb:.1f} GB to download)")


if __name__ == "__main__":
    main()

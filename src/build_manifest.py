## Stage 0 — decide WHICH files to download, before downloading anything.
##
## THE PROBLEM
## The full dataset is ~270 GB (528 KB/file x ~520,000 files). You have 3 days.
## Downloading all of it would eat the entire deadline. So we sample.
##
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


def list_aviary_files(repo_id: str, aviary: str, cache_dir: Path) -> list[str]:
    """Get the .wav paths for ONE aviary. Cached per aviary.

    WHY PER-AVIARY instead of one call for the whole repo:
    The repo holds ~520,000 files, but `--split dev` only needs the ~141,000
    in the six dev aviaries. Listing everything meant waiting on ~4x more
    paginated API requests than necessary — which on an unauthenticated
    connection is the difference between minutes and many minutes.

    Caching per aviary also makes this RESUMABLE. If the listing dies or you
    Ctrl+C on aviary 4, aviaries 1-3 are already saved and skipped on re-run.

    NOTE: this fetches only FILENAMES, not audio. A few MB of text total.
    """
    listing_dir = cache_dir / "listings"
    listing_dir.mkdir(parents=True, exist_ok=True)
    cache = listing_dir / f"{aviary}.json"

    # 1) already listed? read it straight back
    if cache.exists():
        paths = json.loads(cache.read_text())
        print(f"  {aviary}: {len(paths)} files (cached)")
        return paths

    # 2) MIGRATION: an earlier version of this script cached one big listing of
    ##   the whole repo. If that file is still around, slice this aviary out of
    ##   it instead of hitting the API again — saves re-listing 520k paths.
    legacy = cache_dir / "repo_files.json"
    if legacy.exists():
        allp = json.loads(legacy.read_text())
        paths = [p for p in allp if p.startswith(aviary + "/") and p.endswith(".wav")]
        if paths:
            cache.write_text(json.dumps(paths))
            print(f"  {aviary}: {len(paths)} files (from legacy repo_files.json)")
            return paths

    # 3) otherwise walk just this aviary's subtree.
    ##   recursive=True descends into the chunk_NNN/ subfolders for us.
    api = HfApi()
    paths: list[str] = []
    print(f"  {aviary}: listing...", end="", flush=True)
    for item in api.list_repo_tree(
        repo_id=repo_id, path_in_repo=aviary, recursive=True, repo_type="dataset"
    ):
        ## the tree yields both files and folders; folders have no .size and we
        ## only want .wav anyway, so filter on the extension.
        p = getattr(item, "path", None)
        if p and p.endswith(".wav"):
            paths.append(p)
            ## heartbeat so a slow listing never looks frozen
            if len(paths) % 5000 == 0:
                print(f" {len(paths)}", end="", flush=True)

    # 3) save so this cost is paid exactly once
    cache.write_text(json.dumps(paths))
    print(f" done ({len(paths)} files)")
    return paths


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

        # 2d) ROUND-ROBIN: walk the cells in a loop, taking one window from each
        #     per pass. This interleaves them: cell0, cell1, cell2, ... then
        #     back to cell0, so every cell gets one window before any gets two.
        ##
        ## The shuffle on the next line is NOT cosmetic — it fixes a real bias.
        ## If we walk cells in sorted order they come out day-major:
        ##     (d1,b0) (d1,b1) ... (d1,b5) (d2,b0) ...
        ## A dense aviary only needs ~12 windows to hit its file budget, so the
        ## loop would stop partway through day 2 and NEVER SAMPLE DAY 3.
        ## Observed for real: dev_aviary_4 has 3 recording days but the sample
        ## contained only 2. Shuffling makes the truncation land uniformly
        ## across days and time-of-day instead of always chopping off the end.
        ordered: list[tuple[int, int]] = []
        bucket_names = sorted(by_bucket.keys())
        rng.shuffle(bucket_names)
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

    # 3) work out which aviaries we actually need to list
    splits = ["dev", "eval"] if args.split == "all" else [args.split]
    wanted = [a for split in splits for a in cfg["splits"][split]]
    print(f"Listing {len(wanted)} aviaries from {cfg['repo_id']} (cached per aviary)...")

    # 4) list + sample one aviary at a time.
    ##   Doing both in the same loop means we never hold all 141k paths in
    ##   memory at once, and you see progress aviary by aviary.
    rows = []
    for split in splits:
        for aviary in cfg["splits"][split]:
            paths = list_aviary_files(cfg["repo_id"], aviary, cache_dir)

            ## parse_repo_path returns None for anything that isn't an audio
            ## file, so this quietly drops stray non-.wav entries.
            clips = [c for c in (parse_repo_path(p) for p in paths) if c is not None]
            if not clips:
                print(f"  !! {aviary}: no parseable audio files — check the listing")
                continue

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

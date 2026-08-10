## Stage 0b — download exactly the files listed in the manifest.
##
## Two properties this script must have, because it runs for hours:
##   1. RESUMABLE — if your wifi drops or you Ctrl+C, re-running continues
##      from where it stopped instead of starting over.
##   2. PARALLEL  — downloading one file at a time wastes most of your
##      bandwidth waiting for network round-trips.
##
## Usage:
##     python -m src.download --split dev --limit 20    <- ALWAYS smoke test first
##     python -m src.download --split dev               <- then the real thing

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .common import load_config, resolve_dir


def check_auth() -> bool:
    """Warn loudly if we're unauthenticated.

    Anonymous requests to the Hub get a much lower rate limit. Downloading
    ~18,000 files anonymously reliably triggers HTTP 429. Being logged in is
    the single biggest factor in whether this script succeeds.
    """
    try:
        from huggingface_hub import get_token
        return get_token() is not None
    except Exception:  # noqa: BLE001 - older hub versions
        import os
        return bool(os.environ.get("HF_TOKEN"))


def is_rate_limited(msg: str) -> bool:
    """Is this error the server telling us to slow down?"""
    ## NOTE: lowercase BOTH sides. Writing `"Too Many Requests" in msg.lower()`
    ## silently never matches, because the haystack is lowercased but the
    ## needle isn't. Caught by tests/test_download_backoff.py.
    return "429" in msg or "too many requests" in msg.lower()


def fetch_one(repo_id: str, repo_path: str, dest_root: Path, retries: int) -> tuple[str, str | None]:
    """Download a single file. Returns (path, error_message_or_None).

    Why return the error instead of raising it?
    Because this runs inside a thread pool over thousands of files. If one
    file raises, we don't want the whole multi-hour download to die. We collect
    failures and report them at the end so they can be retried.
    """
    target = dest_root / repo_path

    # 1) SKIP if we already have it. This single check is what makes the whole
    #    script resumable — re-running is cheap because finished files are
    #    instantly skipped.
    ##  We check size > 0 too: a zero-byte file means a previous run was
    ##  interrupted mid-write, so it needs re-downloading, not skipping.
    if target.exists() and target.stat().st_size > 0:
        return repo_path, None

    # 2) try a few times, BACKING OFF between attempts.
    ##
    ## WHY THE BACKOFF MATTERS (learned the hard way):
    ## An earlier version of this function retried immediately, 3 times, with
    ## no delay. When the Hub started returning 429 Too Many Requests, 8 worker
    ## threads each retried instantly — so the response to "you are sending too
    ## many requests" was to send three times as many. 16,220 of 18,237 files
    ## failed.
    ##
    ## Correct behaviour is exponential backoff: wait 2s, then 4s, then 8s...
    ## doubling each time. Rate-limit errors get an extra multiplier because
    ## the server has explicitly told us to slow down.
    ##
    ## The random jitter matters too. Without it, all 8 threads would sleep the
    ## same amount and wake up simultaneously, producing a synchronised burst —
    ## the "thundering herd" problem. Jitter spreads their retries out.
    last = None
    for attempt in range(retries):
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=repo_path,
                local_dir=dest_root,   ## keeps the aviary/chunk folder structure
            )
            return repo_path, None
        except Exception as e:  # noqa: BLE001
            ## Deliberately catching bare Exception: we want to survive ANY
            ## failure mode (timeout, 500, disk full, rate limit) and keep the
            ## other threads working. The error text is preserved and logged.
            last = f"{type(e).__name__}: {e}"

            if attempt == retries - 1:
                break  ## out of attempts, don't sleep pointlessly

            delay = 2.0 * (2 ** attempt)          ## 2s, 4s, 8s, 16s, ...
            if is_rate_limited(last):
                delay *= 3                        ## server asked us to back off
            delay += random.uniform(0, delay * 0.3)   ## jitter
            time.sleep(min(delay, 120.0))         ## cap so one file can't stall forever

    return repo_path, last


def main() -> None:
    # 1) arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval", "all"], default="dev")
    ap.add_argument("--limit", type=int, default=None,
                    help="download only N files — use this to smoke test and to measure your bandwidth")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = resolve_dir(cfg, "cache")
    data_dir = resolve_dir(cfg, "data")

    # 2) load the manifest built in the previous stage
    manifest = cache_dir / f"manifest_{args.split}.csv"
    if not manifest.exists():
        raise SystemExit(
            f"No manifest at {manifest}.\n"
            f"Run first:  python -m src.build_manifest --split {args.split}"
        )

    df = pd.read_csv(manifest)
    if args.limit:
        df = df.head(args.limit)

    paths = df["repo_path"].tolist()

    # 2a) how many do we still actually need? (resumability, and an honest ETA)
    todo = [p for p in paths if not (data_dir / p).exists()]
    print(f"{len(paths)} files in manifest, {len(paths) - len(todo)} already on disk, "
          f"{len(todo)} to fetch (~{len(todo) * 528_044 / 1e9:.1f} GB)")

    # 2b) authentication check — this is the difference between success and 429
    if not check_auth():
        print("\n" + "!" * 70)
        print("NOT LOGGED IN to Hugging Face.")
        print("Anonymous downloads are rate-limited and WILL fail at this volume.")
        print("Run:  hf auth login          (huggingface-cli is deprecated in hub 1.x)")
        print("Token: https://huggingface.co/settings/tokens  (a READ token is enough)")
        print("!" * 70 + "\n")
        if input("Continue anyway? [y/N] ").strip().lower() != "y":
            raise SystemExit("Aborted. Log in, then re-run.")

    # 3) THE PARALLEL DOWNLOAD
    ##  ThreadPoolExecutor (not ProcessPool) because this is I/O-bound, not
    ##  CPU-bound — the threads spend their time waiting on the network, and
    ##  Python's GIL doesn't block waiting. 8 workers is a reasonable default;
    ##  more isn't always faster and HF may rate-limit you.
    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=cfg["download"]["workers"]) as ex:

        # 3a) submit() queues every job immediately and returns a Future
        #     (a placeholder for "a result that will exist later")
        futs = [
            ex.submit(fetch_one, cfg["repo_id"], p, data_dir, cfg["download"]["max_retries"])
            for p in paths
        ]

        # 3b) as_completed() yields futures in the order they FINISH, not the
        #     order submitted — so the progress bar moves smoothly instead of
        #     stalling on one slow file.
        for f in tqdm(as_completed(futs), total=len(futs), unit="file"):
            p, err = f.result()
            if err:
                errors.append((p, err))

    # 4) report failures so they can be retried (just re-run this script —
    #    completed files get skipped by the check in step 1 of fetch_one)
    if errors:
        log = cache_dir / f"download_errors_{args.split}.txt"
        log.write_text("\n".join(f"{p}\t{e}" for p, e in errors))
        print(f"\n{len(errors)} failures logged to {log}.")
        print("Re-run the same command to retry — already-downloaded files are skipped.")
    else:
        print("\nAll files downloaded.")


if __name__ == "__main__":
    main()

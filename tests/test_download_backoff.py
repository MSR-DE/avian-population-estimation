"""Regression tests for the download retry logic.

The bug these guard against actually happened: 16,220 of 18,237 files failed
with HTTP 429 because fetch_one retried immediately with no delay, turning
"you are sending too many requests" into three times as many requests.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.download as dl
from src.download import fetch_one, is_rate_limited

# --- 1. rate-limit detection -------------------------------------------------
assert is_rate_limited("RuntimeError: HTTP status client error (429 Too Many Requests)")
assert is_rate_limited("429")
## the case that exposed the original bug: lowercase haystack, capitalised needle
assert is_rate_limited("Too Many Requests")
assert is_rate_limited("too many requests")
assert not is_rate_limited("ConnectionError: DNS failure")
assert not is_rate_limited("FileNotFoundError")
print("rate-limit detection: 6/6 OK")

# --- 2. backoff actually delays, and gives up cleanly ------------------------
calls = {"n": 0}


def always_429(**kwargs):
    calls["n"] += 1
    raise RuntimeError("HTTP status client error (429 Too Many Requests)")


dl.hf_hub_download = always_429
t0 = time.time()
path, err = fetch_one("repo", "dev_aviary_1/chunk_000/rec_d1_00_00_01.wav",
                      Path("/tmp/does-not-exist"), retries=3)
elapsed = time.time() - t0

assert calls["n"] == 3, f"expected 3 attempts, got {calls['n']}"
assert err is not None and is_rate_limited(err)
## attempt 0 sleeps ~2s*3, attempt 1 sleeps ~4s*3, attempt 2 doesn't sleep.
## So a correct implementation takes >= ~18s; the buggy one took ~0s.
assert elapsed > 6, f"no backoff happened — slept only {elapsed:.1f}s"
print(f"backoff: 3 attempts over {elapsed:.1f}s, error preserved")

# --- 3. no pointless sleep after the final attempt ---------------------------
calls["n"] = 0
t0 = time.time()
fetch_one("repo", "dev_aviary_1/chunk_000/rec_d1_00_00_02.wav",
          Path("/tmp/does-not-exist"), retries=1)
assert calls["n"] == 1
assert time.time() - t0 < 1.0, "should not sleep when there are no retries left"
print("no trailing sleep on final attempt: OK")

# --- 4. success path returns no error ----------------------------------------
def works(**kwargs):
    calls["n"] += 1
    return "/fake/path"


dl.hf_hub_download = works
calls["n"] = 0
path, err = fetch_one("repo", "dev_aviary_1/chunk_000/rec_d1_00_00_03.wav",
                      Path("/tmp/does-not-exist"), retries=3)
assert err is None and calls["n"] == 1, (err, calls)
print("success path: single call, no error")

print("\nALL CHECKS PASSED")

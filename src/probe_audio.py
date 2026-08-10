## Stage 0c — check what the audio ACTUALLY is, instead of trusting the brief.
##
## THE OPEN QUESTION
## Every file on the hub is exactly 528,044 bytes.
## The challenge doc says "3-second WAV recordings sampled at 48 kHz".
##
## Let's do that arithmetic:
##     3 seconds x 48,000 samples/sec x 2 bytes/sample (16-bit) = 288,000 bytes
##     + 44 bytes of WAV header                                 = 288,044 bytes
##
## But the real files are 528,044 bytes — about 1.83x bigger.
## So at least one of {3 seconds, mono, 16-bit} in the brief is WRONG.
## (Notice both numbers end in 044 — that's the 44-byte header, which tells us
##  it's uncompressed PCM. The difference is entirely in the audio payload.)
##
## WHY THIS MATTERS — it is not pedantry.
## Every feature downstream is "per file": detections per file, calls per file.
## If a file is 5.5 seconds rather than 3, then every rate we compute is on a
## different scale than the brief implies, and comparisons against the
## published baseline are off by a constant factor.
##
## Run this on the first ~20 downloaded files, then write the answer into the
## README and use it as the normalisation constant in features.py.
##
## Usage:
##     python -m src.probe_audio

from __future__ import annotations

import argparse
from collections import Counter

import soundfile as sf

from .common import load_config, resolve_dir

## What the brief's description would predict, for comparison in the output.
EXPECTED_BYTES_3S_48K_MONO_16 = 3 * 48_000 * 2 + 44


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="how many files to inspect")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = resolve_dir(cfg, "data")

    # 1) find whatever audio we've downloaded so far.
    ##   rglob = "recursive glob", searches all subfolders. sorted() makes the
    ##   selection deterministic so re-running inspects the same files.
    wavs = sorted(data_dir.rglob("*.wav"))[: args.n]
    if not wavs:
        raise SystemExit(
            f"No .wav files under {data_dir}.\n"
            "Run:  python -m src.build_manifest --split dev\n"
            "then: python -m src.download --split dev --limit 20"
        )

    ## Counter is a dict subclass that counts things — Counter[key] += 1 works
    ## even on keys that don't exist yet.
    ## Key includes `frames` so that a file of a different LENGTH shows up as a
    ## separate profile — otherwise mixed-duration clips would hide in one row.
    profiles: Counter = Counter()   ## (samplerate, channels, subtype, frames) -> n
    sizes: Counter = Counter()      ## byte size -> n

    # 2) inspect each file's HEADER.
    ##   sf.info() reads only the header, not the audio samples — so this is
    ##   fast even on large files. We are not decoding anything.
    print(f"{'file':<44} {'bytes':>9} {'sr':>7} {'ch':>3} {'subtype':<10} {'sec':>7}")
    print("-" * 88)
    for w in wavs:
        info = sf.info(str(w))
        nbytes = w.stat().st_size

        ## duration = number of samples / samples per second
        dur = info.frames / info.samplerate

        profiles[(info.samplerate, info.channels, info.subtype, info.frames)] += 1
        sizes[nbytes] += 1

        print(
            f"{w.name:<44} {nbytes:>9} {info.samplerate:>7} {info.channels:>3} "
            f"{info.subtype:<10} {dur:>7.3f}"
        )

    # 3) summarise what we found
    print("\n--- summary ---")
    for (sr, ch, sub, frames), n in profiles.items():
        ## soundfile reports bit depth as a 'subtype' string, so map it back
        ## to a number for the arithmetic check.
        bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32}.get(sub, 16)
        secs = frames / sr
        print(f"{n} files: {sr} Hz, {ch} ch, {sub} ({bits}-bit), {secs:.4f} s")

        ## Reconstruct the file size from the properties we actually observed.
        ## If this lands on 528,044 we have fully explained the discrepancy —
        ## which is the point of the whole script.
        predicted = int(frames * ch * bits / 8) + 44
        match = "MATCHES" if predicted in sizes else "does NOT match"
        print(f"   -> these properties predict {predicted} bytes ({match} observed)")

    print(f"\ndistinct file sizes observed: {dict(sizes)}")
    print(f"brief predicts (3s/48k/mono/16-bit): {EXPECTED_BYTES_3S_48K_MONO_16}")

    # 4) THE ANSWER — the number to carry into features.py
    info = sf.info(str(wavs[0]))
    actual_dur = info.frames / info.samplerate
    print(f"\nACTUAL clip duration: {actual_dur:.4f} s   (brief claims 3 s)")

    if abs(actual_dur - 3.0) > 0.05:
        print(
            ">>> The brief is WRONG about clip duration.\n"
            f">>> Use {actual_dur:.4f} s as the per-file normalisation constant\n"
            ">>> in features.py, and note the discrepancy in the report."
        )
    else:
        print(
            ">>> Duration matches the brief — so the extra bytes come from\n"
            ">>> channel count or bit depth instead. Check the summary above."
        )


if __name__ == "__main__":
    main()

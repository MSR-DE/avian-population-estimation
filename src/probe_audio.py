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

    # 1) find audio, sampling from EVERY aviary — not just the first N files.
    ##
    ## WHY PER-AVIARY: an earlier version took `sorted(rglob('*.wav'))[:20]`,
    ## which returned 20 files that all happened to come from dev_aviary_1. It
    ## concluded "clips are 2.75 s, 528,044 bytes" — true of aviary_1 and FALSE
    ## of aviary_2, whose files are ~1.01 MB. Sampling one group and
    ## generalising to all of them is exactly the mistake this script exists to
    ## prevent, so it must look at each aviary separately.
    per_aviary: dict[str, list] = {}
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        files = sorted(d.rglob("*.wav"))[: args.n]
        if files:
            per_aviary[d.name] = files

    wavs = [w for files in per_aviary.values() for w in files]
    if not wavs:
        raise SystemExit(
            f"No .wav files under {data_dir}.\n"
            "Run:  python -m src.build_manifest --split dev\n"
            "then: python -m src.download --split dev --limit 20"
        )
    print(f"Inspecting up to {args.n} files from each of "
          f"{len(per_aviary)} aviaries: {', '.join(per_aviary)}\n")

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

    # 4) THE ANSWER — per-aviary normalisation constants for features.py
    ##
    ## If these differ across aviaries then "detections per file" is measured in
    ## different units per aviary, and rates are NOT comparable until each is
    ## divided by its own clip duration. Getting this wrong would silently
    ## corrupt every cross-aviary comparison — which is the entire task.
    print("\n" + "=" * 76)
    print("PER-AVIARY AUDIO PROFILE")
    print("=" * 76)
    print(f"{'aviary':<16}{'sr':>7}{'ch':>4}{'subtype':<9}{'seconds':>9}{'bytes':>10}")
    durations = {}
    for aviary, files in per_aviary.items():
        info = sf.info(str(files[0]))
        dur = info.frames / info.samplerate
        durations[aviary] = dur
        nb = files[0].stat().st_size
        print(f"{aviary:<16}{info.samplerate:>7}{info.channels:>4}"
              f"{info.subtype:<9}{dur:>9.4f}{nb:>10}")

    # 5) FULL SCAN — is duration constant WITHIN each aviary?
    ##
    ## Steps 1-4 only opened `--n` files per aviary. That is the same sampling
    ## mistake one level down: a mixed-duration aviary would look uniform.
    ##
    ## We don't need to decode anything to check. For uncompressed PCM,
    ##     bytes = seconds * sample_rate * channels * (bit_depth/8) + 44
    ## so duration is recoverable from the file size alone. os.stat() over
    ## 10k files takes a second; opening them all would take minutes.
    profile_set = {(sr, ch, sub) for (sr, ch, sub, _f) in profiles}
    if len(profile_set) == 1:
        sr, ch, sub = profile_set.pop()
        bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32}.get(sub, 16)
        bytes_per_sec = sr * ch * bits // 8

        print("\n" + "=" * 76)
        print(f"FULL SCAN — every downloaded file, duration inferred from size")
        print(f"(assuming {sr} Hz / {ch} ch / {sub}, i.e. {bytes_per_sec} bytes per second)")
        print("=" * 76)

        for aviary in sorted(p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            sz: Counter = Counter(
                f.stat().st_size for f in (data_dir / aviary).rglob("*.wav")
            )
            total = sum(sz.values())
            parts = []
            for nbytes, n in sorted(sz.items()):
                secs = (nbytes - 44) / bytes_per_sec
                parts.append(f"{secs:.4f}s x{n}")
                ## record the longest duration seen, so a mixed aviary is not
                ## silently normalised by whichever file happened to be first
                durations[aviary] = max(durations.get(aviary, 0), secs)
            flag = "  <-- MIXED" if len(sz) > 1 else ""
            print(f"  {aviary:<16}{total:>7} files   {', '.join(parts)}{flag}")
    else:
        print("\n!! Mixed sample-rate/channel profiles — size-based scan skipped.")

    uniq = sorted(set(round(v, 4) for v in durations.values()))
    print()
    if len(uniq) == 1:
        d = uniq[0]
        print(f">>> All aviaries share clip duration {d:.4f} s.")
        if abs(d - 3.0) > 0.05:
            print(f">>> The brief claims 3 s and is WRONG. Set "
                  f"audio.clip_seconds = {d:.4f} in config.yaml.")
    else:
        print(f">>> WARNING: clip duration VARIES across aviaries: {uniq}")
        print(">>> A single audio.clip_seconds in config.yaml is therefore wrong.")
        print(">>> features.py must normalise each aviary by ITS OWN duration:")
        for a, d in sorted(durations.items()):
            print(f"      {a:<16} {d:.4f} s")


if __name__ == "__main__":
    main()

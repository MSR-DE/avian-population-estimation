## Stage 1 — run ARIA over the sampled audio, once, and cache the result.
##
## Detection is the expensive step (a BirdNET + PERCH + fusion ensemble over
## ~18,000 clips). It must run EXACTLY ONCE. Everything downstream reads the
## cached parquet, so feature engineering can be re-run freely.
##
## THREE THINGS THIS SCRIPT FIXES ABOUT ARIA'S RAW OUTPUT
##
## 1. The CSV's "File name" column is a BASENAME, not a path:
##        rec_d1_06_59_00.500000.wav
##    Filenames repeat across aviaries — dev_aviary_1 and dev_aviary_3 both
##    contain files named rec_d1_00_00_*.wav. If you ran the detector over a
##    merged directory you could not tell them apart afterwards. So we run it
##    ONE AVIARY AT A TIME and tag each row with its aviary.
##
## 2. Detections per file are NOT comparable across aviaries. ARIA analyses
##    fixed 3.0 s segments. A 2.75 s clip (aviaries 1,3,4,5,6) is padded to one
##    segment; a 5.25 s clip (aviary 2) yields two. Identical birds would
##    therefore produce ~2x the detections per file in aviary 2. We record
##    duration and segment count per file so features.py can normalise by
##    SECONDS OF AUDIO rather than by file.
##
## 3. It has no notion of recording time. We parse day and time-of-day back out
##    of the filename so bout and diel features are possible later.
##
## Usage:
##     python -m src.detect --split dev
##     python -m src.detect --split dev --aviary dev_aviary_1   # single aviary
##     python -m src.detect --split dev --dry-run               # print commands

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

from .common import load_config, parse_repo_path, resolve_dir

SEGMENT_SECONDS = 3.0  ## ARIA's default --segment-length; BirdNET's native window


def audio_profile(aviary_dir: Path) -> tuple[int, int, int]:
    """(sample_rate, channels, bytes_per_second) from the first file found.

    We read ONE header, then infer every other file's duration from its byte
    size. os.stat over thousands of files is ~instant; opening them all is not.
    """
    first = next(aviary_dir.rglob("*.wav"))
    info = sf.info(str(first))
    bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32}.get(info.subtype, 16)
    return info.samplerate, info.channels, info.samplerate * info.channels * bits // 8


def run_aria(
    aviary_dir: Path,
    out_csv: Path,
    model_dir: Path,
    species_file: Path | None,
    extra: list[str],
    dry_run: bool,
) -> bool:
    """Invoke the aria-inference CLI for one aviary. Returns True on success."""
    cmd = [
        "aria-inference", "detect",
        "--input", str(aviary_dir),
        "--output", str(out_csv),
        "--model-dir", str(model_dir),
    ]
    ## Restricting to the species actually present in this aviary removes a
    ## large class of false positives — notably Chilean/American Flamingo
    ## being mistaken for our Greater Flamingo target. See species_lists.py.
    if species_file is not None and species_file.exists():
        cmd += ["--allowed-species-file", str(species_file)]
    cmd += extra

    print("  $ " + " ".join(cmd))
    if dry_run:
        return False

    t0 = time.time()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  !! aria-inference exited {proc.returncode}", file=sys.stderr)
        return False
    print(f"  done in {time.time() - t0:.0f}s")
    return True


def file_inventory(aviary: str, aviary_dir: Path, bytes_per_sec: int) -> pd.DataFrame:
    """One row per AUDIO FILE that was analysed — including files with no
    detections at all.

    WHY THIS IS SEPARATE FROM THE DETECTIONS TABLE (and why it matters)
    The detections parquet only contains rows where the detector fired. Files
    where nothing was detected leave no trace in it. But those files are not
    missing data — they are *evidence of silence*, and two of the most useful
    features depend on counting them:

      p_active   = fraction of files containing >=1 target detection.
                   The occupancy model N ~ -ln(1 - p_active) is built on it.
      rate       = detections per SECOND OF AUDIO ANALYSED.

    Both need the denominator "all files analysed". Deriving it from the
    detections table is simply wrong: summing clip_seconds over detection rows
    double-counts busy files and drops silent ones entirely, so the ratio
    degenerates to 1/clip_seconds and carries no information at all.
    """
    rows = []
    for p in sorted(aviary_dir.rglob("*.wav")):
        secs = (p.stat().st_size - 44) / bytes_per_sec
        clip = parse_repo_path(f"{aviary}/chunk_000/{p.name}")
        rows.append({
            "aviary": aviary,
            "filename": p.name,
            "day": clip.day if clip else pd.NA,
            "t_sec": clip.t_sec if clip else pd.NA,
            "clip_seconds": secs,
            "n_segments": int(-(-secs // SEGMENT_SECONDS)),
        })
    return pd.DataFrame(rows)


def tidy(raw_csv: Path, aviary: str, bytes_per_sec: int, aviary_dir: Path) -> pd.DataFrame:
    """Turn ARIA's CSV into tidy rows with aviary, time, and duration attached."""
    df = pd.read_csv(raw_csv)

    # 1) normalise ARIA's spaced column names into snake_case
    df = df.rename(columns={
        "File name": "filename",
        "Start (s)": "start_s",
        "End (s)": "end_s",
        "Species": "species_common",
        "Confidence": "confidence",
        "Method": "method",
    })
    df["aviary"] = aviary

    # 2) recover recording time from the filename.
    ##   parse_repo_path expects "<aviary>/<chunk>/<file>", so we synthesise
    ##   that shape — only the filename part actually matters to the parser.
    parsed = df["filename"].map(lambda f: parse_repo_path(f"{aviary}/chunk_000/{f}"))
    df["day"] = [p.day if p else pd.NA for p in parsed]
    df["t_sec"] = [p.t_sec if p else pd.NA for p in parsed]

    # 3) attach per-file duration, inferred from byte size (see audio_profile).
    ##   Without this, a detection in aviary 2 (5.25 s clips) and one in
    ##   aviary 1 (2.75 s clips) would be treated as equivalent evidence.
    sizes = {p.name: p.stat().st_size for p in aviary_dir.rglob("*.wav")}
    df["clip_seconds"] = df["filename"].map(
        lambda f: (sizes.get(f, 0) - 44) / bytes_per_sec if f in sizes else pd.NA
    )
    ## how many 3 s windows ARIA actually analysed for this clip
    df["n_segments"] = df["clip_seconds"].map(
        lambda s: int(-(-float(s) // SEGMENT_SECONDS)) if pd.notna(s) else pd.NA
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--aviary", default=None, help="run just this one aviary")
    ap.add_argument("--model-dir", default="~/aria-models")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-species-filter", action="store_true",
                    help="skip the per-aviary whitelist (for an ablation)")
    ap.add_argument("--config", default=None)
    ap.add_argument("extra", nargs="*",
                    help="extra flags passed straight to aria-inference detect")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = resolve_dir(cfg, "cache")
    data_dir = resolve_dir(cfg, "data")
    model_dir = Path(args.model_dir).expanduser()

    raw_dir = cache_dir / "detections_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    aviaries = [args.aviary] if args.aviary else cfg["splits"][args.split]

    frames: list[pd.DataFrame] = []
    inventories: list[pd.DataFrame] = []
    for aviary in aviaries:
        aviary_dir = data_dir / aviary
        if not aviary_dir.exists() or not any(aviary_dir.rglob("*.wav")):
            print(f"{aviary}: no audio downloaded yet — skipping")
            continue

        n_files = sum(1 for _ in aviary_dir.rglob("*.wav"))
        sr, ch, bps = audio_profile(aviary_dir)
        print(f"\n{aviary}: {n_files} files, {sr} Hz / {ch} ch")

        out_csv = raw_dir / f"{aviary}.csv"
        species_file = None if args.no_species_filter else (
            cache_dir / "species_lists" / f"{aviary}.txt"
        )

        # 1) skip aviaries already detected — this stage is expensive, and it
        #    must be safe to interrupt and resume like the download is.
        if out_csv.exists():
            print(f"  cached: {out_csv}")
        elif not run_aria(aviary_dir, out_csv, model_dir, species_file,
                          args.extra, args.dry_run):
            continue

        if not out_csv.exists():
            continue

        # 2) tidy detections AND record every file analysed (see file_inventory)
        df = tidy(out_csv, aviary, bps, aviary_dir)
        inv = file_inventory(aviary, aviary_dir, bps)
        frames.append(df)
        inventories.append(inv)

        ## denominator is total audio ANALYSED, from the file inventory —
        ## never from the detection rows
        total_secs = inv["clip_seconds"].sum()
        active = df["filename"].nunique()
        print(f"  {len(df)} detections over {len(inv)} files "
              f"({total_secs/3600:.2f} h audio), {df.species_common.nunique()} species")
        print(f"  {len(df)/total_secs:.4f} detections/audio-second, "
              f"p_active={active/len(inv):.3f}")

    if not frames:
        raise SystemExit("\nNo detections produced.")

    # 3) two cached tables — features.py reads both.
    ##   detections_*.parquet : one row per DETECTION
    ##   files_*.parquet      : one row per FILE ANALYSED (incl. silent ones)
    all_df = pd.concat(frames, ignore_index=True)
    all_inv = pd.concat(inventories, ignore_index=True)

    det_out = cache_dir / f"detections_{args.split}.parquet"
    inv_out = cache_dir / f"files_{args.split}.parquet"
    all_df.to_parquet(det_out, index=False)
    all_inv.to_parquet(inv_out, index=False)

    print(f"\nWrote {det_out}  ({len(all_df)} detections)")
    print(f"Wrote {inv_out}  ({len(all_inv)} files analysed)")

    # 4) summary with the CORRECT denominator: audio actually analysed
    det = all_df.groupby("aviary").agg(
        detections=("species_common", "size"),
        species=("species_common", "nunique"),
        active_files=("filename", "nunique"),
    )
    inv = all_inv.groupby("aviary").agg(
        files=("filename", "size"),
        audio_hours=("clip_seconds", lambda s: s.sum() / 3600),
        clip_s=("clip_seconds", "max"),
    )
    summ = inv.join(det, how="left").fillna(0)
    summ["det_per_sec"] = (summ.detections / (summ.audio_hours * 3600)).round(4)
    summ["p_active"] = (summ.active_files / summ.files).round(3)
    summ["audio_hours"] = summ.audio_hours.round(2)
    print("\nPer aviary (denominator = audio analysed, not detection rows):")
    print(summ[["files", "clip_s", "audio_hours", "detections",
                "species", "det_per_sec", "p_active"]].to_string())


if __name__ == "__main__":
    main()

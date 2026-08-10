## Shared helpers used by every other script.
## If you only read one file first, read this one — the Clip object and the
## filename parser defined here are what the whole pipeline moves around.

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

## ROOT = the repo folder (Aviation-Population-Estimation/).
## parents[0] would be src/, so parents[1] climbs one more level up.
## We do this instead of relative paths so scripts work from any directory.
ROOT = Path(__file__).resolve().parents[1]


## ---------------------------------------------------------------------------
## The filename parser
## ---------------------------------------------------------------------------
## Real filenames from the dataset look like:
##     rec_d1_00_01_49.wav            <- day 1, 00:01:49
##     rec_d1_00_00_45.750000.wav     <- day 1, 00:00:45.75  (fractional seconds!)
##
## So the seconds field is sometimes an integer and sometimes has 6 decimals.
## If you write a parser that assumes integers it will silently drop ~half the
## files. That is why the regex has `(?:\.\d+)?` — the "optional decimal part".
##
## Named groups (?P<day>...) let us write m["day"] instead of m.group(1),
## which is much easier to read later.
FNAME_RE = re.compile(
    r"rec_d(?P<day>\d+)_(?P<hh>\d{1,2})_(?P<mm>\d{1,2})_(?P<ss>\d+(?:\.\d+)?)\.wav$"
)


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml into a plain dict.

    Every script calls this instead of hard-coding numbers. That means you can
    change the sample size in ONE place and the whole pipeline follows.
    """
    # 1) default to the config.yaml sitting next to this repo's root
    path = Path(path) if path else ROOT / "config.yaml"

    # 2) safe_load (not load) — safe_load refuses to execute arbitrary Python
    #    embedded in the YAML. Habit worth keeping even on files you wrote.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dir(cfg: dict, key: str) -> Path:
    """Turn a config path key like 'data' into a real folder, creating it.

    parents=True  -> also create missing parent folders
    exist_ok=True -> don't crash if it already exists (so this is re-runnable)
    """
    d = ROOT / cfg["paths"][key]
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class Clip:
    """One audio file, placed in aviary-local time.

    frozen=True makes it immutable (you can't accidentally reassign a field),
    which also makes it hashable so we can use Clips in sets and dict keys.

    Why store day and t_sec separately instead of one timestamp?
    Because the challenge anonymised the dates — 'd1' in one aviary has NO
    relationship to 'd1' in another. There is no real calendar to map onto,
    so we keep day as an opaque label and seconds-since-midnight as the
    thing that actually carries meaning (time of day drives bird behaviour).
    """

    repo_path: str  ## e.g. "dev_aviary_1/chunk_000/rec_d1_00_01_49.wav"
    aviary: str     ## e.g. "dev_aviary_1"
    day: int        ## 1, 2, 3... local to this aviary only
    t_sec: float    ## seconds since midnight

    @property
    def hour(self) -> float:
        """Hour of day as a float, e.g. 13.5 == 13:30.

        Used for diel (time-of-day) stratification — birds call far more at
        dawn, so a sample clumped at 6am would badly overestimate call rate.
        """
        return self.t_sec / 3600.0

    @property
    def abs_sec(self) -> float:
        """Seconds since the start of day 1, for THIS aviary.

        Use this whenever you need to sort clips or compute the gap between
        two consecutive recordings. Sorting by t_sec alone would interleave
        days (day 2 at 00:01 would sort before day 1 at 23:59).
        """
        return (self.day - 1) * 86400.0 + self.t_sec


def parse_repo_path(repo_path: str) -> Clip | None:
    """Turn a repo path string into a Clip. Returns None if it isn't audio.

    The repo contains non-audio files (README.md, metadata/*.csv,
    .gitattributes). Rather than filtering those out at every call site, this
    function returns None for them and callers just skip Nones. One rule,
    one place.
    """
    # 1) every audio file sits exactly 3 levels deep:
    #    aviary / chunk_NNN / rec_....wav
    #    Anything with a different shape is not audio.
    parts = repo_path.split("/")
    if len(parts) != 3:
        return None

    aviary, _chunk, fname = parts

    # 2) folder must actually be an aviary (guards against future folders)
    if "aviary" not in aviary:
        return None

    # 3) filename must match the timestamp pattern
    m = FNAME_RE.search(fname)
    if not m:
        return None

    # 4) collapse HH:MM:SS into a single seconds-since-midnight number.
    #    float() on the seconds field handles both "49" and "45.750000".
    t = int(m["hh"]) * 3600 + int(m["mm"]) * 60 + float(m["ss"])

    return Clip(repo_path=repo_path, aviary=aviary, day=int(m["day"]), t_sec=t)


def duplicate_group_map(cfg: dict) -> dict[str, str]:
    """Map every aviary to a canonical id, collapsing duplicates together.

    WHY THIS EXISTS (this is the subtle one):
    Some aviaries have byte-identical ground truth — same species, same counts.
    e.g. dev_aviary_5 and dev_aviary_6, and dev_aviary_1 vs eval_aviary_1.
    They are almost certainly the same physical enclosure recorded twice.

    If you run leave-one-out cross-validation naively, you might train on
    dev_aviary_5 and validate on dev_aviary_6 — which is really validating on
    data you already trained on. Your CV score comes out looking great and
    then collapses on the real eval set.

    Fix: give duplicates a shared canonical id and split CV folds by that id
    instead of by folder name. This is 'GroupKFold' thinking.

    Returns e.g. {"dev_aviary_5": "dev_aviary_5", "dev_aviary_6": "dev_aviary_5", ...}
    """
    canon: dict[str, str] = {}

    # 1) for each duplicate group, pick one member as the canonical name.
    #    sorted()[0] just makes the choice deterministic — any consistent
    #    rule works, what matters is that it's the same every run.
    for group in cfg.get("duplicate_groups", []):
        head = sorted(group)[0]
        for a in group:
            canon[a] = head

    # 2) every aviary NOT in a duplicate group is its own canonical id.
    #    setdefault only writes if the key is missing, so it won't clobber
    #    the mappings we just made in step 1.
    all_aviaries = cfg["splits"]["dev"] + cfg["splits"]["eval"]
    for a in all_aviaries:
        canon.setdefault(a, a)

    return canon

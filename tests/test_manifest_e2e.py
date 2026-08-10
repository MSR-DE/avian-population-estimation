"""End-to-end run of build_manifest.main() with the HF listing stubbed out."""
import sys, json, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.build_manifest as bm

tmp = Path(tempfile.mkdtemp())
(tmp/'cache').mkdir(); (tmp/'results').mkdir()

# Pre-seed the per-aviary listing cache so no network call happens.
(tmp/'cache'/'listings').mkdir()
for n, aviary in enumerate(['dev_aviary_1','dev_aviary_2','dev_aviary_3',
                            'dev_aviary_4','dev_aviary_5','dev_aviary_6']):
    paths=[]
    ndays = 2 if aviary=='dev_aviary_5' else 3
    for d in range(1, ndays+1):
        t=0.0
        while t < 86400:
            paths.append(f"{aviary}/chunk_{int(t//20000):03d}/rec_d{d}_{int(t//3600):02d}_{int(t%3600//60):02d}_{t%60:.6f}.wav")
            t += 25.0
    (tmp/'cache'/'listings'/f'{aviary}.json').write_text(json.dumps(paths))

cfg = f"""
repo_id: fake/repo
paths: {{data: {tmp}/data, cache: {tmp}/cache, results: {tmp}/results}}
sampling: {{window_seconds: 1800, max_files_per_aviary: 3000, seed: 17,
           stratify_by_hour_bucket: true, hour_buckets: 6}}
splits:
  dev: [dev_aviary_1, dev_aviary_2, dev_aviary_3, dev_aviary_4, dev_aviary_5, dev_aviary_6]
  eval: [eval_aviary_1]
duplicate_groups: [[dev_aviary_5, dev_aviary_6]]
download: {{workers: 8, max_retries: 3}}
"""
cfgp = tmp/'config.yaml'; cfgp.write_text(cfg)

# resolve_dir joins against ROOT, but our paths are absolute so they win.
sys.argv = ['x', '--split', 'dev', '--config', str(cfgp)]
bm.main()

import pandas as pd
df = pd.read_csv(tmp/'cache'/'manifest_dev.csv')
print("\n--- assertions ---")
assert set(df.aviary.unique()) == {f'dev_aviary_{i}' for i in range(1,7)}, df.aviary.unique()
print(f"all 6 dev aviaries present, {len(df)} rows total")
per = df.groupby('aviary').size()
assert per.min() >= 3000, per
print(f"every aviary >= 3000 files (min {per.min()}, max {per.max()})")
assert df[df.aviary=='dev_aviary_5'].day.nunique() == 2, "aviary_5 has only 2 days"
print("day counts respected per aviary")
assert not (tmp/'data').exists() or not any((tmp/'data').rglob('*.wav')), "must not download audio"
print("no audio downloaded — manifest only")
print("\nE2E PASSED")

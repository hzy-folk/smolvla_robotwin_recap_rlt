#!/usr/bin/env python3
"""Finalize an audited qpos16 success dataset and its reproducible episode splits."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split-seed', type=int, default=20261112)
    parser.add_argument('--validation-count', type=int, help='default: 20 percent, at least one episode per split')
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((root / 'correctness_audit.json').read_text())
    entries = manifest['episodes']
    count = len(entries)
    assert count >= 2 and audit['all_pass'] and audit['count'] == count
    assert [e['id'] for e in entries] == [f'episode_{i:03d}' for i in range(count)]

    validation_count = args.validation_count if args.validation_count is not None else max(1, round(count * 0.2))
    assert 0 < validation_count < count
    rng = np.random.default_rng(args.split_seed)
    validation = set(rng.choice(count, size=validation_count, replace=False).tolist())
    for i, entry in enumerate(entries):
        entry['split'] = 'validation' if i in validation else 'train'
        entry['task_success'] = True
    manifest['status'] = 'complete_audited'
    manifest['count'] = count
    manifest['split_seed'] = args.split_seed
    manifest['split_counts'] = {'train': count - validation_count, 'validation': validation_count}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    split_root = root / 'splits'
    split_root.mkdir(exist_ok=True)
    for split in ('train', 'validation'):
        rows = [entry for entry in entries if entry['split'] == split]
        (split_root / f'{split}.json').write_text(
            json.dumps({'split': split, 'episodes': rows}, ensure_ascii=False, indent=2))

    print(json.dumps({'status': manifest['status'], 'count': len(entries),
                      'train': sum(e['split'] == 'train' for e in entries),
                      'validation': sum(e['split'] == 'validation' for e in entries),
                      'all_audited_pass': audit['all_pass']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

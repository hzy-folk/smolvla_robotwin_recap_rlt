#!/usr/bin/env python3
"""Audit and replay a numbered RoboTwin success dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def replay(entry, root):
    import sys
    from dataset import SMOL_ROOT
    sys.path.insert(0, str(SMOL_ROOT / "collection"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from robotwin_common import make_env
    from robotwin_qpos_scene import make_left_right_env
    from robotwin_smol_control import Controller

    folder = root / entry['npz'].rsplit('/', 1)[0]
    raw = folder / f"{entry['id']}.npz"
    with np.load(raw, allow_pickle=False) as d:
        seed = int(d['seed'])
        actions = np.asarray(d['action'], dtype=np.float32)
        config = json.loads((folder / 'environment_config.json').read_text())
    env, _ = make_left_right_env(make_env, seed, collect=False,
                                 wrist_camera=config['project2_wrist_camera'],
                                 head_camera=config['project2_head_camera'],
                                 layout=config['project2_layout'])
    try:
        controller = Controller(env)
        for action in actions:
            controller.execute(action)
        result = controller.finish()
        return result
    finally:
        env.close_env()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = manifest['episodes']
    rows = []
    progress_path = root / 'audit_progress.json'
    for entry in entries:
        folder = root / entry['npz'].rsplit('/', 1)[0]
        raw = folder / f"{entry['id']}.npz"
        checks = {'id': entry['id'], 'npz': str(raw.relative_to(root))}
        with np.load(raw, allow_pickle=False) as d:
            n = len(d['state'])
            checks.update(frames=n, state_shape=list(d['state'].shape),
                          action_shape=list(d['action'].shape),
                          finite=bool(np.isfinite(d['state']).all() and np.isfinite(d['action']).all()),
                          success=bool(d['success']), sample_hz=int(d['sample_hz']))
            checks['arrays_aligned'] = all(d[k].shape[0] == n for k in ('action', 'images', 'left_wrist', 'right_wrist'))
        video_checks = []
        for name in ('head', 'left_wrist', 'right_wrist'):
            path = folder / f'{name}.mp4'
            cap = cv2.VideoCapture(str(path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
            cap.release()
            video_checks.append({'file': path.name, 'exists': path.is_file(), 'frames': count, 'fps': fps,
                                 'aligned': count == checks['frames']})
        checks['videos'] = video_checks
        checks['duration_ok'] = 11.0 <= checks['frames'] / 25.0 <= 15.0
        if args.replay:
            checks['replay'] = replay(entry, root)
            checks['replay_success'] = bool(checks['replay'].get('stable_success', False))
        checks['pass'] = bool(checks['finite'] and checks['success'] and checks['sample_hz'] == 25
                              and checks['arrays_aligned'] and checks['duration_ok']
                              and all(v['exists'] and v['aligned'] for v in video_checks)
                              and (not args.replay or checks['replay_success']))
        rows.append(checks)
        progress_path.write_text(json.dumps({'completed': len(rows), 'total': len(entries),
            'passed': sum(bool(x['pass']) for x in rows), 'last_episode': entry['id'],
            'rows': rows}, ensure_ascii=False, indent=2))
        print(json.dumps({'id': entry['id'], 'pass': checks['pass']}, ensure_ascii=False), flush=True)
    audit = {'all_pass': all(x['pass'] for x in rows), 'count': len(rows),
             'total_frames': sum(x['frames'] for x in rows), 'replay': args.replay,
             'rows': [{**x, 'npz_sha256': sha256(root / x['npz'])} for x in rows]}
    out = args.out or root / 'correctness_audit.json'
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({'audit': str(out), 'all_pass': audit['all_pass']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

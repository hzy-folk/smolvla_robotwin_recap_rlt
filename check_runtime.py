"""Read-only checks for configured source, interpreter and weight locations."""
import argparse
import hashlib
import json
from pathlib import Path
from runtime_config import ROOT, path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile', choices=('all', 'model', 'robotwin', 'libero', 'value'), default='all')
    parser.add_argument('--verify-hashes', action='store_true', help='also hash task assets and base SmolVLA weights')
    args = parser.parse_args()
    needed = []
    if args.profile in ('all', 'model', 'robotwin', 'libero'):
        needed += [path('model_dir') / 'config.json', path('model_dir') / 'model.safetensors',
                   path('vlm_dir') / 'config.json', path('vlm_dir') / 'tokenizer_config.json']
    if args.profile in ('all', 'robotwin'):
        needed += [path('robotwin_python'), path('robotwin_root') / 'envs/stack_blocks_two.py',
                   path('robotwin_root') / 'assets/embodiments/piper/config.yml']
    if args.profile in ('all', 'model'):
        needed += [path('model_python')]
    if args.profile in ('all', 'libero'):
        needed += [path('libero_python'), path('libero_root') / 'libero/libero/envs/env_wrapper.py']
        task = json.loads((ROOT / 'assets.required.json').read_text())['libero_pro_task']
        needed += [path('libero_pro_root') / name for name in task['files']]
    if args.profile in ('all', 'value'):
        needed += [path('value_python'), path('pistar_root') / 'scripts/train_value.py',
                   path('pistar_root') / 'gemma/gemma/gm/nn/_gemma.py']
    missing = [str(p) for p in needed if not p.is_file()]
    if args.verify_hashes:
        def sha(p):
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            return h.hexdigest()
        if args.profile in ('all', 'model', 'robotwin', 'libero'):
            weight = path('model_dir') / 'model.safetensors'
            if weight.is_file() and sha(weight) != '7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb':
                missing.append('Base weight hash differs from the retained experiments; use the pinned revision in DEPENDENCIES.md')
        if args.profile in ('all', 'libero'):
            for name, record in task['files'].items():
                file = path('libero_pro_root') / name
                if file.is_file() and sha(file) != record['sha256']:
                    missing.append(f'Task asset hash mismatch: {file}')
    print(json.dumps({'profile': args.profile, 'status': 'incomplete' if missing else 'paths_present',
                      'missing': missing,
                      'scope': 'paths/hashes only; not a fresh installation or full simulator test'}, indent=2))
    raise SystemExit(bool(missing))


if __name__ == '__main__':
    main()

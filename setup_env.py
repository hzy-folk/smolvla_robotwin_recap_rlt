"""Plan or build isolated environments; never downloads data or model weights."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent


def source_commands(name):
    lock = json.loads((ROOT / 'sources.lock.json').read_text())[name]
    dest = ROOT / lock['directory']
    return dest, lock


def fetch(name):
    dest, lock = source_commands(name)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', str(dest)], check=True)
        subprocess.run(['git', '-C', str(dest), 'remote', 'add', 'origin', lock['url']], check=True)
        subprocess.run(['git', '-C', str(dest), 'fetch', '--depth', '1', 'origin', lock['revision']], check=True)
        subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    revision = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != lock['revision']:
        raise RuntimeError(f'{dest} is at {revision}; expected {lock["revision"]}. Existing checkout was not changed.')
    if lock.get('patch'):
        patch = ROOT / lock['patch']
        command = ['git', '-C', str(dest), 'apply']
        already = subprocess.run(command + ['--reverse', '--check', str(patch)], capture_output=True).returncode == 0
        if not already:
            subprocess.run(command + ['--check', str(patch)], check=True)
            subprocess.run(command + [str(patch)], check=True)


def commands(profile, backend):
    env = ROOT / '.venvs' / profile
    python = env / 'bin/python'
    result = [['uv', 'venv', '--python', '3.11' if profile == 'value' else '3.10', str(env)]]
    pip = ['uv', 'pip', 'install', '--python', str(python)]
    if profile == 'value':
        result += [pip + ['--no-deps', '--torch-backend', backend, '-r', str(ROOT / 'requirements/value.txt')],
                   pip + ['--no-deps', '-e', str(ROOT / 'third_party/pistar/packages/openpi-client')]]
    else:
        result.append(pip + ['--torch-backend', backend, '-r', str(ROOT / f'requirements/{profile}.txt')])
        if profile == 'libero':
            result.append(pip + ['--no-deps', '-e', str(ROOT / 'third_party/LIBERO')])
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile', choices=('model', 'robotwin', 'libero', 'value'), required=True)
    parser.add_argument('--torch-backend', default='cu128', help='cu128 or cpu; match GPU/driver support')
    parser.add_argument('--fetch-sources', action='store_true')
    parser.add_argument('--install', action='store_true', help='execute commands; otherwise only print a plan')
    args = parser.parse_args()
    source = {'robotwin': 'robotwin', 'libero': 'libero', 'value': 'pistar'}.get(args.profile)
    if args.fetch_sources and source:
        _, lock = source_commands(source)
        print(f'Source: {lock["url"]} @ {lock["revision"]}', flush=True)
        if args.install:
            fetch(source)
    for command in commands(args.profile, args.torch_backend):
        print(shlex.join(command), flush=True)
        if args.install:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

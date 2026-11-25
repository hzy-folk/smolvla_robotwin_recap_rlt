"""Portable runtime paths shared by collection, RECAP and RLT (stdlib only)."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = Path(os.environ.get('SMOLVLA_RUNTIME_CONFIG', ROOT / 'runtime.local.json')).expanduser()
DEFAULTS = {
    'model_dir': 'assets/smolvla', 'vlm_dir': 'assets/smolvlm',
    'lerobot_src': '', 'robotwin_root': 'third_party/RoboTwin',
    'libero_root': 'third_party/LIBERO', 'libero_pro_root': 'third_party/LIBERO-Pro-task',
    'pistar_root': 'third_party/pistar', 'libero_config': '.runtime/libero',
    'model_python': '.venvs/model/bin/python', 'robotwin_python': '.venvs/robotwin/bin/python',
    'libero_python': '.venvs/libero/bin/python', 'value_python': '.venvs/value/bin/python',
    'model_extra_paths': [], 'sim_extra_paths': [],
}
SETTINGS = dict(DEFAULTS)
if CONFIG_FILE.is_file():
    overrides = json.loads(CONFIG_FILE.read_text())
    unknown = overrides.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'Unknown runtime options: {sorted(unknown)}')
    SETTINGS.update(overrides)
elif 'SMOLVLA_RUNTIME_CONFIG' in os.environ:
    raise FileNotFoundError(CONFIG_FILE)


def path(key):
    value = os.environ.get('SMOLVLA_' + key.upper(), SETTINGS[key])
    if not value:
        return None
    result = Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (ROOT / result).resolve()


def add_paths(role, prepend=False):
    values = SETTINGS.get(role + '_extra_paths', [])
    for value in reversed(values) if prepend else values:
        import re
        version = re.search(r'python(\d+\.\d+)', value)
        if version and version.group(1) != f'{sys.version_info.major}.{sys.version_info.minor}':
            continue
        candidate = Path(value).expanduser()
        candidate = candidate if candidate.is_absolute() else ROOT / candidate
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate)) if prepend else sys.path.append(str(candidate))


def bootstrap_model():
    add_paths('model', prepend=True)
    source = path('lerobot_src')
    if source and source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))


MODEL_DIR = path('model_dir')
VLM_DIR = path('vlm_dir')
MODEL_PYTHON = path('model_python')
ROBOTWIN_PYTHON = path('robotwin_python')
LIBERO_PYTHON = path('libero_python')
PISTAR_ROOT = path('pistar_root')
ROBOTWIN_ROOT = path('robotwin_root')
LIBERO_ROOT = path('libero_root')
LIBERO_PRO_ROOT = path('libero_pro_root')

"""Check independent path configuration and installer plans without installation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='smolvla-runtime-') as folder:
        config = Path(folder) / 'runtime.json'
        config.write_text(json.dumps({'model_dir': 'assets/custom_model',
                                      'robotwin_root': str(Path(folder) / 'robotwin')}))
        env = dict(os.environ, SMOLVLA_RUNTIME_CONFIG=str(config),
                   SMOLVLA_VLM_DIR=str(Path(folder) / 'vlm'), PYTHONPATH=str(ROOT))
        output = subprocess.check_output([sys.executable, '-c',
            'import runtime_config as r,json;print(json.dumps([str(r.MODEL_DIR),str(r.ROBOTWIN_ROOT),str(r.VLM_DIR)]))'],
            cwd=folder, env=env, text=True)
        assert json.loads(output) == [str(ROOT / 'assets/custom_model'),
                                     str(Path(folder) / 'robotwin'), str(Path(folder) / 'vlm')]
        for profile in ('model', 'robotwin', 'libero', 'value'):
            output = subprocess.check_output([sys.executable, str(ROOT / 'setup_env.py'),
                '--profile', profile, '--fetch-sources'], cwd=folder, text=True)
            assert 'uv venv' in output and f'requirements/{profile}.txt' in output
            assert '/home/hzy/' not in output.replace(str(ROOT), '<repository>')
    for base in ('collection', 'recap', 'rlt'):
        for file in (ROOT / base).rglob('*.py'):
            assert '/home/hzy/' not in file.read_text(), file
    print(json.dumps({'status': 'passed', 'isolated_config': True,
                      'installer_profiles': 4, 'installation_executed': False}))


if __name__ == '__main__':
    main()

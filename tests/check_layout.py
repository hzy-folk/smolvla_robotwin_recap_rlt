"""Check task configuration precedence and actual dispatcher paths without models."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'collection'), str(ROOT / 'recap'), str(ROOT)]
from collection_config import parse_task_args
from run_pipeline import commands, STAGES
from rlt.bootstrap import bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int)
    parser.add_argument('--first-seed', type=int)
    with patch.object(sys, 'argv', ['collect']):
        defaults = parse_task_args(parser, 'robotwin', 'collection')
    assert defaults.count == 3 and defaults.first_seed == 930000

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int)
    parser.add_argument('--first-seed', type=int)
    with patch.object(sys, 'argv', ['collect', '--count', '7']):
        overridden = parse_task_args(parser, 'robotwin', 'collection')
    assert overridden.count == 7 and overridden.first_seed == 930000

    config = json.loads((ROOT / 'recap/configs/pipeline.example.json').read_text())
    plan = commands(config)
    assert tuple(plan) == STAGES
    for command in plan.values():
        assert Path(command[1]).is_file()
        assert Path(command[1]).parent == ROOT / 'recap'
    assert ROOT / 'recap_runs/reproduce' == Path(plan['rollout'][plan['rollout'].index('--output') + 1]).parent
    assert bootstrap()['scripts'] == ROOT / 'collection'
    import dataset
    assert dataset.SMOL_ROOT == ROOT and Path(dataset.__file__).parent == ROOT / 'collection'

    for relative in ('collection/robotwin_smol_worker.py',
                     'recap/generate_rlt_actor_100_gallery.py',
                     'rlt/legacy_collect/scripts/infer_smol_robotwin_qpos16.py',
                     'rlt/configs/robotwin_piper_nstep3_bc10warmup5000.yaml'):
        assert (ROOT / relative).is_file(), relative
    assert not (ROOT / 'scripts').exists()
    print(json.dumps({'status': 'passed', 'config_override': True,
                      'recap_dispatcher_stages': len(plan),
                      'shared_modules': 'collection', 'project_root': str(ROOT)}))


if __name__ == '__main__':
    main()

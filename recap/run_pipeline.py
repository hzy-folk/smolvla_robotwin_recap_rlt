#!/usr/bin/env python3
"""Run one RECAP iteration with explicit environments and recorded stage status."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'recap'
STAGES = ('initial_sft', 'rollout', 'prepare_value', 'train_value', 'predict_value',
          'label_advantage', 'final_sft', 'evaluate')


def commands(config):
    def path(key):
        p = Path(config[key]).expanduser()
        return str((ROOT / p).resolve() if not p.is_absolute() else p)
    out = Path(path('output'))
    model, sim, value = (path(k) for k in ('model_python', 'sim_python', 'value_python'))
    initial, final = config.get('initial_steps', 20000), config.get('final_steps', 24000)
    steps = config.get('value_steps', 40000)
    batch = config.get('policy_batch_size', 56)
    common = ['--batch-size', str(batch), '--micro-batch', str(config.get('policy_micro_batch', batch)),
              '--warmup-steps', str(config.get('warmup_steps', 1000))]
    value_data, value_run = out/'value_data', out/'value'
    assets = ['--pistar-root', path('pistar_root'), '--tokenizer', path('tokenizer')]
    value_init = (['--init-checkpoint', path('value_init_checkpoint')]
                  if config.get('value_init_checkpoint') else
                  ['--siglip-checkpoint', path('siglip'), '--gemma-checkpoint', path('gemma')])
    def cmd(python, script, *args):
        return [python, str(SCRIPTS/script), *map(str, args)]
    return {
        'initial_sft': cmd(model, 'train_recap_initial_sft.py', '--data', path('success_data'),
            '--cache', out/'initial_cache', '--output', out/'initial', '--steps', initial, *common),
        'rollout': cmd(sim, 'run_rollout50.py', '--checkpoint', out/f'initial/checkpoint_{initial:06d}.pt',
            '--data', path('scene_data'), '--split', 'train', '--output', out/'rollout',
            '--count', config.get('rollout_count', 50), '--advantage', 'positive'),
        'prepare_value': cmd(sim, 'prepare_recap_value_dataset.py', '--success', path('success_data'),
            '--failure', path('failure_data'), '--rollout', out/'rollout', '--output', value_data),
        'train_value': cmd(value, 'train_recap_value_vision.py', '--data', value_data,
            '--output', value_run/'checkpoints', *assets, *value_init, '--steps', steps,
            '--batch-size', config.get('value_batch_size', 8),
            '--micro-batch-size', config.get('value_micro_batch', config.get('value_batch_size', 8)),
            '--save-every', config.get('value_save_every', 10000),
            '--full-validation-every', config.get('value_validation_every', 10000)),
        'predict_value': cmd(value, 'predict_recap_values.py', '--data', value_data,
            '--checkpoint', value_run/f'checkpoints/step_{steps:08d}', *assets,
            '--output', value_run/'predicted_values.json'),
        'label_advantage': cmd(value, 'label_recap_advantage.py', '--value-data', value_data,
            '--values', value_run/'predicted_values.json', '--output', value_run/'advantages.json'),
        'final_sft': cmd(model, 'train_recap_policy.py', '--data', value_run,
            '--value-data', value_data, '--cache', out/'final_cache', '--output', out/'final',
            '--steps', final, '--lr-decay-steps', config.get('final_lr_decay_steps', final), *common),
        'evaluate': cmd(sim, 'run_rollout50.py', '--checkpoint', out/f'final/checkpoint_{final:06d}.pt',
            '--data', path('scene_data'), '--split', 'validation', '--output', out/'evaluation',
            '--count', config.get('eval_count', 20), '--advantage', 'positive'),
    }



def verify_stage(stage, plan):
    command = plan[stage]
    output = Path(command[command.index('--output') + 1])
    if stage in ('initial_sft', 'final_sft'):
        step = int(command[command.index('--steps') + 1])
        status = json.loads((output/'status.json').read_text())
        if status.get('event') != 'completed' or status.get('step') != step:
            raise RuntimeError(f'{stage} did not finish the requested updates')
        if not (output/f'checkpoint_{step:06d}.pt').is_file():
            raise FileNotFoundError(f'{stage}: final checkpoint is absent')
    elif stage in ('rollout', 'evaluate'):
        status = json.loads((output/'summary.json').read_text())
        if (status.get('status') != 'completed' or status.get('errors', 0)
                or status.get('finished') != int(command[command.index('--count')+1])
                or any(not row.get('completed') for row in status['episodes'])):
            raise RuntimeError(f'{stage}: incomplete episodes; inspect summary.json')
    elif stage == 'prepare_value':
        status = json.loads((output/'manifest.json').read_text())
        if status.get('samples', 0) <= 0:
            raise RuntimeError('prepared value dataset is empty')
    elif stage == 'train_value':
        status = json.loads((output/'report.json').read_text())
        step = int(command[command.index('--steps') + 1])
        if status.get('status') != 'completed' or status.get('steps') != step:
            raise RuntimeError('value trainer stopped before completion')
        if not (output/f'step_{step:08d}').is_dir():
            raise FileNotFoundError('final value checkpoint is absent')
    else:
        status = json.loads(output.read_text())
        if not status or (stage == 'label_advantage' and not status.get('rows')):
            raise RuntimeError(f'{stage}: output is empty')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--plan', action='store_true', help='print commands without executing or writing files')
    parser.add_argument('--resume', action='store_true', help='skip previously completed stages of the same plan')
    parser.add_argument('--from-stage', choices=STAGES, default=STAGES[0])
    parser.add_argument('--through-stage', choices=STAGES, default=STAGES[-1])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan = commands(config)
    selected = STAGES[STAGES.index(args.from_stage):STAGES.index(args.through_stage)+1]
    if not selected:
        parser.error('from-stage must precede through-stage')
    if args.plan:
        print(json.dumps({k: plan[k] for k in selected}, ensure_ascii=False, indent=2))
        return
    out = Path(config['output']).expanduser()
    out = (ROOT/out).resolve() if not out.is_absolute() else out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out/'pipeline_status.json'
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    state = {'plan_sha256': digest, 'completed': [], 'commands': plan}
    if status_path.exists():
        if not args.resume:
            parser.error('output already contains a pipeline; use a new output or --resume')
        state = json.loads(status_path.read_text())
        if state['plan_sha256'] != digest:
            parser.error('configuration changed; use a new output')
    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(status_path)
    for stage in selected:
        if stage in state['completed']:
            continue
        state.update(stage=stage, status='running'); save()
        print(f'RECAP stage: {stage}', flush=True)
        try:
            with (out/f'{stage}.log').open('a') as log:
                command = list(plan[stage])
                stage_output = Path(command[command.index('--output')+1])
                if args.resume and stage in ('initial_sft', 'final_sft') and (stage_output/'latest.json').exists():
                    command.append('--resume')
                if args.resume and stage == 'train_value':
                    checkpoints = sorted(p for p in stage_output.glob('step_*')
                                         if (p/'_CHECKPOINT_METADATA').exists())
                    if checkpoints:
                        if '--init-checkpoint' in command:
                            index = command.index('--init-checkpoint')
                            del command[index:index + 2]
                        command += ['--resume', str(checkpoints[-1])]
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                verify_stage(stage, plan)
        except BaseException as exc:
            state.update(status='failed', error=str(exc)); save()
            raise
        state['completed'].append(stage)
        state.pop('error', None)
        state['status'] = 'completed' if stage == STAGES[-1] else 'stage_completed'
        save()


if __name__ == '__main__':
    main()

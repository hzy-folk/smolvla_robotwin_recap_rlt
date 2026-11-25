#!/usr/bin/env python3
"""Run closed-loop RoboTwin episodes from a manifest split without requiring videos."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime_config import path as runtime_path
PYTHON = runtime_path('robotwin_python')
INFER = ROOT / 'rlt/legacy_collect/scripts/infer_smol_robotwin_qpos16.py'


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--task-prompt', default='stack the red block and then the green block.')
    p.add_argument('--flow-steps', type=int, default=4)
    p.add_argument('--rollout-steps', type=int, default=376)
    p.add_argument('--max-episodes', type=int, help='limit the run for a parity smoke test')
    p.add_argument('--resume', action='store_true', help='resume from completed episode reports and rebuild replay')
    p.add_argument('--rlt-token', type=Path, help='save the frozen z and chunk transitions alongside evaluation')
    p.add_argument('--rlt-config', type=Path, help='RLT config used to encode z')
    p.add_argument('--expected-summary', type=Path,
                   help='stop if per-episode success differs from the old evaluation summary')
    p.add_argument('--replay-output', type=Path, help='assemble RLT replay buffer after the run')
    a = p.parse_args()
    assert not a.replay_output or a.rlt_token, '--replay-output requires --rlt-token'

    checkpoint, data, output = a.checkpoint.resolve(), a.data.resolve(), a.output.resolve()
    assert checkpoint.is_file() and PYTHON.is_file()
    entries = [e for e in json.loads((data / 'manifest.json').read_text())['episodes']
               if e.get('split') == a.split]
    if a.max_episodes is not None:
        assert a.max_episodes > 0
        entries = entries[:a.max_episodes]
    assert entries, f'no episodes in split {a.split!r}'
    expected = None
    if a.expected_summary:
        expected_rows = json.loads(a.expected_summary.read_text())['episodes']
        expected = {row['source_episode']: bool(row['official_success']) for row in expected_rows}

    replay = None
    replay_config = None
    if a.replay_output:
        sys.path.insert(0, str(ROOT))
        from rlt.config import load_config
        from rlt.replay import SOURCE_BASE, ReplayBuffer
        config_path = a.rlt_config or ROOT / 'rlt/configs/robotwin_piper.yaml'
        replay_config = load_config(config_path)
        token = __import__('torch').load(a.rlt_token, map_location='cpu', weights_only=False)
        z_dim = int(token.get('embed_dim', replay_config.token.embed_dim)) * int(token['num_tokens'])
        replay = ReplayBuffer(
            replay_config.runtime.replay_capacity, z_dim, replay_config.data.proprio_dim,
            replay_config.data.action_dim, replay_config.data.chunk_len, seed=replay_config.runtime.seed,
            n_step_chunks=replay_config.objective.n_step_chunks,
            gamma=replay_config.actor_critic.gamma,
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / 'rollouts').mkdir(exist_ok=True)
    selection = {
        'checkpoint': str(checkpoint), 'data': str(data), 'split': a.split,
        'count': len(entries), 'entries': entries,
        'rlt_token': str(a.rlt_token.resolve()) if a.rlt_token else None,
    }
    selection_path = output / 'selection.json'
    if a.resume and selection_path.is_file():
        if json.loads(selection_path.read_text()) != selection:
            raise SystemExit(f'resume selection differs from existing run: {selection_path}')
    else:
        write(selection_path, selection)

    prior_rows = []
    summary_path = output / 'summary.json'
    if a.resume and summary_path.is_file():
        prior_rows = [row for row in json.loads(summary_path.read_text()).get('episodes', [])
                      if row.get('completed')]
    rows = list(prior_rows)
    started = time.time()

    def progress():
        successes = sum(bool(row.get('official_success')) for row in rows)
        completed = sum(bool(row.get('completed')) for row in rows)
        write(output / 'summary.json', {
            'status': 'completed' if len(rows) == len(entries) else 'running',
            'split': a.split, 'finished': len(rows), 'completed': completed,
            'total': len(entries), 'successes': successes,
            'success_rate': successes / len(rows) if rows else None,
            'errors': sum(not row.get('completed', False) for row in rows),
            'episodes': rows, 'elapsed_seconds': time.time() - started,
        })
        rate = f'{successes / len(rows):.1%}' if rows else 'n/a'
        (output / '测试进度.md').write_text(
            f'# RoboTwin {a.split} split rollout\n\n'
            f'- 状态：{"完成" if len(rows) == len(entries) else "运行中"}\n'
            f'- 完成：{len(rows)}/{len(entries)}\n'
            f'- 官方成功：{successes}\n- 成功率：{rate}\n')

    def add_episode_to_replay(index, episode_dir, report):
        if replay is None:
            return
        chunk_path = episode_dir / 'rlt_chunks.npz'
        assert chunk_path.is_file(), f'missing RLT chunk export: {chunk_path}'
        with np.load(chunk_path, allow_pickle=False) as chunks:
            z, proprio, ref = chunks['z'], chunks['proprio'], chunks['ref']
            actions = chunks['action']
            executed_steps = int(report.get('steps', 0))
            chunk_len = replay_config.data.chunk_len
            if executed_steps <= 0:
                raise RuntimeError(f"report has invalid executed step count: {report}")
            count = (executed_steps + chunk_len - 1) // chunk_len
            if count > len(z):
                raise RuntimeError(f"report needs {count} chunks but export only has {len(z)}")
            episode_success = bool(report.get('official_success'))
            episode_rows = []
            for chunk_index in range(count):
                terminal = chunk_index == count - 1
                following = chunk_index + 1
                reward = np.zeros(chunk_len, dtype=np.float32)
                success = episode_success and terminal
                if success:
                    reward[(executed_steps - 1) % chunk_len] = float(replay_config.env.success_reward)
                episode_rows.append({
                    'z': z[chunk_index],
                    'z_next': z[chunk_index] if terminal else z[following],
                    'proprio': proprio[chunk_index],
                    'proprio_next': proprio[chunk_index] if terminal else proprio[following],
                    'ref': ref[chunk_index],
                    'ref_next': ref[chunk_index] if terminal else ref[following],
                    'action': actions[chunk_index],
                    'reward': reward,
                    'done': terminal,
                    'success': success,
                    'source': SOURCE_BASE,
                    'source_chunk': np.zeros(replay_config.data.chunk_len, dtype=np.uint8),
                    'episode': index,
                    'stage': 1,
                })
            replay.add_episode(episode_rows)
        if a.replay_output:
            replay.save(a.replay_output)

    for row in prior_rows:
        add_episode_to_replay(int(row['id'].rsplit('_', 1)[1]),
                              output / 'rollouts' / row['id'], row)
    completed_ids = {row['id'] for row in prior_rows}
    if prior_rows:
        progress()

    for index, entry in enumerate(entries):
        if f'episode_{index:03d}' in completed_ids:
            continue
        episode_dir = output / 'rollouts' / f'episode_{index:03d}'
        if episode_dir.exists():
            raise SystemExit(f'incomplete episode directory exists; inspect before resume: {episode_dir}')
        raw = data / entry['npz']
        cmd = [str(PYTHON), str(INFER), '--mode', 'rollout', '--raw', str(raw),
               '--checkpoint', str(checkpoint), '--output', str(episode_dir),
               '--device', a.device, '--flow-steps', str(a.flow_steps),
               '--rollout-steps', str(a.rollout_steps), '--no-video',
               '--check-success-every-step']
        if a.rlt_token:
            cmd += ['--rlt-token', str(a.rlt_token.resolve())]
            if a.rlt_config:
                cmd += ['--rlt-config', str(a.rlt_config.resolve())]
        cmd.extend(['--task-prompt', a.task_prompt])
        begin = time.time()
        with (output / f'episode_{index:03d}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT.parent, stdout=log,
                                    stderr=subprocess.STDOUT)
        report_path = episode_dir / 'report.json'
        row = json.loads(report_path.read_text()) if report_path.exists() else {
            'completed': False, 'error': 'missing report'}
        row.update(id=f'episode_{index:03d}', source_episode=entry['id'],
                   seed=entry.get('seed'), exit_code=result.returncode,
                   wall_seconds=time.time() - begin)
        rows.append(row)

        if expected is not None and entry['id'] in expected:
            if bool(row.get('official_success')) != expected[entry['id']]:
                progress()
                raise SystemExit(
                    f"success mismatch for {entry['id']} seed={entry.get('seed')}: "
                    f"clone={row.get('official_success')} old={expected[entry['id']]}, stop for investigation"
                )

        if row.get('completed'):
            add_episode_to_replay(index, episode_dir, row)

        progress()
        print(json.dumps({
            'finished': len(rows), 'total': len(entries), 'seed': row['seed'],
            'success': row.get('official_success'), 'completed': row.get('completed'),
        }), flush=True)

    progress()
    if replay is not None:
        replay.save(a.replay_output)
        write(output / 'replay_import.json', {
            'path': str(a.replay_output.resolve()),
            'size': len(replay),
            'capacity': replay.capacity,
            'episodes': len(rows),
            'successes': sum(bool(row.get('official_success')) for row in rows),
            'updates_before_training': 0,
        })


if __name__ == '__main__':
    main()

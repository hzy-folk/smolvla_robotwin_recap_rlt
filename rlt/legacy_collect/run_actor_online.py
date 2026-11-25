#!/usr/bin/env python3
"""Stage 3 online RLT using the copied legacy evaluator execution path."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SMOL_ROOT = ROOT.parent
ROBOTWIN_PYTHON = Path('/home/hzy/git/learn/feasibility_audit/robotwin_venv/bin/python')
INFER = ROOT / 'legacy_collect/scripts/infer_smol_robotwin_qpos16.py'
sys.path.insert(0, str(SMOL_ROOT))

from rlt import bootstrap  # noqa: E402
bootstrap.bootstrap()

import torch  # noqa: E402

from rlt.actor_critic import RLTLearner  # noqa: E402
from rlt.config import load_config  # noqa: E402
from rlt.online import LearnerThread, build_normalizer  # noqa: E402
from rlt.replay import SOURCE_RL, ReplayBuffer  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(path)


def load_entries(manifest: Path) -> tuple[Path, list[dict]]:
    payload = json.loads(manifest.read_text())
    entries = payload.get('episodes', payload)
    if isinstance(entries, dict):
        entries = entries.get('episodes', [])
    entries = [row for row in entries if row.get('split', 'train') == 'train']
    root = manifest.parent.parent if manifest.parent.name == 'splits' else manifest.parent
    if not entries:
        raise ValueError(f'empty train manifest: {manifest}')
    for row in entries:
        if not (root / row['npz']).is_file():
            raise FileNotFoundError(root / row['npz'])
    return root, entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--token', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--flow-steps', type=int, default=4)
    parser.add_argument('--rollout-steps', type=int, help='defaults to env.max_episode_steps in the config')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = load_config(args.config)
    if args.rollout_steps is None:
        args.rollout_steps = config.env.max_episode_steps
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    config.runtime.out_dir = str(run_dir)
    device = args.device or config.runtime.device
    config.env.num_envs = 1
    config.validate()
    manifest = args.manifest.resolve()
    data_root, entries = load_entries(manifest)
    checkpoint = args.checkpoint.resolve()
    token_path = args.token.resolve()
    if not ROBOTWIN_PYTHON.is_file() or not checkpoint.is_file() or not token_path.is_file():
        raise FileNotFoundError('missing Robotwin Python, SmolVLA checkpoint, or RL-token checkpoint')

    stage_out = run_dir / 'stage3_legacy_rollout'
    rollouts = stage_out / 'rollouts'
    rollouts.mkdir(parents=True, exist_ok=True)
    summary_path = stage_out / 'summary.json'
    prior_rows = []
    if args.resume and summary_path.is_file():
        prior_rows = [row for row in json.loads(summary_path.read_text()).get('episodes', [])
                      if row.get('completed')]
    completed_ids = {row['id'] for row in prior_rows}

    replay_path = run_dir / 'replay_buffer.npz'
    with np.load(replay_path, allow_pickle=False) as stored:
        z_dim = int(stored['z'].shape[1])
    replay = ReplayBuffer(config.runtime.replay_capacity, z_dim, config.data.proprio_dim,
                          config.data.action_dim, config.data.chunk_len, seed=config.runtime.seed,
                          n_step_chunks=config.objective.n_step_chunks,
                          gamma=config.actor_critic.gamma)
    replay.load(replay_path)
    normalizer = build_normalizer(config)
    delta_mask = torch.ones(config.data.action_dim, dtype=torch.bool)
    for index in config.data.delta_exclude_dims:
        delta_mask[index] = False
    learner = RLTLearner(
        z_dim, config.data.proprio_dim, config.data.action_dim, config.data.chunk_len,
        config.actor_critic, config.objective, denormalize=normalizer.denormalize,
        delta_joint_mask=delta_mask, device=device,
    )
    checkpoint_dir = run_dir / 'checkpoints'
    stage3_checkpoints = sorted(checkpoint_dir.glob('stage3_learner_step_*.pt'))
    stage2_checkpoint = checkpoint_dir / 'stage2_final.pt'
    resume_path = stage3_checkpoints[-1] if stage3_checkpoints else stage2_checkpoint
    if not resume_path.is_file():
        raise FileNotFoundError(f'no stage-2 learner checkpoint in {checkpoint_dir}')
    resume_state = torch.load(resume_path, map_location=device, weights_only=False)
    learner.load_state_dict(resume_state)
    snapshot_path = run_dir / 'actor_snapshot.pkl'
    if not snapshot_path.is_file():
        raise FileNotFoundError(f'missing trained actor snapshot: {snapshot_path}')
    actor_payload = torch.load(snapshot_path, map_location='cpu', weights_only=False)
    if int(actor_payload.get('version', 0)) < 1:
        raise RuntimeError('stage-2 actor snapshot has no actor updates')

    thread = LearnerThread(
        learner, replay, normalizer, config, snapshot_path,
        run_dir / 'metrics/learner.jsonl', run_dir / 'metrics/learner_status.json',
        run_dir / 'learner.pt', z_dim, device, learning_enabled=True,
        actor_start_updates=0,
        checkpoint_every=config.runtime.checkpoint_every,
        checkpoint_prefix='stage3_learner',
    )
    thread.warmup_ready_adds = resume_state.get('warmup_ready_adds')
    thread.ready_for_online = True

    rows = list(prior_rows)
    start_time = time.time()

    def publish(status: str) -> None:
        successes = sum(bool(row.get('official_success')) for row in rows)
        payload = {
            'status': status,
            'episodes_finished': len(rows),
            'successes': successes,
            'success_rate': successes / len(rows) if rows else None,
            'replay_size': len(replay),
            'replay_capacity': replay.capacity,
            'adds_total': replay.adds_total,
            'learner_steps': learner.step_count,
            'actor_updates': learner.actor_version,
            'elapsed_seconds': time.time() - start_time,
            'episodes': rows,
        }
        write_json(summary_path, payload)
        (stage_out / '训练进度.md').write_text(
            f'# RLT 第三阶段旧评测链路采集\n\n'
            f'- 状态：{status}\n- 已完成 episode：{len(rows)}\n'
            f'- 成功：{successes}\n- Replay：{len(replay)}/{replay.capacity}\n'
            f'- Learner 更新：{learner.step_count}\n- Actor 更新：{learner.actor_version}\n')

    def replay_rows(index: int, chunks_path: Path) -> list[dict]:
        with np.load(chunks_path, allow_pickle=False) as data:
            z, proprio, ref = data['z'], data['proprio'], data['ref']
            action, reward, success = data['action'], data['reward'], data['success']
        rows_out = []
        for chunk_index in range(len(z)):
            terminal = chunk_index == len(z) - 1
            following = chunk_index + 1
            rows_out.append({
                'z': z[chunk_index],
                'z_next': np.zeros_like(z[chunk_index]) if terminal else z[following],
                'proprio': proprio[chunk_index],
                'proprio_next': proprio[chunk_index] if terminal else proprio[following],
                'ref': ref[chunk_index],
                'ref_next': ref[chunk_index] if terminal else ref[following],
                'action': action[chunk_index],
                'reward': reward[chunk_index],
                'done': terminal,
                'success': bool(success[chunk_index]),
                'source': SOURCE_RL,
                'source_chunk': np.full(config.data.chunk_len, SOURCE_RL, dtype=np.uint8),
                'episode': index,
                'stage': 3,
            })
        return rows_out

    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')

    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    thread.start()
    status = 'running'
    try:
        episode_counter = len(prior_rows)
        while len(replay) < replay.capacity:
            if thread.error is not None:
                raise RuntimeError('learner thread failed before rollout') from thread.error
            entry_index = episode_counter % len(entries)
            entry = entries[entry_index]
            episode_id = replay.max_episode + 1
            rollout_id = f'episode_{episode_id:06d}'
            retry = 0
            while (rollouts / rollout_id).exists():
                retry += 1
                rollout_id = f'episode_{episode_id:06d}_retry_{retry:02d}'
            if rollout_id in completed_ids:
                episode_counter += 1
                continue
            episode_dir = rollouts / rollout_id
            if episode_dir.exists():
                raise RuntimeError(f'incomplete rollout directory exists; inspect before resuming: {episode_dir}')
            command = [str(ROBOTWIN_PYTHON), str(INFER), '--mode', 'rollout',
                       '--raw', str((data_root / entry['npz']).resolve()),
                       '--checkpoint', str(checkpoint), '--output', str(episode_dir),
                       '--device', 'cuda' if device.startswith('cuda') else 'cpu',
                       '--flow-steps', str(args.flow_steps), '--rollout-steps', str(args.rollout_steps),
                       '--no-video', '--check-success-every-step', '--rlt-token', str(token_path),
                       '--rlt-config', str(args.config.resolve()), '--actor-snapshot', str(snapshot_path),
                       '--task-prompt', config.data.language]
            begin = time.time()
            while True:
                command[command.index('--output') + 1] = str(episode_dir)
                log_path = stage_out / f'{rollout_id}.log'
                with log_path.open('w') as log:
                    process = subprocess.Popen(
                        command, cwd=SMOL_ROOT, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        while True:
                            try:
                                return_code = process.wait(timeout=1.0)
                                break
                            except subprocess.TimeoutExpired:
                                if thread.error is not None:
                                    try:
                                        os.killpg(process.pid, signal.SIGTERM)
                                    except ProcessLookupError:
                                        pass
                                    process.wait()
                                    raise RuntimeError('learner thread failed during rollout') from thread.error
                    except KeyboardInterrupt:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        raise
                report_path = episode_dir / 'report.json'
                if report_path.is_file():
                    break
                error_log = log_path.read_text(errors='replace')
                if 'getCount is non-monotonic' not in error_log or retry >= 2:
                    raise RuntimeError(f'legacy rollout failed without report; see {log_path}')
                retry += 1
                rollout_id = f'episode_{episode_id:06d}_retry_{retry:02d}'
                while (rollouts / rollout_id).exists():
                    retry += 1
                    rollout_id = f'episode_{episode_id:06d}_retry_{retry:02d}'
                episode_dir = rollouts / rollout_id
                print(json.dumps({
                    'event': 'retry_rollout_after_torch_clock_assert',
                    'episode': episode_id, 'retry': retry, 'rollout_id': rollout_id,
                }), flush=True)
            report = json.loads(report_path.read_text())
            if return_code or not report.get('completed'):
                raise RuntimeError(f'legacy rollout failed: {report}')
            versions = report.get('actor_versions', [])
            if not versions or min(versions) < 1:
                raise RuntimeError(f'rollout contains chunks without a trained actor: {versions}')
            transitions = replay_rows(episode_id, episode_dir / 'rlt_chunks.npz')
            admitted = transitions[:max(0, replay.capacity - len(replay))]
            replay.add_episode(admitted)
            replay.save(replay_path)
            row = {
                'id': rollout_id,
                'source_episode': entry['id'],
                'seed': entry.get('seed'),
                'official_success': bool(report.get('official_success')),
                'completed': True,
                'steps': report.get('steps'),
                'chunks': len(transitions),
                'transitions_added': len(admitted),
                'actor_versions': versions,
                'actor_version': learner.actor_version,
                'learner_steps': learner.step_count,
                'wall_seconds': time.time() - begin,
                'exit_code': return_code,
            }
            rows.append(row)
            completed_ids.add(rollout_id)
            episode_counter += 1
            publish('buffer_full' if len(replay) >= replay.capacity else 'running')
            print(json.dumps(row, ensure_ascii=False), flush=True)

        target = int(replay.adds_total * config.objective.updates_per_sample)
        while learner.step_count < target:
            if thread.error is not None:
                raise RuntimeError('learner thread failed') from thread.error
            time.sleep(0.1)
        status = 'complete'
    except KeyboardInterrupt:
        status = 'paused'
    finally:
        if status == 'complete':
            thread.stop_event.set()
        thread.shutdown()
        replay.save(replay_path)
        publish(status)

    if thread.error is not None:
        raise RuntimeError('learner thread failed') from thread.error
    print(json.dumps({
        'status': status, 'episodes': len(rows), 'replay_size': len(replay),
        'replay_capacity': replay.capacity, 'learner_steps': learner.step_count,
        'actor_updates': learner.actor_version,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

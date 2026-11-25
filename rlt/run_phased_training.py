"""Run one resumable phase of the three-stage Robotwin RLT training plan."""
from __future__ import annotations

import argparse
import json
import signal
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DEFAULT_CONFIG = ROOT / "rlt/configs/robotwin_piper.yaml"
DEFAULT_SEEDS = ROOT / "data/robotwin_success_qpos16_formal100/splits/train.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2", "3"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--token", type=Path, default=None)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = args.config.resolve()
    seed_manifest = args.seed_manifest.resolve()
    manifest_payload = json.loads(seed_manifest.read_text())
    manifest_episodes = manifest_payload.get("episodes", manifest_payload)
    if isinstance(manifest_episodes, dict):
        manifest_episodes = manifest_episodes.get("episodes", [])
    target_episode_count = len(manifest_episodes)
    if target_episode_count < 1:
        raise SystemExit(f"seed manifest 为空：{seed_manifest}")
    command = [PYTHON, "-u"]

    if args.stage == "1":
        if not args.checkpoint or not args.token:
            raise SystemExit("stage 1 需要 --checkpoint 和 --token 才能提取 SmolVLA/RL-token 特征")
        if args.resume:
            metrics_path = run_dir / "metrics/rollout.jsonl"
            if not metrics_path.is_file():
                raise SystemExit(f"不能恢复 stage 1：缺少 {metrics_path}")
            completed = set()
            for line in metrics_path.read_text().splitlines():
                try:
                    completed.add(int(json.loads(line)["episode"]))
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
            replay_path = run_dir / "replay_buffer.npz"
            if replay_path.is_file():
                import numpy as np
                with np.load(replay_path) as replay:
                    size = int(replay["_size"])
                    completed.update(int(item) for item in np.unique(replay["episode"][:size]))
            remaining = max(0, target_episode_count - len(completed))
        else:
            if any(run_dir.iterdir()):
                raise SystemExit(f"stage 1 输出目录非空；新训练请指定新目录：{run_dir}")
            remaining = target_episode_count
        if remaining == 0:
            print(f"stage 1 已有 {target_episode_count} 集数据，无需继续采集")
            return
        command += [str(ROOT / "rlt/run_online.py"), "--config", str(config),
                    "--phase", "collect", "--episodes", str(remaining),
                    "--target-episode-count", str(target_episode_count),
                    "--seeds-file", str(seed_manifest), "--out", str(run_dir)]
        if args.checkpoint:
            command += ["--checkpoint", str(args.checkpoint.resolve())]
        if args.token:
            command += ["--token", str(args.token.resolve())]
        if args.resume:
            command += ["--resume-dir", str(run_dir)]
    elif args.stage == "2":
        command += [str(ROOT / "rlt/run_fixed_buffer.py"), "--config", str(config),
                    "--run-dir", str(run_dir)]
        if args.resume:
            command += ["--resume"]
    else:
        if not args.resume:
            raise SystemExit("stage 3 必须使用 --resume，接续 stage 2 的 replay 和 learner")
        if not args.checkpoint or not args.token:
            raise SystemExit("stage 3 需要 --checkpoint 和 --token 才能执行在线 rollout")
        command += [str(ROOT / "rlt/legacy_collect/run_actor_online.py"),
                    "--config", str(config), "--run-dir", str(run_dir),
                    "--manifest", str(seed_manifest), "--resume"]
        if args.checkpoint:
            command += ["--checkpoint", str(args.checkpoint.resolve())]
        if args.token:
            command += ["--token", str(args.token.resolve())]

    child = subprocess.Popen(command, cwd=ROOT)
    received_signal = None

    def forward_signal(signum, frame):
        nonlocal received_signal
        received_signal = signum
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    return_code = child.wait()
    if received_signal is not None and return_code == 0:
        return_code = 128 + received_signal
    resume_command = [PYTHON, str(Path(__file__).resolve()), '--stage', args.stage,
                      '--config', str(config), '--run-dir', str(run_dir),
                      '--seed-manifest', str(seed_manifest), '--resume']
    if args.checkpoint:
        resume_command += ['--checkpoint', str(args.checkpoint.resolve())]
    if args.token:
        resume_command += ['--token', str(args.token.resolve())]
    status = {
        "stage": int(args.stage),
        "status": "complete" if return_code == 0 else "interrupted_or_failed",
        "return_code": return_code,
        "resume_command": shlex.join(resume_command),
    }
    tmp = run_dir / f"stage_{args.stage}_status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(run_dir / f"stage_{args.stage}_status.json")
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()

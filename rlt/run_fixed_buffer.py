"""Train only from a saved replay buffer until its data-proportional budget is met."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlt import bootstrap  # noqa: E402
bootstrap.bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rlt.actor_critic import RLTLearner  # noqa: E402
from rlt.config import load_config  # noqa: E402
from rlt.online import LearnerThread, build_normalizer  # noqa: E402
from rlt.replay import ReplayBuffer  # noqa: E402


def main() -> None:
    def request_pause(signum, frame):
        raise KeyboardInterrupt(f"收到信号 {signum}，执行保存并暂停")

    signal.signal(signal.SIGTERM, request_pause)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, help="stage 1 output, also used for checkpoints")
    parser.add_argument("--resume", action="store_true", help="resume learner state from the latest checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    replay_path = run_dir / "replay_buffer.npz"
    if not replay_path.is_file():
        raise SystemExit(f"找不到 stage 1 经验池：{replay_path}")
    with np.load(replay_path) as saved:
        z_dim = int(saved["z"].shape[1])
    device = config.runtime.device
    config.runtime.out_dir = str(run_dir)
    normalizer = build_normalizer(config)
    replay = ReplayBuffer(
        capacity=config.runtime.replay_capacity,
        z_dim=z_dim,
        proprio_dim=config.data.proprio_dim,
        action_dim=config.data.action_dim,
        chunk_len=config.data.chunk_len,
        seed=config.runtime.seed,
        n_step_chunks=config.objective.n_step_chunks,
        gamma=config.actor_critic.gamma,
    )
    replay.load(replay_path)

    checkpoint_paths = sorted((run_dir / "checkpoints").glob("stage2_learner_step_*.pt"))
    resume = None
    if args.resume:
        stage2_checkpoint = run_dir / "checkpoints/stage2_final.pt"
        resume_path = stage2_checkpoint if stage2_checkpoint.is_file() else (
            checkpoint_paths[-1] if checkpoint_paths else None
        )
        if resume_path is None:
            raise SystemExit("请求恢复阶段二，但找不到 learner checkpoint")
        resume = torch.load(resume_path, map_location=device, weights_only=False)
    delta_mask = torch.ones(config.data.action_dim, dtype=torch.bool)
    for index in config.data.delta_exclude_dims:
        delta_mask[index] = False
    learner = RLTLearner(
        z_dim, config.data.proprio_dim, config.data.action_dim, config.data.chunk_len,
        config.actor_critic, config.objective, denormalize=normalizer.denormalize,
        delta_joint_mask=delta_mask, device=device,
    )
    if resume is not None:
        learner.load_state_dict(resume)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    thread = LearnerThread(
        learner, replay, normalizer, config, run_dir / "actor_snapshot.pkl",
        run_dir / "metrics/learner.jsonl", run_dir / "metrics/learner_status.json",
        run_dir / "learner.pt", z_dim, device, learning_enabled=True,
        checkpoint_every=config.runtime.checkpoint_every, checkpoint_prefix="stage2_learner",
    )
    if resume is not None:
        thread.warmup_ready_adds = resume.get("warmup_ready_adds")
        thread.ready_for_online = bool(resume.get("ready_for_online", False))
    target = int(replay.adds_total * config.objective.updates_per_sample)
    thread.start()
    try:
        while learner.step_count < target:
            if thread.error is not None:
                raise RuntimeError("learner 线程失败") from thread.error
            time.sleep(0.1)
        thread.stop_event.set()
    finally:
        thread.shutdown()
    checkpoint_dir = run_dir / "checkpoints"
    stage2_steps = sorted(checkpoint_dir.glob("stage2_learner_step_*.pt"))
    if stage2_steps:
        final_path = checkpoint_dir / "stage2_final.pt"
        temporary = final_path.with_suffix(".pt.tmp")
        temporary.write_bytes(stage2_steps[-1].read_bytes())
        temporary.replace(final_path)
    state = {
        "stage": 2,
        "status": "complete" if learner.step_count >= target else "paused",
        "replay_size": len(replay),
        "adds_total": replay.adds_total,
        "learner_steps": learner.step_count,
        "target_steps": target,
        "actor_updates": learner.actor_version,
    }
    (run_dir / "stage_status.json").write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()

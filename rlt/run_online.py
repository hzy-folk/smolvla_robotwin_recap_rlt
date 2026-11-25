"""在线 RLT 的入口脚本：采集 + 训练 + 快照，一条命令跑起来。

仿真里的典型用法：

    python rlt/run_online.py --config rlt/configs/robotwin_piper.yaml \
        --checkpoint runs/<平台>/<检查点>.pt \
        --token rlt_runs/robotwin/rlt_token.pt \
        --episodes 20

没有仿真器时可以先用假环境验证链路：

    python rlt/run_online.py --config ... --dummy --token ...

注意 `--dummy` 只验证代码路径，产生的数据没有任何研究意义。
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt import bootstrap  # noqa: E402

bootstrap.bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rlt.config import load_config  # noqa: E402
from rlt.env_robotwin import DummyEnv, RoboTwinEnv  # noqa: E402
from rlt.online import OnlineRunner, build_normalizer, build_provider  # noqa: E402


class RandomFeatureProvider:
    """随机特征提供者，只用于在没有 GPU 时验证命令行链路，数据无意义。"""

    def __init__(self, z_dim: int, chunk_len: int, action_dim: int) -> None:
        self.z_dim = z_dim
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.rng = np.random.default_rng(0)

    def features(self, observation: dict) -> dict:
        return {
            "z_rl": self.rng.normal(size=self.z_dim).astype(np.float32),
            "proprio": np.asarray(observation["observation.state"], dtype=np.float32),
            "ref_chunk": self.rng.normal(scale=0.05, size=(self.chunk_len, self.action_dim)).astype(np.float32),
        }

    def features_batch(self, observations: list[dict]) -> list[dict]:
        return [self.features(observation) for observation in observations]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLT 在线训练")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="SmolVLA 检查点（本地特征时需要）")
    parser.add_argument("--token", default=None, help="RL token 检查点")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dummy", action="store_true", help="用假环境自检")
    parser.add_argument("--fake-features", action="store_true", help="用随机特征，仅验证链路")
    parser.add_argument("--z-dim", type=int, default=960, help="--fake-features 时的 z 维度")
    parser.add_argument("--stats", default=None, help="覆盖归一化统计量路径")
    parser.add_argument("--out", default=None, help="覆盖输出目录")
    parser.add_argument("--phase", choices=("collect", "online"), default="online",
                        help="collect 只采集且不更新 learner；online 为在线采集+训练")
    parser.add_argument("--resume-dir", default=None, help="从该目录的 replay_buffer.npz 和最新 learner checkpoint 恢复")
    parser.add_argument("--seeds-file", default=None,
                        help="覆盖训练 scene/seed manifest；scene 模式按其中 seed 选择 NPZ")
    parser.add_argument("--target-episode-count", type=int, default=None,
                        help="可恢复的固定 episode ID 总数；按 rollout 日志补跑未完成集")
    parser.add_argument("--actor-start-updates", type=int, default=None,
                        help="启用 actor 控制环境所需 learner 步数；默认等于 warmup_updates")
    parser.add_argument("--require-actor", action="store_true",
                        help="启动前要求已加载训练过的 Actor，且从第一段 rollout 就启用")
    return parser.parse_args()


def resolve_scene_config(config):
    """定位场景配置。配置里的文件被挪走时，自动在同级 episodes 目录里找一个。"""
    path = config.resolve_path(config.env.scene_config) if config.env.scene_config else None
    if path is not None and path.is_file():
        return path
    if path is not None:
        episodes_dir = path.parent.parent
        candidates = sorted(episodes_dir.glob("*/environment_config.json")) if episodes_dir.is_dir() else []
        if candidates:
            print(f"[env] 配置里的场景文件已不存在，改用 {candidates[0]}")
            return candidates[0]
    raise SystemExit(f"找不到场景配置 {path}，请更新 env.scene_config")


def resolve_scenes(config, seed_manifest_override: str | None = None) -> list[str]:
    scenes = [str(config.resolve_path(item)) for item in config.env.scenes]
    if config.env.scene_manifest:
        manifest_path = config.resolve_path(config.env.scene_manifest)
        payload = json.loads(manifest_path.read_text())
        entries = payload.get("episodes", payload)
        if isinstance(entries, dict):
            entries = entries.get("episodes", [])
        dataset_root = manifest_path.parent.parent if manifest_path.parent.name == "splits" else manifest_path.parent
        train_entries = [item for item in entries if item.get("split", "train") == "train"]
        if seed_manifest_override:
            override = json.loads(Path(seed_manifest_override).resolve().read_text())
            requested = override.get("episodes", override)
            if isinstance(requested, dict):
                requested = requested.get("episodes", [])
            by_seed = {int(item["seed"]): item for item in train_entries}
            try:
                train_entries = [by_seed[int(item["seed"])] for item in requested]
            except KeyError as error:
                raise SystemExit(f"manifest 中的 seed 不在训练 scene 集合内: {error.args[0]}") from error
        scenes.extend(str((dataset_root / item["npz"]).resolve()) for item in train_entries)
    missing = [path for path in scenes if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"场景 NPZ 不存在: {missing[0]}")
    if len(scenes) != len(set(scenes)):
        raise SystemExit("scene 列表中存在重复 NPZ")
    return scenes


def build_env(config, dummy: bool, seeds_file: str | None = None):
    if dummy:
        print("[warn] 使用 DummyEnv，仅验证代码路径")
        return DummyEnv(
            proprio_dim=config.data.proprio_dim,
            action_dim=config.data.action_dim,
            chunk_len=config.data.chunk_len,
            episodes=config.env.episodes,
        )
    seeds = list(config.env.seeds)
    configured_seed_file = (seeds_file or config.env.seed_file) if config.env.reset_mode != "scene" else ""
    if configured_seed_file:
        seed_path = Path(configured_seed_file).resolve() if seeds_file else config.resolve_path(configured_seed_file)
        payload = json.loads(seed_path.read_text())
        entries = payload.get("episodes", payload)
        if isinstance(entries, dict):
            entries = entries.get("episodes", [])
        seeds = [int(item["seed"]) for item in entries if item.get("split", "train") == "train"]
        if len(seeds) != len(set(seeds)):
            raise SystemExit(f"seed 文件里有重复 seed: {seed_path}")
        if not seeds:
            raise SystemExit(f"训练 seed 文件为空: {seed_path}")
        mode = "循环采集直到 Buffer 填满" if config.env.stop_when_buffer_full else f"运行 {config.env.episodes} 集（种子循环使用）"
        print(f"[env] 从训练划分加载 {len(seeds)} 个 seed，{mode}")
    scenes = resolve_scenes(
        config,
        seed_manifest_override=seeds_file if config.env.reset_mode == "scene" else None,
    )
    if not scenes and not seeds:
        raise SystemExit("既没有场景 npz 也没有 seeds，请先采集场景、配置种子，或加 --dummy")
    return RoboTwinEnv(
        scenes=scenes,
        task=config.data.language,
        seeds=seeds,
        scene_config=resolve_scene_config(config) if config.env.reset_mode != "scene" else None,
        reset_mode=config.env.reset_mode,
        max_chunks=config.env.max_chunks,
        max_episode_steps=config.env.max_episode_steps,
        success_grace_steps=config.env.success_grace_steps,
        success_reward=config.env.success_reward,
    )


def main() -> None:
    def request_pause(signum, frame):
        raise KeyboardInterrupt(f"收到信号 {signum}，执行保存并暂停")

    signal.signal(signal.SIGTERM, request_pause)
    args = parse_args()
    config = load_config(args.config)
    if args.episodes is not None:
        config.env.episodes = int(args.episodes)
        # 显式集数用于有限的测试运行；正式默认按 Buffer 填满停止。
        config.env.stop_when_buffer_full = False
    if args.num_envs is not None:
        config.env.num_envs = int(args.num_envs)
    if args.out:
        config.runtime.out_dir = args.out
    if args.seed is not None:
        config.runtime.seed = int(args.seed)
    if args.stats:
        config.data.stats_path = str(Path(args.stats).resolve())
    device = args.device or config.runtime.device
    torch.manual_seed(config.runtime.seed)
    config.validate()

    normalizer = build_normalizer(config)
    if args.fake_features:
        print("[warn] 使用随机特征，仅验证链路，不要让这种数据进任何结论")
        provider = RandomFeatureProvider(args.z_dim, config.data.chunk_len, config.data.action_dim)
        z_dim = args.z_dim
    else:
        assert args.token, "本地/远程特征都需要 --token"
        provider = build_provider(config, args.checkpoint, args.token, device)
        if hasattr(provider, "metadata"):
            z_dim = int(provider.metadata["z_dim"])
        elif hasattr(provider, "token"):
            z_dim = int(provider.token.z_dim)
        else:
            raise SystemExit("无法确定 z 的维度")

    if args.phase == "collect":
        config.env.stop_when_buffer_full = False
        if args.episodes is None:
            raise SystemExit("--phase collect 必须指定固定的 --episodes")
    env = [build_env(config, args.dummy, args.seeds_file) for _ in range(config.env.num_envs)]
    runner = OnlineRunner(
        config=config,
        env=env,
        provider=provider,
        normalizer=normalizer,
        z_dim=z_dim,
        device=device,
        seed=config.runtime.seed,
        learning_enabled=args.phase != "collect",
        resume_dir=args.resume_dir,
        target_episode_count=args.target_episode_count,
        actor_start_updates=args.actor_start_updates,
        require_actor=args.require_actor,
    )
    summary = runner.run()
    print(summary)


if __name__ == "__main__":
    main()

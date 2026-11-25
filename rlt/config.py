"""RLT 侧的全部可调参数，集中在一个 dataclass 树里。

这么做不是为了好看，而是为了让"哪些数字是 RLT 自己的、哪些是 SmolVLA 的"
一目了然。所有字段都有默认值，YAML 只需要写想改的那些。加载时会做严格
校验，写错字段名会立刻报错，不会被静默忽略。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TokenConfig:
    """阶段一：RL token 编解码器与 z 的取法。"""

    num_tokens: int = 1
    embed_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # image 只压图像 token（对齐论文），full 压图像+语言+状态的整段 prefix。
    z_source: str = "image"

    def validate(self) -> None:
        assert self.num_tokens >= 1
        assert self.embed_dim >= 1
        assert self.num_layers >= 1
        assert self.num_heads >= 1
        assert self.z_source in ("image", "full"), self.z_source


@dataclass
class DataConfig:
    """数据契约，必须和 SmolVLA 侧保持一致。"""

    stats_path: str = ""
    chunk_len: int = 8
    proprio_dim: int = 16
    action_dim: int = 16
    action_representation: str = "abs"  # abs 或 delta
    # Piper 的夹爪维度不做增量，其余关节都做相对当前位姿的增量。
    delta_exclude_dims: tuple[int, ...] = (6, 7, 14, 15)
    num_cameras: int = 3
    image_size: int = 256
    language: str = "stack the blocks"

    def validate(self) -> None:
        self.delta_exclude_dims = tuple(int(i) for i in self.delta_exclude_dims)
        assert self.chunk_len >= 1
        assert 1 <= self.action_dim <= 32, "SmolVLA 的 max_action_dim 是 32"
        assert self.proprio_dim >= 1
        assert self.action_representation in ("abs", "delta")
        assert len(set(self.delta_exclude_dims)) == len(self.delta_exclude_dims)
        assert all(0 <= i < self.action_dim for i in self.delta_exclude_dims)


@dataclass
class ActorCriticConfig:
    """演员和评论家的结构，数字对齐 RLT 原仓库。"""

    hidden_dim: int = 256
    proprio_hidden_dim: int = 64
    ref_hidden_dim: int = 256
    num_layers: int = 2
    fixed_std: float = 0.02
    reference_dropout: float = 0.5
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 128
    tau: float = 0.005
    actor_update_period: int = 2

    def validate(self) -> None:
        assert 0.0 <= self.reference_dropout < 1.0
        assert 0.0 < self.gamma <= 1.0
        assert self.fixed_std > 0.0
        assert self.actor_update_period >= 1


@dataclass
class ObjectiveConfig:
    """损失权重与训练节奏。warmup 与 online 分开写，对应 RLT 的两套配比。"""

    bc_weight: float = 5.0
    q_weight: float = 0.1
    delta_weight: float = 1.0
    warmup_bc_weight: float = 10.0
    warmup_q_weight: float = 0.1
    warmup_delta_weight: float = 1.0
    updates_per_sample: float = 5.0
    n_step_chunks: int = 5
    warmup_min_size: int = 200
    warmup_updates: int = 2000

    def validate(self) -> None:
        assert self.updates_per_sample > 0
        assert self.n_step_chunks >= 1
        assert self.warmup_min_size >= 1
        assert self.warmup_updates >= 0


@dataclass
class RuntimeConfig:
    """进程与文件落盘相关。"""

    seed: int = 0
    device: str = "cuda"
    out_dir: str = "rlt_runs/default"
    snapshot_every: int = 100
    checkpoint_every: int = 10000
    metrics_every: int = 1
    replay_capacity: int = 50000
    shutdown_flush_seconds: float = 15.0  # <0 表示等待全部更新预算完成
    # local 表示特征在进程内直接算，remote 表示连 serve.py。
    feature_source: str = "local"
    server_url: str = "ws://127.0.0.1:9200"
    service_host: str = "0.0.0.0"
    service_port: int = 9200

    def validate(self) -> None:
        assert self.feature_source in ("local", "remote")
        assert self.replay_capacity > 0


@dataclass
class EnvConfig:
    """RoboTwin 环境适配参数。"""

    env_factory: str = "rlt.env_robotwin:create_env"
    # reset_mode=scene 用预先采集的场景 npz 恢复；seed 用随机种子现场新建。
    reset_mode: str = "auto"
    scenes: tuple[str, ...] = ()
    scene_manifest: str = ""
    seeds: tuple[int, ...] = ()
    seed_file: str = ""
    scene_config: str = ""
    episodes: int = 5
    num_envs: int = 1
    stop_when_buffer_full: bool = False
    max_chunks: int = 40
    max_episode_steps: int = 376
    success_grace_steps: int = 0
    success_reward: float = 1.0
    control_hz: float = 25.0
    reward_fn: str = ""
    success_fn: str = ""
    deterministic_actor: bool = False

    def validate(self) -> None:
        self.scenes = tuple(str(item) for item in self.scenes)
        self.seeds = tuple(int(item) for item in self.seeds)
        assert self.reset_mode in ("auto", "scene", "seed"), self.reset_mode
        assert self.episodes >= 0
        assert self.num_envs >= 1
        assert self.max_chunks >= 1
        assert self.max_episode_steps >= 1
        assert self.success_grace_steps >= 0
        assert self.control_hz > 0
        if self.reset_mode == "scene":
            assert self.scenes or self.scene_manifest, "reset_mode=scene 需要 scenes 或 scene_manifest"
        if self.reset_mode == "seed":
            assert (self.seeds or self.seed_file) and self.scene_config, "reset_mode=seed 需要 seeds/seed_file 与 scene_config"


@dataclass
class RLTConfig:
    token: TokenConfig = field(default_factory=TokenConfig)
    data: DataConfig = field(default_factory=DataConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    source_path: str = ""

    def validate(self) -> "RLTConfig":
        for section in (
            self.token,
            self.data,
            self.actor_critic,
            self.objective,
            self.runtime,
            self.env,
        ):
            section.validate()
        return self

    def out_dir(self) -> Path:
        path = Path(self.runtime.out_dir)
        if not path.is_absolute():
            base = Path(self.source_path).resolve().parent if self.source_path else Path.cwd()
            path = base / path
        return path

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        base = Path(self.source_path).resolve().parent if self.source_path else Path.cwd()
        return base / path


_NESTED: dict[type, dict[str, type]] = {
    RLTConfig: {
        "token": TokenConfig,
        "data": DataConfig,
        "actor_critic": ActorCriticConfig,
        "objective": ObjectiveConfig,
        "runtime": RuntimeConfig,
        "env": EnvConfig,
    }
}


def _build(cls: type, payload: dict[str, Any], where: str) -> Any:
    allowed = {item.name for item in fields(cls)}
    unknown = set(payload) - allowed
    assert not unknown, f"{where} 里有未知字段 {sorted(unknown)}"
    kwargs: dict[str, Any] = {}
    nested: dict[str, tuple[type, Any]] = {}
    for item in fields(cls):
        if item.name not in payload:
            continue
        value = payload[item.name]
        child = _NESTED.get(cls, {}).get(item.name)
        if child is not None:
            assert isinstance(value, dict), f"{where}.{item.name} 必须是映射"
            nested[item.name] = (child, value)
        else:
            kwargs[item.name] = value
    instance = cls(**kwargs)
    for name, (child, value) in nested.items():
        setattr(instance, name, _build(child, value, f"{where}.{name}"))
    return instance


def load_config(path: str | Path | None = None) -> RLTConfig:
    """从 YAML 读配置。相对路径以 YAML 所在目录为基准。"""
    if path is None:
        return RLTConfig().validate()
    path = Path(path).resolve()
    assert path.is_file(), f"配置文件不存在 {path}"
    payload = yaml.safe_load(path.read_text()) or {}
    assert isinstance(payload, dict), "YAML 顶层必须是映射"
    config = _build(RLTConfig, payload, "root")
    config.source_path = str(path)
    return config.validate()

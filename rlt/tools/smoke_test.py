"""不依赖 GPU、仿真器和权重，把 RLT 的全部新代码路径过一遍。

覆盖八件事：RL token 的重建能收敛、归一化往返一致、演员评论家能更新、
经验池能存能取、序列化能往返、主循环能跑完两集、快照和状态文件能落盘、
配置解析严格。加了 `--with-vlm` 之后额外验证真实 SmolVLA 的 prefix 抽取，
这一步需要给定检查点，只在 CPU 上跑一条样本。

    python rlt/tools/smoke_test.py
    python rlt/tools/smoke_test.py --with-vlm --checkpoint runs/<平台>/<检查点>.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlt import bootstrap  # noqa: E402

bootstrap.bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rlt import wire  # noqa: E402
from rlt.actor_critic import RLTLearner  # noqa: E402
from rlt.config import RLTConfig, load_config  # noqa: E402
from rlt.env_robotwin import DummyEnv, RoboTwinEnv  # noqa: E402
from rlt.model import RLTokenModel  # noqa: E402
from rlt.norm import ActionNormalizer, NormStats  # noqa: E402
from rlt.online import OnlineRunner  # noqa: E402
from rlt.replay import ReplayBuffer  # noqa: E402
from rlt.run_online import resolve_scenes  # noqa: E402


class FakeProvider:
    """假装是 SmolVLA：返回随机 z 和随机参考动作，形状与真实接口一致。"""

    def __init__(self, z_dim: int, chunk_len: int, action_dim: int, seed: int = 0) -> None:
        self.z_dim = z_dim
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def features(self, observation: dict) -> dict:
        return {
            "z_rl": self.rng.normal(size=self.z_dim).astype(np.float32),
            "proprio": np.asarray(observation["observation.state"], dtype=np.float32),
            "ref_chunk": self.rng.normal(scale=0.05, size=(self.chunk_len, self.action_dim)).astype(np.float32),
        }

    def features_batch(self, observations: list[dict]) -> list[dict]:
        return [self.features(item) for item in observations]


def make_stats_file(path: Path, action_dim: int, proprio_dim: int) -> None:
    action_low = [-1.0 + 0.1 * i for i in range(action_dim)]
    action_high = [1.0 + 0.1 * i for i in range(action_dim)]
    payload = {
        "action": {"q01": action_low, "q99": action_high},
        "state": {"q01": [-3.0] * proprio_dim, "q99": [3.0] * proprio_dim},
    }
    path.write_text(json.dumps(payload))


def check_token(device: torch.device) -> None:
    model = RLTokenModel(dim=64, embed_dim=32, num_tokens=1, layers=2, heads=4, ratio=2.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    target = torch.randn(4, 12, 64, device=device)
    losses = []
    for _ in range(30):
        loss = model.reconstruction_loss(target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    z = model.encode(target)
    assert model.z_dim == 32
    rebuilt = model.decode(z, 12)
    assert z.shape == (4, 32), z.shape
    assert rebuilt.shape == target.shape, rebuilt.shape
    assert losses[-1] < losses[0], f"重建损失没有下降 {losses[0]} -> {losses[-1]}"
    assert torch.isfinite(z).all()
    print(f"[ok] rl token 重建 {losses[0]:.3f} -> {losses[-1]:.3f}，z 形状 {tuple(z.shape)}")


def check_norm() -> ActionNormalizer:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "stats.json"
        make_stats_file(path, action_dim=6, proprio_dim=6)
        stats = NormStats.load(path, action_dim=6, proprio_dim=6)
    for representation in ("abs", "delta"):
        normalizer = ActionNormalizer(
            stats, 6, 6, representation=representation, delta_exclude_dims=(3,)
        )
        chunk = torch.randn(2, 4, 6) * 0.2
        proprio = torch.randn(2, 6) * 0.3
        normalized = normalizer.normalize(chunk, proprio)
        restored = normalizer.denormalize(normalized, proprio)
        error = float((restored - chunk).abs().max())
        assert error < 1e-4, f"{representation} 往返误差过大 {error}"
        padded = torch.zeros(1, 3, 6)
        assert float(normalizer.normalize(padded, torch.zeros(1, 6)).abs().sum()) == 0.0
        assert float(normalizer.denormalize(torch.zeros(1, 3, 6), torch.zeros(1, 6)).abs().sum()) == 0.0
    print("[ok] 归一化往返一致，补零行保持为零")
    return normalizer


def check_learner(device: torch.device) -> None:
    config = RLTConfig()
    config.actor_critic.hidden_dim = 32
    config.actor_critic.proprio_hidden_dim = 16
    config.actor_critic.ref_hidden_dim = 32
    config.actor_critic.batch_size = 8
    config.actor_critic.learning_rate = 1e-3
    config.actor_critic.actor_update_period = 1
    config.validate()

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "stats.json"
        make_stats_file(path, 6, 6)
        stats = NormStats.load(path, 6, 6)
    normalizer = ActionNormalizer(stats, 6, 6)
    learner = RLTLearner(
        16, 6, 6, 4, config.actor_critic, config.objective,
        denormalize=normalizer.denormalize,
        delta_joint_mask=torch.tensor([True, True, True, False, True, True]),
        device=str(device),
    )
    batch = {
        "z": torch.randn(8, 16),
        "z_next": torch.randn(8, 16),
        "proprio": torch.randn(8, 6),
        "proprio_next": torch.randn(8, 6),
        "ref": torch.randn(8, 4, 6) * 0.1,
        "ref_next": torch.randn(8, 4, 6) * 0.1,
        "action": torch.randn(8, 4, 6) * 0.1,
        "reward": torch.randn(8, 4),
        "done": torch.zeros(8),
        "z_n": torch.randn(8, 16),
        "proprio_n": torch.randn(8, 6),
        "ref_n": torch.randn(8, 4, 6) * 0.1,
        "nstep_reward": torch.randn(8),
        "nstep_discount": torch.full((8,), 0.99**4),
        "nstep_done": torch.zeros(8),
        "human_mask": torch.zeros(8, 4),
    }
    metrics = learner.step(batch, warmup=True)
    assert np.isfinite(metrics.critic_loss) and np.isfinite(metrics.actor_loss), metrics
    assert learner.actor_version == 1, learner.actor_version
    mean = learner.actor.mean(
        torch.randn(2, 16), torch.randn(2, 6), torch.randn(2, 4, 6)
    )
    assert mean.shape == (2, 4, 6), mean.shape
    q1, q2 = learner.critic(torch.randn(2, 16), torch.randn(2, 6), torch.randn(2, 4, 6))
    assert q1.shape == (2,) and q2.shape == (2,), (q1.shape, q2.shape)
    print(f"[ok] 演员/评论家更新一次，critic_loss={metrics.critic_loss:.4f}")


def check_replay() -> None:
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=8, z_dim=16, proprio_dim=6, action_dim=6, chunk_len=4)
    for index in range(10):
        buffer.add(
            {
                "z": rng.normal(size=16).astype(np.float32),
                "z_next": rng.normal(size=16).astype(np.float32),
                "proprio": rng.normal(size=6).astype(np.float32),
                "proprio_next": rng.normal(size=6).astype(np.float32),
                "ref": rng.normal(size=(4, 6)).astype(np.float32),
                "ref_next": rng.normal(size=(4, 6)).astype(np.float32),
                "action": rng.normal(size=(4, 6)).astype(np.float32),
                "reward": rng.normal(size=4).astype(np.float32),
                "done": index % 4 == 0,
                "success": False,
                "source": 1,
                "source_chunk": np.ones(4, np.uint8),
                "episode": index // 4,
            }
        )
    assert len(buffer) == 8 and buffer.adds_total == 10
    batch = buffer.sample(5)
    assert batch["z"].shape == (5, 16) and batch["ref"].dtype == np.float16
    assert batch["human_mask"].shape == (5, 4)
    stats = buffer.stats()
    assert stats.adds_total == 10 and stats.max_episode == 2
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "replay.npz"
        buffer.save(path)
        other = ReplayBuffer(capacity=8, z_dim=16, proprio_dim=6, action_dim=6, chunk_len=4)
        other.load(path)
        assert len(other) == 8 and other.adds_total == 10
        expected_sample = buffer.sample(3)
        actual_sample = other.sample(3)
        assert np.array_equal(expected_sample["z"], actual_sample["z"]), "resume 未恢复经验池采样随机状态"
        assert other.max_episode == buffer.max_episode
        assert actual_sample["z"].shape == (3, 16)
    print("[ok] 经验池写入/采样/统计/落盘一致")


def check_n_step_replay() -> None:
    gamma = 0.5
    buffer = ReplayBuffer(8, 2, 1, 1, 2, n_step_chunks=3, gamma=gamma)
    rows = []
    for index in range(3):
        rows.append({
            "z": np.full(2, index, np.float32),
            "z_next": np.full(2, index + 1, np.float32),
            "proprio": np.array([index], np.float32),
            "proprio_next": np.array([index + 1], np.float32),
            "ref": np.full((2, 1), index, np.float32),
            "ref_next": np.full((2, 1), index + 1, np.float32),
            "action": np.full((2, 1), index, np.float32),
            "reward": np.array([1.0, 2.0], np.float32),
            "done": index == 2,
            "success": index == 2,
            "source": 0,
            "source_chunk": np.zeros(2, np.uint8),
            "episode": 7,
            "chunk_index": index,
        })
    buffer.add_episode(rows)
    stored = buffer._fields
    per_chunk_return = 1.0 + gamma * 2.0
    expected_start_return = per_chunk_return * (1 + gamma**2 + gamma**4)
    assert np.isclose(stored["nstep_reward"][0], expected_start_return)
    assert stored["nstep_done"][:3].tolist() == [1, 1, 1]
    assert stored["nstep_discount"][:3].tolist() == [0.0, 0.0, 0.0]
    assert stored["chunk_index"][:3].tolist() == [0, 1, 2]
    assert stored["z_n"][:3, 0].tolist() == [3.0, 3.0, 3.0]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "nstep.npz"
        buffer.save(path)
        restored = ReplayBuffer(8, 2, 1, 1, 2, n_step_chunks=3, gamma=gamma)
        restored.load(path)
        assert np.array_equal(restored._fields["nstep_reward"][:3], stored["nstep_reward"][:3])
    print("[ok] 5-step replay return stops at terminal, uses correct bootstrap state, and round-trips")


def check_success_reward_and_tail() -> None:
    class FakeSimulator:
        def check_success(self):
            return controller.executed >= 2

    class FakeController:
        def __init__(self):
            self.executed = 0
            self.reanchors = 0

        def reanchor(self):
            self.reanchors += 1

        def observe(self):
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            return image, image, image, np.zeros(16, dtype=np.float32)

        def execute(self, action):
            self.executed += 1
            return action

    controller = FakeController()
    env = RoboTwinEnv(
        seeds=[0],
        scene_config=Path(__file__).resolve(),
        reset_mode="seed",
        max_chunks=50,
        success_grace_steps=0,
        success_reward=1.0,
    )
    env.env = FakeSimulator()
    env.controller = controller
    actions = np.zeros((80, 16), dtype=np.float32)
    result = env.execute_chunk(actions, actions, source=0)
    rewards = np.asarray([step.reward for step in result.steps])
    assert result.done and result.success
    assert len(result.steps) == 2, len(result.steps)
    assert np.flatnonzero(rewards).tolist() == [1], rewards
    assert rewards.sum() == 1.0
    assert result.steps[-1].done and result.steps[-1].success
    assert controller.reanchors == 1
    print("[ok] 首次成功动作记录一次 +1，并立即终止 episode")

    class HorizonSimulator:
        def check_success(self):
            return False

    class HorizonController(FakeController):
        def __init__(self):
            super().__init__()
            self.finish_calls = 0

        def finish(self):
            self.finish_calls += 1
            raise AssertionError("RLT collection must not add a settling tail")

    controller = HorizonController()
    env = RoboTwinEnv(
        seeds=[0],
        scene_config=Path(__file__).resolve(),
        reset_mode="seed",
        max_chunks=50,
        max_episode_steps=3,
        success_grace_steps=0,
        success_reward=1.0,
    )
    env.env = HorizonSimulator()
    env.controller = controller
    result = env.execute_chunk(actions, actions, source=0)
    assert result.done and not result.success and len(result.steps) == 3
    assert controller.finish_calls == 0
    assert result.steps[-1].reward == 0.0 and result.steps[-1].done
    print("[ok] 步数上限直接按逐步检查结果终止，不追加稳定等待")


def check_scene_cleanup() -> None:
    import types
    import weakref
    from unittest.mock import patch

    events = []
    class Scene:
        def clear(self):
            events.append("scene.clear")
    class Task:
        def __init__(self):
            self.scene = Scene()
            self.callback = lambda: self  # Match camera/task reference cycles.
        def close_env(self):
            events.append("close_env")
    class Controller:
        def __init__(self, task):
            self.env = task
    adapter = RoboTwinEnv(seeds=[0], scene_config=Path(__file__).resolve(), reset_mode="seed")
    task = Task()
    task_ref = weakref.ref(task)
    adapter.env = task
    adapter.controller = Controller(task)
    del task
    fake_sapien = types.ModuleType("sapien")
    fake_sapien.__path__ = []
    fake_render = types.ModuleType("sapien.render")
    def clear_cache():
        assert task_ref() is None, "清理缓存前旧场景仍被Python引用"
        events.append("clear_cache")
    fake_render.clear_cache = clear_cache
    fake_sapien.render = fake_render
    with patch.dict(sys.modules, {"sapien": fake_sapien, "sapien.render": fake_render}):
        adapter.close()
        adapter.close()
    assert events == ["close_env", "scene.clear", "clear_cache"], events
    assert adapter.env is None and adapter.controller is None
    print("[ok] 重置清理场景实体、控制器和闭包循环后释放渲染缓存；重复close安全")


def check_wire() -> None:
    payload = {
        "z_rl": np.arange(6, dtype=np.float32).reshape(2, 3),
        "ref_chunk": np.zeros((2, 4, 6), dtype=np.float16),
        "meta": {"z_dim": 3, "task": "叠积木", "flag": True},
        "count": 2,
    }
    restored = wire.decode(wire.encode(payload))
    assert np.array_equal(restored["z_rl"], payload["z_rl"])
    assert restored["ref_chunk"].dtype == np.float16
    assert restored["meta"] == payload["meta"]
    assert restored["count"] == 2
    print("[ok] 序列化往返一致")


def online_config(folder: Path) -> RLTConfig:
    stats_path = folder / "stats.json"
    make_stats_file(stats_path, action_dim=6, proprio_dim=6)
    config = RLTConfig()
    config.data.stats_path = str(stats_path)
    config.data.chunk_len = 4
    config.data.action_dim = 6
    config.data.proprio_dim = 6
    config.data.delta_exclude_dims = (3,)
    config.data.language = "dummy"
    config.actor_critic.hidden_dim = 32
    config.actor_critic.proprio_hidden_dim = 16
    config.actor_critic.ref_hidden_dim = 32
    config.actor_critic.batch_size = 8
    config.actor_critic.learning_rate = 1e-3
    config.objective.warmup_min_size = 2
    config.objective.warmup_updates = 2
    config.objective.updates_per_sample = 5
    config.runtime.out_dir = str(folder / "run")
    config.runtime.replay_capacity = 64
    config.runtime.snapshot_every = 1
    config.runtime.checkpoint_every = 2
    config.runtime.shutdown_flush_seconds = 20.0
    config.env.episodes = 2
    config.env.max_chunks = 3
    return config.validate()


def check_online(device: torch.device) -> None:
    with tempfile.TemporaryDirectory() as folder_name:
        folder = Path(folder_name)
        config = online_config(folder)
        from rlt.online import build_normalizer

        normalizer = build_normalizer(config)
        provider = FakeProvider(z_dim=16, chunk_len=config.data.chunk_len, action_dim=6)
        env = DummyEnv(proprio_dim=6, action_dim=6, chunk_len=4, episodes=2)
        runner = OnlineRunner(
            config, env, provider, normalizer, z_dim=16, device=str(device), seed=0
        )
        summary = runner.run()
        assert summary["episodes"] == 2, summary
        assert len(runner.replay) > 0, "经验池没有数据"
        assert (folder / "run/actor_snapshot.pkl").is_file(), "快照没有写出"
        status = json.loads((folder / "run/metrics/learner_status.json").read_text())
        assert {"global_step", "ready_for_online", "replay_size"} <= set(status)
        assert (folder / "run/metrics/learner.jsonl").is_file()
        assert (folder / "run/metrics/rollout.jsonl").is_file()
        assert (folder / "run/online_summary.json").is_file()
        assert summary["learner_steps"] > 0, "学习线程一步都没跑"
        checkpoints = sorted((folder / "run/checkpoints").glob("learner_step_*.pt"))
        checkpoint_steps = [int(path.stem.rsplit("_", 1)[1]) for path in checkpoints]
        assert checkpoint_steps and checkpoint_steps == sorted(set(checkpoint_steps))
        assert all(step % config.runtime.checkpoint_every == 0 for step in checkpoint_steps)
        assert max(checkpoint_steps) == summary["learner_steps"]
        runner.actor_runtime.maybe_reload()
        assert runner.actor_runtime.version >= 0, "演员没有加载到快照"
        print(f"[ok] 在线主循环跑完 {summary['episodes']} 集，"
              f"经验 {summary['replay_size']} 条，训练 {summary['learner_steps']} 步，"
              f"演员版本 {runner.actor_runtime.version}")


def check_multi_env_online(device: torch.device) -> None:
    """不同长度 episode、循环种子、批量推理与精确容量停止，验证不跨轨迹连 TD。"""
    class MarkedEnv(DummyEnv):
        def __init__(self, worker):
            super().__init__(proprio_dim=6, action_dim=6, chunk_len=4)
            self.worker = worker
            self.seeds = [11, 22, 33]
            self.assigned = []
            self.closed = False

        def reset_episode(self, episode):
            self.assigned.append(episode)
            self.global_episode = episode
            self.chunk_index = 0
            self.state[:] = episode * 100
            return self._observe()

        def execute_chunk(self, actions, reference, source):
            result = super().execute_chunk(actions, reference, source)
            self.state[:] = self.global_episode * 100 + self.chunk_index
            result.observation = self._observe()
            result.done = self.chunk_index >= self.worker + 2
            result.success = result.done
            return result

        def close(self):
            self.closed = True

    class MarkedProvider:
        def __init__(self):
            self.batch_sizes = []

        def features_batch(self, observations):
            self.batch_sizes.append(len(observations))
            return [{"z_rl": np.full(16, obs["observation.state"][0], dtype=np.float32),
                     "proprio": obs["observation.state"],
                     "ref_chunk": np.full((4, 6), .05, dtype=np.float32)}
                    for obs in observations]

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        config = online_config(folder)
        config.env.num_envs = 4
        config.env.max_chunks = 6
        config.env.stop_when_buffer_full = True
        config.runtime.replay_capacity = 23
        config.runtime.shutdown_flush_seconds = -1
        envs = [MarkedEnv(i) for i in range(4)]
        provider = MarkedProvider()
        from rlt.online import build_normalizer
        runner = OnlineRunner(config, envs, provider, build_normalizer(config), 16, str(device))
        summary = runner.run()
        assert summary["stop_reason"] == "buffer_full", summary
        assert summary["replay_size"] == summary["adds_total"] == 23, summary
        assert summary["episodes_started"] > config.env.episodes
        assert summary["learner_steps"] == summary["learner_target_steps"] == 115
        assert provider.batch_sizes and set(provider.batch_sizes) == {4}
        assert all(env.closed for env in envs)
        assigned = sorted(ep for env in envs for ep in env.assigned)
        assert assigned == list(range(len(assigned))), assigned
        rows = runner.replay._fields
        for i in range(23):
            z = float(rows["z"][i, 0])
            assert int(z // 100) == int(rows["episode"][i]), "跨 episode 状态串线"
            if rows["done"][i]:
                assert float(rows["z_next"][i, 0]) == 0
            else:
                assert float(rows["z_next"][i, 0]) == z + 1, "TD 下一状态不是同一轨迹"
        logs = [json.loads(line) for line in (folder / "run/metrics/rollout.jsonl").read_text().splitlines()]
        assert all(row["seed"] == envs[0].seeds[row["episode"] % 3] for row in logs)
        assert any(row["transitions_added"] < row["transitions"] for row in logs), "没有覆盖最后一集截断"
        print("[ok] 四环境批量采集、循环种子、容量恰好23、115次更新补完，TD未跨episode")

        failure_folder = folder / "failure"
        failure_folder.mkdir()
        failure_config = online_config(failure_folder)
        failure_config.env.stop_when_buffer_full = True
        failure_config.runtime.replay_capacity = 8
        failure_config.runtime.shutdown_flush_seconds = -1
        failed_env = MarkedEnv(0)
        failed_runner = OnlineRunner(failure_config, failed_env, MarkedProvider(),
                                    build_normalizer(failure_config), 16, str(device))
        def fail_step(*args, **kwargs):
            raise RuntimeError("test learner failure")
        failed_runner.learner.step = fail_step
        try:
            failed_runner.run()
        except RuntimeError as error:
            assert "Learner" in str(error)
        else:
            raise AssertionError("训练线程异常后没有让主循环失败")
        assert failed_env.closed
        print("[ok] Learner异常传播到主线程，无限等待模式不会死等")


def check_pause_resume_buffer(device: torch.device) -> None:
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        config = online_config(folder)
        config.env.episodes = 2
        config.runtime.shutdown_flush_seconds = -1
        from rlt.online import build_normalizer
        normalizer = build_normalizer(config)
        first = OnlineRunner(
            config,
            DummyEnv(proprio_dim=6, action_dim=6, chunk_len=4, episodes=2),
            FakeProvider(16, 4, 6), normalizer, 16, str(device),
            learning_enabled=False,
        )
        first_summary = first.run()
        assert first_summary["learner_steps"] == 0
        assert (folder / "run/replay_buffer.npz").is_file()
        assert (folder / "run/checkpoints/learner_step_000000000.pt").is_file()
        saved_adds = first.replay.adds_total

        config.env.episodes = 1
        # Keep the rollout gate independent from loss-weight warmup: Stage 3
        # may use the trained actor before update 5000 while warmup loss remains active.
        config.objective.warmup_updates = 5000
        resumed = OnlineRunner(
            config,
            DummyEnv(proprio_dim=6, action_dim=6, chunk_len=4, episodes=1),
            FakeProvider(16, 4, 6, seed=1), normalizer, 16, str(device),
            learning_enabled=True,
            resume_dir=folder / "run",
            actor_start_updates=0,
        )
        resumed_summary = resumed.run()
        assert resumed.replay.adds_total > saved_adds
        assert resumed_summary["learner_steps"] == resumed_summary["learner_target_steps"]
        with np.load(folder / "run/replay_buffer.npz") as state:
            assert int(state["_adds"]) == resumed.replay.adds_total
        resumed.actor_runtime.maybe_reload()
        assert resumed.learner.step_count < config.objective.warmup_updates
        assert resumed.actor_runtime.version >= 1
        assert not resumed.actor_runtime.ready(
            len(resumed.replay), resumed.learner.step_count, config
        )
        assert not resumed.actor_runtime.ready(
            config.objective.warmup_min_size - 1,
            resumed.learner.step_count,
            config,
            minimum_updates=0,
        )
        assert resumed.actor_runtime.ready(
            len(resumed.replay), resumed.learner.step_count, config,
            minimum_updates=resumed.actor_start_updates,
        )
        # Stage 3 must use the trained actor from its first chunk even when
        # replay is below the ordinary warmup threshold and updates are < 5000.
        config.objective.warmup_min_size = 100
        config.env.episodes = 1
        stage3 = OnlineRunner(
            config,
            DummyEnv(proprio_dim=6, action_dim=6, chunk_len=4, episodes=1),
            FakeProvider(16, 4, 6, seed=2), normalizer, 16, str(device),
            learning_enabled=True,
            resume_dir=folder / "run",
            actor_start_updates=0,
            require_actor=True,
        )
        stage3_summary = stage3.run()
        assert stage3_summary["episodes"] == 1
        latest_episode = stage3.replay.max_episode
        episode_mask = stage3.replay._fields["episode"][: len(stage3.replay)] == latest_episode
        assert episode_mask.any()
        assert np.all(stage3.replay._fields["source"][: len(stage3.replay)][episode_mask] == 1)
        print(f"[ok] 暂停后恢复保留 {saved_adds} 条 buffer，继续到 {resumed.replay.adds_total} 条并补齐更新预算")



def check_config() -> None:
    config = load_config(None)
    assert config.data.chunk_len == 8
    assert config.runtime.checkpoint_every == 10000
    project_config = load_config(Path(__file__).resolve().parents[1] / "configs/robotwin_piper.yaml")
    assert project_config.env.episodes == 80
    assert project_config.env.reset_mode == "scene"
    assert project_config.env.scene_manifest
    scenes = resolve_scenes(project_config)
    assert len(scenes) == 80 and all(Path(scene).is_file() for scene in scenes)
    stage1_manifest = Path(__file__).resolve().parents[1] / "../data/robotwin_success_qpos16_formal100/splits/train.json"
    stage1_scenes = resolve_scenes(project_config, str(stage1_manifest.resolve()))
    assert len(stage1_scenes) == 80 and stage1_scenes == scenes
    assert project_config.env.max_chunks == 47
    assert project_config.env.max_episode_steps == 376
    assert project_config.runtime.checkpoint_every == 3305
    assert project_config.env.num_envs == 1
    assert project_config.env.stop_when_buffer_full
    assert project_config.actor_critic.batch_size == 256
    assert project_config.actor_critic.learning_rate == 1e-4
    assert project_config.objective.n_step_chunks == 5
    assert project_config.objective.updates_per_sample == 5.0
    assert project_config.env.success_grace_steps == 0
    assert len(stage1_scenes) == 80
    assert project_config.objective.warmup_updates == 5000
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "bad.yaml"
        path.write_text("data:\n  chunk_len: 4\n  unknown_field: 1\n")
        try:
            load_config(path)
        except AssertionError as error:
            assert "unknown_field" in str(error)
        else:
            raise AssertionError("未知字段没有被拦住")
    print("[ok] 配置解析与未知字段校验")


class StubPolicy:
    """按 SmolVLA 的接口约定造一个假策略，用来验证胶水层的形状和顺序。

    真正的权重加载需要 lerobot 全套依赖，这里只模仿 `embed_prefix` 的拼接
    顺序（图像、语言、状态）、掩码形状和前向的返回值结构。
    """

    class _Config:
        use_cache = True
        chunk_size = 4
        max_action_dim = 6
        num_steps = 2

    class _Expert:
        def __init__(self, dim: int) -> None:
            self.dim = dim

        def forward(self, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, fill_kv_cache):
            prefix, suffix = inputs_embeds
            assert (prefix is None) != (suffix is None)
            if prefix is not None:
                return [prefix * 1.0, None], {"cached": True}
            return [None, suffix * 0.5], None

        def denoise_step(self, pad_masks, past_key_values, x_t, timestep):
            return torch.full_like(x_t, 0.1)

        def sample_noise(self, shape, device):
            return torch.ones(shape, device=device) * 0.5

    def __init__(self, dim: int = 32, chunk_size: int = 4, action_dim: int = 6) -> None:
        self.dim = dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.config = self._Config()
        self.config.chunk_size = chunk_size
        self.config.max_action_dim = action_dim
        self.vlm_with_expert = self._Expert(dim)
        self.num_images = 3
        self.tokens_per_image = 16

    def sample_noise(self, shape, device):
        return self.vlm_with_expert.sample_noise(shape, device)

    def denoise_step(self, *args):
        return self.vlm_with_expert.denoise_step(*args)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        batch = state.shape[0]
        assert len(images) == self.num_images
        assert state.shape[1] == 32
        image_tokens = self.tokens_per_image * self.num_images
        language_tokens = lang_tokens.shape[1]
        length = image_tokens + language_tokens + 1
        prefix = torch.arange(length, dtype=torch.float32)[None, :, None].expand(batch, length, self.dim)
        pad = torch.ones(batch, length, dtype=torch.bool)
        att = torch.zeros(batch, length, dtype=torch.long)
        att[:, image_tokens + language_tokens :] = 1
        return prefix.contiguous(), pad, att

    # ---- 策略层接口 ----
    model = None

    def _prepare_batch(self, batch):
        return batch

    def prepare_images(self, batch):
        keys = [key for key in batch if key.startswith("observation.images.")]
        batch_size = len(batch["task"])
        images = [torch.zeros(batch_size, self.tokens_per_image, self.dim) for _ in keys]
        masks = [torch.ones(batch_size, self.tokens_per_image, dtype=torch.bool) for _ in keys]
        return images, masks

    def prepare_state(self, batch):
        state = batch["observation.state"]
        return torch.nn.functional.pad(state, (0, 32 - state.shape[1]))

    def prepare_language(self, batch):
        size = len(batch["task"])
        tokens = torch.zeros(size, 4, dtype=torch.long)
        masks = torch.ones(size, 4, dtype=torch.bool)
        return tokens, masks

    def unnormalize_outputs(self, outputs):
        return outputs


def check_stub_vlm() -> None:
    from rlt.model import infer_z_and_reference, prefix_and_actions
    from rlt.serve import LocalFeatureProvider, observations_to_batch

    policy = StubPolicy(dim=32, chunk_size=4, action_dim=6)
    policy.model = policy
    observation = {
        "observation.state": np.zeros(16, dtype=np.float32),
        "task": "stack",
    }
    for index in (1, 2, 3):
        observation[f"observation.images.camera{index}"] = np.zeros((256, 256, 3), dtype=np.uint8)
    batch = observations_to_batch([observation], "stack", torch.device("cpu"))
    result = prefix_and_actions(policy, batch)
    assert result.image_tokens == 48, result.image_tokens
    assert result.prefix.shape == (1, 53, 32), result.prefix.shape
    assert result.actions.shape == (1, 4, 6), result.actions.shape
    token = RLTokenModel(32, embed_dim=32, num_tokens=1, layers=2, heads=4, ratio=2.0)
    z, actions = infer_z_and_reference(policy, token, batch, action_dim=6, z_source="image")
    assert z.shape == (1, 32), z.shape
    assert actions.shape == (1, 4, 6), actions.shape

    from types import SimpleNamespace

    noise_policy = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(
            config=SimpleNamespace(chunk_size=4, max_action_dim=6)
        ))
    )

    provider = object.__new__(LocalFeatureProvider)
    provider.policy = noise_policy
    provider.device = torch.device("cpu")
    provider._noise_generators = {}
    first = provider._next_noise(0)
    second = provider._next_noise(0)
    other_episode_first = provider._next_noise(1)
    other_episode_second = provider._next_noise(1)
    assert not torch.equal(first, second), "同一 episode 的连续 chunk 应消耗不同噪声"
    assert torch.equal(first, other_episode_first), "每个 episode 都应从 seed 300 的首个噪声块开始"
    assert torch.equal(second, other_episode_second), "每个 episode 内的噪声序列应一致推进"
    expected_generator = torch.Generator(device="cpu").manual_seed(300)
    expected_first = torch.normal(0.0, 1.0, (1, 4, 6), generator=expected_generator)
    expected_second = torch.normal(0.0, 1.0, (1, 4, 6), generator=expected_generator)
    assert torch.equal(first, expected_first) and torch.equal(second, expected_second)
    print("[ok] 胶水层：prefix 顺序、图像 token 切分、z 与参考动作形状正确")


def check_service() -> None:
    """起一个真实的服务端线程，用远程客户端连上去，验证协议闭环。"""
    import socket
    import threading
    import time

    from rlt.serve import RemoteFeatureProvider, serve_forever

    config = RLTConfig()
    config.data.chunk_len = 4
    config.data.action_dim = 6
    config.data.proprio_dim = 16
    config.data.delta_exclude_dims = (3,)
    config.data.language = "stack"
    config.validate()

    policy = StubPolicy(dim=32, chunk_size=4, action_dim=6)
    policy.model = policy
    token = RLTokenModel(32, embed_dim=32, num_tokens=1, layers=2, heads=4, ratio=2.0).eval()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    thread = threading.Thread(
        target=serve_forever,
        args=(config, policy, token, "127.0.0.1", port, "cpu"),
        daemon=True,
    )
    thread.start()

    observation = {"observation.state": np.zeros(16, dtype=np.float32), "task": "stack"}
    for index in (1, 2, 3):
        observation[f"observation.images.camera{index}"] = np.zeros((256, 256, 3), dtype=np.uint8)

    provider = None
    for _ in range(100):
        try:
            provider = RemoteFeatureProvider(f"ws://127.0.0.1:{port}", chunk_len=4)
            break
        except OSError:
            time.sleep(0.05)
    assert provider is not None, "服务端 5 秒内没有起来"
    assert provider.metadata["z_dim"] == 32, provider.metadata
    result = provider.features(observation)
    assert result["z_rl"].shape == (32,), result["z_rl"].shape
    assert result["ref_chunk"].shape == (4, 6), result["ref_chunk"].shape
    assert result["proprio"].shape == (16,)
    batch = provider.features_batch([observation, observation])
    assert len(batch) == 2 and batch[1]["z_rl"].shape == (32,)
    provider.close()
    print("[ok] 服务端与远程客户端协议闭环，单条与批量请求都正确")


def check_vlm(checkpoint: str) -> None:
    from rlt.model import infer_z_and_reference, load_policy, prefix_dim

    device = torch.device("cpu")
    policy = load_policy(checkpoint, device="cpu", flow_steps=2)
    dim = prefix_dim(policy)
    rng = np.random.default_rng(0)
    observation = {
        "observation.state": rng.normal(size=16).astype(np.float32),
        "task": "stack the blocks",
    }
    for index in (1, 2, 3):
        observation[f"observation.images.camera{index}"] = (
            rng.random((256, 256, 3)) * 255
        ).astype(np.uint8)
    from rlt.serve import observations_to_batch
    from rlt.model import prefix_and_actions, select_prefix

    batch = observations_to_batch([observation], "stack the blocks", device)
    result = prefix_and_actions(policy, batch)
    assert result.prefix.shape[0] == 1 and result.prefix.shape[2] == dim, result.prefix.shape
    assert result.image_tokens == 48, f"三路 256 图像应该有 48 个 token，实际 {result.image_tokens}"
    # sample_actions 输出的是 32 维填充空间，截到 16 维是在策略层的
    # _get_action_chunk 里做的，这里保持和上游一致。
    assert result.actions.shape == (1, 16, 32), result.actions.shape
    selected, mask = select_prefix(result.prefix, result.mask, result.image_tokens, "image")
    token = RLTokenModel(dim, embed_dim=32, num_tokens=1, layers=2, heads=8).to(device)
    z = token.encode(selected.float(), mask)
    assert z.shape == (1, dim), z.shape
    loss = token.reconstruction_loss(selected.float(), mask)
    assert torch.isfinite(loss)
    z_full, actions = infer_z_and_reference(policy, token, batch, action_dim=16, z_source="image")
    assert z_full.shape == (1, dim), z_full.shape
    assert actions.shape == (1, 16, 16), actions.shape
    print(
        f"[ok] 真实 SmolVLA prefix {tuple(result.prefix.shape)}，z {tuple(z.shape)}，"
        f"参考动作 {tuple(actions.shape)}，重建损失 {float(loss.detach()):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RLT 自检")
    parser.add_argument("--with-vlm", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    device = torch.device("cpu")

    check_config()
    check_token(device)
    check_norm()
    check_learner(device)
    check_replay()
    check_n_step_replay()
    check_success_reward_and_tail()
    check_scene_cleanup()
    check_wire()
    check_stub_vlm()
    check_service()
    check_online(device)
    check_multi_env_online(device)
    check_pause_resume_buffer(device)
    if args.with_vlm:
        assert args.checkpoint, "--with-vlm 需要 --checkpoint"
        try:
            check_vlm(args.checkpoint)
        except ModuleNotFoundError as error:
            print(f"[skip] 真实权重检查未执行：{error}")
            raise SystemExit(2)
    print("全部自检通过")


if __name__ == "__main__":
    main()

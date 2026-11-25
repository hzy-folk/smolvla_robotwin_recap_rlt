"""在线主循环：采集、改写、执行、入池、训练、推快照。

一个进程里跑两个线程。主线程按控制节奏和仿真交互，学习线程从经验池采样并
更新演员和评论家。两者只通过经验池和一个快照文件握手，和 RLT 原仓库里
"两台机器、四个进程"的边界一一对应，只是把 HTTP 换成了进程内对象。

默认只在关键阶段使用演员：经验池够大、学习器训够步数之前，演员一律返回
参考动作，这一点由 `ActorRuntime.refine` 里的两个判断保证。
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rlt.actor_critic import ChunkActor, RLTLearner
from rlt.config import RLTConfig
from rlt.env_robotwin import ChunkResult
from rlt.norm import ActionNormalizer, NormStats
from rlt.replay import SOURCE_BASE, SOURCE_RL, ReplayBuffer
from rlt.serve import LocalFeatureProvider, RemoteFeatureProvider


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


@dataclass
class ChunkFeature:
    z: np.ndarray
    proprio: np.ndarray
    reference: np.ndarray


class ActorRuntime:
    """演员的在线副本，负责快照热加载和 warmup 阶段的降级。"""

    def __init__(
        self,
        z_dim: int,
        config: RLTConfig,
        snapshot_path: Path,
        device: str,
        deterministic: bool = False,
    ) -> None:
        self.actor = ChunkActor(
            z_dim,
            config.data.proprio_dim,
            config.data.action_dim,
            config.data.chunk_len,
            config.actor_critic.hidden_dim,
            config.actor_critic.proprio_hidden_dim,
            config.actor_critic.ref_hidden_dim,
            config.actor_critic.num_layers,
            config.actor_critic.fixed_std,
        ).to(device).eval()
        self.snapshot_path = snapshot_path
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.version = -1
        self._mtime = 0.0

    def maybe_reload(self) -> None:
        if not self.snapshot_path.is_file():
            return
        mtime = self.snapshot_path.stat().st_mtime
        if mtime <= self._mtime:
            return
        try:
            payload = torch.load(self.snapshot_path, map_location="cpu", weights_only=False)
            version = int(payload.get("version", 0))
            if version <= self.version:
                self._mtime = mtime
                return
            self.actor.load_state_dict(payload["actor"])
            self.actor.eval()
            self.version = version
            self._mtime = mtime
            with torch.no_grad():
                self.actor.mean(
                    torch.zeros(1, payload["z_dim"], device=self.device),
                    torch.zeros(1, payload["proprio_dim"], device=self.device),
                    torch.zeros(1, payload["chunk_len"], payload["action_dim"], device=self.device),
                )
        except (EOFError, RuntimeError, KeyError) as error:
            print(f"[actor] 快照加载失败，继续用旧参数：{error}")

    def ready(
        self,
        replay_size: int,
        learner_steps: int,
        config: RLTConfig,
        minimum_updates: int | None = None,
    ) -> bool:
        required_updates = (
            config.objective.warmup_updates if minimum_updates is None else minimum_updates
        )
        return (
            replay_size >= config.objective.warmup_min_size
            and learner_steps >= required_updates
            and self.version >= 1
        )

    @torch.no_grad()
    def refine(self, feature: ChunkFeature, normalizer: ActionNormalizer, use_actor: bool) -> tuple[np.ndarray, int]:
        if not use_actor:
            return feature.reference, SOURCE_BASE
        reference = torch.as_tensor(feature.reference)[None].to(self.device)
        proprio = torch.as_tensor(feature.proprio)[None].to(self.device)
        normalized_reference = normalizer.normalize(reference, proprio)
        z = torch.as_tensor(feature.z)[None].to(self.device)
        action = self.actor.sample(z, proprio, normalized_reference, deterministic=self.deterministic)
        refined = normalizer.denormalize(action, proprio)[0].cpu().numpy()
        return refined.astype(np.float32), SOURCE_RL


class LearnerThread(threading.Thread):
    """按"数据进来多少，补多少更新"的节奏训练。"""

    def __init__(
        self,
        learner: RLTLearner,
        replay: ReplayBuffer,
        normalizer: ActionNormalizer,
        config: RLTConfig,
        snapshot_path: Path,
        metrics_path: Path,
        status_path: Path,
        checkpoint_path: Path,
        z_dim: int,
        device: str,
        learning_enabled: bool = True,
        actor_start_updates: int | None = None,
        checkpoint_every: int | None = None,
        checkpoint_name: str | None = None,
        checkpoint_prefix: str | None = None,
    ) -> None:
        super().__init__(daemon=True, name="rlt-learner")
        self.learner = learner
        self.replay = replay
        self.normalizer = normalizer
        self.config = config
        self.snapshot_path = snapshot_path
        self.metrics_path = metrics_path
        self.status_path = status_path
        self.checkpoint_path = checkpoint_path
        self.z_dim = z_dim
        self.device = device
        self.learning_enabled = learning_enabled
        self.checkpoint_every = (
            config.runtime.checkpoint_every if checkpoint_every is None else checkpoint_every
        )
        self.checkpoint_name = checkpoint_name
        self.checkpoint_prefix = checkpoint_prefix
        self.actor_start_updates = (
            config.objective.warmup_updates
            if actor_start_updates is None else actor_start_updates
        )
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.warmup_ready_adds: int | None = None
        self.ready_for_online = False
        self._last_status = ""
        self._metrics = metrics_path.open("a", encoding="utf-8")
        self._started_at = time.perf_counter()

    # ---- 采样前的归一化 ----
    def prepare_batch(self, raw: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        proprio = torch.as_tensor(raw["proprio"], dtype=torch.float32)
        proprio_next = torch.as_tensor(raw["proprio_next"], dtype=torch.float32)
        ref = self.normalizer.normalize(torch.as_tensor(raw["ref"], dtype=torch.float32), proprio)
        ref_next = self.normalizer.normalize(torch.as_tensor(raw["ref_next"], dtype=torch.float32), proprio_next)
        proprio_n = torch.as_tensor(raw["proprio_n"], dtype=torch.float32)
        ref_n = self.normalizer.normalize(torch.as_tensor(raw["ref_n"], dtype=torch.float32), proprio_n)
        action = self.normalizer.normalize(torch.as_tensor(raw["action"], dtype=torch.float32), proprio)
        return {
            "z": torch.as_tensor(raw["z"], dtype=torch.float32),
            "z_next": torch.as_tensor(raw["z_next"], dtype=torch.float32),
            "proprio": proprio,
            "proprio_next": proprio_next,
            "ref": ref,
            "ref_next": ref_next,
            "z_n": torch.as_tensor(raw["z_n"], dtype=torch.float32),
            "proprio_n": proprio_n,
            "ref_n": ref_n,
            "action": action,
            "reward": torch.as_tensor(raw["reward"], dtype=torch.float32),
            "done": torch.as_tensor(raw["done"], dtype=torch.float32),
            "nstep_reward": torch.as_tensor(raw["nstep_reward"], dtype=torch.float32),
            "nstep_discount": torch.as_tensor(raw["nstep_discount"], dtype=torch.float32),
            "nstep_done": torch.as_tensor(raw["nstep_done"], dtype=torch.float32),
            "human_mask": torch.as_tensor(raw["human_mask"], dtype=torch.float32),
        }

    # ---- 训练预算 ----
    def budget(self) -> int:
        if not self.learning_enabled:
            return 0
        adds = self.replay.adds_total
        if self.warmup_ready_adds is None and self.replay.__len__() >= self.config.objective.warmup_min_size:
            self.warmup_ready_adds = adds
        if self.warmup_ready_adds is None:
            return 0
        desired = adds * self.config.objective.updates_per_sample
        budget = int(desired - self.learner.step_count)
        self.ready_for_online = (
            self.replay.__len__() >= self.config.objective.warmup_min_size
            and self.learner.step_count >= self.actor_start_updates
            and self.learner.actor_version >= 1
        )
        return max(0, budget)

    def run(self) -> None:
        try:
            self._run_updates()
        except BaseException as error:
            self.error = error
            self.stop_event.set()

    def _run_updates(self) -> None:
        while not self.stop_event.is_set():
            budget = self.budget()
            if budget <= 0:
                self.publish_status()
                time.sleep(0.02)
                continue
            for _ in range(min(budget, 32)):
                update_started = time.perf_counter()
                warmup = self.learner.step_count < self.config.objective.warmup_updates
                raw = self.replay.sample(self.config.actor_critic.batch_size)
                batch = self.prepare_batch(raw)
                metrics = self.learner.step(batch, warmup=warmup)
                self.record(metrics, warmup, time.perf_counter() - update_started)
                if self.learner.step_count % self.config.runtime.snapshot_every == 0:
                    self.export_snapshot()
                if self.checkpoint_every > 0 and self.learner.step_count % self.checkpoint_every == 0:
                    self.save_checkpoint()
            self.publish_status()

    def record(self, metrics, warmup: bool, update_seconds: float = 0.0) -> None:
        row = {
            "step": self.learner.step_count,
            "elapsed_seconds": time.perf_counter() - self._started_at,
            "update_seconds": update_seconds,
            "actor_version": self.learner.actor_version,
            "critic_loss": metrics.critic_loss,
            "actor_loss": metrics.actor_loss,
            "q_mean": metrics.q_mean,
            "bc_penalty": metrics.bc_penalty,
            "delta_penalty": metrics.delta_penalty,
            "human_ratio": metrics.human_ratio,
            "warmup": bool(warmup),
            "replay_size": len(self.replay),
            "adds_total": self.replay.adds_total,
        }
        self._metrics.write(json.dumps(row) + "\n")
        self._metrics.flush()

    def publish_status(self) -> None:
        stats = self.replay.stats()
        payload = {
            "global_step": self.learner.step_count,
            "actor_version": self.learner.actor_version,
            "replay_size": stats.size,
            "adds_total": stats.adds_total,
            "replay_capacity": stats.capacity,
            "num_envs": self.config.env.num_envs,
            "learner_target_steps": int(stats.adds_total * self.config.objective.updates_per_sample),
            "warmup_min_size": self.config.objective.warmup_min_size,
            "warmup_required_updates": self.config.objective.warmup_updates,
            "actor_start_updates": self.actor_start_updates,
            "warmup_complete": self.learner.step_count >= self.config.objective.warmup_updates,
            "ready_for_online": bool(self.ready_for_online),
        }
        text = json.dumps(payload, indent=2)
        if text != self._last_status:
            _atomic_write(self.status_path, text)
            self._last_status = text

    def export_snapshot(self) -> None:
        payload = {
            "version": int(self.learner.actor_version),
            "z_dim": int(self.z_dim),
            "proprio_dim": int(self.config.data.proprio_dim),
            "chunk_len": int(self.config.data.chunk_len),
            "action_dim": int(self.config.data.action_dim),
            "actor": self.learner.actor.state_dict(),
        }
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.snapshot_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(self.snapshot_path)

    def save_checkpoint(self) -> None:
        payload = self.learner.state_dict()
        payload["warmup_ready_adds"] = self.warmup_ready_adds
        payload["ready_for_online"] = bool(self.ready_for_online)
        payload["replay_size"] = len(self.replay)
        payload["replay_adds_total"] = self.replay.adds_total
        replay_path = self.checkpoint_path.parent / "replay_buffer.npz"
        self.replay.save(replay_path)
        payload["replay_path"] = str(replay_path)
        checkpoint_dir = self.checkpoint_path.parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = self.checkpoint_name
        if filename is None:
            prefix = self.checkpoint_prefix or "learner"
            filename = f"{prefix}_step_{self.learner.step_count:09d}.pt"
        destination = checkpoint_dir / filename
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)

    def shutdown(self) -> None:
        self.stop_event.set()
        self.join()
        self.export_snapshot()
        self.save_checkpoint()
        self.publish_status()
        self._metrics.close()


def _pad_chunk(values: list[np.ndarray], length: int, fallback: np.ndarray | None = None) -> np.ndarray:
    """把不满一块的序列补到固定长度，补位用最后一步或指定的填充值。"""
    if not values:
        assert fallback is not None, "空块必须给填充值"
        base = np.asarray(fallback)
        return np.repeat(base[None], length, axis=0)
    stacked = np.stack(values)
    if len(values) >= length:
        return stacked[:length]
    filler = stacked[-1] if fallback is None else np.asarray(fallback)
    padding = np.repeat(filler[None], length - len(values), axis=0)
    return np.concatenate([stacked, padding], axis=0)


class OnlineRunner:
    def __init__(
        self,
        config: RLTConfig,
        env,
        provider,
        normalizer: ActionNormalizer,
        z_dim: int,
        device: str,
        seed: int = 0,
        learning_enabled: bool = True,
        resume_dir: str | Path | None = None,
        target_episode_count: int | None = None,
        actor_start_updates: int | None = None,
        require_actor: bool = False,
    ) -> None:
        self.config = config
        self.envs = list(env) if isinstance(env, (list, tuple)) else [env]
        assert self.envs, "至少需要一个环境"
        assert len(self.envs) == config.env.num_envs, "环境实例数与 env.num_envs 不一致"
        assert len({id(item) for item in self.envs}) == len(self.envs), "每个采集槽必须使用独立环境"
        self.env = self.envs[0]
        self.provider = provider
        self.normalizer = normalizer
        self.device = device
        self.z_dim = z_dim
        self.rng = np.random.default_rng(seed)
        self.target_episode_count = target_episode_count
        self.actor_start_updates = actor_start_updates
        self.require_actor = require_actor

        out = config.out_dir()
        self.out = out
        (out / "metrics").mkdir(parents=True, exist_ok=True)
        self.rollout_metrics = out / "metrics/rollout.jsonl"
        self.status_path = out / "metrics/learner_status.json"
        self.snapshot_path = out / "actor_snapshot.pkl"
        self.checkpoint_path = out / "learner.pt"

        self.replay = ReplayBuffer(
            capacity=config.runtime.replay_capacity,
            z_dim=z_dim,
            proprio_dim=config.data.proprio_dim,
            action_dim=config.data.action_dim,
            chunk_len=config.data.chunk_len,
            seed=seed,
            n_step_chunks=config.objective.n_step_chunks,
            gamma=config.actor_critic.gamma,
        )

        delta_mask = torch.ones(config.data.action_dim, dtype=torch.bool)
        for index in config.data.delta_exclude_dims:
            delta_mask[index] = False
        learner = RLTLearner(
            z_dim=z_dim,
            proprio_dim=config.data.proprio_dim,
            action_dim=config.data.action_dim,
            chunk_len=config.data.chunk_len,
            config=config.actor_critic,
            objective=config.objective,
            denormalize=normalizer.denormalize,
            delta_joint_mask=delta_mask,
            device=device,
        )
        self.learner = learner
        resume_payload = None
        if resume_dir is not None:
            resume_path = Path(resume_dir)
            replay_path = resume_path / "replay_buffer.npz"
            checkpoint_paths = sorted((resume_path / "checkpoints").glob("learner_step_*.pt"))
            if not replay_path.is_file() or not checkpoint_paths:
                raise FileNotFoundError(f"恢复目录缺少 replay_buffer.npz 或 learner checkpoint: {resume_path}")
            self.replay.load(replay_path)
            resume_payload = torch.load(checkpoint_paths[-1], map_location=device, weights_only=False)
            self.learner.load_state_dict(resume_payload)
        self.learner_thread = LearnerThread(
            learner,
            self.replay,
            normalizer,
            config,
            self.snapshot_path,
            out / "metrics/learner.jsonl",
            self.status_path,
            self.checkpoint_path,
            z_dim,
            device,
            learning_enabled=learning_enabled,
            actor_start_updates=actor_start_updates,
        )
        if resume_payload is not None:
            self.learner_thread.warmup_ready_adds = resume_payload.get("warmup_ready_adds")
            self.learner_thread.ready_for_online = bool(resume_payload.get("ready_for_online", False))
        self.actor_runtime = ActorRuntime(
            z_dim, config, self.snapshot_path, device, deterministic=config.env.deterministic_actor
        )

    # ---- 采集 ----
    def _features(self, observation: dict) -> ChunkFeature:
        return self._feature_from_payload(self.provider.features(observation))

    def _features_batch(self, observations: list[dict]) -> list[ChunkFeature]:
        if hasattr(self.provider, "features_batch"):
            payloads = self.provider.features_batch(observations)
        else:
            if len(observations) > 1:
                raise RuntimeError("多环境采集要求特征提供者支持 features_batch")
            payloads = [self.provider.features(observations[0])]
        if len(payloads) != len(observations):
            raise RuntimeError("批量特征数与环境数不一致")
        return [self._feature_from_payload(payload) for payload in payloads]

    def _feature_from_payload(self, payload: dict) -> ChunkFeature:
        reference = np.asarray(payload["ref_chunk"], dtype=np.float32)
        chunk = self.config.data.chunk_len
        if reference.shape[0] < chunk:
            raise RuntimeError(f"参考动作只有 {reference.shape[0]} 步，不足 {chunk} 步")
        return ChunkFeature(
            z=np.asarray(payload["z_rl"], dtype=np.float32),
            proprio=np.asarray(payload["proprio"], dtype=np.float32),
            reference=reference[:chunk].copy(),
        )

    def _rows_from_episode(self, chunks: list[dict], episode: int) -> list[dict]:
        rows = []
        for index, chunk in enumerate(chunks):
            following = chunks[index + 1] if index + 1 < len(chunks) else None
            terminal = following is None
            rows.append(
                {
                    "z": chunk["z"],
                    "proprio": chunk["proprio"],
                    "ref": chunk["reference"],
                    "action": chunk["action"],
                    "reward": chunk["reward"],
                    "done": bool(terminal or chunk["done"]),
                    "z_next": np.zeros_like(chunk["z"]) if terminal else following["z"],
                    "proprio_next": chunk["proprio"] if terminal else following["proprio"],
                    "ref_next": chunk["reference"] if terminal else following["reference"],
                    "source": int(chunk["source"]),
                    "source_chunk": chunk["source_chunk"],
                    "success": bool(chunk["success"]),
                    "episode": episode,
                    "stage": 1,
                }
            )
        return rows

    def _chunk_record(self, feature: ChunkFeature, result: ChunkResult, source: int) -> dict:
        if not result.steps:
            raise RuntimeError("环境返回空 chunk，无法生成训练经验")
        sources = [np.asarray(step.source, dtype=np.uint8) for step in result.steps]
        return {
            "z": feature.z,
            "proprio": feature.proprio,
            "reference": feature.reference,
            "action": _pad_chunk([step.action for step in result.steps],
                                 self.config.data.chunk_len, feature.reference[-1]),
            "reward": _pad_chunk(
                [np.asarray(step.reward, dtype=np.float32) for step in result.steps],
                self.config.data.chunk_len, np.zeros((), dtype=np.float32),
            ).astype(np.float32),
            "source": source,
            "source_chunk": _pad_chunk(sources, self.config.data.chunk_len,
                                        np.zeros((), dtype=np.uint8)).astype(np.uint8),
            "done": result.done,
            "success": result.success,
        }

    def _finish_episode(self, slot: dict) -> dict:
        rows = self._rows_from_episode(slot["chunks"], slot["episode"])
        admitted = rows
        if self.config.env.stop_when_buffer_full:
            # Exactly fill the buffer without overwriting. Compute next states
            # before slicing, so a retained nonterminal row still bootstraps to
            # its actual next chunk, not to another episode or a fake terminal.
            admitted = rows[:max(0, self.replay.capacity - len(self.replay))]
        self.replay.add_episode(admitted)
        summary = {
            "episode": slot["episode"],
            "elapsed_seconds": time.perf_counter() - self._run_started,
            "env_id": slot["env_id"],
            "seed": slot["seed"],
            "chunks": len(slot["chunks"]),
            "transitions": len(rows),
            "transitions_added": len(admitted),
            "executed_steps": slot["executed_steps"],
            "reward_sum": slot["reward_sum"],
            "reward_events": slot["reward_events"],
            "success": bool(slot["chunks"][-1]["success"]),
            "actor_version": self.actor_runtime.version,
            "replay_size": len(self.replay),
            "adds_total": self.replay.adds_total,
        }
        with self.rollout_metrics.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary) + "\n")
        print(f"[rollout] {summary}", flush=True)
        return summary

    def _check_learner(self) -> None:
        if self.learner_thread.error is not None:
            raise RuntimeError("Learner 更新线程失败，停止采集") from self.learner_thread.error

    def _flush_learner(self) -> None:
        """负数等待上限表示补完全部预算；正式配置采用此模式。"""
        seconds = float(self.config.runtime.shutdown_flush_seconds)
        deadline = None if seconds < 0 else time.time() + seconds
        while deadline is None or time.time() < deadline:
            self._check_learner()
            if self.learner_thread.budget() <= 0:
                break
            time.sleep(0.02)
        self._check_learner()
        self.actor_runtime.maybe_reload()

    def run(self) -> dict:
        """独立环境同步批量推理；完成一集立即领取下一个循环种子。"""
        active: dict[int, dict] = {}
        results = []
        start = self.replay.max_episode + 1
        if self.target_episode_count is not None:
            completed = set()
            if self.rollout_metrics.is_file():
                with self.rollout_metrics.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            completed.add(int(json.loads(line)["episode"]))
                        except (ValueError, KeyError, json.JSONDecodeError):
                            continue
            if self.replay._fields:
                completed.update(
                    int(value) for value in np.unique(
                        self.replay._fields["episode"][: len(self.replay)]
                    )
                )
            episode_queue = [i for i in range(self.target_episode_count) if i not in completed]
        elif self.config.env.stop_when_buffer_full:
            episode_queue = None
        else:
            episode_queue = list(range(start, start + self.config.env.episodes))
        run_started = time.perf_counter()
        self._run_started = run_started
        timing = {"reset_seconds": 0.0, "feature_seconds": 0.0, "actor_seconds": 0.0, "environment_seconds": 0.0}
        executed_chunks = 0
        executed_steps = 0
        next_episode = start
        queue_cursor = 0
        batches = 0
        stopping = False

        def can_start() -> bool:
            if self.config.env.stop_when_buffer_full:
                return len(self.replay) < self.replay.capacity
            if episode_queue is not None:
                return queue_cursor < len(episode_queue)
            return next_episode - start < self.config.env.episodes

        def begin(env_id: int) -> None:
            nonlocal next_episode, queue_cursor
            if episode_queue is not None:
                episode = episode_queue[queue_cursor]
                queue_cursor += 1
            else:
                episode = next_episode
                next_episode += 1
            env = self.envs[env_id]
            reset_started = time.perf_counter()
            observation = env.reset_episode(episode) if hasattr(env, "reset_episode") else env.reset()
            timing["reset_seconds"] += time.perf_counter() - reset_started
            seeds = getattr(env, "seeds", ())
            active[env_id] = {
                "episode": episode, "env_id": env_id,
                "seed": int(seeds[episode % len(seeds)]) if seeds else None,
                "observation": observation, "chunks": [],
                "executed_steps": 0, "reward_sum": 0.0, "reward_events": 0,
            }

        self.learner_thread.start()
        try:
            if self.require_actor:
                self.actor_runtime.maybe_reload()
                if self.actor_runtime.version < 1:
                    raise RuntimeError(
                        "Stage 3 要求从首个 rollout chunk 使用 Stage 2 训练过的 Actor，"
                        f"但未能加载有效快照：{self.snapshot_path}"
                    )
            for env_id in range(len(self.envs)):
                if can_start():
                    begin(env_id)
            while active and not stopping:
                self._check_learner()
                ids = list(active)
                feature_started = time.perf_counter()
                observations = []
                for env_id in ids:
                    observation = dict(active[env_id]["observation"])
                    observation["_rlt_episode_id"] = active[env_id]["episode"]
                    observations.append(observation)
                features = self._features_batch(observations)
                timing["feature_seconds"] += time.perf_counter() - feature_started
                batches += 1
                self.actor_runtime.maybe_reload()
                for env_id, feature in zip(ids, features):
                    self._check_learner()
                    slot = active[env_id]
                    use_actor = self.require_actor or self.actor_runtime.ready(
                        len(self.replay), self.learner.step_count, self.config,
                        minimum_updates=self.actor_start_updates,
                    )
                    actor_started = time.perf_counter()
                    action, source = self.actor_runtime.refine(feature, self.normalizer, use_actor)
                    timing["actor_seconds"] += time.perf_counter() - actor_started
                    environment_started = time.perf_counter()
                    result = self.envs[env_id].execute_chunk(action, feature.reference, source)
                    timing["environment_seconds"] += time.perf_counter() - environment_started
                    executed_chunks += 1
                    executed_steps += len(result.steps)
                    slot["chunks"].append(self._chunk_record(feature, result, source))
                    slot["observation"] = result.observation
                    slot["executed_steps"] += len(result.steps)
                    slot["reward_sum"] += sum(float(step.reward) for step in result.steps)
                    slot["reward_events"] += sum(step.reward > 0 for step in result.steps)
                    if result.done or len(slot["chunks"]) >= self.config.env.max_chunks:
                        results.append(self._finish_episode(slot))
                        del active[env_id]
                        if self.config.env.stop_when_buffer_full and len(self.replay) >= self.replay.capacity:
                            stopping = True
                            break
                        if can_start():
                            begin(env_id)
            rollout_finished = time.perf_counter()
            self._flush_learner()
            flush_finished = time.perf_counter()
        finally:
            self.learner_thread.shutdown()
            for env in self.envs:
                if hasattr(env, "close"):
                    env.close()
            if hasattr(self.provider, "close"):
                self.provider.close()
        summary = {
            "episodes": len(results),
            "episodes_started": queue_cursor if episode_queue is not None else next_episode - start,
            "rollout_seconds": rollout_finished - run_started,
            "learner_flush_seconds": flush_finished - rollout_finished,
            "total_seconds": time.perf_counter() - run_started,
            "executed_chunks": executed_chunks,
            "executed_steps": executed_steps,
            **timing,
            "successes": int(sum(item["success"] for item in results)),
            "num_envs": len(self.envs),
            "feature_batches": batches,
            "replay_size": len(self.replay),
            "adds_total": self.replay.adds_total,
            "replay_capacity": self.replay.capacity,
            "stop_reason": "buffer_full" if stopping else "episode_limit",
            "discarded_inflight_chunks": sum(len(slot["chunks"]) for slot in active.values()),
            "learner_steps": self.learner.step_count,
            "learner_target_steps": int(self.replay.adds_total * self.config.objective.updates_per_sample),
            "out": str(self.out),
        }
        _atomic_write(self.out / "online_summary.json", json.dumps(summary, indent=2))
        return summary


def build_normalizer(config: RLTConfig) -> ActionNormalizer:
    assert config.data.stats_path, "缺少 data.stats_path，先运行 rlt/tools/make_stats.py"
    stats = NormStats.load(
        config.resolve_path(config.data.stats_path),
        action_dim=config.data.action_dim,
        proprio_dim=config.data.proprio_dim,
    )
    return ActionNormalizer(
        stats,
        action_dim=config.data.action_dim,
        proprio_dim=config.data.proprio_dim,
        representation=config.data.action_representation,
        delta_exclude_dims=config.data.delta_exclude_dims,
    )


def build_provider(config: RLTConfig, checkpoint: str | None, token: str | None, device: str):
    if config.runtime.feature_source == "remote":
        return RemoteFeatureProvider(config.runtime.server_url, config.data.chunk_len)
    assert checkpoint and token, "本地特征需要一个 SmolVLA 检查点和一个 RL token 检查点"
    from rlt.serve import load_everything

    policy, token_model = load_everything(config, checkpoint, token, device)
    return LocalFeatureProvider(policy, token_model, config, device=device)

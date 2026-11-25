"""RoboTwin 环境适配器，把仿真包装成 RLT 主循环期望的接口。

只做三件事：复位到一个给定场景、接收一整块动作并逐步执行、把每一步的
观测和奖励记下来。仿真器相关的一切都放在函数内部延迟导入，这样没有装
RoboTwin 的机器上也能 import 本模块并跑自检。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class StepRecord:
    observation: dict
    action: np.ndarray
    reference: np.ndarray
    reward: float
    next_observation: dict
    done: bool
    success: bool
    source: int


@dataclass
class ChunkResult:
    observation: dict
    steps: list[StepRecord] = field(default_factory=list)
    done: bool = False
    success: bool = False
    info: dict = field(default_factory=dict)


def observations_from_raw(head: np.ndarray, left: np.ndarray, right: np.ndarray, state: np.ndarray, task: str) -> dict:
    assert state.shape == (16,), f"Piper 的 qpos 应该是 16 维，收到 {state.shape}"
    return {
        "observation.state": np.asarray(state, dtype=np.float32),
        "observation.images.camera1": np.asarray(head, dtype=np.uint8),
        "observation.images.camera2": np.asarray(left, dtype=np.uint8),
        "observation.images.camera3": np.asarray(right, dtype=np.uint8),
        "task": task,
    }


def create_from_seed(seed: int, config_path: str | Path):
    """按种子现场新建一个 Piper 双臂场景。

    走的是项目现有的 `robotwin_common.make_env` 加 `robotwin_qpos_scene`，
    和采集脚本用的是同一条构造路径，因此机器人 URDF、相机外参、方块摆放
    约束都一致。和 `restore()` 的区别只是不要求事先存好场景 npz。
    """
    import json
    import sys

    from rlt.bootstrap import SCRIPTS_DIR

    runs_dir = SCRIPTS_DIR
    if str(runs_dir) not in sys.path:
        sys.path.insert(0, str(runs_dir))

    from robotwin_common import make_env
    from robotwin_qpos_scene import make_left_right_env
    from robotwin_smol_control import Controller
    from envs.robot.robot import Robot

    config = json.loads(Path(config_path).read_text())
    # Match robotwin_smol_control.restore: qpos control never uses the arm
    # planner, whose MPlib collision-geometry conversion can segfault here.
    # Keep the gripper schedule used by RoboTwin's action execution.
    class GripperSchedule:
        def plan_grippers(self, now_val, target_val):
            num_step = 200
            return dict(num_step=num_step,
                        per_step=(target_val - now_val) / num_step,
                        result=np.linspace(now_val, target_val, num_step))

    original_set_planner = Robot.set_planner

    def skip_unused_planner(robot, scene=None):
        robot.communication_flag = False
        robot.left_planner = GripperSchedule()
        robot.right_planner = GripperSchedule()

    Robot.set_planner = skip_unused_planner
    try:
        env, _ = make_left_right_env(
            make_env,
            int(seed),
            collect=False,
            wrist_camera=config["project2_wrist_camera"],
            head_camera=config["project2_head_camera"],
            layout=config["project2_layout"],
        )
    finally:
        Robot.set_planner = original_set_planner
    return env, Controller(env)


class RoboTwinEnv:
    """一个 episode 一个仿真进程，复位可以恢复场景，也可以按种子新建。"""

    def __init__(
        self,
        scenes: list[str] | None = None,
        task: str = "",
        seeds: list[int] | None = None,
        scene_config: str | Path | None = None,
        reset_mode: str = "auto",
        terminate_on_success: bool = True,
        max_chunks: int = 40,
        max_episode_steps: int = 376,
        success_grace_steps: int = 0,
        success_reward: float = 1.0,
        reward_fn=None,
    ) -> None:
        self.scenes = [Path(item).resolve() for item in (scenes or [])]
        self.seeds = [int(item) for item in (seeds or [])]
        self.scene_config = Path(scene_config).resolve() if scene_config else None
        for path in self.scenes:
            assert path.is_file(), f"场景文件不存在 {path}"
        if reset_mode == "auto":
            reset_mode = "scene" if self.scenes else "seed"
        assert reset_mode in ("scene", "seed"), reset_mode
        if reset_mode == "scene":
            assert self.scenes, "reset_mode=scene 需要 scenes"
        else:
            assert self.seeds, "reset_mode=seed 需要 seeds"
            assert self.scene_config and self.scene_config.is_file(), "reset_mode=seed 需要 scene_config"
        self.reset_mode = reset_mode
        self.task = task
        self.terminate_on_success = terminate_on_success
        self.max_chunks = max_chunks
        self.max_episode_steps = int(max_episode_steps)
        assert self.max_episode_steps >= 1
        self.success_grace_steps = int(success_grace_steps)
        self.success_reward = float(success_reward)
        self.reward_fn = reward_fn
        self._cursor = 0
        self._seed_cursor = 0
        self.env = None
        self.controller = None
        self.chunk_index = 0
        self.episode_steps = 0
        self.done = False
        self.success_seen = False
        self.steps_after_success = 0

    # ---- 生命周期 ----
    def reset_episode(self, episode: int) -> dict:
        """由采集器统一分配全局 episode，依次循环种子，避免各环境重复取首 seed。"""
        self._cursor = int(episode)
        self._seed_cursor = int(episode)
        return self.reset()

    def reset(self) -> dict:
        self.close()
        if self.reset_mode == "scene":
            scene = self.scenes[self._cursor % len(self.scenes)]
            self._cursor += 1
            from robotwin_smol_control import restore

            self.env, self.controller, _ = restore(scene)
        else:
            seed = self.seeds[self._seed_cursor % len(self.seeds)]
            self._seed_cursor += 1
            self.env, self.controller = create_from_seed(seed, self.scene_config)
        self.chunk_index = 0
        self.episode_steps = 0
        self.done = False
        self.success_seen = False
        self.steps_after_success = 0
        return self._observe()

    def close(self) -> None:
        if self.env is not None:
            self.env.close_env()
            # Base_Task.close_env delegates to gym.Env.close (a no-op here).
            # Explicitly detach SAPIEN entities and collect the task/camera
            # closures before constructing another scene. Otherwise every
            # episode reset can retain another renderer's GPU allocations.
            scene = getattr(self.env, "scene", None)
            if scene is not None:
                scene.clear()
            self.controller = None
            self.env = None
            del scene
            import gc

            gc.collect()
            from sapien.render import clear_cache

            clear_cache()
        self.env = None
        self.controller = None

    def _observe(self) -> dict:
        head, left, right, state = self.controller.observe()
        return observations_from_raw(head, left, right, state, self.task)

    def check_success(self) -> bool:
        return bool(self.env.check_success())

    def current_phase_name(self) -> str:
        return "critical"

    # ---- 执行 ----
    def execute_chunk(self, actions: np.ndarray, reference: np.ndarray, source: int) -> ChunkResult:
        actions = np.asarray(actions, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        assert actions.ndim == 2 and actions.shape == reference.shape, "动作块与参考块形状必须一致"
        remaining_steps = self.max_episode_steps - self.episode_steps
        actions = actions[:remaining_steps]
        reference = reference[:len(actions)]
        if not len(actions):
            raise RuntimeError("episode 已达到低层动作上限，不能继续执行新 chunk")
        observation = self._observe()
        self.controller.reanchor()
        result = ChunkResult(observation=observation)
        for index in range(len(actions)):
            before = observation
            executed = self.controller.execute(actions[index])
            self.episode_steps += 1
            observation = self._observe()
            success = self.check_success()
            success_event = success and not self.success_seen
            if success_event:
                self.success_seen = True
            if self.reward_fn is not None:
                reward = float(self.reward_fn(before, actions[index], index, success_event))
            else:
                reward = self.success_reward if success_event else 0.0
            if self.success_seen and not success_event:
                self.steps_after_success += 1
            done = bool(
                self.terminate_on_success
                and self.success_seen
                and self.steps_after_success >= self.success_grace_steps
            )
            result.steps.append(
                StepRecord(
                    observation=before,
                    action=np.asarray(executed, dtype=np.float32),
                    reference=reference[index],
                    reward=reward,
                    next_observation=observation,
                    done=done,
                    success=self.success_seen,
                    source=source,
                )
            )
            if done:
                result.done = True
                result.success = self.success_seen
                break
        self.chunk_index += 1
        horizon_reached = (
            self.episode_steps >= self.max_episode_steps
            or self.chunk_index >= self.max_chunks
        )
        if horizon_reached and not result.done:
            last = result.steps[-1]
            last.done = True
            last.success = self.success_seen
            result.done = True
            result.success = self.success_seen
        result.success = self.success_seen
        result.observation = observation
        result.info = {"chunk_index": self.chunk_index, "success": result.success}
        return result


def create_env(
    scenes=(), task="", seeds=(), scene_config=None, reset_mode="auto", max_chunks=40,
    max_episode_steps=376,
    success_grace_steps=0, success_reward=1.0, reward_fn=None, **_: object
) -> RoboTwinEnv:
    """工厂函数，对应配置里的 `rlt.env_robotwin:create_env`。"""
    return RoboTwinEnv(
        scenes=list(scenes),
        task=task,
        seeds=list(seeds),
        scene_config=scene_config,
        reset_mode=reset_mode,
        max_chunks=max_chunks,
        max_episode_steps=max_episode_steps,
        success_grace_steps=success_grace_steps,
        success_reward=success_reward,
        reward_fn=reward_fn,
    )


class DummyEnv:
    """没有仿真器时用的假环境，只验证链路，不产生任何有意义的结论。"""

    def __init__(self, proprio_dim: int = 16, action_dim: int = 16, chunk_len: int = 8, episodes: int = 2) -> None:
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.episodes = episodes
        self.chunk_index = 0
        self.episode = 0
        self.state = np.zeros(proprio_dim, dtype=np.float32)
        self.rng = np.random.default_rng(0)
        self._image = (self.rng.random((256, 256, 3)) * 255).astype(np.uint8)

    def reset(self) -> dict:
        self.episode += 1
        self.chunk_index = 0
        self.state = self.rng.normal(0, 0.1, self.proprio_dim).astype(np.float32)
        return self._observe()

    def _observe(self) -> dict:
        image = self._image
        return {
            "observation.state": self.state.copy(),
            "observation.images.camera1": image,
            "observation.images.camera2": image,
            "observation.images.camera3": image,
            "task": "dummy",
        }

    def execute_chunk(self, actions: np.ndarray, reference: np.ndarray, source: int) -> ChunkResult:
        actions = np.asarray(actions, dtype=np.float32)
        result = ChunkResult(observation=self._observe())
        for index in range(len(actions)):
            before = self._observe()
            self.state = self.state + 0.01 * np.tanh(actions[index])
            after = self._observe()
            reward = -float(np.abs(actions[index]).mean())
            result.steps.append(
                StepRecord(
                    observation=before,
                    action=actions[index].copy(),
                    reference=np.asarray(reference[index], dtype=np.float32),
                    reward=reward,
                    next_observation=after,
                    done=False,
                    success=False,
                    source=source,
                )
            )
        self.chunk_index += 1
        result.done = self.chunk_index >= 2
        result.success = bool(result.done and self.episode >= self.episodes)
        result.observation = self._observe()
        return result

    def check_success(self) -> bool:
        return False

    def current_phase_name(self) -> str:
        return "critical"

    def close(self) -> None:
        return None

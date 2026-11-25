"""25 Hz measured-position trajectory tracker, ten real SAPIEN steps per target."""
import json
import sys
from pathlib import Path

import numpy as np

from dataset import AUDIT


def restore(raw):
    # Import simulator in its own runtime, not inside the model worker.
    # Use the same project-local left/right layout and camera adapter as the
    # collector; feasibility_audit contains the simulator package, not this
    # task-specific scene helper.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(1, str(AUDIT / 'robotwin_runs'))
    from robotwin_common import make_env
    from envs.robot.robot import Robot
    from robotwin_qpos_scene import make_left_right_env
    raw = Path(raw).resolve()
    config = json.loads((raw.parent / 'environment_config.json').read_text())
    with np.load(raw, allow_pickle=False) as d:
        seed, initial = int(d['seed']), d['state'][0]
        initial_blocks = d['block_poses'][0, :, :3] if 'block_poses' in d else None
        names, units = d['joint_names'].tolist(), d['joint_units'].tolist()
    # Closed-loop inference directly commands qpos and never asks the task
    # planner to generate paths. Avoid constructing MPlib's collision scene:
    # its SAPIEN geometry conversion segfaults for these replay scenes.
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
        env, _ = make_left_right_env(make_env, seed, collect=False,
            wrist_camera=config['project2_wrist_camera'], head_camera=config['project2_head_camera'],
            layout=config['project2_layout'])
    finally:
        Robot.set_planner = original_set_planner
    try:
        controller = Controller(env)
        assert controller.names == names and units == (['rad'] * 6 + ['m'] * 2) * 2
        state_error = float(np.max(np.abs(controller.measure() - initial)))
        if initial_blocks is not None:
            block_error = float(np.max(np.abs(np.asarray([
                env.block1.get_pose().p, env.block2.get_pose().p]) - initial_blocks)))
            assert block_error < 1e-5, 'restored block positions differ from the collected initial scene'
        else:
            # The formal success dataset stores the deterministic seed and
            # environment config, but not block_poses. Rebuilding from those
            # recreates the sampled scene; verify its robot state against the
            # recorded initial observation and report that blocks were rebuilt.
            block_error = None
        assert state_error < 1e-5, 'restored robot state differs from the collected initial scene'
        return env, controller, dict(initial_qpos_max_error=state_error,
            initial_block_position_max_error_m=block_error,
            block_positions_reconstructed_from_seed=initial_blocks is None)
    except BaseException:
        env.close_env()
        raise


class Controller:
    def __init__(self, env):
        self.env, self.entities, self.indices, self.joints, self.names = env, [], [], [], []
        for side in ('left', 'right'):
            entity = getattr(env.robot, side + '_entity')
            active = list(entity.get_active_joints())
            group = list(getattr(env.robot, side + '_arm_joints')) + [g[0] for g in getattr(env.robot, side + '_gripper')]
            assert len(group) == 8
            self.entities.append(entity)
            self.indices.append([active.index(j) for j in group])
            self.joints.extend(group)
            self.names.extend(side + '/' + j.get_name() for j in group)
        self.limits = np.asarray([joint.get_limits()[0] for joint in self.joints])
        assert abs(float(env.scene.get_timestep()) - .004) < 1e-8
        self.anchor = self.measure()
        self.clipped_dimensions = 0
        self.max_clip = np.zeros(16)

    def measure(self):
        return np.concatenate([e.get_qpos()[ids] for e, ids in zip(self.entities, self.indices)]).astype(np.float32)

    def observe(self):
        import cv2
        obs = self.env.get_obs()['observation']
        rgb = [cv2.resize(obs[k]['rgb'], (256, 256)) for k in ('head_camera', 'left_camera', 'right_camera')]
        return (*rgb, self.measure())

    def reanchor(self):
        self.anchor = self.measure()

    def execute(self, target):
        target = np.asarray(target, dtype=np.float32)
        assert target.shape == (16,) and np.isfinite(target).all()
        end = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        # Ignore sub-nanometre/radian floating point noise at hard limits.
        self.clipped_dimensions += int(np.count_nonzero(np.abs(end - target) > 1e-6))
        self.max_clip = np.maximum(self.max_clip, np.abs(end - target))
        start = self.anchor.copy()
        velocity = (end - start) / .04
        for tick in range(10):
            point = start + (end - start) * ((tick + 1) / 10)
            for entity in self.entities:
                self.env.robot._entity_qf(entity)
            for joint, value, speed in zip(self.joints, point, velocity):
                joint.set_drive_target(float(value))
                joint.set_drive_velocity_target(float(speed))
            self.env.scene.step()
        self.anchor = end.copy()
        return end

    def finish(self):
        # Extra stability test only; not used to fabricate model inputs/actions.
        before = np.asarray([self.env.block1.get_pose().p.copy(), self.env.block2.get_pose().p.copy()])
        for joint in self.joints:
            joint.set_drive_velocity_target(0.)
        for _ in range(80):
            for entity in self.entities:
                self.env.robot._entity_qf(entity)
            self.env.scene.step()
        after = np.asarray([self.env.block1.get_pose().p.copy(), self.env.block2.get_pose().p.copy()])
        shift = np.linalg.norm(after - before, axis=1)
        released = bool(np.min(self.measure()[[6, 7, 14, 15]]) > .035)
        official = bool(self.env.check_success())
        return dict(official_success=official, final_measured_grippers_released=released,
            stable_success=bool(official and released and np.max(shift) < .002),
            block_shift_during_hold_m=shift.tolist(), diagnostic_hold_seconds=.32,
            gripper_flags_note='native initial command flags retained; measured release also required',
            clipped_dimensions=self.clipped_dimensions, max_clip_per_dimension=self.max_clip.tolist())

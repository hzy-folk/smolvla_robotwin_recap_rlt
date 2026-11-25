#!/usr/bin/env python3
"""Collect complete successful RoboTwin stack demonstrations in the SmolVLA qpos16 format."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collection_config import parse_task_args

import cv2
import numpy as np

from dataset import AUDIT

ROOT = Path(__file__).resolve().parents[1]
PY_TASK = "stack the red block and then the green block"

# One recorded frame is exactly ten 4 ms simulator steps, i.e. 40 ms / 25 Hz.
TICKS_PER_FRAME = 10
SAMPLE_HZ = 25
TIMESTEP_S = .004
# Measured finger qpos: fully open ~= .0400 m, closed on a block ~= .0227-.0260 m.
# The 3 mm inward correction may only be active inside this closed window.
GRIPPER_CLOSED_MAX_M = .030
GRIPPER_INWARD_M = .003
# Required episode duration range (seconds) and the lift the right arm must still show.
DURATION_RANGE_S = (11., 15.)
# The screw planner fails once the wrist passes ~4 cm above the release pose, so a
# 4 cm vertical lift is the largest clean post-release motion for this embodiment.
# The tracking lag is ~3 mm, so command slightly more to reach 4 cm measured.
LIFT_HEIGHT_M = .043
MIN_RIGHT_LIFT_M = .034
# The left-arm return runs at the fastest joint speed already observed in the same
# episode, so it can never look faster than the other motions. These bounds only
# guard against degenerate measurements.
RETURN_SPEED_BOUNDS = (.2, 3.)


def extend14(x):
    """Expand RoboTwin's one-value-per-gripper command to two finger values."""
    x = np.asarray(x, dtype=np.float32).reshape(14)
    return np.asarray((*x[:6], x[6], x[6], *x[7:13], x[13], x[13]), dtype=np.float32)


def target_duration(seed):
    """Per-episode duration drawn from the required 11-15 s window.

    Consecutive seeds must not produce near-identical durations, so this uses an
    independent seed stream instead of a function of `seed % N`.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20261]))
    return float(rng.uniform(*DURATION_RANGE_S))


def gripper_windows(state):
    """Boolean masks for the frames where each gripper actually holds a block.

    Both fingers have to be closed, otherwise the opening ramp would be treated
    as a grasp. Opening is detected immediately because the measured qpos rises
    above the closed threshold.
    """
    state = np.asarray(state, dtype=np.float32)
    left = (state[:, 6] <= GRIPPER_CLOSED_MAX_M) & (state[:, 7] <= GRIPPER_CLOSED_MAX_M)
    right = (state[:, 14] <= GRIPPER_CLOSED_MAX_M) & (state[:, 15] <= GRIPPER_CLOSED_MAX_M)
    return left, right


def build_actions(state):
    """Absolute qpos actions with the inward correction only inside closed windows."""
    state = np.asarray(state, dtype=np.float32)
    action = state.copy()
    left, right = gripper_windows(state)
    action[left, 6] -= np.float32(GRIPPER_INWARD_M)
    action[left, 7] -= np.float32(GRIPPER_INWARD_M)
    action[right, 14] -= np.float32(GRIPPER_INWARD_M)
    action[right, 15] -= np.float32(GRIPPER_INWARD_M)
    return action, dict(left_closed_frames=int(left.sum()), right_closed_frames=int(right.sum()),
                        left_open_frames=int((~left).sum()), right_open_frames=int((~right).sum()))


def resample_episode(images, state, command14, right_ee_z, target_n):
    """Retime a complete successful trajectory to `target_n` frames at 25 Hz.

    Joint positions are interpolated linearly and images use nearest valid frame,
    so only the playback speed changes; the geometry and the ordering of the
    expert motions are preserved. Unlike the previous version this never keeps
    the native length, which is what made every episode the same duration.
    """
    target_n = int(target_n)
    src = np.linspace(0., len(state) - 1, target_n)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, len(state) - 1)
    alpha = (src - lo).astype(np.float32)[:, None]
    out_state = state[lo] * (1. - alpha) + state[hi] * alpha
    # Hold native commands and diagnostics with nearest-frame selection: they are
    # only used to report the gripper window, never as action inputs.
    hold = np.rint(src).astype(np.int64)
    out_images = [images[i] for i in hold]
    out_command = command14[hold]
    out_ee_z = np.asarray(right_ee_z, dtype=np.float32)[hold]
    return out_images, out_state.astype(np.float32), out_command, out_ee_z, hold


def direct_joint_move(env, side, target, steps):
    joints = list(getattr(env.robot, side + '_arm_joints'))
    start = np.asarray([np.asarray(j.get_drive_target()).reshape(-1)[0] for j in joints], dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    # Smoothstep speed profile with explicit velocity targets: the old version
    # finished the whole return in 60 ticks (0.24 s), which read as a snap.
    duration = steps * TIMESTEP_S
    points = []
    for k in range(steps):
        u = (k + 1) / steps
        point = start + (target - start) * (3. * u * u - 2. * u ** 3)
        velocity = (target - start) * (6. * u * (1. - u) / duration)
        points.append(point)
        env.robot.set_arm_joints(point, velocity, side)
        env.scene.step()
    return frame_peak_speed(np.asarray(points))


def frame_peak_speed(points_per_tick):
    """Peak joint speed (rad/s) of a 250 Hz command sequence, sampled per 25 Hz frame."""
    points = np.asarray(points_per_tick, dtype=np.float64)
    if len(points) < TICKS_PER_FRAME + 1:
        return 0.
    frame_step = np.abs(points[TICKS_PER_FRAME:] - points[:-TICKS_PER_FRAME]).max()
    return float(frame_step) / (TICKS_PER_FRAME * TIMESTEP_S)


def plan_straight(env, side, target_pose):
    """Plan a straight-line Cartesian move with the screw planner; None on failure."""
    planner = getattr(env.robot, side + '_planner')
    original = planner.planner_type
    planner.planner_type = 'mplib_screw'
    try:
        result = getattr(env.robot, side + '_plan_path')(
            np.asarray(target_pose, dtype=np.float64).tolist())
    finally:
        planner.planner_type = original
    if not result or result.get('status') != 'Success':
        return None
    return result


def play_path(env, side, result, joint_speed):
    """Play a planner path, holding each point long enough to hit `joint_speed`."""
    positions = np.asarray(result['position'], dtype=np.float64)
    velocities = np.asarray(result['velocity'], dtype=np.float64)
    step = float(np.abs(np.diff(positions, axis=0)).max()) if len(positions) > 1 else 0.
    hold = max(1, int(np.ceil(step / (joint_speed * TIMESTEP_S))))
    for position, velocity in zip(positions, velocities / hold):
        env.robot.set_arm_joints(position, velocity, side)
        for _ in range(hold):
            env.scene.step()
    return dict(hold=hold, seconds=len(positions) * hold * TIMESTEP_S,
                peak_rad_s=step / (hold * TIMESTEP_S))


def left_return_home(env, home_pose, home_joints, joint_speed):
    """Send the left arm home at the speed measured from the other motions."""
    result = plan_straight(env, 'left', home_pose)
    if result is not None:
        played = play_path(env, 'left', result, joint_speed)
        error = float(np.linalg.norm(np.asarray(env.robot.get_left_ee_pose()[:3], dtype=np.float64)
                                     - np.asarray(home_pose[:3], dtype=np.float64)))
        if error < .01:
            return dict(mode='screw', error_m=error, reference_rad_s=joint_speed, **played)
    # Fallback: joint-space smoothstep, timed so its peak speed equals the reference.
    joints = list(env.robot.left_arm_joints)
    start = np.asarray([np.asarray(j.get_drive_target()).reshape(-1)[0] for j in joints],
                       dtype=np.float64)
    travel = float(np.abs(np.asarray(home_joints) - start).max())
    steps = max(50, int(round(1.5 * travel / (joint_speed * TIMESTEP_S))))
    peak = direct_joint_move(env, 'left', home_joints, steps)
    return dict(mode='joint', steps=steps, seconds=steps * TIMESTEP_S,
                reference_rad_s=joint_speed, peak_rad_s=peak)


def vertical_lift(env, dz=LIFT_HEIGHT_M, parts=4):
    """Raise the right arm straight up in world z after the gripper opens.

    The RRT planner interpolates joint space, so the old fallback produced a
    diagonal retreat and the direct Cartesian request frequently failed IK.
    Plan each segment with the screw (straight-line Cartesian) planner instead:
    position moves purely along +z while the orientation is held fixed.
    """
    from envs.utils.action import ArmTag, Action
    planner = env.robot.right_planner
    original_type = planner.planner_type
    z0 = float(np.asarray(env.robot.get_right_ee_pose(), dtype=np.float64)[2])
    reached = 0.
    planner.planner_type = 'mplib_screw'
    try:
        for k in range(parts):
            current = np.asarray(env.robot.get_right_ee_pose(), dtype=np.float64)
            target = current.copy()
            target[2] = z0 + dz * (k + 1) / parts
            env.plan_success = True
            ok = env.move((ArmTag('right'), [Action(ArmTag('right'), 'move', target_pose=target.tolist())]))
            reached = float(np.asarray(env.robot.get_right_ee_pose(), dtype=np.float64)[2]) - z0
            if not ok or not env.plan_success:
                return reached
    finally:
        planner.planner_type = original_type
    return reached


def sequential_expert(env, left_home_pose, joint_speed):
    """Run the official two-block plan with explicit left-return-before-right ordering."""
    from envs.utils.action import ArmTag, Action
    env.last_gripper = None
    env.last_actor = None
    # Prevent the task's early-success guard from cutting off the required
    # right-arm lift immediately after the right gripper opens.
    original_check = env.check_success
    env.check_success = lambda: False
    try:
        left = ArmTag('left')
        right = ArmTag('right')
        block1, block2 = env.block1, env.block2
        left_home = np.asarray([np.asarray(j.get_drive_target()).reshape(-1)[0]
                                for j in env.robot.left_arm_joints],
                               dtype=np.float64)
        env.move(env.grasp_actor(block1, arm_tag=left, pre_grasp_dis=.09))
        env.move(env.move_by_displacement(arm_tag=left, z=.07))
        env.move(env.place_actor(block1, target_pose=env.block1_target_pose,
                                 arm_tag=left, functional_point_id=0,
                                 pre_dis=.05, dis=0., pre_dis_axis='fp'))
        env.move(env.move_by_displacement(arm_tag=left, z=.07))
        env.last_gripper, env.last_actor = left, block1
        # Separate move from the right grasp: bring the left arm back home along a
        # straight line at the same paced speed as the rest of the demonstration.
        left_return = left_return_home(env, left_home_pose, left_home, joint_speed())
        env.move(env.grasp_actor(block2, arm_tag=right, pre_grasp_dis=.09))
        env.move(env.move_by_displacement(arm_tag=right, z=.07))
        target = block1.get_functional_point(1)
        env.move(env.place_actor(block2, target_pose=target, arm_tag=right,
                                 functional_point_id=0, pre_dis=.05, dis=0.,
                                 pre_dis_axis='fp'))
        # Explicit post-release retreat: strictly vertical, fully recorded.
        env.plan_success = True
        lift = vertical_lift(env)
        env.last_gripper, env.last_actor = right, block2
        return dict(right_lift_m=float(lift), right_lift_ok=bool(lift >= MIN_RIGHT_LIFT_M),
                    left_return=left_return)
    finally:
        env.check_success = original_check


def collect_one(seed, destination, episode_id=None):
    import sys
    sys.path.insert(0, str(AUDIT / 'robotwin_runs'))
    from robotwin_common import make_env
    from robotwin_qpos_scene import make_left_right_env
    from robotwin_smol_control import Controller

    env, config = make_left_right_env(make_env, seed, collect=True,
                                      fixed_block_yaw=True, stack_xy_jitter_m=0.0)
    # Recording is driven by simulator ticks below, not by RoboTwin's save hook.
    env.save_freq = None
    controller = Controller(env)
    images, states, commands, right_ee_z = [], [], [], []
    ticks = {'n': 0}

    def record():
        obs = env.get_obs()
        view = obs['observation']
        images.append({
            'images': cv2.resize(view['head_camera']['rgb'], (256, 256)),
            'left_wrist': cv2.resize(view['left_camera']['rgb'], (256, 256)),
            'right_wrist': cv2.resize(view['right_camera']['rgb'], (256, 256)),
        })
        states.append(controller.measure())
        right_ee_z.append(float(np.asarray(env.robot.get_right_ee_pose())[2]))
        # Keep the native 14D command for gripper-window diagnostics. Expanding
        # it before indexing would move the right gripper from column 13.
        commands.append(np.asarray(obs['joint_action']['vector'], dtype=np.float32).reshape(14))

    def reference_joint_speed():
        """Fastest per-frame joint speed recorded so far in this episode (rad/s)."""
        if len(states) < 2:
            return RETURN_SPEED_BOUNDS[0]
        frames = np.asarray(states, dtype=np.float32)
        speed = float(np.abs(np.diff(frames, axis=0)).max()) / (TICKS_PER_FRAME * TIMESTEP_S)
        return float(np.clip(speed, *RETURN_SPEED_BOUNDS))

    original_step = env.scene.step

    def stepping():
        original_step()
        ticks['n'] += 1
        if ticks['n'] % TICKS_PER_FRAME == 0:
            record()

    started = time.perf_counter()
    # Home pose of the left arm before any motion; the return targets exactly this.
    left_home_pose = np.asarray(env.robot.get_left_ee_pose(), dtype=np.float64)
    env.scene.step = stepping
    try:
        record()
        try:
            lift = sequential_expert(env, left_home_pose, reference_joint_speed)
        except RuntimeError as error:
            return dict(seed=seed, success=False, planning_success=False,
                        frames=len(states), accepted=False, error=repr(error))
        success = bool(env.check_success())
        planned = bool(env.plan_success)
        if ticks['n'] % TICKS_PER_FRAME:
            record()
        # RoboTwin may clear plan_success during a trailing no-op IK call after
        # check_success() became true. The task result stays authoritative.
        if not success:
            return dict(seed=seed, success=success, planning_success=planned,
                        frames=len(states), accepted=False, right_lift_m=lift['right_lift_m'])
        if not lift['right_lift_ok']:
            return dict(seed=seed, success=success, planning_success=planned,
                        frames=len(states), accepted=False, reason='right arm lift too small',
                        right_lift_m=lift['right_lift_m'])
        state = np.asarray(states, dtype=np.float32)
        command = np.asarray(commands, dtype=np.float32)
        native_frames = len(state)
        native_seconds = ticks['n'] * TIMESTEP_S
        target_seconds = target_duration(seed)
        target_n = int(round(target_seconds * SAMPLE_HZ))
        images, state, command, right_ee_z, source_index = resample_episode(
            images, state, command, right_ee_z, target_n)
        action, window = build_actions(state)
        n = len(state)
        speed_factor = native_seconds / (n / SAMPLE_HZ)
        episode_name = episode_id or f'episode_{seed:06d}'
        episode = destination / episode_name
        episode.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(episode / f'{episode_name}.npz',
            images=np.asarray([x['images'] for x in images], dtype=np.uint8),
            left_wrist=np.asarray([x['left_wrist'] for x in images], dtype=np.uint8),
            right_wrist=np.asarray([x['right_wrist'] for x in images], dtype=np.uint8),
            state=state, action=action,
            diagnostic_command14=command[:, :14],
            right_ee_z=right_ee_z,
            joint_names=np.asarray(controller.names),
            joint_units=np.asarray((['rad'] * 6 + ['m'] * 2) * 2),
            seed=np.asarray(seed), task=np.asarray(PY_TASK), success=np.asarray(True),
            sample_hz=np.asarray(SAMPLE_HZ), timestamp=np.arange(n, dtype=np.float64) / SAMPLE_HZ,
            tick=np.arange(n, dtype=np.int64) * TICKS_PER_FRAME,
            source_index=source_index.astype(np.int64))
        for key, label in (('images', 'head'), ('left_wrist', 'left_wrist'), ('right_wrist', 'right_wrist')):
            writer = cv2.VideoWriter(str(episode / f'{label}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                     SAMPLE_HZ, (256, 256))
            for frame in np.asarray([x[key] for x in images]):
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        report = dict(seed=seed, task=PY_TASK, frames=n, sample_hz=SAMPLE_HZ,
                      duration_seconds=n / SAMPLE_HZ, success=True, planning_success=planned,
                      accepted=True, state_dim=16, action_dim=16,
                      action_contract='absolute qpos; closed gripper fingers inward 3 mm',
                      right_lift_m=lift['right_lift_m'],
                      left_return=lift['left_return'],
                      native_frames=native_frames, native_seconds=native_seconds,
                      target_duration_seconds=target_seconds, speed_factor=speed_factor,
                      gripper_window=window,
                      collection_seconds=time.perf_counter() - started,
                      layout=config['project2_layout'])
        (episode / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (episode / 'environment_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
        return report
    finally:
        env.scene.step = original_step
        env.close_env()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'data/robotwin_success_qpos16_pilot3')
    p.add_argument('--count', type=int, default=3)
    p.add_argument('--first-seed', type=int, default=930000)
    args = parse_task_args(p, "robotwin", "collection")
    # make_env changes cwd into RoboTwin; resolve before creating the env.
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = []
    attempts = 0
    while len(episodes) < args.count:
        seed = args.first_seed + attempts
        attempts += 1
        report = collect_one(seed, args.output / 'episodes')
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if report.get('accepted'):
            episode_name = f'episode_{len(episodes):03d}'
            # The collector names files after the successful sample index; the
            # seed remains in report/NPZ for reproducibility.
            source_episode = args.output / 'episodes' / f'episode_{seed:06d}'
            if source_episode.exists():
                target_episode = args.output / 'episodes' / episode_name
                source_episode.rename(target_episode)
                source_npz = target_episode / f'episode_{seed:06d}.npz'
                target_npz = target_episode / f'{episode_name}.npz'
                source_npz.rename(target_npz)
                report_path = target_episode / 'report.json'
                report_data = json.loads(report_path.read_text())
                report_data['episode_id'] = episode_name
                report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2))
            episodes.append(dict(id=episode_name, seed=seed,
                npz=f'episodes/{episode_name}/{episode_name}.npz',
                frames=report['frames'], duration_seconds=report['duration_seconds'],
                speed_factor=report['speed_factor'], split='pilot', task_success=True))
    manifest = dict(status='pilot_complete', count=len(episodes), task=PY_TASK,
                    randomization=dict(red_x_m=[-.24, -.20], green_x_m=[.20, .24],
                        y_m=[-.04, 0.0], fixed_yaw=True, target_xy_m=[0., -.13],
                        target_jitter_m=0.0), episodes=episodes,
                    attempts=attempts, collected_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(output=str(args.output), count=len(episodes), attempts=attempts), ensure_ascii=False))
    # SAPIEN's renderer leaves non-daemon threads behind, so a normal interpreter
    # shutdown hangs the process after the work is done. Exit hard instead of
    # leaving an idle collector holding the GPU.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

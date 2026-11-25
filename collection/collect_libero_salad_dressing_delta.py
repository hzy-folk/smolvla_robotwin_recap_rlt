"""Collect one seed-controlled LIBERO salad-dressing episode with OSC deltas."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from collection_config import parse_task_args

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT.parents[1] / "feasibility_audit"
LIBERO = AUDIT / "third_party" / "LIBERO-filtered"
PRO_DATA = AUDIT / "third_party" / "LIBERO-Pro-task"
TASK = "pick_up_the_salad_dressing_and_place_it_in_the_basket"
PRO_SOURCE_TASK = "pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
PRO_TASK_LANGUAGE = "Pick the salad dressing and place it in the basket"
INSTRUCTION = "pick up the salad dressing"
FPS = 20
SIZE = 256
APPROACH_OFFSET_MIN_M = 0.16
APPROACH_OFFSET_MAX_M = 0.20
MOVE_STALL_EPS_M = 0.0015
MOVE_STALL_PATIENCE = 5

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(AUDIT / "platform_runs" / "libero_config"))
sys.path.insert(0, str(AUDIT / "libero_venv" / "lib" / "python3.10" / "site-packages"))
sys.path.insert(0, str(LIBERO))
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def load_official_init_states(variant: str):
    """Load the paired benchmark BDDL and official initial-state array."""
    if variant == "pro":
        bddl_path = PRO_DATA / "bddl_files" / "libero_object_task" / f"{PRO_SOURCE_TASK}.bddl"
        init_path = PRO_DATA / "init_files" / "libero_object_task" / f"{PRO_SOURCE_TASK}.pruned_init"
        bddl = bddl_path.read_text(encoding="utf-8")
        required = (
            f"(:language {PRO_TASK_LANGUAGE})",
            "salad_dressing_1\n  basket_1_contain_region",
            "(And (In salad_dressing_1 basket_1_contain_region))",
        )
        if any(part not in bddl for part in required):
            raise RuntimeError(f"Pro BDDL does not describe the salad-dressing task: {bddl_path}")
        # LIBERO-Pro task perturbation suites retain the base suite task order;
        # resolve the source task name for metadata instead of selecting by ID.
        base_suite = benchmark.get_benchmark_dict()["libero_object"]()
        matches = [(i, task) for i, task in enumerate(base_suite.tasks) if task.name == PRO_SOURCE_TASK]
        if len(matches) != 1:
            raise RuntimeError(f"Could not resolve Pro source task {PRO_SOURCE_TASK!r} in libero_object")
        task_id, task_language = matches[0][0], PRO_TASK_LANGUAGE
        suite_name = "libero_object_task"
    elif variant == "standard":
        suite = benchmark.get_benchmark_dict()["libero_object"]()
        matches = [(i, task) for i, task in enumerate(suite.tasks) if task.name == TASK]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {TASK!r} task in libero_object; found {len(matches)}")
        task_id, task = matches[0]
        if "salad dressing" not in task.language.lower():
            raise RuntimeError(f"Task index {task_id} has unexpected language: {task.language!r}")
        bddl_path = LIBERO / "libero" / "libero" / "bddl_files" / task.problem_folder / f"{task.name}.bddl"
        init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        task_language = task.language
        suite_name = "libero_object"
    else:
        raise ValueError(f"Unknown benchmark variant: {variant!r}")

    if not bddl_path.is_file() or not init_path.is_file():
        raise FileNotFoundError(f"Missing benchmark task files: {bddl_path} or {init_path}")
    try:
        import torch

        try:
            states = torch.load(init_path, map_location="cpu", weights_only=False)
        except TypeError:
            states = torch.load(init_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Could not load LIBERO initial states from {init_path}") from exc
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 110 or len(states) != 50:
        raise RuntimeError(f"Unexpected init-state array shape {states.shape} at {init_path}")
    return task_id, task_language, states, bddl_path, init_path, suite_name


def make_env(seed: int, init_state: np.ndarray, dressing_jitter_m: float, bddl_path: Path):
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=SIZE,
        camera_widths=SIZE,
        controller="OSC_POSE",
        control_freq=FPS,
        ignore_done=True,
    )
    env.seed(seed)
    env.reset()
    state = np.asarray(init_state, dtype=np.float64).copy()
    sim = env.env.sim
    if state.shape != (1 + sim.model.nq + sim.model.nv,):
        env.close()
        raise RuntimeError(f"Unexpected flattened init-state shape {state.shape}")
    qpos_start, qpos_end = sim.model.get_joint_qpos_addr("salad_dressing_1_joint0")
    if qpos_end - qpos_start != 7:
        env.close()
        raise RuntimeError("Salad dressing is not represented by a 7-D free joint")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53414C41]))
    dressing_offset = rng.uniform(-dressing_jitter_m, dressing_jitter_m, size=2)
    state[1 + qpos_start : 1 + qpos_start + 2] += dressing_offset
    obs = env.set_init_state(state)
    return env, obs, dressing_offset


def collect(
    seed: int,
    output: Path,
    variant: str = "standard",
    dressing_jitter_m: float = 0.02,
    approach_offset_min_m: float = APPROACH_OFFSET_MIN_M,
    approach_offset_max_m: float = APPROACH_OFFSET_MAX_M,
):
    output.mkdir(parents=True, exist_ok=True)
    if dressing_jitter_m < 0 or not 0 <= approach_offset_min_m <= approach_offset_max_m:
        raise ValueError("Invalid jitter or approach-height range")
    task_id, task_language, init_states, bddl_path, init_path, suite_name = load_official_init_states(variant)
    if not 0 <= seed < len(init_states):
        raise ValueError(f"seed must select an official init-state index in [0, {len(init_states) - 1}], got {seed}")
    env, obs, dressing_offset = make_env(seed, init_states[seed], dressing_jitter_m, bddl_path)
    approach_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x41505052]))
    approach_offset_m = float(approach_rng.uniform(approach_offset_min_m, approach_offset_max_m))
    reset_settle_steps = 8
    neutral = np.r_[np.zeros(6, dtype=np.float32), np.float32(-1.0)]
    for _ in range(reset_settle_steps):
        obs, _, _, _ = env.step(neutral)
    initial_sim_state = env.get_sim_state().copy()
    robot = env.env.robots[0]
    controller = robot.controller
    xyz_step = np.asarray(controller.output_max[:3], dtype=np.float32)
    dressing = np.asarray(obs["salad_dressing_1_pos"], dtype=np.float64).copy()
    basket = np.asarray(obs["basket_1_pos"], dtype=np.float64).copy()
    stamp = time.time()
    data = {"agentview": [], "wrist": [], "state": [], "action": [], "object_pos": [],
            "basket_pos": [], "timestamps": [], "stage": [], "terminated": [],
            "truncated": []}
    stages = []

    def state8(current):
        import robosuite.utils.transform_utils as transform_utils

        # Keep robosuite's quaternion branch, matching the Pi05 LIBERO state
        # preprocessing. Converting through a matrix and SciPy canonicalizes
        # rotations to [-pi, pi], which flips the vector near pi.
        rotvec = transform_utils.quat2axisangle(
            np.asarray(current["robot0_eef_quat"], dtype=np.float64).copy()
        )
        return np.r_[current["robot0_eef_pos"], rotvec, current["robot0_gripper_qpos"]].astype(np.float32)

    def orientation(current):
        import robosuite.utils.transform_utils as transform_utils

        matrix = transform_utils.quat2mat(current["robot0_eef_quat"])
        return Rotation.from_matrix(matrix)

    # The controller's cached matrix can differ from the measured pose after reset.
    grasp_rotation = orientation(obs)

    def step(action, stage):
        nonlocal obs
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (7,)
        assert np.all(np.abs(action) <= 1.00001)
        data["agentview"].append(obs["agentview_image"][::-1].copy())
        data["wrist"].append(obs["robot0_eye_in_hand_image"][::-1].copy())
        data["state"].append(state8(obs))
        data["action"].append(action.copy())
        data["object_pos"].append(np.asarray(obs["salad_dressing_1_pos"]).copy())
        data["basket_pos"].append(np.asarray(obs["basket_1_pos"]).copy())
        data["timestamps"].append((len(data["state"]) - 1) / FPS)
        data["stage"].append(stage)
        obs, _, _, _ = env.step(action)
        terminated = bool(env.check_success())
        data["terminated"].append(terminated)
        data["truncated"].append(False)
        return terminated

    def is_terminated():
        return bool(data["terminated"] and data["terminated"][-1])

    def hold(gripper: float, count: int, stage: str, *, stop_on_success: bool = True):
        for _ in range(count):
            terminated = step(
                np.r_[np.zeros(6, dtype=np.float32), np.float32(gripper)], stage
            )
            if stop_on_success and terminated:
                break

    def wait_until_still(gripper: float, stage: str, *, max_steps=2):
        previous_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        stable_steps = 0
        for _ in range(max_steps):
            step(np.r_[np.zeros(6, dtype=np.float32), np.float32(gripper)], stage)
            current_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
            motion = float(np.linalg.norm(current_eef - previous_eef))
            stable_steps = stable_steps + 1 if motion <= 0.0003 else 0
            previous_eef = current_eef
            if stable_steps >= 2:
                break

    def pose_action(target, gripper: float, rotation_target=None):
        target = np.asarray(target, dtype=np.float64)
        delta = target - np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        increment = np.clip(delta, -xyz_step, xyz_step)
        rotation_target = grasp_rotation if rotation_target is None else rotation_target
        rot_delta = (rotation_target * orientation(obs).inv()).as_rotvec()
        rot_step = np.asarray(controller.output_max[3:], dtype=np.float32)
        rot_increment = np.clip(rot_delta, -rot_step, rot_step)
        return np.r_[increment / xyz_step,
                     rot_increment / rot_step,
                     np.float32(gripper)].astype(np.float32)

    def move(
        target,
        gripper: float,
        stage: str,
        max_steps: int = 80,
        settle_steps: int = 2,
        stop_on_success: bool = True,
        rotation_target=None,
        smooth_stop: bool = False,
    ):
        target = np.asarray(target, dtype=np.float64)
        rotation_target = grasp_rotation if rotation_target is None else rotation_target
        started = len(data["state"])
        steps = 0
        stalled_steps = 0
        stagnated = False
        while steps < max_steps:
            previous_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
            rot_delta = (rotation_target * orientation(obs).inv()).as_rotvec()
            osc = pose_action(target, gripper, rotation_target)
            terminated = step(osc, stage)
            steps += 1
            measured_motion = float(
                np.linalg.norm(np.asarray(obs["robot0_eef_pos"], dtype=np.float64) - previous_eef)
            )
            stalled_steps = stalled_steps + 1 if measured_motion <= MOVE_STALL_EPS_M else 0
            if terminated and stop_on_success:
                break
            if np.linalg.norm(target - obs["robot0_eef_pos"]) <= 0.008 and np.linalg.norm(rot_delta) <= 0.035:
                break
            if stalled_steps >= MOVE_STALL_PATIENCE:
                stagnated = True
                break
        if stagnated and smooth_stop:
            arm_action = osc[:6].copy()
            for scale in (0.65, 0.35, 0.12, 0.0):
                step(
                    np.r_[arm_action * scale, np.float32(gripper)],
                    f"{stage}_brake",
                )
        for _ in range(settle_steps):
            if is_terminated():
                break
            if step(np.r_[np.zeros(6, dtype=np.float32), np.float32(gripper)], f"{stage}_settle"):
                break
        position_error = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - target))
        orientation_error = float(np.linalg.norm((rotation_target * orientation(obs).inv()).as_rotvec()))
        stages.append({"name": stage, "sample_start": started, "sample_end": len(data["state"]),
                       "move_steps": steps, "position_error_m": position_error,
                       "orientation_error_rad": orientation_error,
                       "step_limit_reached": steps == max_steps,
                       "stagnated": stagnated})

    try:
        # The bottle lies sideways; approach above the body and keep the reset wrist pose.
        grasp = dressing + np.array([0.0, 0.0, 0.040])
        move(grasp + np.array([0.0, 0.0, approach_offset_m]), -1.0, "approach", max_steps=60)
        if not is_terminated():
            move(grasp, -1.0, "descend", smooth_stop=True)
        if not is_terminated():
            wait_until_still(-1.0, "pre_close_wait")
            hold(1.0, 8, "close")
            move(grasp + np.array([0.0, 0.0, 0.13]), 1.0, "lift")
        if not is_terminated():
            # Carry above the rim, then use the basket's live pose for final alignment.
            offset = np.asarray(obs["robot0_eef_pos"], dtype=np.float64) - np.asarray(obs["salad_dressing_1_pos"], dtype=np.float64)
            carry_orientation = orientation(obs)
            carry = basket + np.array([0.0, 0.0, 0.26]) + offset
            move(carry, 1.0, "transport", settle_steps=0, rotation_target=carry_orientation)
            basket_at_place = np.asarray(obs["basket_1_pos"], dtype=np.float64).copy()
            # Release above the rim to avoid the wrist settling against the basket edge.
            release_pose = basket_at_place + np.array([0.0, 0.0, 0.24]) + offset
            move(
                release_pose,
                1.0,
                "lower",
                settle_steps=0,
                rotation_target=carry_orientation,
                smooth_stop=True,
            )
        if not is_terminated():
            wait_until_still(1.0, "pre_release_wait")
            release_hold = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
            release_orientation = orientation(obs)
            hold(-1.0, 8, "release")
            hold(-1.0, 6, "drop_settle")
            retreat = release_hold.copy()
            retreat[2] += 0.08
            move(
                retreat,
                -1.0,
                "retreat",
                max_steps=40,
                settle_steps=0,
                stop_on_success=False,
                rotation_target=release_orientation,
            )
        success = is_terminated()
        final_object = np.asarray(obs["salad_dressing_1_pos"], dtype=np.float32)
        arrays = {k: np.asarray(v) for k, v in data.items()}
        arrays["state"] = arrays["state"].astype(np.float32)
        arrays["action"] = arrays["action"].astype(np.float32)
        # Flat frame aliases match the fields emitted by RPent's LIBERO Flywheel exporter.
        arrays["image"] = arrays["agentview"]
        arrays["wrist_image"] = arrays["wrist"]
        arrays["actions"] = arrays["action"]
        arrays["prompt"] = INSTRUCTION
        arrays["action_format"] = "libero.osc_pose.delta.v1"
        arrays["camera_order"] = "agentview,wrist"
        arrays["collector_image_orientation"] = "vertical_flip"
        arrays["state_schema"] = "eef_xyz_m,eef_rotvec_xyz_rad,gripper_qpos_2"
        arrays["action_range"] = np.asarray([-1.0, 1.0], dtype=np.float32)
        arrays["fps"] = np.int32(FPS)
        arrays["seed"] = np.int64(seed)
        arrays["init_state_index"] = np.int64(seed)
        arrays["initial_dressing_offset_xy"] = dressing_offset.astype(np.float32)
        arrays["approach_offset_above_grasp_m"] = np.float32(approach_offset_m)
        arrays["suite"] = suite_name
        arrays["suite_task_id"] = np.int64(task_id)
        arrays["suite_task_language"] = task_language
        arrays["task"] = TASK
        arrays["source_task_file"] = bddl_path.name
        arrays["language_instruction"] = INSTRUCTION
        arrays["success"] = np.bool_(success)
        arrays["initial_sim_state"] = initial_sim_state
        arrays["initial_dressing_pos"] = dressing.astype(np.float32)
        arrays["initial_basket_pos"] = basket.astype(np.float32)
        np.savez_compressed(output / "episode_000.npz", **arrays)
        for key, file_name in (("agentview", "agentview.mp4"), ("wrist", "wrist.mp4")):
            writer = cv2.VideoWriter(str(output / file_name), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
            for frame in data[key]:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        report = {
            "seed": seed,
            "suite": suite_name,
            "suite_task_id": task_id,
            "suite_task_language": task_language,
            "source_task_file": bddl_path.name,
            "init_state_file": init_path.name,
            "init_state_index": seed,
            "init_state_count": len(init_states),
            "initial_dressing_offset_xy": dressing_offset.tolist(),
            "dressing_jitter_half_range_m": dressing_jitter_m,
            "approach_offset_above_grasp_m": approach_offset_m,
            "approach_offset_range_m": [approach_offset_min_m, approach_offset_max_m],
            "approach_height_above_dressing_m": 0.04 + approach_offset_m,
            "approach_target_z_m": float(dressing[2] + 0.04 + approach_offset_m),
            "task": TASK,
            "language_instruction": INSTRUCTION,
            "success": success,
            "fps": FPS,
            "reset_settle_steps": reset_settle_steps,
            "samples": len(data["state"]),
            "duration_s": len(data["state"]) / FPS,
            "initial_dressing_pos": dressing.tolist(),
            "initial_basket_pos": basket.tolist(),
            "final_dressing_pos": final_object.tolist(),
            "elapsed_s": time.time() - stamp,
            "stages": stages,
            "action_contract": "the exact normalized 7-D OSC_POSE action passed to env.step at each recorded state",
            "action_format": "libero.osc_pose.delta.v1",
            "camera_order": "agentview,wrist",
            "collector_image_orientation": "vertical_flip; training adds horizontal flip for RPent 180-degree input",
            "action_range": [-1.0, 1.0],
            "state_contract": "eef xyz, eef rotation vector, and two gripper finger positions",
            "frame_alignment": "state/images are pre-action; terminated/truncated label the transition caused by that action",
            "post_success_tail": "the arm holds its placement pose while opening the gripper and recording the object settling into the basket, then retracts vertically",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        make_viewer(output, arrays, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        env.close()


def make_viewer(output: Path, arrays: dict, report: dict):
    ffmpeg = AUDIT / "venv" / "lib" / "python3.10" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg-linux-x86_64-v7.0.2"
    for view in ("agentview", "wrist"):
        subprocess.run([
            str(ffmpeg), "-y", "-i", str(output / f"{view}.mp4"), "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output / f"{view}_browser.mp4"),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LIBERO Episode Viewer</title>
<style>
:root{color-scheme:light;--ink:#1d2929;--muted:#657271;--line:#d7dfdc;--paper:#fff;--bg:#f2f5f3;--teal:#087f78;--orange:#d36b36}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif}
header{padding:18px 24px 14px;background:#fff;border-bottom:1px solid var(--line)}h1{font-size:20px;margin:0 0 5px}header p{margin:0;color:var(--muted)}main{max-width:1440px;padding:18px 24px 40px;margin:auto}
.views{display:flex;justify-content:center;gap:20px;align-items:flex-start}.view{margin:0;width:320px;flex:0 1 320px;min-width:0}.view figcaption{font-weight:650;margin-bottom:6px;text-align:center}.view video{width:320px;height:320px;max-width:100%;background:#151a19;object-fit:contain;display:block;margin:0 auto}
.transport{display:grid;grid-template-columns:auto auto minmax(80px,1fr) auto auto;gap:9px;align-items:center;padding:10px 0 16px}button{font:inherit;color:var(--ink);background:white;border:1px solid var(--line);border-radius:4px;padding:6px 11px;cursor:pointer}button:hover{border-color:var(--teal)}input[type=range]{width:100%;accent-color:var(--teal)}output{font-variant-numeric:tabular-nums;color:var(--muted);min-width:95px;text-align:right}.status{font-size:12px;color:var(--muted);min-height:17px}
h2{font-size:16px;margin:20px 0 8px}.group{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.chart{background:var(--paper);border:1px solid var(--line);padding:10px 11px 8px;min-width:0}.chart h3{font-size:13px;margin:0 0 5px;font-weight:650}.chart canvas{display:block;width:100%;height:155px}.legend{display:flex;flex-wrap:wrap;gap:4px 12px;color:var(--muted);font-size:11px}.legend span:before{content:"";display:inline-block;width:9px;height:3px;background:var(--c);vertical-align:middle;margin-right:4px}
@media(max-width:850px){main{padding:14px}.group{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:560px){header{padding:14px}.views,.group{grid-template-columns:1fr}.transport{grid-template-columns:auto auto 1fr auto}.transport output{grid-column:1/-1;text-align:left}.chart canvas{height:145px}}
</style></head><body><header><h1>Pick the salad dressing and place it in the basket</h1><p>Seed __SEED__ · __FPS__ Hz · __SAMPLES__ samples · __SUCCESS__</p></header>
<main><section class="views"><figure class="view"><figcaption>Agentview · 桌面全景</figcaption><video id="agent" muted playsinline preload="metadata" src="agentview_browser.mp4"></video></figure><figure class="view"><figcaption>Wrist · 手腕相机</figcaption><video id="wrist" muted playsinline preload="metadata" src="wrist_browser.mp4"></video></figure></section>
<div class="transport"><button id="play" title="播放/暂停">播放</button><button id="back" title="后退一帧">−1 帧</button><input id="seek" type="range" min="0" max="__LAST__" value="0"><button id="forward" title="前进一帧">+1 帧</button><output id="clock">00:00.00 / 00:00.00</output></div><div class="status" id="status"></div>
<h2>Robot state</h2><section class="group" id="states"></section><h2>Executed action · normalized OSC_POSE delta</h2><section class="group" id="actions"></section></main>
<script>
const S=__STATE__, A=__ACTION__, fps=__FPS__, count=S.length;
const videos=[document.querySelector('#agent'),document.querySelector('#wrist')],seek=document.querySelector('#seek'),clock=document.querySelector('#clock'),status=document.querySelector('#status');
const specs=[
 ['states',S,[{title:'End-effector position (m)',cols:[0,1,2],names:['x','y','z'],colors:['#087f78','#d36b36','#4969a5']},{title:'End-effector orientation (rad)',cols:[3,4,5],names:['rx','ry','rz'],colors:['#087f78','#d36b36','#4969a5']},{title:'Gripper finger position',cols:[6,7],names:['left','right'],colors:['#087f78','#d36b36']}]],
 ['actions',A,[{title:'Translation delta (normalized)',cols:[0,1,2],names:['dx','dy','dz'],colors:['#087f78','#d36b36','#4969a5']},{title:'Rotation delta (normalized)',cols:[3,4,5],names:['dRx','dRy','dRz'],colors:['#087f78','#d36b36','#4969a5']},{title:'Gripper command',cols:[6],names:['gripper'],colors:['#d36b36']}]]
];
function mmss(t){let m=Math.floor(t/60),s=(t%60).toFixed(2).padStart(5,'0');return String(m).padStart(2,'0')+':'+s}
function makeChart(root,data,spec){let box=document.createElement('article');box.className='chart';let title=document.createElement('h3');title.textContent=spec.title;let canvas=document.createElement('canvas');let legend=document.createElement('div');legend.className='legend';spec.names.forEach((n,i)=>{let e=document.createElement('span');e.textContent=n;e.style.setProperty('--c',spec.colors[i]);legend.append(e)});box.append(title,canvas,legend);root.append(box);let ctx=canvas.getContext('2d');
 function draw(cursor=-1){let w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio||1;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);let L=39,R=8,T=8,B=20,pw=w-L-R,ph=h-T-B;let values=spec.cols.flatMap(c=>data.map(r=>r[c]));let lo=Math.min(...values),hi=Math.max(...values);if(!isFinite(lo)||!isFinite(hi)){return}if(hi===lo){lo-=.5;hi+=.5}else{let p=(hi-lo)*.08;lo-=p;hi+=p}
 ctx.font='10px system-ui';ctx.textAlign='right';ctx.textBaseline='middle';for(let k=0;k<=4;k++){let y=T+ph*k/4,v=hi-(hi-lo)*k/4;ctx.strokeStyle='#e8edeb';ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(w-R,y);ctx.stroke();ctx.fillStyle='#687572';ctx.fillText(v.toFixed(2),L-5,y)}ctx.textAlign='center';ctx.textBaseline='top';for(let k=0;k<=4;k++){let i=Math.round((count-1)*k/4),x=L+pw*k/4;ctx.fillStyle='#687572';ctx.fillText(mmss(i/fps),x,h-B+4)}
 spec.cols.forEach((c,j)=>{ctx.strokeStyle=spec.colors[j];ctx.lineWidth=1.6;ctx.beginPath();data.forEach((r,i)=>{let x=L+pw*i/Math.max(1,count-1),y=T+ph*(hi-r[c])/(hi-lo);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke()});if(cursor>=0){let x=L+pw*cursor/Math.max(1,count-1);ctx.strokeStyle='#202b2a';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,T+ph);ctx.stroke();spec.cols.forEach((c,j)=>{let y=T+ph*(hi-data[cursor][c])/(hi-lo);ctx.fillStyle=spec.colors[j];ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()})}}
 canvas.addEventListener('pointermove',e=>{let rect=canvas.getBoundingClientRect(),i=Math.round(Math.max(0,Math.min(count-1,(e.clientX-rect.left-39)/(rect.width-47)*(count-1))));setFrame(i);});canvas.addEventListener('pointerleave',()=>draw(-1));canvas.addEventListener('click',e=>{let rect=canvas.getBoundingClientRect(),i=Math.round(Math.max(0,Math.min(count-1,(e.clientX-rect.left-39)/(rect.width-47)*(count-1))));setFrame(i)});return {draw};}
const charts=specs.flatMap(([id,data,items])=>items.map(s=>makeChart(document.getElementById(id),data,s)));
function setFrame(i){i=Math.max(0,Math.min(count-1,i));seek.value=i;let t=i/fps;videos.forEach(v=>{if(v.readyState>0&&Math.abs(v.currentTime-t)>0.06)v.currentTime=t});clock.textContent=mmss(t)+' / '+mmss((count-1)/fps);charts.forEach(c=>c.draw(i))}
seek.addEventListener('input',()=>setFrame(Number(seek.value)));document.querySelector('#back').onclick=()=>setFrame(Number(seek.value)-1);document.querySelector('#forward').onclick=()=>setFrame(Number(seek.value)+1);
let raf=0;async function play(){try{videos.forEach(v=>v.currentTime=Number(seek.value)/fps);await Promise.all(videos.map(v=>v.play()));document.querySelector('#play').textContent='暂停';const tick=()=>{if(videos[0].paused)return;let i=Math.min(count-1,Math.round(videos[0].currentTime*fps));seek.value=i;clock.textContent=mmss(i/fps)+' / '+mmss((count-1)/fps);charts.forEach(c=>c.draw(i));if(!videos[1].seeking&&Math.abs(videos[1].currentTime-videos[0].currentTime)>0.2){videos[1].currentTime=videos[0].currentTime}raf=requestAnimationFrame(tick)};cancelAnimationFrame(raf);tick()}catch(e){status.textContent='视频无法播放：'+e.message}}
function pause(){videos.forEach(v=>v.pause());cancelAnimationFrame(raf);document.querySelector('#play').textContent='播放'}document.querySelector('#play').onclick=()=>videos[0].paused?play():pause();videos.forEach(v=>v.addEventListener('ended',pause));videos.forEach(v=>v.addEventListener('error',()=>{status.textContent='视频加载失败，请确认浏览器版 MP4 文件存在。'}));window.addEventListener('resize',()=>charts.forEach(c=>c.draw(Number(seek.value))));setFrame(0);
</script></body></html>'''
    replacements = {
        "__SEED__": str(report["seed"]), "__FPS__": str(report["fps"]),
        "__SAMPLES__": str(report["samples"]), "__SUCCESS__": "success" if report["success"] else "not successful",
        "__LAST__": str(len(arrays["state"]) - 1),
        "__STATE__": json.dumps(arrays["state"].tolist(), separators=(",", ":")),
        "__ACTION__": json.dumps(arrays["action"].tolist(), separators=(",", ":")),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    (output / "viewer.html").write_text(template, encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark", choices=("standard", "pro"), default="standard")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dressing-jitter-m", type=float, default=0.02)
    parser.add_argument("--approach-offset-min-m", type=float, default=APPROACH_OFFSET_MIN_M)
    parser.add_argument("--approach-offset-max-m", type=float, default=APPROACH_OFFSET_MAX_M)
    args = parse_task_args(parser, "libero", "collection")
    default_name = "libero_pro_object_task_salad_dressing_seed0" if args.benchmark == "pro" else "libero_salad_dressing_official_init_seed0"
    collect(
        args.seed,
        args.output or ROOT / "data" / default_name,
        variant=args.benchmark,
        dressing_jitter_m=args.dressing_jitter_m,
        approach_offset_min_m=args.approach_offset_min_m,
        approach_offset_max_m=args.approach_offset_max_m,
    )

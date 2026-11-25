#!/usr/bin/env python3
"""Closed-loop LIBERO-Pro salad-dressing evaluation on training scenes 00..39.

Uses saved initial_sim_state, NOT recorded actions or oracle trajectories.
Run with the existing libero_venv Python (NumPy 1.26 / MuJoCo 2.3.7). See --help.
"""
from __future__ import annotations

import argparse
import dataclasses
import html
import json
import os
from pathlib import Path
import random
import site
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT.parent / "smolvla_rpent_bridge"
AUDIT = ROOT.parents[1] / "feasibility_audit"
DEFAULT_DATA = ROOT / "data/libero_pro_salad_dressing_seeds00_49_current"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, required=True, help="new, empty output directory")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-end", type=int, default=39, help="inclusive; default evaluates 40 training scenes")
    p.add_argument("--max-steps", type=int, default=400, help="20 Hz simulation steps, default 20 simulated seconds")
    p.add_argument("--random-seed", type=int, default=191)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--check-scenes", action="store_true", help="restore scenes and verify initial observations; no model required")
    args = p.parse_args()
    if not 0 <= args.seed_start <= args.seed_end <= 49 or args.max_steps < 1:
        p.error("require 0 <= seed-start <= seed-end <= 49 and max-steps > 0")
    if not args.check_scenes and (args.checkpoint is None or not args.checkpoint.is_file()):
        p.error("rollout requires --checkpoint pointing to a trained LIBERO OSC7 .pt file")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("output directory must be empty; use a new path to preserve previous results")
    return args


def state8(obs):
    from robosuite.utils.transform_utils import quat2axisangle
    rot = quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy())
    return np.r_[obs["robot0_eef_pos"], rot, obs["robot0_gripper_qpos"]].astype(np.float32)


def wire_obs(obs):
    """Same boundary as RPent -> VLA: raw images rotated 180 degrees."""
    return {
        "main_images": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])[None],
        "wrist_images": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])[None],
        "states": state8(obs)[None],
        "task_descriptions": ["pick up the salad dressing"],
    }


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def write_summary(args, rows):
    good = [r for r in rows if "error" not in r]
    rollout = [r for r in good if "task_success" in r]
    summary = {
        "mode": "scene_check" if args.check_scenes else "closed_loop_rollout",
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "data": str(args.data.resolve()), "requested": args.seed_end - args.seed_start + 1,
        "completed": len(rows), "errors": len(rows) - len(good),
        "grasp_successes": sum(r["grasp_success"] for r in rollout),
        "task_successes": sum(r["task_success"] for r in rollout),
        "released_in_basket_successes": sum(r["released_in_basket"] for r in rollout),
        # Errors count as failures in rates; never silently omit broken scenes.
        "task_success_rate": sum(r["task_success"] for r in rollout) / len(rows) if rollout else None,
        "grasp_success_rate": sum(r["grasp_success"] for r in rollout) / len(rows) if rollout else None,
        "fps": 20, "predict_chunk": 16, "execute_steps": 5, "denoising_steps": 10,
        "scope": "SmolVLA + actual bridge + LIBERO; no GPT planner or SAM3",
        "grasp_definition": "both-finger contact according to robosuite _check_grasp and object lifted >= 3cm for 3 consecutive steps",
        "task_definition": "LIBERO official check_success observed at any step",
        "release_definition": "official success and no gripper grasp for 5 consecutive steps",
        "episodes": rows,
    }
    save_json(args.output / "summary.json", summary)
    cards = []
    for r in rows:
        seed = r["seed"]
        status = ("ERROR: " + r["error"]) if "error" in r else (
            f'抓取 {r.get("grasp_success", "—")} · 任务 {r.get("task_success", "—")} · 松爪入篮 {r.get("released_in_basket", "—")}')
        video = f'<video controls preload="metadata" src="seed_{seed:02d}/rollout.mp4"></video>' if r.get("video") else ""
        cards.append(f'<article><h3>seed {seed:02d}</h3><p>{html.escape(status)}</p>{video}<p><a href="seed_{seed:02d}/result.json">详情</a></p></article>')
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>SmolVLA LIBERO 训练场景评测</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#f5f6f8}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px}article{background:white;padding:16px;border-radius:10px}video{width:100%;max-width:512px}</style>'
        f'<h1>SmolVLA · LIBERO Salad Dressing</h1><p>已完成 {len(rows)} / {summary["requested"]}；'
        f'抓取 {summary["grasp_successes"]}；任务成功 {summary["task_successes"]}；错误 {summary["errors"]}</p>'
        '<p>左：全景，右：腕部。训练场景闭环测试，未使用 GPT 规划器或 SAM3。</p><main>' + ''.join(cards) + '</main>', encoding="utf-8")
    return summary


def load_backend(args):
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint.get("state_dim"), checkpoint.get("action_dim")) != (8, 7) or not checkpoint.get("trainable"):
        raise ValueError("expected a LIBERO state8/action7 task checkpoint from train_smol_libero_osc7.py")
    contract = checkpoint.get("config", {}).get("contract", {})
    required = {"action": "libero.osc_pose.delta.v1", "instruction": "pick up the salad dressing",
                "fps": 20, "chunk_size": 16, "n_action_steps": 5, "num_steps": 10}
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(f"checkpoint contract {key}={contract.get(key)!r}, expected {expected!r}")
    del checkpoint
    sys.path.insert(0, str(BRIDGE))
    from bridge.backend import SmolVLABackend
    from bridge.profiles import Profile
    # Use the deployment profile, including checkpoint statistics and clipping.
    os.environ["SMOLVLA_OSC7_CHECKPOINT"] = str(args.checkpoint.resolve())
    profile = dataclasses.replace(Profile.load(BRIDGE / "profiles/osc7_template.json"), device=args.device)
    backend = SmolVLABackend(profile, log=lambda text: print(text, flush=True))
    backend.load()
    return backend


def evaluate_scene(args, seed, backend):
    from collect_libero_salad_dressing_delta import OffScreenRenderEnv, load_official_init_states
    directory = args.output / f"seed_{seed:02d}"
    directory.mkdir()
    source = args.data / f"seed_{seed:02d}" / "episode_000.npz"
    with np.load(source, allow_pickle=False) as data:
        if str(data["suite"].item()) != "libero_object_task" or str(data["prompt"].item()) != "pick up the salad dressing":
            raise ValueError(f"unexpected task or prompt in {source}")
        if int(data["seed"]) != seed or int(data["fps"]) != 20:
            raise ValueError(f"unexpected seed or control frequency in {source}")
        initial = np.asarray(data["initial_sim_state"], dtype=np.float64).copy()
        expected_state = data["state"][0].copy()
        expected_image = data["agentview"][0].copy()
        expected_wrist = data["wrist"][0].copy()
        expected_object = data["initial_dressing_pos"].copy()
        expected_basket = data["initial_basket_pos"].copy()
        initial_gripper_action = (
            np.asarray(data["initial_gripper_action"], dtype=np.float64).copy()
            if "initial_gripper_action" in data else None
        )
    if initial_gripper_action is None:
        # Legacy episodes omit the gripper's Python-side accumulator. Recreate
        # the collector's startup commands BEFORE restoring the saved scene.
        report = json.loads(source.with_name("report.json").read_text(encoding="utf-8"))
        startup_steps = int(report["reset_settle_steps"])
        if startup_steps < 1:
            raise ValueError("cannot reconstruct legacy gripper state without startup steps")
    _, _, _, bddl, _, _ = load_official_init_states("pro")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                             controller="OSC_POSE", control_freq=20, ignore_done=True)
    writer = None
    try:
        env.seed(seed)
        env.reset()
        gripper = env.env.robots[0].gripper
        if initial_gripper_action is None:
            neutral = np.r_[np.zeros(6, dtype=np.float32), np.float32(-1.0)]
            for _ in range(startup_steps):
                env.step(neutral)
            initial_gripper_action = np.asarray(gripper.current_action).copy()
        # This state is AFTER collection's settling phase. Do not settle again.
        obs = env.set_init_state(initial)
        gripper.current_action = initial_gripper_action.copy()
        state_error = float(np.max(np.abs(state8(obs) - expected_state)))
        image_mae = float(np.abs(obs["agentview_image"][::-1].astype(float) - expected_image).mean())
        wrist_mae = float(np.abs(obs["robot0_eye_in_hand_image"][::-1].astype(float) - expected_wrist).mean())
        # Recomputed robosuite observables can differ slightly from the cached
        # end-of-step sensors saved by the collector (about 2e-5 on seed 00).
        # Separately check object coordinates to catch a wrong scene/jitter.
        object_error = float(np.max(np.abs(obs["salad_dressing_1_pos"] - expected_object)))
        basket_error = float(np.max(np.abs(obs["basket_1_pos"] - expected_basket)))
        if state_error > 1e-4 or max(object_error, basket_error) > 1e-5:
            raise ValueError(f"restored state differs from recorded first frame: max error {state_error}")
        result = {"seed": seed, "source": str(source.resolve()), "restored_state_max_error": state_error,
                  "restored_agentview_pixel_mae": image_mae, "restored_wrist_pixel_mae": wrist_mae,
                  "restored_object_max_error_m": object_error, "restored_basket_max_error_m": basket_error,
                  "restored_gripper_current_action": gripper.current_action.tolist()}
        if args.check_scenes:
            return result
        import torch
        torch.manual_seed(args.random_seed + seed)
        np.random.seed(args.random_seed + seed)
        random.seed(args.random_seed + seed)
        backend.policy.reset()
        if not args.no_video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(directory / "rollout.mp4"), fps=20, codec="libx264", macro_block_size=None)
        def frame(current):
            if writer is not None:
                writer.append_data(np.concatenate([current["agentview_image"][::-1], current["robot0_eye_in_hand_image"][::-1]], axis=1).copy())
        frame(obs)
        base_z = float(obs["salad_dressing_1_pos"][2])
        task_success = grasp_success = released = False
        grasp_streak = release_streak = 0
        first_success = None
        states, actions, positions, contacts, successes, latency = [], [], [], [], [], []
        started = time.monotonic()
        for _ in range((args.max_steps + 4) // 5):
            before = time.monotonic()
            chunk = backend.predict(wire_obs(obs))
            latency.append(time.monotonic() - before)
            if chunk.shape != (1, 5, 7) or not np.isfinite(chunk).all() or np.abs(chunk).max() > 1.00001:
                raise ValueError(f"invalid bridge output {chunk.shape}")
            for action in chunk[0]:
                if len(actions) >= args.max_steps:
                    break
                states.append(state8(obs))
                actions.append(action.copy())
                obs, _, _, _ = env.step(action)
                frame(obs)
                contact = bool(env.env._check_grasp(env.env.robots[0].gripper, env.env.objects_dict["salad_dressing_1"]))
                pos = np.asarray(obs["salad_dressing_1_pos"], dtype=np.float32)
                lifted = float(pos[2]) - base_z >= 0.03
                grasp_streak = grasp_streak + 1 if contact and lifted else 0
                grasp_success |= grasp_streak >= 3
                success = bool(env.check_success())
                task_success |= success
                release_streak = release_streak + 1 if success and not contact else 0
                released |= release_streak >= 5
                if success and first_success is None:
                    first_success = len(actions)
                positions.append(pos.copy()); contacts.append(contact); successes.append(success)
                # Allow one simulated second for policy-controlled release/settling after official success.
                if first_success is not None and len(actions) - first_success >= 20:
                    break
            if first_success is not None and len(actions) - first_success >= 20:
                break
        np.savez_compressed(directory / "rollout.npz", state=np.asarray(states), actions=np.asarray(actions),
                            object_pos=np.asarray(positions), grasp_contact=np.asarray(contacts),
                            official_success=np.asarray(successes), initial_sim_state=initial,
                            timestamps=np.arange(len(actions)) / 20.0)
        result.update(grasp_success=bool(grasp_success), task_success=bool(task_success),
                      released_in_basket=bool(released), official_success_final=bool(successes[-1]),
                      first_success_step=first_success, steps=len(actions), simulated_seconds=len(actions)/20,
                      wall_seconds=time.monotonic()-started, mean_inference_seconds=float(np.mean(latency)),
                      video=not args.no_video)
        return result
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    args = arguments()
    if int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError("LIBERO's installed robosuite requires the existing libero_venv NumPy 1.26 runtime; run /home/hzy/git/learn/feasibility_audit/libero_venv/bin/python")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Match the existing training runtime; do not create/install an environment.
    # NumPy is already imported from libero_venv. Prefer the model runtime's
    # Transformers/Torch, rather than libero_venv's older Transformers.
    sys.path.insert(0, str(AUDIT / "followup_vlm_venv/lib/python3.10/site-packages"))
    # The training venv inherits these locations, while libero_venv is isolated.
    # Torch and requests need typing_extensions / urllib3 from these paths.
    for package_dir in (site.getusersitepackages(), "/usr/lib/python3/dist-packages"):
        if package_dir not in sys.path:
            sys.path.append(package_dir)
    for name in ("venv", "libero_venv"):
        sys.path.append(str(AUDIT / name / "lib/python3.10/site-packages"))
    for seed in range(args.seed_start, args.seed_end + 1):
        path = args.data / f"seed_{seed:02d}/episode_000.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    backend = None
    rows = []
    try:
        if not args.check_scenes:
            backend = load_backend(args)
        for seed in range(args.seed_start, args.seed_end + 1):
            try:
                result = evaluate_scene(args, seed, backend)
            except Exception as exc:
                traceback.print_exc()
                result = {"seed": seed, "error": str(exc), "traceback": traceback.format_exc()}
            directory = args.output / f"seed_{seed:02d}"
            directory.mkdir(exist_ok=True)
            save_json(directory / "result.json", result)
            rows.append(result)
            write_summary(args, rows)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        summary = write_summary(args, rows)
        if backend is not None:
            backend.close()
    print(f"Report: {args.output / 'index.html'}", flush=True)
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

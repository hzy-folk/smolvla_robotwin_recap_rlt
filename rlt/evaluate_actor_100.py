#!/usr/bin/env python3
"""Evaluate a trained RLT actor on all 100 recorded RoboTwin episodes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parent
SMOL_ROOT = ROOT.parent
DATA_ROOT = SMOL_ROOT / "data/robotwin_success_qpos16_formal100"
DEFAULT_BASELINE = ROOT.parent / "rlt_runs/legacy_eval_exact_rlt_stage1_80_20261114/summary.json"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import path as runtime_path
ROBOTWIN_PYTHON = runtime_path('robotwin_python')
INFER = ROOT / "legacy_collect/scripts/infer_smol_robotwin_qpos16.py"
GALLERY = SMOL_ROOT / "recap/generate_rlt_actor_100_gallery.py"


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(path)


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="SmolVLA checkpoint used as the base policy")
    parser.add_argument("--token", type=Path, required=True, help="trained RL-token checkpoint")
    parser.add_argument("--actor-snapshot", type=Path, required=True,
                        help="trained actor snapshot, normally the final stage-3 actor_snapshot.pkl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/robotwin_piper.yaml")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                        help="stage-1 80-episode summary used for per-episode comparison")
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True,
                        help="new evaluation output directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--actor-deterministic", action="store_true",
                        help="evaluate the actor mean action without exploration noise")
    parser.add_argument("--no-video", action="store_true",
                        help="skip writing camera videos during rollout")
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all",
                        help="evaluate all manifest episodes or one explicit dataset split")
    parser.add_argument("--task-prompt", default="stack the red block and then the green block.")
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=376)
    parser.add_argument("--max-episodes", type=int,
                        help="limit for smoke tests; full evaluation omits this")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gallery-name", default="robotwin_rlt_actor_final_eval100.html")
    args = parser.parse_args()

    checkpoint, token = args.checkpoint.resolve(), args.token.resolve()
    actor_snapshot, config = args.actor_snapshot.resolve(), args.config.resolve()
    baseline_path, data = args.baseline.resolve(), args.data.resolve()
    output = args.output.resolve()
    for path in (checkpoint, token, actor_snapshot, config, baseline_path, data / "manifest.json",
                 ROBOTWIN_PYTHON, INFER, GALLERY):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = load_json(data / "manifest.json")
    all_entries = list(manifest["episodes"])
    all_train = [row for row in all_entries if row.get("split") == "train"]
    all_validation = [row for row in all_entries if row.get("split") == "validation"]
    if len(all_entries) != 100 or len(all_train) != 80 or len(all_validation) != 20:
        raise RuntimeError(f"expected manifest split 80 train + 20 validation, got {len(all_train)} + {len(all_validation)}")
    entries = all_entries if args.split == "all" else [row for row in all_entries if row.get("split") == args.split]
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            raise ValueError("--max-episodes must be positive")
        entries = entries[:args.max_episodes]
    train = [row for row in entries if row.get("split") == "train"]
    validation = [row for row in entries if row.get("split") == "validation"]
    for row in entries:
        if not (data / row["npz"]).is_file():
            raise FileNotFoundError(data / row["npz"])

    baseline = load_json(baseline_path)
    if baseline.get("status") != "completed" or baseline.get("total") != 80:
        raise RuntimeError(f"baseline must be the completed 80-episode stage-1 run: {baseline_path}")
    baseline_rows = {row["source_episode"]: bool(row["official_success"])
                     for row in baseline["episodes"] if row.get("completed")}
    train_ids = {row["id"] for row in all_train}
    if len(baseline_rows) != 80 or set(baseline_rows) != train_ids:
        raise RuntimeError("stage-1 baseline IDs do not exactly match the current 80-episode train split")

    selection = {
        "checkpoint": file_identity(checkpoint), "token": file_identity(token),
        "actor_snapshot": file_identity(actor_snapshot), "config": file_identity(config),
        "baseline": file_identity(baseline_path), "manifest": file_identity(data / "manifest.json"),
        "data_root": str(data), "split": args.split, "entries": entries, "device": args.device,
        "task_prompt": args.task_prompt, "flow_steps": args.flow_steps,
        "rollout_steps": args.rollout_steps, "actor_deterministic": args.actor_deterministic,
        "no_video": args.no_video,
        "gallery_name": args.gallery_name,
    }
    selection_path = output / "selection.json"
    if args.resume:
        if not selection_path.is_file() or load_json(selection_path) != selection:
            raise RuntimeError(f"resume inputs differ from original evaluation selection: {selection_path}")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}; use --resume for this run")
        output.mkdir(parents=True, exist_ok=True)
        write_json(selection_path, selection)

    rollouts = output / "rollouts"
    rollouts.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    prior = load_json(summary_path) if args.resume and summary_path.is_file() else {}
    prior_rows = {row["source_episode"]: row for row in prior.get("episodes", [])}
    attempts = list(prior.get("attempts", []))
    start = time.time()

    def publish(status: str) -> None:
        rows = [prior_rows[row["id"]] for row in entries if row["id"] in prior_rows]
        completed = [row for row in rows if row.get("completed")]
        successes = sum(bool(row.get("official_success")) for row in completed)
        train_rows = [row for row in completed if row.get("split") == "train"]
        val_rows = [row for row in completed if row.get("split") == "validation"]
        train_counts = {"baseline_success_actor_success": 0, "baseline_success_actor_failure": 0,
                        "baseline_failure_actor_success": 0, "baseline_failure_actor_failure": 0}
        for row in train_rows:
            key = f"baseline_{'success' if row['baseline_success'] else 'failure'}_actor_{'success' if row['official_success'] else 'failure'}"
            train_counts[key] += 1
        payload = {
            "status": status, "total": len(entries), "finished": len(rows),
            "completed": len(completed), "errors": sum(not row.get("completed") for row in rows),
            "successes": successes, "success_rate": successes / len(completed) if completed else None,
            "train": {"total": len(train_rows), "successes": sum(bool(r["official_success"]) for r in train_rows),
                      "success_rate": (sum(bool(r["official_success"]) for r in train_rows) / len(train_rows)) if train_rows else None,
                      "baseline_successes": sum(bool(r["baseline_success"]) for r in train_rows),
                      "transitions": train_counts},
            "validation": {"total": len(val_rows), "successes": sum(bool(r["official_success"]) for r in val_rows),
                           "success_rate": (sum(bool(r["official_success"]) for r in val_rows) / len(val_rows)) if val_rows else None},
            "baseline_summary": str(baseline_path), "episodes": rows, "attempts": attempts,
            "elapsed_seconds": time.time() - start,
        }
        write_json(summary_path, payload)
        rate = f"{successes}/{len(completed)}" if completed else "0/0"
        (output / "评测进度.md").write_text(
            f"# RLT Actor 100 集评测\n\n- 状态：{status}\n- 完成：{len(completed)}/{len(entries)}\n"
            f"- 当前成功：{rate}\n- 训练集完成：{len(train_rows)}/80\n"
            f"- 验证集完成：{len(val_rows)}/20\n- 运行错误：{payload['errors']}\n",
            encoding="utf-8")

    for index, entry in enumerate(entries):
        episode_id = entry["id"]
        previous = prior_rows.get(episode_id)
        if previous and previous.get("completed"):
            folder = rollouts / previous["id"]
            if (folder / "report.json").is_file():
                continue

        started = time.time()
        report = {"completed": False, "error": "rollout not attempted"}
        versions = []
        process = None
        rollout_id = f"episode_{index:03d}"
        for retry_attempt in range(4):
            if retry_attempt:
                rollout_id = f"episode_{index:03d}_retry_{retry_attempt:02d}"
            while (rollouts / rollout_id).exists():
                retry_attempt += 1
                rollout_id = f"episode_{index:03d}_retry_{retry_attempt:02d}"
            episode_dir = rollouts / rollout_id
            command = [str(ROBOTWIN_PYTHON), str(INFER), "--mode", "rollout",
                       "--raw", str((data / entry["npz"]).resolve()),
                       "--checkpoint", str(checkpoint), "--output", str(episode_dir),
                       "--device", args.device, "--flow-steps", str(args.flow_steps),
                       "--rollout-steps", str(args.rollout_steps), "--check-success-every-step",
                       "--rlt-token", str(token), "--rlt-config", str(config),
                       "--actor-snapshot", str(actor_snapshot),
                       "--task-prompt", args.task_prompt]
            if args.actor_deterministic:
                command.append("--actor-deterministic")
            if args.no_video:
                command.append("--no-video")
            log_path = output / f"{rollout_id}.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=SMOL_ROOT, stdout=log, stderr=subprocess.STDOUT)
            report_path = episode_dir / "report.json"
            report = load_json(report_path) if report_path.is_file() else {
                "completed": False, "error": "missing report.json",
            }
            versions = report.get("actor_versions", [])
            completed = bool(report.get("completed")) and process.returncode == 0
            if completed and (not versions or min(versions) < 1):
                completed = False
                report["error"] = f"actor was not active throughout rollout: versions={versions}"
            if completed:
                break
            if retry_attempt < 3:
                print(json.dumps({"retrying": entry["id"], "attempt": retry_attempt + 2,
                                  "error": report.get("error"), "exit_code": process.returncode},
                                 ensure_ascii=False), flush=True)
        baseline_success = baseline_rows.get(episode_id)
        row = {
            "id": rollout_id, "source_episode": episode_id, "seed": entry.get("seed"),
            "split": entry["split"], "baseline_success": baseline_success,
            "official_success": report.get("official_success") if completed else None,
            "completed": completed, "steps": report.get("steps"),
            "simulated_seconds": report.get("simulated_seconds", 0),
            "wall_seconds": time.time() - started, "exit_code": process.returncode,
            "actor_versions": versions, "error": report.get("error"),
        }
        if entry["split"] == "train":
            row["comparison_label"] = (
                f"阶段一{'成功' if baseline_success else '失败'} → Actor"
                f"{'成功' if row['official_success'] else '失败' if completed else '错误'}"
            )
        prior_rows[episode_id] = row
        attempts.append({"source_episode": episode_id, "id": rollout_id,
                         "completed": completed, "exit_code": process.returncode,
                         "error": report.get("error")})
        publish("running")
        print(json.dumps({"finished": len(prior_rows), "total": len(entries),
                          "episode": episode_id, "split": entry["split"],
                          "success": row["official_success"], "completed": completed},
                         ensure_ascii=False), flush=True)

    all_completed = len(prior_rows) == len(entries) and all(
        prior_rows[row["id"]].get("completed") for row in entries)
    publish("completed" if all_completed else "incomplete")
    if not all_completed:
        raise SystemExit("evaluation has failed episodes; inspect summary/logs and rerun with --resume")

    if args.max_episodes is None and args.split in ("all", "train"):
        subprocess.run([str(ROBOTWIN_PYTHON), str(GALLERY), "--run", str(output),
                        "--output", args.gallery_name], cwd=SMOL_ROOT, check=True)
        print(f"Gallery: {SMOL_ROOT / 'recap_runs/visualizations' / args.gallery_name}", flush=True)
    print(f"Evaluation complete. Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

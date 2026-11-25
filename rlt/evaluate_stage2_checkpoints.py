#!/usr/bin/env python3
"""Evaluate each saved Stage-2 actor snapshot on the same deterministic 80 train episodes."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rlt.config import load_config

ROOT = Path(__file__).resolve().parent
SMOL_ROOT = ROOT.parent
DEFAULT_RUN = SMOL_ROOT / "rlt_runs/legacy_eval_exact_rlt_stage1_80_nstep5_batch256_bc10_20261114"
DEFAULT_SFT = SMOL_ROOT / "recap_runs/formal_success100_batch56_24000_original_prompt/checkpoint_024000.pt"
DEFAULT_TOKEN = SMOL_ROOT / "rlt_runs/formal_success100_batch56_24000_continued_10000/rlt_token_512_continued_step_025000.pt"
DEFAULT_DATA = SMOL_ROOT / "data/robotwin_success_qpos16_formal100"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import path as runtime_path
PYTHON = runtime_path('model_python')
EXPECTED_STEPS = (3305, 6610, 9915, 13220, 16525)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"stage2_learner_step_(\d+)\.pt", path.name)
    if not match:
        raise ValueError(f"unexpected Stage-2 checkpoint filename: {path.name}")
    return int(match.group(1))


def make_actor_snapshot(checkpoint: Path, destination: Path, config_path: Path) -> dict:
    step = checkpoint_step(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step_count", -1)) != step:
        raise RuntimeError(f"checkpoint step mismatch: filename={step}, state={payload.get('step_count')}")
    version = int(payload.get("actor_version", 0))
    actor = payload.get("actor")
    if version < 1 or not isinstance(actor, dict) or not actor:
        raise RuntimeError(f"checkpoint has no trained actor state: {checkpoint}")
    config = load_config(config_path)
    snapshot = {
        "version": version,
        "z_dim": config.token.embed_dim * config.token.num_tokens,
        "proprio_dim": config.data.proprio_dim,
        "chunk_len": config.data.chunk_len,
        "action_dim": config.data.action_dim,
        "actor": actor,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_step": step,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        if (existing.get("source_checkpoint") != snapshot["source_checkpoint"]
                or int(existing.get("source_step", -1)) != step
                or int(existing.get("version", -1)) != version):
            raise RuntimeError(f"existing actor snapshot does not match checkpoint: {destination}")
    else:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(snapshot, temporary)
        temporary.replace(destination)
    return {"step": step, "actor_version": version, "snapshot": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/robotwin_piper_bc10.yaml")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline", type=Path,
                        help="completed 80-episode Stage-1 baseline summary; defaults to RUN_DIR/summary.json")
    parser.add_argument("--output-root", type=Path,
                        help="evaluation output root; defaults to RUN_DIR/stage2_checkpoint_eval/deterministic80_train")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-only", action="store_true", help="test one episode with the first checkpoint")
    parser.add_argument("--steps", type=int, nargs="+", help="explicit learner checkpoint steps; defaults to the standard five")
    parser.add_argument("--summary-doc", type=Path, help="also publish live progress into this aggregate Markdown document")
    args = parser.parse_args()

    run = args.run_dir.resolve()
    checkpoint_dir = run / "checkpoints"
    output_root = (args.output_root.resolve() if args.output_root else
                   run / "stage2_checkpoint_eval/deterministic80_train")
    actor_dir = output_root / "actor_snapshots"
    baseline = args.baseline.resolve() if args.baseline else run / "summary.json"
    config = args.config.resolve()
    eval_config = load_config(config)
    source_checkpoint = args.checkpoint.resolve()
    token = args.token.resolve()
    for path in (PYTHON, source_checkpoint, token, config, args.data / "manifest.json", baseline):
        if not path.is_file():
            raise FileNotFoundError(path)
    status = read_json(run / "stage_status.json")
    steps = tuple(args.steps) if args.steps else EXPECTED_STEPS
    if not steps or len(set(steps)) != len(steps) or any(step < 1 for step in steps):
        raise ValueError(f"invalid requested checkpoint steps: {steps}")
    completed_steps = int(status.get("learner_steps", -1))
    if status.get("status") != "complete" or completed_steps < max(steps):
        raise RuntimeError(f"Stage-2 run is not complete at the expected update count: {status}")
    checkpoints = sorted(checkpoint_dir.glob("stage2_learner_step_*.pt"), key=checkpoint_step)
    available_steps = tuple(checkpoint_step(path) for path in checkpoints)
    if args.steps:
        missing_steps = [step for step in steps if step not in available_steps]
        if missing_steps:
            raise RuntimeError(f"requested checkpoints are missing: {missing_steps}; found {available_steps}")
    elif available_steps != EXPECTED_STEPS:
        raise RuntimeError(f"expected checkpoints at {EXPECTED_STEPS}, found {[p.name for p in checkpoints]}")
    baseline_summary = read_json(baseline)
    if baseline_summary.get("status") != "completed" or baseline_summary.get("total") != 80:
        raise RuntimeError("the matching 80-episode Stage-1 baseline is missing or incomplete")
    if args.smoke_only:
        output_root = run / "stage2_checkpoint_eval/smoke_first_checkpoint"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.smoke_only and any(output_root.iterdir()):
        raise FileExistsError(f"smoke output is not empty: {output_root}")

    results: list[dict] = []
    progress_path = output_root / "评测进度.md"
    overall_path = output_root / "summary.json"
    if args.smoke_only:
        steps = steps[:1]

    def publish_summary_doc(state: str) -> None:
        if args.summary_doc is None:
            return
        doc = args.summary_doc.resolve()
        begin = "<!-- BEGIN CONTINUATION CHECKPOINT EVALUATION -->"
        end = "<!-- END CONTINUATION CHECKPOINT EVALUATION -->"
        status_zh = {"running": "进行中", "completed": "已完成", "failed": "失败"}.get(state, state)
        block = [begin, "", "### BC=5000、n-step=3 续训检查点（80 集确定性评测）", "",
                 f"- 状态：{status_zh}，{len(results)}/{len(steps)} 个检查点完成",
                 f"- 检查点：{', '.join(f'{step:,}' for step in steps)} 步",
                 f"- 评测目录：`{output_root}`"]
        block.extend(f"- {item['step']:,} 步：{item['successes']}/80 成功 ({item['success_rate']:.1%})"
                     for item in results)
        if state == "running" and len(results) < len(steps):
            block.append(f"- 当前检查点：{steps[len(results)]:,} 步")
        block.extend(["", end])
        text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
        if begin in text and end in text:
            start_index = text.index(begin)
            end_index = text.index(end, start_index) + len(end)
            text = text[:start_index] + "\n".join(block) + text[end_index:]
        else:
            text = text.rstrip() + ("\n\n" if text.strip() else "") + "\n".join(block) + "\n"
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(text, encoding="utf-8")

    def publish(current: int | None, state: str) -> None:
        payload = {
            "status": state,
            "completed_checkpoints": len(results),
            "total_checkpoints": len(steps),
            "current_step": current,
            "results": results,
            "updated_at": time.time(),
        }
        write_json(overall_path, payload)
        lines = ["# Stage 2 检查点确定性 80 集评测", "", f"- 状态：{state}",
                 f"- 检查点进度：{len(results)}/{len(steps)}",
                 (f"- BC 权重：前 {eval_config.objective.warmup_updates:,} 次 learner 更新 "
                  f"{eval_config.objective.warmup_bc_weight:g}，之后 {eval_config.objective.bc_weight:g}")]
        for item in results:
            lines.append(f"- {item['step']} 步：{item['successes']}/80 ({item['success_rate']:.1%})")
        if current is not None:
            lines.append(f"- 当前：{current} 步检查点评测中")
        progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        publish_summary_doc(state)

    publish(None, "running")
    for step in steps:
        checkpoint = checkpoint_dir / f"stage2_learner_step_{step:09d}.pt"
        snapshot = make_actor_snapshot(checkpoint, actor_dir / f"actor_step_{step:09d}.pkl", config)
        output = output_root / f"step_{step:09d}"
        publish(step, "running")
        command = [
            str(PYTHON), "-u", str(ROOT / "evaluate_actor_100.py"),
            "--checkpoint", str(source_checkpoint), "--token", str(token),
            "--actor-snapshot", snapshot["snapshot"], "--config", str(config),
            "--baseline", str(baseline), "--data", str(args.data.resolve()),
            "--output", str(output), "--device", args.device,
            "--actor-deterministic", "--no-video", "--split", "train",
        ]
        if args.smoke_only:
            command.extend(["--max-episodes", "1"])
        selection_path = output / "selection.json"
        if selection_path.is_file():
            command.append("--resume")
        log_path = output_root / f"step_{step:09d}.log"
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=SMOL_ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            publish(step, "failed")
            raise RuntimeError(f"checkpoint evaluation failed for step {step}; see {log_path}")
        summary = read_json(output / "summary.json")
        if summary.get("status") != "completed" or summary.get("completed") != (1 if args.smoke_only else 80):
            publish(step, "failed")
            raise RuntimeError(f"incomplete checkpoint evaluation at step {step}: {summary}")
        if summary.get("errors", 0):
            publish(step, "failed")
            raise RuntimeError(f"checkpoint evaluation had rollout errors at step {step}: {summary}")
        if args.smoke_only:
            test_row = summary["episodes"][0]
            if not test_row.get("actor_versions") or min(test_row["actor_versions"]) != snapshot["actor_version"]:
                publish(step, "failed")
                raise RuntimeError(f"actor checkpoint version mismatch: {test_row}")
        result = {
            "step": step,
            "actor_version": snapshot["actor_version"],
            "successes": summary["train"]["successes"],
            "success_rate": summary["train"]["success_rate"],
            "episodes": summary["train"]["total"],
            "summary": str(output / "summary.json"),
            "log": str(log_path),
        }
        results.append(result)
        publish(None, "running")
        print(json.dumps(result, ensure_ascii=False), flush=True)

    publish(None, "completed")


if __name__ == "__main__":
    main()

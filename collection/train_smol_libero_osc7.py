#!/usr/bin/env python3
"""Fine-tune SmolVLA on LIBERO 8-D state / 7-D OSC-delta episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collection_config import parse_task_args

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from dataset import MODEL_DIR, bootstrap_model
bootstrap_model()
from smol_model import make, torch  # noqa: E402


FPS = 20
CHUNK_SIZE = 16
N_ACTION_STEPS = 5
INFERENCE_DENOISING_STEPS = 10
STATE_DIM = 8
ACTION_DIM = 7
ACTION_FORMAT = "libero.osc_pose.delta.v1"
INSTRUCTION = "pick up the salad dressing"
CAMERA_KEYS = ("image", "wrist_image")
BASE_MODEL = MODEL_DIR / "model.safetensors"


def _scalar(data, key, default=None):
    if key not in data:
        return default
    value = np.asarray(data[key])
    return value.item() if value.ndim == 0 else value


def _instruction(value: object) -> str:
    text = str(value).strip().removesuffix(".").rstrip()
    if not text:
        raise ValueError("episode instruction is empty")
    return text.rstrip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_episodes(sources: list[Path]) -> list[Path]:
    found = set()
    for source in sources:
        source = source.expanduser().resolve()
        if source.is_file() and source.suffix == ".npz":
            found.add(source)
        elif source.is_dir():
            found.update(path.resolve() for path in source.rglob("episode_*.npz"))
        else:
            raise FileNotFoundError(f"data source does not exist: {source}")
    paths = sorted(found)
    if not paths:
        raise FileNotFoundError("no episode_*.npz files found in --data sources")
    return paths


@dataclass
class Episode:
    path: Path
    cache_dir: Path
    start: int
    stop: int
    instruction: str
    arrays: dict[str, np.ndarray]


class LiberoDataset:
    """Validated, per-episode mmap cache with episode-isolated action chunks."""

    def __init__(self, sources: list[Path], cache: Path):
        self.cache = cache.expanduser().resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.episodes: list[Episode] = []
        self.fingerprint_rows = []
        cursor = 0
        for path in find_episodes(sources):
            digest = _sha256(path)
            cache_dir = self.cache / digest[:20]
            arrays, instruction = self._load_episode(path, cache_dir)
            count = len(arrays["state"])
            self.episodes.append(
                Episode(path, cache_dir, cursor, cursor + count, instruction, arrays)
            )
            self.fingerprint_rows.append({"path": str(path), "sha256": digest, "frames": count})
            cursor += count

        instructions = {episode.instruction for episode in self.episodes}
        if len(instructions) != 1:
            raise ValueError(
                "all episodes in one training run must use the same instruction; "
                f"found {sorted(instructions)}"
            )
        self.instruction = next(iter(instructions))
        self.frame_count = cursor
        self.ends = np.asarray([episode.stop for episode in self.episodes], dtype=np.int64)
        self.fingerprint = hashlib.sha256(
            json.dumps(self.fingerprint_rows, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _load_episode(path: Path, cache_dir: Path):
        with np.load(path, allow_pickle=False) as data:
            state = np.asarray(data["state"], dtype=np.float32)
            action_key = "actions" if "actions" in data else "action"
            action = np.asarray(data[action_key], dtype=np.float32)
            image = np.asarray(data["image"] if "image" in data else data["agentview"])
            wrist = np.asarray(
                data["wrist_image"] if "wrist_image" in data else data["wrist"]
            )
            instruction = _instruction(
                _scalar(data, "prompt", _scalar(data, "language_instruction", ""))
            )
            success = bool(_scalar(data, "success", False))
            fps = int(_scalar(data, "fps", -1))
            action_format = str(_scalar(data, "action_format", ""))
            camera_order = str(_scalar(data, "camera_order", "agentview,wrist"))
            image_orientation = str(
                _scalar(data, "collector_image_orientation", "vertical_flip")
            )
            state_schema = str(
                _scalar(data, "state_schema", "eef_xyz_m,eef_rotvec_xyz_rad,gripper_qpos_2")
            )
            action_range = np.asarray(
                _scalar(data, "action_range", np.asarray([-1.0, 1.0])), dtype=np.float32
            )
            timestamps = np.asarray(data["timestamps"], dtype=np.float64) if "timestamps" in data else None

        count = len(state)
        if state.shape != (count, STATE_DIM) or action.shape != (count, ACTION_DIM):
            raise ValueError(
                f"{path}: expected state (N,8) and action (N,7), got "
                f"{state.shape} and {action.shape}"
            )
        for name, frames in (("image", image), ("wrist_image", wrist)):
            if frames.shape != (count, 256, 256, 3) or frames.dtype != np.uint8:
                raise ValueError(
                    f"{path}: {name} must be uint8 (N,256,256,3), got "
                    f"{frames.shape} {frames.dtype}"
                )
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: state/action contains NaN or Inf")
        if count > 1:
            rotvec_step = np.linalg.norm(np.diff(state[:, 3:6], axis=0), axis=1)
            discontinuities = np.flatnonzero(rotvec_step > np.pi)
            if discontinuities.size:
                index = int(discontinuities[0])
                raise ValueError(
                    f"{path}: axis-angle state jumps by {rotvec_step[index]:.3f} rad "
                    f"between frames {index} and {index + 1}; recollect with "
                    "collect_libero_salad_dressing_delta.py using robosuite's "
                    "quaternion conversion"
                )
        if np.any(np.abs(action) > 1.00001):
            raise ValueError(f"{path}: normalized OSC action exceeds [-1,1]")
        if not success:
            raise ValueError(f"{path}: unsuccessful episodes are excluded from this trainer")
        if fps != FPS:
            raise ValueError(f"{path}: expected {FPS} Hz, got {fps}")
        if action_format != ACTION_FORMAT:
            raise ValueError(
                f"{path}: expected action_format={ACTION_FORMAT!r}, got {action_format!r}"
            )
        if camera_order != "agentview,wrist":
            raise ValueError(
                f"{path}: expected camera_order='agentview,wrist', got {camera_order!r}"
            )
        if image_orientation != "vertical_flip":
            raise ValueError(
                f"{path}: expected collector_image_orientation='vertical_flip', "
                f"got {image_orientation!r}"
            )
        if state_schema != "eef_xyz_m,eef_rotvec_xyz_rad,gripper_qpos_2":
            raise ValueError(f"{path}: unsupported state_schema={state_schema!r}")
        if action_range.shape != (2,) or not np.array_equal(action_range, [-1.0, 1.0]):
            raise ValueError(f"{path}: expected action_range metadata [-1,1], got {action_range}")
        if timestamps is not None and (
            timestamps.shape != (count,)
            or (count > 1 and not np.allclose(np.diff(timestamps), 1 / FPS, atol=1e-6))
        ):
            raise ValueError(f"{path}: timestamps do not describe a {FPS} Hz sequence")

        cache_dir.mkdir(parents=True, exist_ok=True)
        source_arrays = {"state": state, "action": action, "image": image, "wrist_image": wrist}
        cached = {}
        for key, value in source_arrays.items():
            target = cache_dir / f"{key}.npy"
            if not target.is_file():
                temp = target.with_suffix(".npy.tmp")
                with temp.open("wb") as stream:
                    np.save(stream, value, allow_pickle=False)
                temp.replace(target)
            cached[key] = np.load(target, mmap_mode="r", allow_pickle=False)
        return cached, instruction

    def split_episodes(self, seed: int, validation_fraction: float):
        if len(self.episodes) < 2:
            raise ValueError("formal training needs at least 2 successful episodes for episode-level validation")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation-fraction must be between 0 and 1")
        order = np.random.default_rng(seed).permutation(len(self.episodes))
        val_count = min(len(order) - 1, max(1, round(len(order) * validation_fraction)))
        val_ids = set(int(i) for i in order[:val_count])
        train_ids = set(range(len(self.episodes))) - val_ids
        train_frames = self._global_indices(train_ids)
        val_frames = self._global_indices(val_ids)
        return train_ids, val_ids, train_frames, val_frames

    def split_seed_00_39_40_49(self):
        """Use the requested fixed seed split; refuse ambiguous episode ordering."""
        expected = [f"seed_{seed:02d}" for seed in range(50)]
        actual = [episode.path.parent.name for episode in self.episodes]
        if actual != expected:
            raise ValueError(
                "--split-mode seed-00-39 requires exactly seed_00..seed_49 "
                "directories in the dataset"
            )
        train_ids = set(range(40))
        val_ids = set(range(40, 50))
        return (
            train_ids,
            val_ids,
            self._global_indices(train_ids),
            self._global_indices(val_ids),
        )

    def _global_indices(self, episode_ids: set[int]) -> np.ndarray:
        return np.concatenate(
            [
                np.arange(self.episodes[i].start, self.episodes[i].stop, dtype=np.int64)
                for i in sorted(episode_ids)
            ]
        )

    def _episode_indices(self, indices: np.ndarray):
        ids = np.searchsorted(self.ends, indices, side="right")
        local = indices - np.asarray([self.episodes[i].start for i in ids])
        return ids, local

    def frames(self, key: str, indices: np.ndarray) -> np.ndarray:
        ids, local = self._episode_indices(np.asarray(indices, dtype=np.int64))
        return np.stack([self.episodes[i].arrays[key][j] for i, j in zip(ids, local)])

    def action_chunks(self, starts: np.ndarray) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.int64)
        ids, _ = self._episode_indices(starts)
        target_ids = np.minimum(
            starts[:, None] + np.arange(CHUNK_SIZE, dtype=np.int64),
            self.ends[ids, None] - 1,
        )
        flat = target_ids.reshape(-1)
        episode_ids, local = self._episode_indices(flat)
        values = np.stack(
            [self.episodes[i].arrays["action"][j] for i, j in zip(episode_ids, local)]
        )
        return values.reshape(len(starts), CHUNK_SIZE, ACTION_DIM)

    def statistics(self, train_episode_ids: set[int]):
        stats = {}
        for key, stat_key, dim in (
            ("state", "observation.state", STATE_DIM),
            ("action", "action", ACTION_DIM),
        ):
            total = np.zeros(dim, dtype=np.float64)
            total_sq = np.zeros(dim, dtype=np.float64)
            count = 0
            for i in sorted(train_episode_ids):
                values = np.asarray(self.episodes[i].arrays[key], dtype=np.float64)
                total += values.sum(axis=0)
                total_sq += np.square(values).sum(axis=0)
                count += len(values)
            mean = total / count
            std = np.sqrt(np.maximum(total_sq / count - mean * mean, 0.0))
            stats[stat_key] = {
                "mean": torch.tensor(mean, dtype=torch.float32),
                "std": torch.tensor(np.maximum(std, 1e-3), dtype=torch.float32),
                "raw_std": torch.tensor(std, dtype=torch.float32),
            }
        return stats


def make_batch(data, starts, language, device):
    starts = np.asarray(starts, dtype=np.int64)
    state = data.frames("state", starts)
    images = []
    for key in CAMERA_KEYS:
        frames = data.frames(key, starts)
        # The collector stores raw[::-1] (vertical flip); RLinf's LIBERO
        # get_libero_image applies raw[::-1, ::-1] before RPent's VLA client.
        frames = np.flip(frames, axis=2)
        images.append(
            torch.from_numpy(frames.transpose(0, 3, 1, 2).copy()).to(
                device=device, dtype=torch.float32
            )
            / 255.0
        )
    return {
        "observation.state": torch.from_numpy(state.copy()).to(
            device=device, dtype=torch.float32
        ),
        "observation.images.camera1": images[0],
        "observation.images.camera2": images[1],
        "task": [language] * len(starts),
        "action": torch.from_numpy(data.action_chunks(starts)).to(
            device=device, dtype=torch.float32
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True,
                        help="episode NPZ or a directory containing episode_*.npz; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/.libero_osc7_cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=("random", "seed-00-39"), default="random",
                        help="seed-00-39 uses seed_00..39 for train and seed_40..49 for validation")
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--validation-windows", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=191)
    parser.add_argument("--task-prompt", default=None,
                        help="override episode text; set to the exact RPent task_descriptions string")
    parser.add_argument("--smoke-test", action="store_true",
                        help="one in-memory optimizer update; requires only one episode")
    parser.add_argument("--smoke-checkpoint", type=Path, default=None,
                        help="save the one-step smoke-test checkpoint to this path")
    return parse_task_args(parser, "libero", "finetune")


def validate(model, data, indices, language, device, count, seed):
    rng = np.random.default_rng(seed)
    picks = rng.choice(indices, size=min(count, len(indices)), replace=False)
    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, len(picks), 8):
            batch = make_batch(data, picks[start:start + 8], language, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, _ = model(batch)
            losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def save_checkpoint(path, model, stats, train_ids, data, step, args, language, base_sha):
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    trainable = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }
    payload = {
        "platform": "libero",
        "step": step,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "trainable": trainable,
        "stats": stats,
        "trainable_names": sorted(trainable),
        "base_model": str(BASE_MODEL),
        "config": {
            "contract": {
                "version": "libero_osc7_delta_v1",
                "state": "eef_position_xyz_m, eef_rotation_axis_angle_xyz_rad, gripper_qpos_2_m",
                "action": ACTION_FORMAT,
                "action_alignment": "action[t:t+16] predicts from pre-action observation[t]; repeat episode-final action for tail padding",
                "images": "agentview/wrist RGB uint8; collector vertical flip plus training horizontal flip, matching RPent 180-degree LIBERO VLA input",
                "instruction": language,
                "fps": FPS,
                "chunk_size": CHUNK_SIZE,
                "n_action_steps": N_ACTION_STEPS,
                "num_steps": INFERENCE_DENOISING_STEPS,
            },
            "base_model_sha256": base_sha,
            "dataset_fingerprint": data.fingerprint,
            "dataset_episodes": [str(data.episodes[i].path) for i in sorted(train_ids)],
            "language": language,
            "optimizer": "official_lerobot_smolvla_adamw_cosine_warmup",
            "optimizer_config": {
                "lr": float(model.config.optimizer_lr),
                "betas": list(model.config.optimizer_betas),
                "eps": float(model.config.optimizer_eps),
                "weight_decay": float(model.config.optimizer_weight_decay),
                "grad_clip_norm": float(model.config.optimizer_grad_clip_norm),
            },
            "scheduler_config": {
                "warmup_steps": args.warmup_steps,
                "decay_steps": args.steps,
                "decay_lr": float(model.config.scheduler_decay_lr),
            },
            "batch_size": args.batch_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def main():
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative")
    if not args.smoke_test and args.warmup_steps >= args.steps:
        raise ValueError("formal training needs warmup-steps < steps")
    if min(args.validation_every, args.validation_windows, args.save_every, args.report_every) < 1:
        raise ValueError("validation/save/report intervals and window count must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"SmolVLA base weights not found: {BASE_MODEL}")
    if not args.smoke_test and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output}; choose a new run directory"
        )

    data = LiberoDataset(args.data, args.cache)
    if args.smoke_test:
        train_ids = set(range(len(data.episodes)))
        train_frames = data._global_indices(train_ids)
        val_ids, val_frames = set(), np.empty(0, dtype=np.int64)
    else:
        if args.split_mode == "seed-00-39":
            train_ids, val_ids, train_frames, val_frames = data.split_seed_00_39_40_49()
        else:
            train_ids, val_ids, train_frames, val_frames = data.split_episodes(
                args.seed, args.validation_fraction
            )
    language = _instruction(args.task_prompt or data.instruction)
    if language != INSTRUCTION or data.instruction != INSTRUCTION:
        raise ValueError(
            f"LIBERO OSC7 training and RPent pi0_pick must use {INSTRUCTION!r}; "
            f"dataset={data.instruction!r}, requested={language!r}"
        )
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    stats = data.statistics(train_ids)
    model = make(STATE_DIM, ACTION_DIM, stats, cameras=2).to(device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("SmolVLA config produced no trainable parameters")
    optimizer_cfg = model.config.get_optimizer_preset()
    optimizer = optimizer_cfg.build(trainable)
    scheduler_cfg = model.config.get_scheduler_preset()
    scheduler_cfg.num_warmup_steps = args.warmup_steps
    scheduler_cfg.num_decay_steps = args.steps
    scheduler = scheduler_cfg.build(optimizer, args.steps)
    base_sha = _sha256(BASE_MODEL)

    if args.smoke_test:
        starts = np.random.default_rng(args.seed).choice(
            train_frames, size=min(args.batch_size, len(train_frames)), replace=False
        )
        batch = make_batch(data, starts, language, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, _ = model(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite smoke loss: {float(loss)}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, optimizer_cfg.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite smoke gradient norm")
        optimizer.step()
        if args.smoke_checkpoint is not None:
            save_checkpoint(
                args.smoke_checkpoint, model, stats, train_ids, data, 1,
                args, language, base_sha,
            )
        print(json.dumps({
            "smoke_test": "passed", "checkpoint_written": args.smoke_checkpoint is not None,
            "checkpoint": str(args.smoke_checkpoint) if args.smoke_checkpoint else None,
            "batch_size": len(starts), "loss": float(loss.detach()),
            "grad_norm": float(grad_norm), "episodes": len(data.episodes),
            "frames": data.frame_count, "instruction": language,
            "contract": "libero_osc7_delta_v1",
        }, ensure_ascii=False), flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "status": "training",
        "platform": "libero",
        "contract": "libero_osc7_delta_v1",
        "data_fingerprint": data.fingerprint,
        "episodes": len(data.episodes),
        "train_episodes": len(train_ids),
        "validation_episodes": len(val_ids),
        "frames": data.frame_count,
        "train_frames": len(train_frames),
        "validation_frames": len(val_frames),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "fps": FPS,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": N_ACTION_STEPS,
        "num_steps": INFERENCE_DENOISING_STEPS,
        "image_orientation": "180-degree rotation; matches RPent LIBERO VLA input",
        "instruction": language,
        "split_mode": args.split_mode,
        "train_episode_paths": [str(data.episodes[i].path) for i in sorted(train_ids)],
        "validation_episode_paths": [str(data.episodes[i].path) for i in sorted(val_ids)],
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": float(model.config.optimizer_lr),
        "optimizer": {
            "name": "AdamW",
            "betas": list(model.config.optimizer_betas),
            "eps": float(model.config.optimizer_eps),
            "weight_decay": float(model.config.optimizer_weight_decay),
            "grad_clip_norm": float(optimizer_cfg.grad_clip_norm),
        },
        "scheduler": {
            "name": "cosine_with_warmup",
            "warmup_steps": args.warmup_steps,
            "decay_steps": args.steps,
            "decay_lr": float(model.config.scheduler_decay_lr),
        },
        "validation_every": args.validation_every,
        "save_every": args.save_every,
        "base_model_sha256": base_sha,
    }
    (args.output / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rng = np.random.default_rng(args.seed)
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    running = []
    for step in range(1, args.steps + 1):
        starts = rng.choice(train_frames, size=args.batch_size, replace=len(train_frames) < args.batch_size)
        batch = make_batch(data, starts, language, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = model(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, optimizer_cfg.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running.append(float(metrics["loss"]))

        validation_loss = None
        if step % args.validation_every == 0 or step == 1:
            validation_loss = validate(
                model, data, val_frames, language, device,
                args.validation_windows, args.seed + step,
            )
        if step % args.report_every == 0 or step == 1 or step == args.steps:
            row = {
                "step": step,
                "train_loss": float(np.mean(running)),
                "validation_loss": validation_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - start_time,
            }
            with (args.output / "training_progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            running.clear()
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                args.output / f"checkpoint_{step:06d}.pt",
                model, stats, train_ids, data, step, args, language, base_sha,
            )
    (args.output / "status.json").write_text(
        json.dumps({"status": "completed", "step": args.steps}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

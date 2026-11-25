#!/usr/bin/env python3
"""Train PiStar's SigLIP + Gemma3 distributional value model on RECAP arrays."""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import cv2

from recap_value_model import load_pistar_value_components, load_value_checkpoint
from recap_training_control import PauseTraining, pause_if_requested


CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pistar-root", type=Path)
    parser.add_argument("--siglip-checkpoint", type=Path)
    parser.add_argument("--gemma-checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=None)
    parser.add_argument("--validation-steps", type=int, nargs="*", default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="initialize weights from a PiStar checkpoint, resetting optimizer and step")
    parser.add_argument("--skip-validation-steps", type=int, nargs="*", default=[])
    parser.add_argument(
        "--full-validation-every",
        type=int,
        default=0,
        help="Run full validation every N steps; 0 means only at the final step.",
    )
    parser.add_argument("--freeze-mode", choices=("none", "siglip_only", "all_backbones"), default="all_backbones")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.micro_batch_size <= 0 or args.batch_size % args.micro_batch_size:
        raise ValueError("batch-size must be a positive multiple of micro-batch-size")
    accumulation_steps = args.batch_size // args.micro_batch_size

    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    assets = [args.tokenizer]
    if not (args.resume or args.init_checkpoint):
        if args.siglip_checkpoint is None or args.gemma_checkpoint is None:
            parser.error("fresh training requires --siglip-checkpoint and --gemma-checkpoint")
        assets += [args.siglip_checkpoint, args.gemma_checkpoint]
    for path in assets:
        if not path.exists():
            raise FileNotFoundError(f"Required PiStar pretrained asset is missing: {path}")
    if args.siglip_checkpoint:
        os.environ["PISTAR_SIGLIP_CHECKPOINT"] = str(args.siglip_checkpoint.resolve())
    if args.gemma_checkpoint:
        os.environ["PISTAR_GEMMA_CHECKPOINT"] = str(args.gemma_checkpoint.resolve())

    root, trainer, openpi_model, ValueModelConfig = load_pistar_value_components(args.pistar_root)
    from recap_value_validation import make_value_evaluator
    import jax
    import jax.numpy as jnp
    from openpi.models import value_model as value_module

    meta = json.loads((args.data / "manifest.json").read_text())
    if meta.get("schema") != "recap_value_pistar_v1":
        raise ValueError(f"Expected recap_value_pistar_v1 data, got {meta.get('schema')!r}")
    if meta["image_size"] != 224 or tuple(meta["camera_order"]) != ("images", "left_wrist", "right_wrist"):
        raise ValueError("PiStar value training requires three RGB views resized to 224x224")
    targets = np.load(args.data / "returns.npy", mmap_mode="r")
    if meta.get("image_storage") == "jpeg_indexed_v1":
        offsets = np.load(args.data / "image_offsets.npy", mmap_mode="r")
        image_file = (args.data / "images.jpgbin").open("rb")
        images = None
    else:
        offsets = None
        image_file = None
        images = np.load(args.data / "images.npy", mmap_mode="r")
    if len(targets) != meta["samples"] or (offsets is not None and len(offsets) != len(targets)) or (images is not None and len(images) != len(targets)):
        raise ValueError("Prepared value arrays do not match their manifest")

    rng = np.random.default_rng(391)
    val_keys = {episode["key"] for episode in meta["episodes"] if episode.get("split") == "validation"}
    if not val_keys:
        order = rng.permutation(len(meta["episodes"]))
        nval = max(1, int(0.2 * len(order)))
        val_keys = {meta["episodes"][int(i)]["key"] for i in order[:nval]}
    train_ids: list[int] = []
    val_ids: list[int] = []
    tasks: dict[str, str] = {}
    sample_tasks = np.empty(meta["samples"], dtype=object)
    for episode in meta["episodes"]:
        ids = list(range(episode["start"], episode["stop"]))
        (val_ids if episode["key"] in val_keys else train_ids).extend(ids)
        tasks[episode["key"]] = episode["task"]
        sample_tasks[episode["start"]:episode["stop"]] = episode["task"]

    config = ValueModelConfig(siglip_variant="So400m/14")
    tokenizer = trainer.GemmaValueTokenizer(config.max_token_len, tokenizer_path=str(args.tokenizer))
    token_cache = {}
    for task in set(tasks.values()):
        token_cache[task] = tokenizer.tokenize(task)

    def make_observation(ids: np.ndarray):
        if offsets is not None:
            view_batch = np.empty((len(ids), 3, meta["image_size"], meta["image_size"], 3), dtype=np.uint8)
            for batch_index, sample_id in enumerate(ids):
                for camera_index in range(3):
                    offset, length = offsets[sample_id, camera_index]
                    image_file.seek(int(offset))
                    encoded = np.frombuffer(image_file.read(int(length)), dtype=np.uint8)
                    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if decoded is None:
                        raise ValueError(f"Corrupt JPEG at sample {sample_id}, camera {camera_index}")
                    view_batch[batch_index, camera_index] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        else:
            view_batch = np.asarray(images[ids])
        image_dict = {
            key: jnp.asarray(view_batch[:, i].astype(np.float32) / 127.5 - 1.0)
            for i, key in enumerate(CAMERA_KEYS)
        }
        prompt_tokens, prompt_masks = [], []
        for task in sample_tasks[ids]:
            tokens, mask = token_cache[str(task)]
            prompt_tokens.append(tokens)
            prompt_masks.append(mask)
        return openpi_model.Observation(
            images=image_dict,
            image_masks={key: jnp.ones((len(ids),), dtype=jnp.bool_) for key in CAMERA_KEYS},
            # PiStar's current value forward accepts state in Observation but does not consume it.
            state=jnp.zeros((len(ids), config.action_dim), dtype=jnp.float32),
            tokenized_prompt=jnp.stack(prompt_tokens),
            tokenized_prompt_mask=jnp.stack(prompt_masks),
        )

    seed = jax.random.PRNGKey(391)
    seed, init_rng = jax.random.split(seed)
    state, tx, _ = trainer.create_train_state(
        config,
        args.steps,
        init_rng,
        load_pretrained=not (args.resume or args.init_checkpoint),
        ema_decay=None,
        freeze_mode=args.freeze_mode,
    )
    model_def = state.model_def
    trainable_filter = trainer._make_trainable_filter(args.freeze_mode)
    start_step = 0
    if args.init_checkpoint:
        loaded = load_value_checkpoint(trainer, args.init_checkpoint, state)
        state = trainer.TrainState(0, loaded.params, state.model_def,
                                   tx.init(loaded.params.filter(trainable_filter)), None)
        print(json.dumps({"phase": "initialized_from_checkpoint", "source": str(args.init_checkpoint),
                          "optimizer_reset": True, "step": 0}), flush=True)
        del loaded
    if args.resume:
        state = load_value_checkpoint(trainer, args.resume, state)
        start_step = int(state.step)
        resume_state = args.resume / "training_state.pkl"
        if resume_state.exists():
            with resume_state.open("rb") as stream:
                saved = pickle.load(stream)
            for key, expected in (("total_steps", args.steps), ("batch_size", args.batch_size),
                                  ("micro_batch_size", args.micro_batch_size),
                                  ("freeze_mode", args.freeze_mode), ("step", start_step)):
                if key in saved and saved[key] != expected:
                    raise ValueError(f"Resume configuration mismatch for {key}: {saved[key]} != {expected}")
            state = trainer.TrainState(state.step, state.params, state.model_def,
                                       jax.tree.map(jnp.asarray, saved["opt_state"]), state.ema_params)
            seed = jnp.asarray(saved["seed"])
            rng.bit_generator.state = saved["numpy_rng"]
        else:
            # Old checkpoints contain weights only; moments restart, but the
            # schedule counter advances to the actual completed update count.
            def advance_counts(tree):
                if hasattr(tree, "_fields"):
                    return tree._replace(**{name: jnp.asarray(start_step, dtype=value.dtype)
                                            if name == "count" else advance_counts(value)
                                            for name, value in zip(tree._fields, tree)})
                if isinstance(tree, tuple):
                    return tuple(advance_counts(value) for value in tree)
                return tree
            state = trainer.TrainState(state.step, state.params, state.model_def,
                                       advance_counts(state.opt_state), state.ema_params)
        print(json.dumps({"phase": "resumed", "step": start_step,
                          "optimizer_restored": resume_state.exists()}), flush=True)

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def compiled_train_step(params, opt_state, step, ema_params, step_rng, observation, target):
        micro_observation = jax.tree.map(
            lambda x: x.reshape((accumulation_steps, args.micro_batch_size, *x.shape[1:])), observation
        )
        micro_target = target.reshape((accumulation_steps, args.micro_batch_size))
        trainable_params = params.filter(trainable_filter)
        zero_grads = jax.tree.map(jnp.zeros_like, trainable_params)

        def accumulate(carry, micro_batch):
            loss_sum, grad_sum = carry
            micro_index, micro_obs, micro_y = micro_batch
            model = trainer.nnx.merge(model_def, params)
            model.train()
            micro_rng = jax.random.fold_in(jax.random.fold_in(step_rng, step), micro_index)

            def loss_fn(module):
                return module.compute_loss(micro_rng, micro_obs, micro_y, train=True)

            diff_state = trainer.nnx.DiffState(0, trainable_filter)
            loss, grads = trainer.nnx.value_and_grad(loss_fn, argnums=diff_state)(model)
            grad_sum = jax.tree.map(lambda total, value: total + value, grad_sum, grads)
            return (loss_sum + loss, grad_sum), None

        indices = jnp.arange(accumulation_steps, dtype=jnp.int32)
        (loss_sum, grad_sum), _ = jax.lax.scan(
            accumulate,
            (jnp.asarray(0.0, dtype=jnp.float32), zero_grads),
            (indices, micro_observation, micro_target),
        )
        grads = jax.tree.map(lambda value: value / accumulation_steps, grad_sum)
        updates, new_opt_state = tx.update(grads, opt_state, trainable_params)
        new_params = trainer.optax.apply_updates(trainable_params, updates)
        updated_model = trainer.nnx.merge(model_def, params)
        trainer.nnx.update(updated_model, new_params)
        full_params = trainer.nnx.state(updated_model)
        new_ema_params = None
        if ema_params is not None:
            new_ema_params = jax.tree.map(lambda old, new: 0.99 * old + 0.01 * new, ema_params, full_params)
        return full_params, new_opt_state, new_ema_params, step + 1, {"loss": loss_sum / accumulation_steps}

    compiled_eval = make_value_evaluator(model_def, trainer.nnx, openpi_model, value_module)

    def emit(payload):
        print(json.dumps(payload), flush=True)

    def save_full_checkpoint(step):
        trainer.save_checkpoint(state, args.output, step)
        checkpoint = args.output / f"step_{step:08d}"
        # An existing committed checkpoint already represents this step.
        if not (checkpoint / "training_state.pkl").exists():
            temporary = checkpoint / "training_state.pkl.tmp"
            with temporary.open("wb") as stream:
                pickle.dump({"opt_state": jax.device_get(state.opt_state),
                             "seed": np.asarray(seed), "numpy_rng": rng.bit_generator.state,
                             "step": step, "total_steps": args.steps,
                             "batch_size": args.batch_size,
                             "micro_batch_size": args.micro_batch_size,
                             "freeze_mode": args.freeze_mode}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(checkpoint / "training_state.pkl")
        emit({"phase": "checkpoint_saved", "step": step, "checkpoint": checkpoint.name})
        return checkpoint

    def check_pause():
        pause_if_requested(args.output, int(state.step), save_full_checkpoint, emit)

    def evaluate_full_validation(params, eval_rng, step):
        """Evaluate every held-out frame and return frame-weighted mean metrics."""
        if not val_ids:
            raise ValueError("validation split is empty")
        total_loss = 0.0
        total_abs_error = 0.0
        total_count = 0
        validation_start = time.time()
        print(json.dumps({"phase": "validation", "step": step,
                          "frames_completed": 0, "frame_count": len(val_ids),
                          "elapsed_seconds": 0}), flush=True)
        for start in range(0, len(val_ids), args.micro_batch_size):
            check_pause()
            batch_ids = np.asarray(val_ids[start:start + args.micro_batch_size], dtype=np.int64)
            val_obs = make_observation(batch_ids)
            val_target = jnp.asarray(np.asarray(targets[batch_ids]), dtype=jnp.float32)
            eval_rng, batch_rng = jax.random.split(eval_rng)
            batch_loss_sum, batch_error_sum = jax.device_get(
                compiled_eval(params, batch_rng, val_obs, val_target)
            )
            batch_count = len(batch_ids)
            total_loss += float(batch_loss_sum)
            total_abs_error += float(batch_error_sum)
            total_count += batch_count
            if (start // args.micro_batch_size + 1) % 25 == 0 or total_count == len(val_ids):
                print(json.dumps({"phase": "validation", "step": step,
                                  "frames_completed": total_count, "frame_count": len(val_ids),
                                  "elapsed_seconds": time.time() - validation_start}), flush=True)
        return total_loss / total_count, total_abs_error / total_count

    args.output.mkdir(parents=True, exist_ok=True)
    events = args.output / "events.jsonl"
    start = time.time()
    losses: list[float] = []
    for step in range(start_step + 1, args.steps + 1):
        check_pause()
        ids = rng.choice(train_ids, args.batch_size, replace=len(train_ids) < args.batch_size)
        observation = make_observation(ids)
        target = jnp.asarray(np.asarray(targets[ids]), dtype=jnp.float32)
        seed, step_rng = jax.random.split(seed)
        params, opt_state, ema_params, new_step, info = compiled_train_step(
            state.params, state.opt_state, state.step, state.ema_params, step_rng, observation, target
        )
        state = trainer.TrainState(new_step, params, model_def, opt_state, ema_params)
        loss = float(info["loss"])
        losses.append(loss)
        check_pause()

        should_report = step % args.report_every == 0 or step == args.steps
        should_validate = step == args.steps or (
            step in args.validation_steps if args.validation_steps is not None else
            args.full_validation_every > 0 and step % args.full_validation_every == 0
        )
        should_validate = should_validate and step not in args.skip_validation_steps
        # Commit weights before any potentially slow validation work.
        should_save = step == args.steps or (
            step in args.checkpoint_steps if args.checkpoint_steps is not None else
            step % args.save_every == 0
        )
        if should_save:
            save_full_checkpoint(step)

        if should_report or should_validate:
            val_loss = None
            value_mae = None
            if should_validate:
                val_loss, value_mae = evaluate_full_validation(state.params, step_rng, step)
            row = {
                "step": step,
                "train_loss": float(np.mean(losses[-args.report_every:])),
                "validation_loss": val_loss,
                "validation_value_mae": value_mae,
                "validation_scope": "all_validation_frames" if should_validate else "not_evaluated",
                "elapsed_seconds": time.time() - start,
            }
            with events.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    report = {
        "status": "completed",
        "architecture": "PiStar SigLIP So400m/14 + Gemma3-270M + 201-bin C51 value head",
        "pistar_root": str(root),
        "pretrained_siglip": str(args.siglip_checkpoint.resolve()) if args.siglip_checkpoint else None,
        "pretrained_gemma": str(args.gemma_checkpoint.resolve()) if args.gemma_checkpoint else None,
        "tokenizer": str(args.tokenizer.resolve()),
        "data_manifest": str((args.data / "manifest.json").resolve()),
        "samples": meta["samples"],
        "source_counts": meta["source_counts"],
        "success_count": meta["success_count"],
        "failure_count": meta["failure_count"],
        "steps": args.steps,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "validation_episode_count": len(val_keys),
        "freeze_mode": args.freeze_mode,
        "ema_enabled": False,
        "elapsed_seconds": time.time() - start,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except PauseTraining:
        pass

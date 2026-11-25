#!/usr/bin/env python3
"""Run the PiStar value model over every prepared RECAP episode frame."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import cv2

from recap_value_model import load_pistar_value_components, load_value_checkpoint


CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="PiStar Orbax step checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pistar-root", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"PiStar Orbax checkpoint directory not found: {args.checkpoint}")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(args.tokenizer)

    _, trainer, openpi_model, ValueModelConfig = load_pistar_value_components(args.pistar_root)
    import jax
    import jax.numpy as jnp

    meta = json.loads((args.data / "manifest.json").read_text())
    if meta.get("image_storage") == "jpeg_indexed_v1":
        offsets = np.load(args.data / "image_offsets.npy", mmap_mode="r")
        image_file = (args.data / "images.jpgbin").open("rb")
        images = None
    else:
        offsets = None
        image_file = None
        images = np.load(args.data / "images.npy", mmap_mode="r")
    config = ValueModelConfig(siglip_variant="So400m/14")
    tokenizer = trainer.GemmaValueTokenizer(config.max_token_len, tokenizer_path=str(args.tokenizer))
    token_cache = {task: tokenizer.tokenize(task) for task in {e["task"] for e in meta["episodes"]}}

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    state, _, _ = trainer.create_train_state(
        config, 1, init_rng, load_pretrained=False, freeze_mode="all_backbones", ema_decay=None
    )
    state = load_value_checkpoint(trainer, args.checkpoint, state)
    model_def = state.model_def

    @jax.jit
    def infer(params, step_rng, observation):
        model = trainer.nnx.merge(model_def, params)
        model.eval()
        return model.compute_value(step_rng, observation, train=False)

    predictions = {}
    frames_completed = 0
    for episode_index, episode in enumerate(meta["episodes"], start=1):
        start, stop = int(episode["start"]), int(episode["stop"])
        values: list[float] = []
        tokens, token_mask = token_cache[episode["task"]]
        for offset in range(start, stop, args.batch_size):
            ids = np.arange(offset, min(offset + args.batch_size, stop))
            if offsets is not None:
                view_batch = np.empty((len(ids), 3, meta["image_size"], meta["image_size"], 3), dtype=np.uint8)
                for batch_index, sample_id in enumerate(ids):
                    for camera_index in range(3):
                        image_offset, length = offsets[sample_id, camera_index]
                        image_file.seek(int(image_offset))
                        encoded = np.frombuffer(image_file.read(int(length)), dtype=np.uint8)
                        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                        if decoded is None:
                            raise ValueError(f"Corrupt JPEG at sample {sample_id}, camera {camera_index}")
                        view_batch[batch_index, camera_index] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            else:
                view_batch = np.asarray(images[ids])
            observation = openpi_model.Observation(
                images={
                    key: jnp.asarray(view_batch[:, i].astype(np.float32) / 127.5 - 1.0)
                    for i, key in enumerate(CAMERA_KEYS)
                },
                image_masks={key: jnp.ones((len(ids),), dtype=jnp.bool_) for key in CAMERA_KEYS},
                state=jnp.zeros((len(ids), config.action_dim), dtype=jnp.float32),
                tokenized_prompt=jnp.broadcast_to(tokens, (len(ids), tokens.shape[-1])),
                tokenized_prompt_mask=jnp.broadcast_to(token_mask, (len(ids), token_mask.shape[-1])),
            )
            rng, step_rng = jax.random.split(rng)
            values.extend(np.asarray(infer(state.params, step_rng, observation)).astype(float).tolist())
        predictions[f"{episode['source']}__{episode['id']}"] = values
        frames_completed += len(values)
        print(json.dumps({
            "phase": "inference",
            "episodes_completed": episode_index,
            "episode_count": len(meta["episodes"]),
            "frames_completed": frames_completed,
            "frame_count": meta["samples"],
        }), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, ensure_ascii=False) + "\n")
    print(json.dumps({"episodes": len(predictions), "frames": sum(map(len, predictions.values())), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()

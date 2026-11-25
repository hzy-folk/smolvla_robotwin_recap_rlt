#!/usr/bin/env python3
"""Prepare RECAP return targets and indexed JPEG frames without large RAM spikes."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SCALE = 468.75
CAMERAS = ("images", "left_wrist", "right_wrist")


def targets(n: int, success: bool) -> np.ndarray:
    rewards = np.full(n, -1.0 / SCALE, dtype=np.float32)
    rewards[-1] = 0.0 if success else -93.75 / SCALE
    return np.clip(np.cumsum(rewards[::-1])[::-1], -1.0, 0.0)


def resized_jpeg(frame: np.ndarray, size: int) -> bytes:
    if frame.shape[0] != size or frame.shape[1] != size:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    episodes: list[dict] = []
    for root, source in ((args.success, "original_success"), (args.failure, "original_failure")):
        manifest = json.loads((root / "manifest.json").read_text())
        for episode in manifest["episodes"]:
            episodes.append({"kind": "npz", "root": root, "entry": episode, "source": source})

    summary = json.loads((args.rollout / "summary.json").read_text())
    if not summary.get("episodes") or summary.get("status") != "completed" or summary.get("finished") != summary.get("total", len(summary.get("episodes", []))):
        raise ValueError("Expected a completed, nonempty rollout")
    for row in summary["episodes"]:
        episodes.append({"kind": "rollout", "root": args.rollout, "entry": row,
                         "source": "rollout_success" if row.get("official_success", False) else "rollout_failure"})

    total = 0
    for episode in episodes:
        entry, root = episode["entry"], episode["root"]
        state_path = root / entry["npz"] if episode["kind"] == "npz" else root / "rollouts" / entry["id"] / "rollout.npz"
        with np.load(state_path, allow_pickle=False) as data:
            episode["length"] = len(data["state"])
        total += episode["length"]
    offsets = np.lib.format.open_memmap(args.output / "image_offsets.npy", mode="w+", dtype=np.int64,
                                        shape=(total, 3, 2))
    states = np.lib.format.open_memmap(args.output / "states.npy", mode="w+", dtype=np.float32, shape=(total, 16))
    values = np.lib.format.open_memmap(args.output / "returns.npy", mode="w+", dtype=np.float32, shape=(total,))
    records = []
    cursor = 0
    with (args.output / "images.jpgbin").open("wb") as image_stream:
        for episode in episodes:
            entry, root = episode["entry"], episode["root"]
            if episode["kind"] == "npz":
                npz_path = root / entry["npz"]
                with np.load(npz_path, allow_pickle=False) as data:
                    state = data["state"]
                    n = len(state)
                    success = bool(data["official_success"][-1]) if "official_success" in data else bool(data["success"])
                    task = str(data["task"].item())
                    states[cursor:cursor+n] = state
                    values[cursor:cursor+n] = targets(n, success)
                    for camera, key in enumerate(CAMERAS):
                        frames = data[key]
                        if len(frames) != n:
                            raise ValueError(f"Frame count mismatch in {npz_path} ({key})")
                        for t in range(n):
                            payload = resized_jpeg(frames[t], args.size)
                            offset = image_stream.tell()
                            image_stream.write(payload)
                            offsets[cursor+t, camera] = (offset, len(payload))
                episode_id = entry["id"]
            else:
                rollout_dir = root / "rollouts" / entry["id"]
                with np.load(rollout_dir / "rollout.npz", allow_pickle=False) as data:
                    state = data["state"]
                    n = len(state)
                    states[cursor:cursor+n] = state
                    success = bool(entry.get("official_success", False))
                    values[cursor:cursor+n] = targets(n, success)
                task = "stack the red block and then the green block"
                for camera, name in enumerate(("head", "left_wrist", "right_wrist")):
                    cap = cv2.VideoCapture(str(rollout_dir / f"{name}.mp4"))
                    t = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if t >= n:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        payload = resized_jpeg(frame, args.size)
                        offset = image_stream.tell()
                        image_stream.write(payload)
                        offsets[cursor+t, camera] = (offset, len(payload))
                        t += 1
                    cap.release()
                    if t != n:
                        raise ValueError(f"Expected {n} {name} video frames for {entry['id']}, got {t}")
                episode_id = entry["id"]

            records.append({"source": episode["source"], "id": episode_id,
                            "key": f"{episode['source']}:{episode_id}", "start": cursor,
                            "stop": cursor+n, "success": success, "task": task,
                            "split": entry.get("split", "train") if episode["kind"] == "npz" else "train"})
            cursor += n
            print(f"prepared {len(records)}/{len(episodes)}: {episode['source']} {episode_id} ({n} frames)", flush=True)

    if cursor != total:
        raise ValueError(f"Prepared {cursor} frames, expected {total}")
    offsets.flush()
    states.flush()
    values.flush()
    counts = {source: sum(ep["source"] == source for ep in records)
              for source in ("original_success", "original_failure", "rollout_success", "rollout_failure")}
    meta = {"schema": "recap_value_pistar_v1", "image_storage": "jpeg_indexed_v1", "jpeg_quality": 95,
            "fps": 25, "image_size": args.size, "camera_order": list(CAMERAS),
            "normalization_scale": SCALE, "lookahead": 8, "episodes": records, "samples": total,
            "source_roots": {"original_success": str(args.success.resolve()),
                             "original_failure": str(args.failure.resolve()),
                             "rollout": str(args.rollout.resolve())},
            "source_counts": counts, "success_count": sum(ep["success"] for ep in records),
            "failure_count": sum(not ep["success"] for ep in records)}
    (args.output / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "episodes"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

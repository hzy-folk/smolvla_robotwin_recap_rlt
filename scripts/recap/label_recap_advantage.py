#!/usr/bin/env python3
"""Compute N-step Advantage and labels for every source in the final dataset."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from recap_config import RecapConfig
from recap_rewards import reward_sequence


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("no advantage values")
    values = sorted(values)
    index = (len(values) - 1) * percent / 100.0
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def advantages(rewards: list[float], values: list[float], lookahead: int) -> list[float]:
    if len(rewards) != len(values):
        raise ValueError("reward and value lengths differ")
    result = []
    for t in range(len(values)):
        end = min(t + lookahead, len(values))
        future = 0.0 if end >= len(values) else values[end]
        result.append(sum(rewards[t:end]) + future - values[t])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, help="Existing per-episode label JSON directory")
    parser.add_argument("--value-data", type=Path,
                        help="Prepared PiStar value dataset directory (manifest.json + returns.npy)")
    parser.add_argument("--values", type=Path, required=True, help="JSON: {source__episode_id: [value, ...]}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = RecapConfig()
    cfg.validate()
    if bool(args.labels) == bool(args.value_data):
        parser.error("provide exactly one of --labels or --value-data")
    predicted = json.loads(args.values.read_text())
    rows, all_advantages = [], []
    if args.value_data:
        import numpy as np

        manifest = json.loads((args.value_data / "manifest.json").read_text())
        returns = np.load(args.value_data / "returns.npy", mmap_mode="r")
        episode_rows = []
        for episode in manifest["episodes"]:
            start, stop = int(episode["start"]), int(episode["stop"])
            episode_rows.append({
                "source": episode["source"],
                "episode_id": episode["id"],
                "frames": stop - start,
                "task_success": bool(episode["success"]),
                "return_label": np.asarray(returns[start:stop]).tolist(),
            })
    else:
        episode_rows = [json.loads(path.read_text()) for path in sorted(args.labels.glob("*.json"))]

    for row in episode_rows:
        key = f"{row['source']}__{row.get('episode_id', row.get('id'))}"
        if key not in predicted:
            raise KeyError(f"missing value predictions: {key}")
        if "reward_label" in row:
            rewards = row["reward_label"]
        else:
            rewards = reward_sequence(int(row["frames"]), bool(row["task_success"]), cfg)
        adv = advantages(rewards, predicted[key], cfg.lookahead)
        # The final policy update uses the critic for every source, including
        # the original demonstrations used by the first SFT stage.
        all_advantages.extend(adv)
        rows.append({"key": key, "source": row["source"],
                     "episode_id": row.get("episode_id", row.get("id")), "advantage": adv})
    threshold = percentile(all_advantages, 100.0 - cfg.positive_top_percent)
    for row in rows:
        row["adv_ind"] = ["positive" if x >= threshold else "negative" for x in row["advantage"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "threshold": threshold,
        "threshold_percentile": 100.0 - cfg.positive_top_percent,
        "threshold_source": "all_sources",
        "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"threshold": threshold, "episodes": len(rows), "advantage_values": len(all_advantages)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

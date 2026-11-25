"""从项目数据集里统计归一化用的分位数。

RLT 用分位数而不是均值方差，统计对象必须和 `ActionNormalizer` 实际处理的
数值一致。abs 模式下就是绝对关节目标；delta 模式下要先减掉当前关节角再
统计，否则尺度会差一个数量级，网络输入会全挤在零附近。

    python rlt/tools/make_stats.py --dataset <数据目录> \
        --config rlt/configs/robotwin_piper.yaml --out rlt_runs/robotwin/stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlt import bootstrap  # noqa: E402

bootstrap.bootstrap()

import numpy as np  # noqa: E402

from rlt.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 RLT 的动作/状态分位数统计")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--low", type=float, default=0.01)
    parser.add_argument("--high", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    from robotwin_smol_data import RobotwinDataset

    args = parse_args()
    config = load_config(args.config)
    data_config = config.data
    dataset = RobotwinDataset(args.dataset)
    indices = dataset.indices(args.split)
    assert len(indices) > 0, "数据划分是空的"

    actions = np.asarray(dataset.action[indices], dtype=np.float32)
    states = np.asarray(dataset.state[indices], dtype=np.float32)
    assert actions.shape[1] == data_config.action_dim, f"动作维度 {actions.shape[1]} 与配置不一致"

    if data_config.action_representation == "delta":
        relative = actions - states[:, : data_config.action_dim]
        keep = np.ones(data_config.action_dim, dtype=bool)
        for index in data_config.delta_exclude_dims:
            keep[index] = False
        actions = np.where(keep[None, :], relative, actions)

    payload = {
        "action": {
            "q01": np.quantile(actions, args.low, axis=0).tolist(),
            "q99": np.quantile(actions, args.high, axis=0).tolist(),
        },
        "state": {
            "q01": np.quantile(states, args.low, axis=0).tolist(),
            "q99": np.quantile(states, args.high, axis=0).tolist(),
        },
        "meta": {
            "dataset": str(Path(args.dataset).resolve()),
            "split": args.split,
            "frames": int(len(indices)),
            "representation": data_config.action_representation,
            "low": args.low,
            "high": args.high,
        },
    }
    for key in ("action", "state"):
        low = np.asarray(payload[key]["q01"])
        high = np.asarray(payload[key]["q99"])
        assert np.all(high - low > 1e-6), f"{key} 有维度分位区间退化，检查数据"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["meta"], indent=2))
    print(f"[save] {out}")


if __name__ == "__main__":
    main()

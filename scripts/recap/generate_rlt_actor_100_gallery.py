#!/usr/bin/env python3
"""Build a video gallery with stage-1 vs RLT-actor results for 100 episodes."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from generate_rollout_galleries import OUT, make_entry, write_page

CAMERAS = ("head", "left_wrist", "right_wrist")
JOINT_NAMES = [
    "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
    "left_joint_5", "left_joint_6", "left_finger_1", "left_finger_2",
    "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
    "right_joint_5", "right_joint_6", "right_finger_1", "right_finger_2",
]
SHIM = (
    '<script>window.__CURVES__=__DATA__;'
    '(function(){var raw=window.fetch;window.fetch=function(input,init){'
    'var key=String((input&&input.url)||input);'
    'if(Object.prototype.hasOwnProperty.call(window.__CURVES__,key))'
    'return Promise.resolve(new Response(JSON.stringify(window.__CURVES__[key]),'
    "{status:200,headers:{'Content-Type':'application/json'}}));return raw(input,init);};})();</script>"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", default="robotwin_rlt_actor_final_eval100.html")
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    split = selection.get("split", "all")
    rows = summary["episodes"]
    if summary.get("status") != "completed" or len(rows) != summary.get("total"):
        raise RuntimeError(f"evaluation incomplete: {len(rows)}/{summary.get('total')}")

    curves, entries = {}, []
    for row in rows:
        folder = run / "rollouts" / row["id"]
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if not report.get("completed"):
            raise RuntimeError(f"incomplete episode report: {folder}")
        with np.load(folder / "rollout.npz", allow_pickle=False) as data:
            state, action = data["state"], data["applied_targets"]
        if len(action) < len(state):
            action = np.concatenate((action, action[-1:]), axis=0)
        curve = {
            "state": np.round(state, 5).tolist(),
            "action": np.round(action, 5).tolist(),
            "names": JOINT_NAMES,
            "units": (["rad"] * 6 + ["m"] * 2) * 2,
        }
        chart = folder / "chart_data.json"
        chart.write_text(json.dumps(curve, separators=(",", ":")), encoding="utf-8")
        curves[os.path.relpath(chart.resolve(), OUT).replace(os.sep, "/")] = curve
        videos = [folder / f"{camera}.mp4" for camera in CAMERAS]
        links = [("评测报告", folder / "report.json"),
                 ("关节状态与动作", folder / "rollout.npz")]
        entries.append(make_entry(
            OUT, row["id"], row["source_episode"], row.get("seed"),
            bool(row["official_success"]), True,
            report.get("simulated_seconds", 0), report.get("steps", 0), None,
            videos, links, chart=chart,
        ) | {"comparison_label": row.get("comparison_label"), "split": row["split"]})

    train, validation = summary["train"], summary["validation"]
    trans = train["transitions"]
    train_rate = f"{train['success_rate']:.1%}" if train["success_rate"] is not None else "n/a"
    validation_rate = f"{validation['success_rate']:.1%}" if validation["success_rate"] is not None else "n/a"
    if split == "train":
        title = "RLT Actor 最终评测 · 训练集 80"
        subtitle = "Actor 确定性评测 · 训练集"
        metrics = [
            {"label": "Actor 成功", "value": f"{train['successes']}/{train['total']} ({train_rate})"},
            {"label": "阶段一基线成功", "value": f"{train['baseline_successes']}/{train['total']}"},
            {"label": "基线失败后成功", "value": str(trans["baseline_failure_actor_success"])},
            {"label": "基线成功后失败", "value": str(trans["baseline_success_actor_failure"])},
        ]
    else:
        title = "RLT Actor 最终评测 · 训练 80 + 验证 20"
        subtitle = "Actor 最终评测"
        metrics = [
            {"label": "训练集 Actor 成功", "value": f"{train['successes']}/{train['total']} ({train_rate})"},
            {"label": "阶段一训练集", "value": f"{train['baseline_successes']}/{train['total']}"},
            {"label": "验证集 Actor 成功", "value": f"{validation['successes']}/{validation['total']} ({validation_rate})"},
            {"label": "训练集改善 / 退化", "value": f"{trans['baseline_failure_actor_success']} / {trans['baseline_success_actor_failure']}"},
        ]
    OUT.mkdir(parents=True, exist_ok=True)
    write_page(
        args.output, title, subtitle,
        entries, {"errors": 0, "count": len(entries), "successes": summary["successes"],
                  "success_rate": summary["success_rate"], "metrics": metrics},
    )
    page_path = OUT / args.output
    page = page_path.read_text(encoding="utf-8")
    shim = SHIM.replace("__DATA__", json.dumps(curves, separators=(",", ":")).replace("<", "\\u003c"))
    marker = '<meta charset="utf-8">'
    if marker not in page:
        raise RuntimeError("gallery template changed: charset meta tag missing")
    page_path.write_text(page.replace(marker, marker + shim, 1), encoding="utf-8")
    print(f"Wrote {page_path} ({len(entries)} episodes)", flush=True)


if __name__ == "__main__":
    main()

"""运行环境引导：把项目现有脚本、lerobot 与 torch 所在路径接进来。

这个模块只做 `sys.path` 和环境变量设置，不导入任何重依赖，也不改变项目里
任何既有文件的行为。任何 RLT 入口脚本第一件事就是调用 `bootstrap()`。

注意路径优先级。项目 `scripts/` 目录里有一个叫 `dataset.py` 的模块，它是
本项目的数据契约实现，而 lerobot 内部也有同名概念。把 `scripts/` 插在
`sys.path` 靠前的位置可以保证 `from dataset import ...` 解析到项目版本。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RLT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = RLT_DIR.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"

# 项目既有的运行环境分布在这两个 site-packages 里，torch 和 lerobot 依赖都在。
AUDIT_DIR = Path(os.environ.get("RLT_AUDIT_DIR", "/home/hzy/git/learn/feasibility_audit"))
EXTRA_SITE = (
    AUDIT_DIR / "venv/lib/python3.10/site-packages",
    Path("/home/hzy/venvs/legged-model/lib/python3.10/site-packages"),
)
LEROBOT_SRC = AUDIT_DIR / "lerobot_stable_src"

# 项目自带的 SmolVLA 权重与配置，全部只读引用，不做任何拷贝或覆盖。
DEFAULT_MODEL_DIR = AUDIT_DIR / "followup_runs/models/smolvla"
DEFAULT_VLM_DIR = AUDIT_DIR / "followup_runs/models/smolvlm"


def bootstrap(verbose: bool = False) -> dict[str, Path]:
    """把依赖路径补齐，返回用到的关键目录，方便脚本里打日志。"""
    for path in (SCRIPTS_DIR, PROJECT_DIR):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    for path in EXTRA_SITE:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))

    # 目前机器上没有任何一个环境同时具备 lerobot 全部依赖和 sapien。
    # 需要时用这个变量按优先级补一个 site-packages，例如：
    #   RLT_EXTRA_SITE=/path/to/robotwin_venv/lib/python3.10/site-packages
    extra = os.environ.get("RLT_EXTRA_SITE", "")
    for item in extra.split(os.pathsep):
        if item and Path(item).is_dir() and item not in sys.path:
            sys.path.append(item)

    if LEROBOT_SRC.is_dir():
        path = str(LEROBOT_SRC)
        if path not in sys.path:
            sys.path.insert(0, path)

    # 离线运行，避免脚本在没有网络时卡在 hub 请求上。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    info = {
        "rlt": RLT_DIR,
        "project": PROJECT_DIR,
        "scripts": SCRIPTS_DIR,
        "audit": AUDIT_DIR,
        "lerobot_src": LEROBOT_SRC,
        "model": DEFAULT_MODEL_DIR,
        "vlm": DEFAULT_VLM_DIR,
    }
    if verbose:
        for key, value in info.items():
            print(f"[bootstrap] {key:11s} {value}")
    return info


def entrypoint() -> None:
    """给直接执行的脚本用：把项目根目录放进路径，这样才能 `import rlt`。"""
    root = str(PROJECT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    bootstrap()

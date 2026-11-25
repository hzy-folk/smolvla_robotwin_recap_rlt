"""运行环境引导：把项目现有脚本、lerobot 与 torch 所在路径接进来。

这个模块只做 `sys.path` 和环境变量设置，不导入任何重依赖，也不改变项目里
任何既有文件的行为。任何 RLT 入口脚本第一件事就是调用 `bootstrap()`。

注意路径优先级。项目 `collection/` 目录里有一个叫 `dataset.py` 的模块，它是
本项目的数据契约实现，而 lerobot 内部也有同名概念。把 `collection/` 插在
`sys.path` 靠前的位置可以保证 `from dataset import ...` 解析到项目版本。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RLT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = RLT_DIR.parent
SCRIPTS_DIR = PROJECT_DIR / "collection"

sys.path.insert(0, str(PROJECT_DIR))
from runtime_config import bootstrap_model, path, MODEL_DIR, VLM_DIR
LEROBOT_SRC = path('lerobot_src')
DEFAULT_MODEL_DIR = MODEL_DIR
DEFAULT_VLM_DIR = VLM_DIR


def bootstrap(verbose: bool = False) -> dict[str, Path]:
    """把依赖路径补齐，返回用到的关键目录，方便脚本里打日志。"""
    for path in (SCRIPTS_DIR, PROJECT_DIR):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    bootstrap_model()

    # 离线运行，避免脚本在没有网络时卡在 hub 请求上。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    info = {
        "rlt": RLT_DIR,
        "project": PROJECT_DIR,
        "scripts": SCRIPTS_DIR,
        "runtime": PROJECT_DIR,
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

"""SmolVLA 上的 RLT（RL Token）复现子系统。

本包只在显式调用 `rlt.*` 的脚本时被导入，不参与项目原有的训练、推理、
采集与回放路径。所有对外接口都是新增文件，没有修改任何既有脚本。
"""
from pathlib import Path

RLT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = RLT_DIR.parent

__all__ = ["RLT_DIR", "PROJECT_DIR"]

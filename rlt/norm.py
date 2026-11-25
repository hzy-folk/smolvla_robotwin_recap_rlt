"""动作与状态的归一化，把绝对关节目标压到网络友好的范围。

SmolVLA 内部用均值方差归一化，RLT 原仓库用分位数归一化。这里选后者，因为
分位数对长尾更稳，而且统计量可以由 `tools/make_stats.py` 从本项目数据集里
直接算出来，不依赖模型内部的 normalizer，训练和部署两边可以共用一套数字。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

EPS = 1e-6


@dataclass
class NormStats:
    action_q01: np.ndarray
    action_q99: np.ndarray
    state_q01: np.ndarray
    state_q99: np.ndarray

    @staticmethod
    def load(path: str | Path, action_dim: int, proprio_dim: int) -> "NormStats":
        payload = json.loads(Path(path).read_text())
        action = payload["action"]
        state = payload["state"]
        stats = NormStats(
            action_q01=np.asarray(action["q01"], dtype=np.float32),
            action_q99=np.asarray(action["q99"], dtype=np.float32),
            state_q01=np.asarray(state["q01"], dtype=np.float32),
            state_q99=np.asarray(state["q99"], dtype=np.float32),
        )
        assert stats.action_q01.shape == (action_dim,), f"动作统计量维度不是 {action_dim}"
        assert stats.state_q01.shape == (proprio_dim,), f"状态统计量维度不是 {proprio_dim}"
        assert np.all(stats.action_q99 - stats.action_q01 > EPS), "动作分位数区间退化"
        return stats


class ActionNormalizer:
    """在绝对动作和网络输入空间之间来回变换。"""

    def __init__(
        self,
        stats: NormStats,
        action_dim: int,
        proprio_dim: int,
        representation: str = "abs",
        delta_exclude_dims: tuple[int, ...] = (),
    ) -> None:
        assert representation in ("abs", "delta")
        assert proprio_dim >= action_dim, "delta 表示需要 proprio 至少覆盖动作维度"
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.representation = representation
        self.exclude = set(int(i) for i in delta_exclude_dims)
        self.action_q01 = torch.tensor(stats.action_q01)
        self.action_scale = torch.tensor(stats.action_q99 - stats.action_q01)
        self.state_q01 = torch.tensor(stats.state_q01)
        self.state_scale = torch.tensor(stats.state_q99 - stats.state_q01)

    # ---- 内部工具 ----
    @staticmethod
    def _nonzero_rows(chunk: torch.Tensor) -> torch.Tensor:
        return chunk.abs().sum(dim=-1, keepdim=True) > 0

    def _to_relative(self, chunk: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if self.representation == "abs":
            return chunk
        base = proprio[:, None, : self.action_dim]
        delta = chunk - base
        if self.exclude:
            keep = torch.ones(self.action_dim, dtype=torch.bool, device=chunk.device)
            keep[sorted(self.exclude)] = False
            delta = torch.where(keep[None, None, :], delta, chunk)
        return delta

    def _from_relative(self, chunk: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if self.representation == "abs":
            return chunk
        base = proprio[:, None, : self.action_dim]
        absolute = chunk + base
        if self.exclude:
            keep = torch.ones(self.action_dim, dtype=torch.bool, device=chunk.device)
            keep[sorted(self.exclude)] = False
            absolute = torch.where(keep[None, None, :], absolute, chunk)
        return absolute

    # ---- 对外接口（torch） ----
    def normalize(self, chunk: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        chunk = chunk.float()
        relative = self._to_relative(chunk, proprio.float())
        scale = self.action_scale.to(chunk.device)
        center = self.action_q01.to(chunk.device)
        value = (relative - center) / scale * 2.0 - 1.0
        return torch.where(self._nonzero_rows(chunk), value, torch.zeros_like(value))

    def denormalize(self, chunk: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        chunk = chunk.float()
        scale = self.action_scale.to(chunk.device)
        center = self.action_q01.to(chunk.device)
        relative = (chunk + 1.0) / 2.0 * scale + center
        absolute = self._from_relative(relative, proprio.float())
        return torch.where(self._nonzero_rows(chunk), absolute, torch.zeros_like(absolute))

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        scale = self.state_scale.to(state.device)
        center = self.state_q01.to(state.device)
        return (state.float() - center) / scale * 2.0 - 1.0

    # ---- 对外接口（numpy），供环境侧使用 ----
    def normalize_np(self, chunk: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.normalize(
                torch.as_tensor(np.asarray(chunk)), torch.as_tensor(np.asarray(proprio))
            ).numpy()

    def denormalize_np(self, chunk: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.denormalize(
                torch.as_tensor(np.asarray(chunk)), torch.as_tensor(np.asarray(proprio))
            ).numpy()

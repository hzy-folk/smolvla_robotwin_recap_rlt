"""经验池。列存的环形缓冲区，一条经验就是一次 chunk 决策。

字段和 RLT 原仓库对齐，只是把跨进程的 HTTP 传输换成了本进程内存，因为仿真
里不需要隔离延迟。大数组统一存 float16，一个 z 是 960 维，二十万条经验大约
两三个 GB，用 float32 会翻倍。
"""
from __future__ import annotations

import threading
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCE_BASE = 0
SOURCE_RL = 1
SOURCE_HUMAN = 2
SOURCE_MIXED = 3


@dataclass
class ReplayStats:
    size: int
    capacity: int
    adds_total: int
    max_episode: int
    human_ratio: float
    success_ratio: float


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        chunk_len: int,
        seed: int = 0,
        n_step_chunks: int = 1,
        gamma: float = 0.99,
    ) -> None:
        assert capacity > 0
        self.capacity = int(capacity)
        self.z_dim = int(z_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.n_step_chunks = int(n_step_chunks)
        self.gamma = float(gamma)
        if self.n_step_chunks < 1 or not 0.0 < self.gamma <= 1.0:
            raise ValueError("n_step_chunks must be >= 1 and gamma must be in (0, 1]")
        self.rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self._fields: dict[str, np.ndarray] = {}
        self._size = 0
        self._cursor = 0
        self._adds = 0
        self._max_episode = -1

    # ---- 存储布局 ----
    def _allocate(self) -> None:
        h, a, p, z = self.chunk_len, self.action_dim, self.proprio_dim, self.z_dim
        self._fields = {
            "z": np.zeros((self.capacity, z), np.float16),
            "z_next": np.zeros((self.capacity, z), np.float16),
            "proprio": np.zeros((self.capacity, p), np.float32),
            "proprio_next": np.zeros((self.capacity, p), np.float32),
            "ref": np.zeros((self.capacity, h, a), np.float16),
            "ref_next": np.zeros((self.capacity, h, a), np.float16),
            "action": np.zeros((self.capacity, h, a), np.float16),
            "reward": np.zeros((self.capacity, h), np.float32),
            "nstep_reward": np.zeros((self.capacity,), np.float32),
            "nstep_discount": np.zeros((self.capacity,), np.float32),
            "nstep_done": np.zeros((self.capacity,), np.uint8),
            "z_n": np.zeros((self.capacity, z), np.float16),
            "proprio_n": np.zeros((self.capacity, p), np.float32),
            "ref_n": np.zeros((self.capacity, h, a), np.float16),
            "done": np.zeros((self.capacity,), np.uint8),
            "success": np.zeros((self.capacity,), np.uint8),
            "source": np.zeros((self.capacity,), np.uint8),
            "source_chunk": np.zeros((self.capacity, h), np.uint8),
            "episode": np.zeros((self.capacity,), np.int32),
            "chunk_index": np.zeros((self.capacity,), np.int32),
            "stage": np.zeros((self.capacity,), np.uint8),
        }

    def _validate(self, row: dict) -> None:
        assert row["z"].shape == (self.z_dim,)
        assert row["ref"].shape == (self.chunk_len, self.action_dim)
        assert row["action"].shape == (self.chunk_len, self.action_dim)
        assert row["reward"].shape == (self.chunk_len,)
        assert row["proprio"].shape == (self.proprio_dim,)
        assert row["source_chunk"].shape == (self.chunk_len,)

    # ---- 写入 ----
    def add(self, row: dict) -> None:
        """主线程写入，学习线程可能同时在采样，加锁保证读到完整的一行。"""
        if self.n_step_chunks != 1:
            raise ValueError("n-step replay requires complete ordered episodes; use add_episode")
        with self._lock:
            self.add_episode([row])

    def _add(self, row: dict) -> None:
        if not self._fields:
            self._allocate()
        self._validate(row)
        index = self._cursor
        for key in self._fields:
            if key == "episode":
                self._fields[key][index] = int(row.get("episode", 0))
            elif key == "chunk_index":
                self._fields[key][index] = int(row.get("chunk_index", 0))
            elif key == "stage":
                self._fields[key][index] = int(row.get("stage", 0))
            elif key == "success":
                self._fields[key][index] = int(bool(row.get("success", False)))
            elif key == "done":
                self._fields[key][index] = int(bool(row.get("done", False)))
            else:
                self._fields[key][index] = row[key]
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._adds += 1
        self._max_episode = max(self._max_episode, int(row.get("episode", 0)))

    def add_episode(self, rows: list[dict]) -> None:
        with self._lock:
            chunk_gamma = self.gamma ** self.chunk_len
            chunk_returns = [
                float(np.dot(np.asarray(row["reward"], dtype=np.float64),
                             self.gamma ** np.arange(self.chunk_len)))
                for row in rows
            ]
            for index, original in enumerate(rows):
                accumulated = 0.0
                discount = 1.0
                last_index = index
                terminal = False
                for offset in range(self.n_step_chunks):
                    current = index + offset
                    if current >= len(rows):
                        break
                    accumulated += discount * chunk_returns[current]
                    last_index = current
                    if bool(rows[current].get("done", False)):
                        terminal = True
                        break
                    discount *= chunk_gamma
                last = rows[last_index]
                row = dict(original)
                row.update({
                    "chunk_index": index,
                    "nstep_reward": accumulated,
                    "nstep_discount": 0.0 if terminal else discount,
                    "nstep_done": terminal,
                    "z_n": last["z_next"],
                    "proprio_n": last["proprio_next"],
                    "ref_n": last["ref_next"],
                })
                self._add(row)

    def __len__(self) -> int:
        return self._size

    @property
    def adds_total(self) -> int:
        return self._adds

    @property
    def max_episode(self) -> int:
        return self._max_episode

    # ---- 采样 ----
    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        with self._lock:
            assert self._size > 0, "经验池为空"
            indices = self.rng.integers(0, self._size, size=int(batch_size))
            batch = {key: value[indices] for key, value in self._fields.items()}
            batch["human_mask"] = (batch["source_chunk"] >= SOURCE_HUMAN).astype(np.float32)
            return batch

    def stats(self) -> ReplayStats:
        with self._lock:
            return self._stats()

    def _stats(self) -> ReplayStats:
        if self._size == 0:
            return ReplayStats(0, self.capacity, self._adds, self._max_episode, 0.0, 0.0)
        used = slice(0, self._size)
        return ReplayStats(
            size=self._size,
            capacity=self.capacity,
            adds_total=self._adds,
            max_episode=self._max_episode,
            human_ratio=float((self._fields["source_chunk"][used] >= SOURCE_HUMAN).mean()),
            success_ratio=float(self._fields["success"][used].mean()),
        )

    # ---- 落盘 ----
    def save(self, path: str | Path) -> None:
        with self._lock:
            self._save(path)

    def _save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in self._fields.items()}
        payload.update(
            {
                "_capacity": np.int64(self.capacity),
                "_z_dim": np.int64(self.z_dim),
                "_proprio_dim": np.int64(self.proprio_dim),
                "_action_dim": np.int64(self.action_dim),
                "_chunk_len": np.int64(self.chunk_len),
                "_n_step_chunks": np.int64(self.n_step_chunks),
                "_gamma": np.float64(self.gamma),
                "_rng_state": np.asarray(json.dumps(self.rng.bit_generator.state)),
                "_size": np.int64(self._size),
                "_cursor": np.int64(self._cursor),
                "_adds": np.int64(self._adds),
                "_max_episode": np.int64(self._max_episode),
            }
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **payload,
            )
        temporary.replace(path)

    def load(self, path: str | Path) -> None:
        with self._lock:
            self._load(path)

    def _load(self, path: str | Path) -> None:
        if not self._fields:
            self._allocate()
        with np.load(path) as payload:
            expected = {
                "_capacity": self.capacity,
                "_z_dim": self.z_dim,
                "_proprio_dim": self.proprio_dim,
                "_action_dim": self.action_dim,
                "_chunk_len": self.chunk_len,
                "_n_step_chunks": self.n_step_chunks,
            }
            for key, value in expected.items():
                if key in payload and int(payload[key]) != value:
                    raise ValueError(f"ReplayBuffer {key[1:]} 不匹配：文件={int(payload[key])}，配置={value}")
            if "_gamma" not in payload or not np.isclose(float(payload["_gamma"]), self.gamma):
                raise ValueError(f"ReplayBuffer gamma 不匹配：配置={self.gamma}")
            missing_fields = set(self._fields).difference(payload.files)
            if missing_fields:
                raise ValueError(f"经验池缺少字段：{sorted(missing_fields)}；需使用当前格式重建")
            for key in payload.files:
                if key.startswith("_"):
                    continue
                if key not in self._fields:
                    raise ValueError(f"经验池文件包含未知字段：{key}")
                if payload[key].shape != self._fields[key].shape:
                    raise ValueError(f"经验池字段形状不匹配：{key} {payload[key].shape} != {self._fields[key].shape}")
                self._fields[key][:] = payload[key]
            self._size = int(payload["_size"])
            self._cursor = int(payload["_cursor"])
            self._adds = int(payload["_adds"])
            self._max_episode = int(payload["_max_episode"])
            if "_rng_state" in payload:
                self.rng.bit_generator.state = json.loads(str(payload["_rng_state"].item()))
            if not 0 <= self._size <= self.capacity or not 0 <= self._cursor < self.capacity:
                raise ValueError("经验池文件中的 size/cursor 超出范围")

"""WebSocket 上的轻量序列化，把 numpy 数组安全地塞进 JSON。

项目环境里没有装 msgpack，而 RLT 原仓库的协议依赖它。这里用 JSON 加 base64
自己实现一份等价能力，只影响本包新增的服务端和客户端。格式刻意做得直白：
数组写成 {"__array__": {"dtype": ..., "shape": [...], "data": base64}}，
其余类型保持 JSON 原生表示。

体积上比 msgpack 大一倍左右，但链路只在同一台机器或同一局域网内使用，
而且只在需要远程分机部署时才会启用，本地仿真默认走进程内直连。
"""
from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

ARRAY_KEY = "__array__"


def _encode_array(array: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(array)
    return {
        ARRAY_KEY: {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    }


def _to_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    return value


def _from_json(value: Any) -> Any:
    if isinstance(value, dict):
        if ARRAY_KEY in value and len(value) == 1:
            payload = value[ARRAY_KEY]
            raw = base64.b64decode(payload["data"])
            array = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
            return array.reshape(tuple(payload["shape"])).copy()
        return {key: _from_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_json(item) for item in value]
    return value


def encode(payload: dict[str, Any]) -> str:
    return json.dumps(_to_json(payload), separators=(",", ":"))


def decode(text: str | bytes) -> dict[str, Any]:
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return _from_json(json.loads(text))

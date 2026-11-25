"""把 SmolVLA 包装成 RLT 需要的特征服务，输出 z 和参考动作块。

两种使用方式。仿真默认用同进程的 `LocalFeatureProvider`，零通信开销；需要
把大模型放到另一台机器时用 `serve.py` 起服务，再用 `RemoteFeatureProvider`
连过去。两种提供者的返回值完全一致，主循环不需要知道用的是哪一种。

请求格式（一次一条或一批）：

    {"batch": [{"observation.state": (16,), "observation.images.camera1": (256,256,3),
                "task": "..."}, ...]}

响应格式：

    {"z_rl": (B, z_dim), "proprio": (B, 16), "ref_chunk": (B, H, A),
     "meta": {"z_dim": ..., "chunk_size": ..., "action_dim": ...}}
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt import bootstrap  # noqa: E402

bootstrap.bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rlt import wire  # noqa: E402
from rlt.config import RLTConfig, load_config  # noqa: E402
from rlt.model import RLTokenModel, infer_z_and_reference, load_policy, prefix_dim  # noqa: E402


def camera_keys(count: int) -> list[str]:
    return [f"observation.images.camera{i}" for i in range(1, count + 1)]


def observations_to_batch(observations: list[dict], default_task: str, device: torch.device) -> dict:
    """把环境给的观测字典拼成 lerobot 期望的批。"""
    assert observations, "空批次"
    keys = [key for key in observations[0] if key.startswith("observation.images.")]
    states = np.stack([np.asarray(item["observation.state"], dtype=np.float32) for item in observations])
    batch = {
        "observation.state": torch.tensor(states, device=device, dtype=torch.float32),
        "task": [str(item.get("task", default_task)) for item in observations],
    }
    for key in keys:
        pictures = np.stack([np.asarray(item[key]) for item in observations])
        assert pictures.ndim == 4 and pictures.shape[-1] == 3, f"{key} 需要是 HWC 彩图"
        tensor = torch.tensor(pictures.transpose(0, 3, 1, 2).copy(), device=device, dtype=torch.float32) / 255.0
        batch[key] = tensor
    return batch


def slice_reference(actions: torch.Tensor, chunk_len: int) -> torch.Tensor:
    assert actions.shape[1] >= chunk_len, f"VLA 只给了 {actions.shape[1]} 步，不够 {chunk_len} 步"
    return actions[:, :chunk_len]


class LocalFeatureProvider:
    """进程内直接调用 SmolVLA，仿真默认走这条。"""

    def __init__(self, policy, token: RLTokenModel, config: RLTConfig, device: str = "cuda") -> None:
        self.policy = policy
        self.token = token.eval()
        self.config = config
        self.device = torch.device(device)
        self._noise_generators: dict[int, torch.Generator] = {}

    def _next_noise(self, episode_id: int = 0) -> torch.Tensor:
        # The evaluator starts a worker per episode, so each episode gets its own
        # seed-300 stream while successive chunks within it consume that stream.
        generator = self._noise_generators.get(episode_id)
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(300)
            self._noise_generators[episode_id] = generator
        flow = self.policy.model.model
        shape = (1, flow.config.chunk_size, flow.config.max_action_dim)
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )

    def _features_one(self, observation: dict) -> dict:
        batch = observations_to_batch([observation], self.config.data.language, self.device)
        z, actions = infer_z_and_reference(
            self.policy,
            self.token,
            batch,
            action_dim=self.config.data.action_dim,
            z_source=self.config.token.z_source,
            noise=self._next_noise(int(observation.get("_rlt_episode_id", 0))),
        )
        reference = slice_reference(actions, self.config.data.chunk_len)
        return {
            "z_rl": z[0].float().cpu().numpy(),
            "proprio": np.asarray(observation["observation.state"], dtype=np.float32),
            "ref_chunk": reference[0].float().cpu().numpy(),
        }

    @torch.no_grad()
    def features(self, observation: dict) -> dict:
        return self._features_one(observation)

    @torch.no_grad()
    def features_batch(self, observations: list[dict]) -> list[dict]:
        # Keep the native evaluator's B=1 inference numerics while simulations
        # continue to run concurrently in separate environments.
        return [self._features_one(observation) for observation in observations]


class RemoteFeatureProvider:
    """连 serve.py 的客户端，接口和本地版一致。"""

    def __init__(self, url: str, chunk_len: int, timeout: float = 30.0) -> None:
        import websockets.sync.client as ws_client

        self.url = url
        self.chunk_len = chunk_len
        self.timeout = timeout
        self._client = ws_client.connect(url, open_timeout=timeout, max_size=None)
        self.metadata = wire.decode(self._client.recv(timeout=timeout))

    def _request(self, observations: list[dict]) -> list[dict]:
        payload = {"batch": observations}
        self._client.send(wire.encode(payload))
        response = wire.decode(self._client.recv(timeout=self.timeout))
        z = np.asarray(response["z_rl"])
        reference = np.asarray(response["ref_chunk"])[:, : self.chunk_len]
        proprio = np.asarray(response["proprio"])
        return [
            {"z_rl": z[i], "proprio": proprio[i], "ref_chunk": reference[i]}
            for i in range(len(observations))
        ]

    def features(self, observation: dict) -> dict:
        return self._request([observation])[0]

    def features_batch(self, observations: list[dict]) -> list[dict]:
        return self._request(observations)

    def close(self) -> None:
        self._client.close()


def load_everything(config: RLTConfig, checkpoint: str, token_checkpoint: str, device: str):
    policy = load_policy(checkpoint, device=device)
    saved = torch.load(token_checkpoint, map_location="cpu", weights_only=False)
    dim = int(saved["dim"])
    assert dim == prefix_dim(policy), f"RL token 维度 {dim} 与模型 {prefix_dim(policy)} 不一致"
    token = RLTokenModel(
        dim=dim,
        embed_dim=int(saved.get("embed_dim", config.token.embed_dim)),
        num_tokens=int(saved["num_tokens"]),
        layers=config.token.num_layers,
        heads=config.token.num_heads,
        ratio=config.token.mlp_ratio,
        dropout=config.token.dropout,
    )
    token.load_state_dict(saved["state_dict"])
    token = token.to(device).eval()
    return policy, token


def serve_forever(config: RLTConfig, policy, token: RLTokenModel, host: str, port: int, device: str) -> None:
    from websockets.sync.server import serve

    provider = LocalFeatureProvider(policy, token, config, device=device)
    metadata = {
        "z_dim": int(token.z_dim),
        "proprio_dim": int(config.data.proprio_dim),
        "chunk_size": int(config.data.chunk_len),
        "action_dim": int(config.data.action_dim),
        "task": config.data.language,
    }
    lock = threading.Lock()

    def handler(connection) -> None:
        connection.send(wire.encode(metadata))
        for message in connection:
            try:
                payload = wire.decode(message)
                observations = payload["batch"] if "batch" in payload else [payload]
                with lock:
                    results = provider.features_batch(observations)
                response = {
                    "z_rl": np.stack([item["z_rl"] for item in results]),
                    "proprio": np.stack([item["proprio"] for item in results]),
                    "ref_chunk": np.stack([item["ref_chunk"] for item in results]),
                    "meta": metadata,
                }
                connection.send(wire.encode(response))
            except Exception as error:  # noqa: BLE001
                connection.send(wire.encode({"error": f"{type(error).__name__}: {error}"}))

    print(f"[serve] listening on ws://{host}:{port} z_dim={metadata['z_dim']}")
    # 一条观测大约 0.6MB（三路 256 图），批量时必须放开默认的 1MB 上限。
    with serve(handler, host, port, max_size=None) as server:
        server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLT 特征服务：z 与参考动作")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True, help="SmolVLA 微调检查点")
    parser.add_argument("--token", required=True, help="阶段一产出的 RL token 检查点")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = args.device or config.runtime.device
    policy, token = load_everything(config, args.checkpoint, args.token, device)
    serve_forever(
        config,
        policy,
        token,
        host=args.host or config.runtime.service_host,
        port=args.port or config.runtime.service_port,
        device=device,
    )


if __name__ == "__main__":
    main()

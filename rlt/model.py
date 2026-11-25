"""RL token 模块，以及从 SmolVLA 里取 prefix 表示的那一层胶水。

分成两半。上半部分是纯 PyTorch 的小编码器-解码器，和具体 VLA 无关，可以在
没有权重、没有 GPU 的情况下单独测试。下半部分负责调用 SmolVLA 的内部接口，
把一次前向里的 prefix 输出留下来，并且顺手把参考动作块采出来。

刻意没有修改 lerobot 的任何文件。所有对内部的调用都是"读"，包括
`embed_prefix`、`make_att_2d_masks`、`vlm_with_expert.forward`、`denoise_step`。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def as_vla(policy):
    """兼容两种传入形式：项目的 `Policy` 封装，或者裸的 `SmolVLAPolicy`。

    项目的 `Policy.model` 是 SmolVLAPolicy，而 SmolVLAPolicy.model 才是真正做
    前向的 VLAFlowMatching。用"有没有 prepare_images"来判断当前对象是不是策略层，
    避免猜层数。
    """
    candidate = getattr(policy, "model", None)
    if candidate is not None and hasattr(candidate, "prepare_images"):
        return candidate
    return policy


def build_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """复刻 SmolVLA 的注意力掩码构造，免去对 lerobot 内部函数名的依赖。

    语义和上游完全一致：按 `att_masks` 的累积值决定可见范围，再与 padding
    掩码取交集。抽出来放在这里，既少了耦合，也让自检可以在没有 lerobot 的
    环境里跑。
    """
    assert att_masks.ndim == 2 and pad_masks.ndim == 2
    cumulative = torch.cumsum(att_masks, dim=1)
    visible = cumulative[:, None, :] <= cumulative[:, :, None]
    padded = pad_masks[:, None, :] * pad_masks[:, :, None]
    return visible & padded


def sinusoidal_positions(length: int, dim: int, device: torch.device | None = None) -> torch.Tensor:
    """标准正弦位置编码，形状 (length, dim)，不需要训练。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half, 1)
    )
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    angles = positions * freqs[None, :]
    encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if encoding.shape[-1] < dim:
        encoding = F.pad(encoding, (0, dim - encoding.shape[-1]))
    return encoding


class CrossAttention(nn.Module):
    """手写的单头组多头交叉注意力，带 key padding 掩码。"""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        assert dim % heads == 0, "dim 必须能被 heads 整除"
        self.heads = heads
        self.head_dim = dim // heads
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, length, dim = x.shape
        source = context.shape[1]
        query = self.q_proj(self.ln_q(x)).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key_value = self.ln_kv(context)
        key = self.k_proj(key_value).view(batch, source, self.heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(key_value).view(batch, source, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2)) * (self.head_dim**-0.5)
        if context_mask is not None:
            keep = context_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        merged = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, dim)
        return x + self.dropout(self.out_proj(merged))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RLTokenEncoder(nn.Module):
    """把一串 prefix 表示压成 num_tokens 个向量。"""

    def __init__(self, dim: int, embed_dim: int, num_tokens: int, layers: int, heads: int, ratio: float, dropout: float) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.input_proj = nn.Linear(dim, embed_dim) if dim != embed_dim else nn.Identity()
        self.query = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)
        blocks: list[nn.Module] = []
        for _ in range(layers):
            blocks.append(CrossAttention(embed_dim, heads, dropout))
            blocks.append(FeedForward(embed_dim, ratio, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        prefix = self.input_proj(prefix)
        batch = prefix.shape[0]
        query = self.query.expand(batch, -1, -1)
        for block in self.blocks:
            query = block(query, prefix, mask) if isinstance(block, CrossAttention) else block(query)
        return self.norm(query)


class RLTokenDecoder(nn.Module):
    """从压缩向量重建整串 prefix 表示。"""

    def __init__(self, dim: int, embed_dim: int, layers: int, heads: int, ratio: float, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        blocks: list[nn.Module] = []
        for _ in range(layers):
            blocks.append(CrossAttention(embed_dim, heads, dropout))
            blocks.append(FeedForward(embed_dim, ratio, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, dim) if embed_dim != dim else nn.Identity()

    def forward(self, tokens: torch.Tensor, length: int) -> torch.Tensor:
        batch = tokens.shape[0]
        query = self.query.expand(batch, length, -1) + sinusoidal_positions(length, tokens.shape[-1], tokens.device)
        for block in self.blocks:
            if isinstance(block, CrossAttention):
                query = block(query, tokens, None)
            else:
                query = block(query)
        return self.output_proj(self.norm(query))


class RLTokenModel(nn.Module):
    """编码器加解码器的组合，训练目标就是重建 prefix 表示。"""

    def __init__(
        self,
        dim: int,
        embed_dim: int = 512,
        num_tokens: int = 1,
        layers: int = 2,
        heads: int = 8,
        ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.encoder = RLTokenEncoder(dim, embed_dim, num_tokens, layers, heads, ratio, dropout)
        self.decoder = RLTokenDecoder(dim, embed_dim, layers, heads, ratio, dropout)

    @property
    def z_dim(self) -> int:
        return self.embed_dim * self.num_tokens

    def encode(self, prefix: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        assert prefix.shape[-1] == self.dim, f"prefix 维度 {prefix.shape[-1]} 与模型维度 {self.dim} 不一致"
        return self.encoder(prefix, mask).reshape(prefix.shape[0], -1)

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode(self.encode(prefix, mask), prefix.shape[1], mask)

    def decode(self, tokens: torch.Tensor, length: int, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = tokens.shape[0]
        view = tokens.reshape(batch, self.num_tokens, self.embed_dim)
        return self.decoder(view, length)

    def reconstruction_loss(
        self, prefix: torch.Tensor, mask: torch.Tensor | None = None, detach_target: bool = True
    ) -> torch.Tensor:
        target = prefix.detach() if detach_target else prefix
        rebuilt = self.forward(prefix, mask)
        error = (rebuilt.float() - target.float()) ** 2
        if mask is None:
            return error.mean()
        weight = mask.to(error.dtype)[:, :, None]
        return (error * weight).sum() / (weight.sum() * error.shape[-1]).clamp_min(1.0)


@dataclass
class PrefixBatch:
    """一次 SmolVLA 前向里我们关心的三样东西。"""

    prefix: torch.Tensor
    mask: torch.Tensor
    image_tokens: int
    actions: torch.Tensor | None


def _image_token_count(pad_masks: torch.Tensor, language_tokens: int, state_tokens: int = 1) -> int:
    """prefix 的拼装顺序是 图像、语言、状态，图像 token 数等于总长减去后两段。

    SmolVLA 的 `prepare_images` 返回的掩码是"这一路相机是否存在"的布尔量，
    形状是 (B,)，不是每个 token 一位，所以不能直接数掩码宽度。
    """
    total = int(pad_masks.shape[1]) - int(language_tokens) - int(state_tokens)
    assert total > 0, "图像 token 数不为正，检查 prefix 拼装顺序"
    return total


@torch.no_grad()
def prefix_and_actions(policy, batch: dict, with_actions: bool = True) -> PrefixBatch:
    """跑一次 prefix 前向，同时拿到 z 的来源和参考动作块。

    和 `SmolVLAPolicy.predict_action_chunk` 的唯一区别是：这里保留了
    `vlm_with_expert.forward` 的第一个返回值，也就是 prefix 的逐 token 输出。
    原实现把它丢掉了，所以多拿这一份不额外消耗算力。
    """
    vla = as_vla(policy)
    model = vla.model
    assert model.config.use_cache, "需要 use_cache=True，否则拿不到可复用的 prefix KV 缓存"
    prepared = vla._prepare_batch(dict(batch))
    images, img_masks = vla.prepare_images(prepared)
    state = vla.prepare_state(prepared)
    lang_tokens, lang_masks = vla.prepare_language(prepared)

    prefix_embs, pad_masks, att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_2d_masks = build_att_2d_masks(pad_masks, att_masks)
    prefix_position_ids = torch.cumsum(pad_masks, dim=1) - 1

    outputs, past_key_values = model.vlm_with_expert.forward(
        attention_mask=prefix_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )
    prefix_out = outputs[0]
    assert prefix_out is not None, "prefix 前向没有返回表示"
    image_tokens = _image_token_count(pad_masks, lang_tokens.shape[1])
    if not with_actions:
        return PrefixBatch(
            prefix=prefix_out,
            mask=pad_masks.to(torch.bool),
            image_tokens=image_tokens,
            actions=None,
        )

    noise = model.sample_noise(
        (state.shape[0], model.config.chunk_size, model.config.max_action_dim), state.device
    )
    dt = torch.tensor(-1.0 / model.config.num_steps, dtype=torch.float32, device=state.device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=state.device)
    while time >= -dt / 2:
        velocity = model.denoise_step(pad_masks, past_key_values, x_t, time.expand(state.shape[0]))
        x_t = x_t + dt * velocity
        time = time + dt

    return PrefixBatch(
        prefix=prefix_out,
        mask=pad_masks.to(torch.bool),
        image_tokens=image_tokens,
        actions=x_t,
    )


def select_prefix(prefix: torch.Tensor, mask: torch.Tensor, image_tokens: int, source: str) -> tuple[torch.Tensor, torch.Tensor]:
    """按配置决定把哪一段 prefix 交给压缩器。"""
    if source == "image":
        image_mask = torch.zeros_like(mask)
        image_mask[:, :image_tokens] = True
        return prefix[:, :image_tokens], image_mask[:, :image_tokens]
    assert source == "full", source
    return prefix, mask


def _copy_batch(batch: dict) -> dict:
    return {
        key: value.clone() if torch.is_tensor(value) else list(value) if isinstance(value, list) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def infer_z_and_reference(
    policy,
    rlt: RLTokenModel,
    batch: dict,
    action_dim: int,
    z_source: str,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the token-training prefix path and SmolVLA's native action inference."""
    vla = as_vla(policy)
    result = prefix_and_actions(vla, _copy_batch(batch), with_actions=False)
    selected, mask = select_prefix(result.prefix, result.mask, result.image_tokens, z_source)
    z = rlt.encode(selected.float(), mask)

    if hasattr(vla, "predict_action_chunk"):
        # Match the project rollout evaluator's bfloat16 SmolVLA inference path.
        action_batch = _copy_batch(batch)
        if next(vla.parameters()).device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actions = vla.predict_action_chunk(action_batch, noise=noise)
        else:
            actions = vla.predict_action_chunk(action_batch, noise=noise)
        actions = actions[:, :, :action_dim].float()
    else:
        # Lightweight test doubles do not expose SmolVLA's public inference API.
        result = prefix_and_actions(vla, _copy_batch(batch), with_actions=True)
        actions = result.actions[:, :, :action_dim].float()
        actions = vla.unnormalize_outputs({"action": actions})["action"]
    return z, actions


def load_policy(checkpoint: str, device: str = "cuda", flow_steps: int | None = None):
    """加载本项目微调过的 SmolVLA 策略，复用既有 Policy 封装，不做任何改动。"""
    try:
        from robotwin_smol_model import Policy
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"加载 SmolVLA 需要完整的 lerobot 依赖，当前环境缺少 `{error.name}`。"
            "请先激活项目训练时用的 Python 环境，再运行需要真实权重的脚本；"
            "只验证链路可以改用 `rlt/tools/smoke_test.py`（不含 --with-vlm）。"
        ) from error

    policy = Policy(checkpoint, device=device, flow_steps=flow_steps or 4)
    policy.model.eval()
    for parameter in policy.model.parameters():
        parameter.requires_grad_(False)
    return policy


def prefix_dim(policy) -> int:
    return int(as_vla(policy).model.vlm_with_expert.vlm.config.text_config.hidden_size)

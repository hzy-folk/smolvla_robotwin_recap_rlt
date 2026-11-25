"""演员与评论家，以及 chunk 级的时序差分损失。

结构和 RLT 原仓库一致，只是从 JAX 换成 PyTorch。三路输入分别投影后拼接，
演员输出一整块动作，评论家给这块动作打分。参考动作在训练时整块丢弃，
用来避免演员退化成"抄草稿"。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEncoder(nn.Module):
    """把 z、proprio 和第三路输入（参考动作或待评价动作）拼成一个向量。"""

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        third_dim: int,
        hidden: int,
        proprio_hidden: int,
        third_hidden: int,
    ) -> None:
        super().__init__()
        self.z_proj = nn.Linear(z_dim, hidden)
        self.z_norm = nn.LayerNorm(hidden)
        self.proprio_proj = nn.Linear(proprio_dim, proprio_hidden)
        self.proprio_norm = nn.LayerNorm(proprio_hidden)
        self.third_proj = nn.Linear(third_dim, third_hidden)
        self.third_norm = nn.LayerNorm(third_hidden)
        self.width = hidden + proprio_hidden + third_hidden

    def forward(self, z: torch.Tensor, proprio: torch.Tensor, third: torch.Tensor) -> torch.Tensor:
        if third.ndim == 3:
            third = third.reshape(third.shape[0], -1)
        z_part = self.z_norm(self.z_proj(z))
        proprio_part = torch.tanh(self.proprio_norm(self.proprio_proj(proprio)))
        third_part = torch.tanh(self.third_norm(self.third_proj(third)))
        return torch.cat([z_part, proprio_part, third_part], dim=-1)


def _mlp(width: int, hidden: int, layers: int, output: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    current = width
    for _ in range(layers):
        modules += [nn.Linear(current, hidden), nn.LayerNorm(hidden), nn.GELU()]
        current = hidden
    modules.append(nn.Linear(current, output))
    return nn.Sequential(*modules)


class ChunkActor(nn.Module):
    """输出一整块归一化动作的均值，标准差是固定常数。"""

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        chunk_len: int,
        hidden: int = 256,
        proprio_hidden: int = 64,
        third_hidden: int = 256,
        layers: int = 2,
        fixed_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.fixed_std = fixed_std
        self.encoder = InputEncoder(
            z_dim, proprio_dim, action_dim * chunk_len, hidden, proprio_hidden, third_hidden
        )
        self.head = _mlp(self.encoder.width, hidden, layers, action_dim * chunk_len)

    def mean(self, z: torch.Tensor, proprio: torch.Tensor, third: torch.Tensor) -> torch.Tensor:
        flat = self.head(self.encoder(z, proprio, third))
        return flat.reshape(z.shape[0], self.chunk_len, self.action_dim)

    def sample(
        self, z: torch.Tensor, proprio: torch.Tensor, third: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        mu = self.mean(z, proprio, third)
        if deterministic:
            return mu
        return mu + self.fixed_std * torch.randn_like(mu)


class QNetwork(nn.Module):
    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        chunk_len: int,
        hidden: int = 256,
        proprio_hidden: int = 64,
        third_hidden: int = 256,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = InputEncoder(
            z_dim, proprio_dim, action_dim * chunk_len, hidden, proprio_hidden, third_hidden
        )
        self.head = _mlp(self.encoder.width, hidden, layers, 1)

    def forward(self, z: torch.Tensor, proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(z, proprio, action)).squeeze(-1)


class TwinCritic(nn.Module):
    """两条独立的 Q，训练目标取两者最小值，抑制过估计。"""

    def __init__(self, z_dim: int, proprio_dim: int, action_dim: int, chunk_len: int, **kwargs) -> None:
        super().__init__()
        self.q1 = QNetwork(z_dim, proprio_dim, action_dim, chunk_len, **kwargs)
        self.q2 = QNetwork(z_dim, proprio_dim, action_dim, chunk_len, **kwargs)

    def forward(self, z: torch.Tensor, proprio: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(z, proprio, action), self.q2(z, proprio, action)


@dataclass
class LearnerMetrics:
    critic_loss: float
    actor_loss: float
    q_mean: float
    bc_penalty: float
    delta_penalty: float
    human_ratio: float
    actor_updated: bool


def reference_dropout(ref: torch.Tensor, probability: float) -> torch.Tensor:
    """整块丢弃参考动作。掩码形状 (B,1,1)，保证要么全留要么全清。"""
    if probability <= 0.0:
        return ref
    keep = torch.rand(ref.shape[0], 1, 1, device=ref.device) >= probability
    return torch.where(keep, ref, torch.zeros_like(ref))


def chunk_td_target(
    reward: torch.Tensor, done: torch.Tensor, next_value: torch.Tensor, gamma: float
) -> torch.Tensor:
    """把块内奖励折扣累加，再加上下一块开头的价值。"""
    horizon = reward.shape[1]
    discount = torch.tensor(
        [gamma**i for i in range(horizon)], device=reward.device, dtype=reward.dtype
    )
    returns = (reward * discount[None, :]).sum(dim=1)
    bootstrap = (1.0 - done.float()) * (gamma**horizon) * next_value
    return returns + bootstrap


def n_step_td_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    discount: torch.Tensor,
    next_value: torch.Tensor,
) -> torch.Tensor:
    """N-step chunk return plus a bootstrap value if the episode continues."""
    return reward + (1.0 - done.float()) * discount * next_value


class RLTLearner:
    """把两套网络、两个优化器和全部损失装在一起，外部只调 `step`。"""

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        chunk_len: int,
        config,
        objective,
        denormalize=None,
        delta_joint_mask: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.cfg = config
        self.objective = objective
        self.denormalize = denormalize
        self.delta_joint_mask = delta_joint_mask
        common = dict(
            hidden=config.hidden_dim,
            proprio_hidden=config.proprio_hidden_dim,
            third_hidden=config.ref_hidden_dim,
            layers=config.num_layers,
        )
        self.actor = ChunkActor(
            z_dim, proprio_dim, action_dim, chunk_len, fixed_std=config.fixed_std, **common
        ).to(self.device)
        self.target_actor = ChunkActor(
            z_dim, proprio_dim, action_dim, chunk_len, fixed_std=config.fixed_std, **common
        ).to(self.device)
        self.critic = TwinCritic(z_dim, proprio_dim, action_dim, chunk_len, **common).to(self.device)
        self.target_critic = TwinCritic(z_dim, proprio_dim, action_dim, chunk_len, **common).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        for parameter in list(self.target_actor.parameters()) + list(self.target_critic.parameters()):
            parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.step_count = 0
        self.actor_version = 0

    def _critic_step(self, batch: dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            next_action = self.target_actor.sample(
                batch["z_n"], batch["proprio_n"], batch["ref_n"], deterministic=False
            )
            q1_next, q2_next = self.target_critic(batch["z_n"], batch["proprio_n"], next_action)
            target = n_step_td_target(
                batch["nstep_reward"], batch["nstep_done"], batch["nstep_discount"],
                torch.minimum(q1_next, q2_next),
            )
        q1, q2 = self.critic(batch["z"], batch["proprio"], batch["action"])
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()
        return float(loss.detach())

    def _delta_penalty(self, predicted_abs: torch.Tensor, target_abs: torch.Tensor) -> torch.Tensor:
        diff_predicted = predicted_abs[:, 1:] - predicted_abs[:, :-1]
        diff_target = target_abs[:, 1:] - target_abs[:, :-1]
        error = (diff_predicted - diff_target) ** 2
        if self.delta_joint_mask is not None:
            mask = self.delta_joint_mask.to(torch.bool).to(self.device)
            error = error[..., mask]
        return error.mean()

    def _actor_step(self, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, float]:
        dropped = reference_dropout(batch["ref"], self.cfg.reference_dropout)
        action = self.actor.sample(batch["z"], batch["proprio"], dropped, deterministic=False)
        q_value = self.critic.q1(batch["z"], batch["proprio"], action)

        human = batch["human_mask"].bool()
        has_human = human.any()
        target_action = torch.where(human[..., None], batch["action"], batch["ref"])
        bc = ((action - target_action) ** 2).mean()
        if has_human:
            human_bc = ((action - batch["action"]) ** 2).mean(dim=-1)[human].mean()
        else:
            human_bc = torch.zeros((), device=self.device)
        if (~human).any():
            policy_bc = ((action - batch["ref"]) ** 2).mean(dim=-1)[~human].mean()
        else:
            policy_bc = torch.zeros((), device=self.device)

        delta = torch.zeros((), device=self.device)
        if weights["delta"] > 0.0 and self.denormalize is not None:
            delta = self._delta_penalty(
                self.denormalize(action, batch["proprio"]),
                self.denormalize(target_action, batch["proprio"]),
            )

        loss = weights["bc"] * bc - weights["q"] * q_value.mean() + weights["delta"] * delta
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        self.actor_version += 1
        return {
            "actor_loss": float(loss.detach()),
            "q_mean": float(q_value.detach().mean()),
            "bc_penalty": float(bc.detach()),
            "policy_bc": float(policy_bc.detach()),
            "human_bc": float(human_bc.detach()),
            "delta_penalty": float(delta.detach()),
            "human_ratio": float(human.float().mean()),
        }

    def _soft_update(self) -> None:
        tau = self.cfg.tau
        pairs = ((self.target_actor, self.actor), (self.target_critic, self.critic))
        for target, source in pairs:
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.data.mul_(1.0 - tau).add_(source_parameter.data, alpha=tau)

    def step(self, batch: dict[str, torch.Tensor], warmup: bool) -> LearnerMetrics:
        if warmup:
            weights = {
                "bc": self.objective.warmup_bc_weight,
                "q": self.objective.warmup_q_weight,
                "delta": self.objective.warmup_delta_weight,
            }
        else:
            weights = {
                "bc": self.objective.bc_weight,
                "q": self.objective.q_weight,
                "delta": self.objective.delta_weight,
            }
        batch = {key: value.to(self.device) for key, value in batch.items()}
        critic_loss = self._critic_step(batch)
        actor_updated = (self.step_count + 1) % self.cfg.actor_update_period == 0
        stats = {"actor_loss": 0.0, "q_mean": 0.0, "bc_penalty": 0.0, "delta_penalty": 0.0, "human_ratio": 0.0}
        if actor_updated:
            stats = self._actor_step(batch, weights)
            self._soft_update()
        self.step_count += 1
        return LearnerMetrics(
            critic_loss=critic_loss,
            actor_loss=stats["actor_loss"],
            q_mean=stats["q_mean"],
            bc_penalty=stats["bc_penalty"],
            delta_penalty=stats["delta_penalty"],
            human_ratio=stats["human_ratio"],
            actor_updated=actor_updated,
        )

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "step_count": self.step_count,
            "actor_version": self.actor_version,
        }

    def load_state_dict(self, payload: dict) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_actor.load_state_dict(payload["target_actor"])
        self.target_critic.load_state_dict(payload["target_critic"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.step_count = int(payload["step_count"])
        self.actor_version = int(payload["actor_version"])

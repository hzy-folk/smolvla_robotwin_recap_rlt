from __future__ import annotations

try:
    from .recap_config import RecapConfig
except ImportError:  # direct execution: python recap/prepare_recap_value_dataset.py
    from recap_config import RecapConfig


def reward_sequence(frames: int, task_success: bool, cfg: RecapConfig) -> list[float]:
    """Return normalized per-transition rewards for one episode."""
    if frames <= 0:
        raise ValueError("episode must contain at least one frame")
    cfg.validate()
    rewards = [-1.0 / cfg.normalization_scale] * frames
    rewards[-1] = (0.0 if task_success else -cfg.failure_penalty) / cfg.normalization_scale
    return rewards


def value_targets(frames: int, task_success: bool, cfg: RecapConfig) -> list[float]:
    rewards = reward_sequence(frames, task_success, cfg)
    targets = []
    running = 0.0
    for reward in reversed(rewards):
        running += reward
        targets.append(max(-1.0, min(0.0, running)))
    return list(reversed(targets))

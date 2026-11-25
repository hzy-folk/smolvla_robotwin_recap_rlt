from dataclasses import dataclass


@dataclass(frozen=True)
class RecapConfig:
    fps: int = 25
    chunk_size: int = 16
    n_action_steps: int = 8
    lookahead: int = 8
    max_episode_seconds: int = 15
    failure_penalty_ratio: float = 0.25
    positive_top_percent: float = 50.0

    @property
    def max_frames(self) -> int:
        return self.fps * self.max_episode_seconds

    @property
    def failure_penalty(self) -> float:
        return self.failure_penalty_ratio * self.max_frames

    @property
    def normalization_scale(self) -> float:
        return self.max_frames + self.failure_penalty

    def validate(self) -> None:
        assert self.fps > 0
        assert self.chunk_size > 0
        assert 0 < self.n_action_steps <= self.chunk_size
        assert self.lookahead == self.n_action_steps
        assert self.max_episode_seconds > 0
        assert self.failure_penalty_ratio > 0
        assert 0 < self.positive_top_percent < 100

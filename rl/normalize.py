"""Running mean/std normalization for RL observation vectors."""
from __future__ import annotations

import numpy as np


class RunningNormalizer:
    """Welford online algorithm for running mean/variance, used to normalize observations."""

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a single observation or batch."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / max(total_count, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / max(total_count, 1)
        self.mean = new_mean
        self.var = m2 / max(total_count, 1)
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize an observation using running statistics."""
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip,
            self.clip,
        ).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]

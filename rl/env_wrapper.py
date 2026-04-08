"""Gymnasium-compatible wrapper around LLMServeEnvironment for RL training."""
from __future__ import annotations

from typing import Any

import numpy as np

from llmserve_env.models import ServeAction, ServeObservation
from rl.normalize import RunningNormalizer
from server.llmserve_environment import LLMServeEnvironment


# The 15 numeric observation fields in fixed order.
OBS_FIELDS: list[str] = [
    "queue_depth",
    "active_requests",
    "kv_cache_occupancy",
    "mean_prompt_length",
    "p50_ttft_ms",
    "p99_ttft_ms",
    "p50_itl_ms",
    "throughput_tps",
    "slo_compliance_rate",
    "gpu_memory_used_gb",
    "estimated_cost_per_1k",
    "request_arrival_rate",
    "spec_acceptance_rate",
    "eviction_events",
    "step_index",
]
OBS_DIM = len(OBS_FIELDS)


def obs_to_vector(obs: ServeObservation) -> np.ndarray:
    """Flatten a ServeObservation into a float32 array of shape (15,)."""
    return np.array([float(getattr(obs, f)) for f in OBS_FIELDS], dtype=np.float32)


class GymEnvWrapper:
    """Thin wrapper that gives the LLMServeEnvironment a Gymnasium-like interface.

    Supports:
        - reset() -> obs (np.ndarray)
        - step(action_dict) -> (obs, reward, done, info)
        - Optional running normalization of observations
    """

    def __init__(
        self,
        task_id: str = "static_workload",
        seed: int = 42,
        normalize: bool = True,
        mode: str = "sim",
    ) -> None:
        self.task_id = task_id
        self.seed = seed
        self._env = LLMServeEnvironment(seed=seed, mode=mode)
        self.normalizer = RunningNormalizer(shape=(OBS_DIM,)) if normalize else None
        self._last_obs: ServeObservation | None = None
        self._episode_step = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        ep_seed = seed if seed is not None else self.seed
        obs = self._env.reset(seed=ep_seed, task_id=self.task_id)
        self._last_obs = obs
        self._episode_step = 0
        vec = obs_to_vector(obs)
        if self.normalizer is not None:
            self.normalizer.update(vec)
            vec = self.normalizer.normalize(vec)
        return vec

    def step(self, action: dict[str, Any] | ServeAction) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if isinstance(action, dict):
            action = ServeAction(**action)
        obs = self._env.step(action)
        self._last_obs = obs
        self._episode_step += 1
        reward = float(getattr(obs, "reward", 0.0) or 0.0)
        done = bool(getattr(obs, "done", False))
        vec = obs_to_vector(obs)
        if self.normalizer is not None:
            self.normalizer.update(vec)
            vec = self.normalizer.normalize(vec)
        info = {"task_id": self.task_id, "step": self._episode_step, "raw_obs": obs}
        return vec, reward, done, info

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def last_observation(self) -> ServeObservation | None:
        return self._last_obs

"""Lightweight PPO implementation for InferenceGym.

No external RL library dependency — just PyTorch.
Supports mixed action spaces via the PolicyNetwork heads.
Designed to train on CPU in <10 minutes for Task 1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rl.env_wrapper import GymEnvWrapper
from rl.policy_network import PolicyNetwork, action_dict_to_tensors, batch_action_tensors


@dataclass
class RolloutBuffer:
    """Stores one rollout of experience for PPO update."""
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOTrainer:
    """Proximal Policy Optimisation trainer."""

    def __init__(
        self,
        env: GymEnvWrapper,
        policy: PolicyNetwork,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_length: int = 512,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
    ) -> None:
        self.env = env
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

        # State
        self._obs: np.ndarray | None = None
        self._total_steps = 0
        self._episodes_done = 0
        self._episode_reward = 0.0

    def collect_rollout(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run self.rollout_length steps in the environment, filling the buffer."""
        buffer.clear()
        self.policy.eval()
        episode_rewards: list[float] = []

        if self._obs is None:
            self._obs = self.env.reset()
            self._episode_reward = 0.0

        with torch.no_grad():
            for _ in range(self.rollout_length):
                obs_t = torch.from_numpy(self._obs).unsqueeze(0)
                sample = self.policy.sample_action(obs_t)
                _, value = self.policy.get_distributions(obs_t)

                next_obs, reward, done, info = self.env.step(sample.action_dict)

                buffer.observations.append(self._obs.copy())
                buffer.actions.append(sample.action_dict)
                buffer.log_probs.append(sample.log_prob.squeeze())
                buffer.rewards.append(reward)
                buffer.dones.append(done)
                buffer.values.append(value.item())

                self._obs = next_obs
                self._total_steps += 1
                self._episode_reward += reward

                if done:
                    episode_rewards.append(self._episode_reward)
                    self._episodes_done += 1
                    self._obs = self.env.reset()
                    self._episode_reward = 0.0

        # Bootstrap value for incomplete episode
        with torch.no_grad():
            obs_t = torch.from_numpy(self._obs).unsqueeze(0)
            _, last_value = self.policy.get_distributions(obs_t)
            last_value = last_value.item()

        stats = {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "episodes": len(episode_rewards),
            "total_steps": self._total_steps,
        }

        # Compute GAE
        self._compute_gae(buffer, last_value)
        return stats

    def _compute_gae(self, buffer: RolloutBuffer, last_value: float) -> None:
        """Compute generalized advantage estimates in-place."""
        n = len(buffer)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = False
            else:
                next_value = buffer.values[t + 1]
                next_done = buffer.dones[t + 1]

            mask = 0.0 if buffer.dones[t] else 1.0
            delta = buffer.rewards[t] + self.gamma * next_value * mask - buffer.values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t] = gae + buffer.values[t]

        # Store as attributes for update
        buffer._advantages = advantages  # type: ignore[attr-defined]
        buffer._returns = returns  # type: ignore[attr-defined]

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO update on the collected rollout buffer."""
        self.policy.train()
        n = len(buffer)

        # Prepare tensors
        obs_batch = torch.from_numpy(np.stack(buffer.observations))
        old_log_probs = torch.stack(buffer.log_probs).detach()
        action_tensors = batch_action_tensors(
            [action_dict_to_tensors(a) for a in buffer.actions]
        )
        advantages = torch.from_numpy(buffer._advantages)  # type: ignore[attr-defined]
        returns = torch.from_numpy(buffer._returns)  # type: ignore[attr-defined]

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            # Create random minibatch indices
            indices = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                idx = indices[start:end]
                idx_t = torch.from_numpy(idx).long()

                mb_obs = obs_batch[idx_t]
                mb_old_log_probs = old_log_probs[idx_t]
                mb_advantages = advantages[idx_t]
                mb_returns = returns[idx_t]
                mb_actions = {k: v[idx_t] for k, v in action_tensors.items()}

                new_log_probs, entropy, values = self.policy.evaluate_actions(mb_obs, mb_actions)

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                vf_loss = nn.functional.mse_loss(values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = pg_loss + self.value_coef * vf_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "pg_loss": total_pg_loss / max(num_updates, 1),
            "vf_loss": total_vf_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def train(
        self,
        total_steps: int,
        log_interval: int = 2000,
        checkpoint_interval: int = 10000,
        checkpoint_path: str | None = None,
    ) -> list[dict[str, float]]:
        """Main training loop. Returns history of stats per rollout."""
        history: list[dict[str, float]] = []
        buffer = RolloutBuffer()
        start_time = time.time()
        last_log_step = 0

        while self._total_steps < total_steps:
            rollout_stats = self.collect_rollout(buffer)
            update_stats = self.update(buffer)
            combined = {**rollout_stats, **update_stats}
            history.append(combined)

            # Log progress
            if self._total_steps - last_log_step >= log_interval:
                elapsed = time.time() - start_time
                sps = self._total_steps / max(elapsed, 1.0)
                print(
                    f"[TRAIN] steps={self._total_steps:>7d}/{total_steps} "
                    f"episodes={self._episodes_done:>4d} "
                    f"mean_reward={combined['mean_reward']:>7.3f} "
                    f"pg_loss={combined['pg_loss']:.4f} "
                    f"entropy={combined['entropy']:.2f} "
                    f"sps={sps:.0f}"
                )
                last_log_step = self._total_steps

            # Checkpoint
            if checkpoint_path and self._total_steps % checkpoint_interval < self.rollout_length:
                self.save(checkpoint_path.replace(".pt", f"_step{self._total_steps}.pt"))

        elapsed = time.time() - start_time
        print(f"[TRAIN] Done. Total steps: {self._total_steps}, Time: {elapsed:.1f}s")
        return history

    def save(self, path: str) -> None:
        """Save policy weights and normalizer state."""
        state = {"policy": self.policy.state_dict()}
        if self.env.normalizer is not None:
            state["normalizer"] = self.env.normalizer.state_dict()
        torch.save(state, path)
        print(f"[SAVE] Weights saved to {path}")

    def load(self, path: str) -> None:
        """Load policy weights and normalizer state."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(state["policy"])
        if "normalizer" in state and self.env.normalizer is not None:
            self.env.normalizer.load_state_dict(state["normalizer"])
        print(f"[LOAD] Weights loaded from {path}")

"""MLP policy + value network for mixed discrete/continuous action space.

Output heads:
  1. batch_cap       — Gaussian (mean + log_std), clipped to [1, 512]
  2. kv_budget_frac  — Gaussian (mean + log_std), clipped to [0.10, 1.0]
  3. spec_depth      — Categorical over 9 values (0–8)
  4. quant_tier      — Categorical over 3 values (FP16, INT8, INT4)
  5. prefill_split   — Bernoulli (single logit)
  6. priority_route  — Bernoulli (single logit)

Total params ~40k — small enough for fast CPU training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, Normal

from llmserve_env.models import QuantizationTier, ServeAction


QUANT_OPTIONS = [QuantizationTier.FP16.value, QuantizationTier.INT8.value, QuantizationTier.INT4.value]


@dataclass
class ActionSample:
    """Container for a sampled action and its log-probability."""
    action_dict: dict[str, Any]
    log_prob: torch.Tensor
    entropy: torch.Tensor


class PolicyNetwork(nn.Module):
    """Shared-trunk MLP with 6 output heads for mixed action space."""

    def __init__(self, obs_dim: int = 15, hidden: int = 128, hidden2: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
        )

        # --- Continuous heads (Gaussian) ---
        self.batch_cap_mean = nn.Linear(hidden2, 1)
        self.batch_cap_log_std = nn.Parameter(torch.zeros(1))
        self.kv_budget_mean = nn.Linear(hidden2, 1)
        self.kv_budget_log_std = nn.Parameter(torch.zeros(1))

        # --- Discrete heads ---
        self.spec_depth_logits = nn.Linear(hidden2, 9)    # 0–8
        self.quant_tier_logits = nn.Linear(hidden2, 3)    # FP16, INT8, INT4
        self.prefill_split_logit = nn.Linear(hidden2, 1)  # Bernoulli
        self.priority_route_logit = nn.Linear(hidden2, 1) # Bernoulli

        # --- Value head (separate final layer) ---
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
        """Return distribution parameters and value estimate."""
        features = self.trunk(obs)
        value = self.value_head(obs).squeeze(-1)
        return {
            "batch_cap_mean": self.batch_cap_mean(features).squeeze(-1),
            "batch_cap_log_std": self.batch_cap_log_std.expand_as(self.batch_cap_mean(features).squeeze(-1)),
            "kv_budget_mean": self.kv_budget_mean(features).squeeze(-1),
            "kv_budget_log_std": self.kv_budget_log_std.expand_as(self.kv_budget_mean(features).squeeze(-1)),
            "spec_depth_logits": self.spec_depth_logits(features),
            "quant_tier_logits": self.quant_tier_logits(features),
            "prefill_split_logit": self.prefill_split_logit(features).squeeze(-1),
            "priority_route_logit": self.priority_route_logit(features).squeeze(-1),
        }, value

    def get_distributions(self, obs: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
        """Build actual distribution objects from network outputs."""
        params, value = self.forward(obs)
        dists = {
            "batch_cap": Normal(params["batch_cap_mean"], params["batch_cap_log_std"].exp().clamp(min=0.01)),
            "kv_budget": Normal(params["kv_budget_mean"], params["kv_budget_log_std"].exp().clamp(min=0.01)),
            "spec_depth": Categorical(logits=params["spec_depth_logits"]),
            "quant_tier": Categorical(logits=params["quant_tier_logits"]),
            "prefill_split": Bernoulli(logits=params["prefill_split_logit"]),
            "priority_route": Bernoulli(logits=params["priority_route_logit"]),
        }
        return dists, value

    def sample_action(self, obs: torch.Tensor) -> ActionSample:
        """Sample an action from the policy and compute log-probability."""
        dists, _ = self.get_distributions(obs)

        # Sample from each head
        batch_cap_raw = dists["batch_cap"].sample()
        kv_budget_raw = dists["kv_budget"].sample()
        spec_depth_idx = dists["spec_depth"].sample()
        quant_tier_idx = dists["quant_tier"].sample()
        prefill_split = dists["prefill_split"].sample()
        priority_route = dists["priority_route"].sample()

        # Compute joint log-prob as sum of individual log-probs
        log_prob = (
            dists["batch_cap"].log_prob(batch_cap_raw)
            + dists["kv_budget"].log_prob(kv_budget_raw)
            + dists["spec_depth"].log_prob(spec_depth_idx)
            + dists["quant_tier"].log_prob(quant_tier_idx)
            + dists["prefill_split"].log_prob(prefill_split)
            + dists["priority_route"].log_prob(priority_route)
        )

        # Compute joint entropy
        entropy = (
            dists["batch_cap"].entropy()
            + dists["kv_budget"].entropy()
            + dists["spec_depth"].entropy()
            + dists["quant_tier"].entropy()
            + dists["prefill_split"].entropy()
            + dists["priority_route"].entropy()
        )

        # Clip continuous values to valid ranges
        batch_cap = int(torch.clamp(batch_cap_raw, 1.0, 512.0).round().item())
        kv_budget = float(torch.clamp(kv_budget_raw, 0.10, 1.0).item())

        action_dict = {
            "batch_cap": batch_cap,
            "kv_budget_fraction": round(kv_budget, 2),
            "speculation_depth": int(spec_depth_idx.item()),
            "quantization_tier": QUANT_OPTIONS[int(quant_tier_idx.item())],
            "prefill_decode_split": bool(prefill_split.item() > 0.5),
            "priority_routing": bool(priority_route.item() > 0.5),
        }
        return ActionSample(action_dict=action_dict, log_prob=log_prob, entropy=entropy)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-probs, entropy, and values for stored actions (for PPO update)."""
        dists, values = self.get_distributions(obs)

        log_prob = (
            dists["batch_cap"].log_prob(actions["batch_cap"])
            + dists["kv_budget"].log_prob(actions["kv_budget"])
            + dists["spec_depth"].log_prob(actions["spec_depth"])
            + dists["quant_tier"].log_prob(actions["quant_tier"])
            + dists["prefill_split"].log_prob(actions["prefill_split"])
            + dists["priority_route"].log_prob(actions["priority_route"])
        )
        entropy = (
            dists["batch_cap"].entropy()
            + dists["kv_budget"].entropy()
            + dists["spec_depth"].entropy()
            + dists["quant_tier"].entropy()
            + dists["prefill_split"].entropy()
            + dists["priority_route"].entropy()
        )
        return log_prob, entropy, values


def action_dict_to_tensors(action_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert an action dict into tensors for evaluate_actions."""
    return {
        "batch_cap": torch.tensor(float(action_dict["batch_cap"]), dtype=torch.float32),
        "kv_budget": torch.tensor(float(action_dict["kv_budget_fraction"]), dtype=torch.float32),
        "spec_depth": torch.tensor(
            action_dict["speculation_depth"], dtype=torch.long
        ),
        "quant_tier": torch.tensor(
            QUANT_OPTIONS.index(action_dict["quantization_tier"]), dtype=torch.long
        ),
        "prefill_split": torch.tensor(
            1.0 if action_dict["prefill_decode_split"] else 0.0, dtype=torch.float32
        ),
        "priority_route": torch.tensor(
            1.0 if action_dict["priority_routing"] else 0.0, dtype=torch.float32
        ),
    }


def batch_action_tensors(action_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of single-step action tensors into batched tensors."""
    keys = action_list[0].keys()
    return {k: torch.stack([a[k] for a in action_list]) for k in keys}

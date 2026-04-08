#!/usr/bin/env python3
"""PPO agent — loads pre-trained weights and runs inference only.

Usage:
    from agents.ppo_agent import PPOAgent
    agent = PPOAgent("weights/ppo_task1_static.pt")
    action = agent.act(observation, task_id)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from llmserve_env.models import ServeAction, ServeObservation  # noqa: E402
from rl.env_wrapper import obs_to_vector  # noqa: E402
from rl.normalize import RunningNormalizer  # noqa: E402
from rl.policy_network import PolicyNetwork  # noqa: E402


class PPOAgent:
    """Inference-only agent that loads trained PPO weights."""

    def __init__(self, weights_path: str, obs_dim: int = 15) -> None:
        self.policy = PolicyNetwork(obs_dim=obs_dim)
        self.normalizer: RunningNormalizer | None = None

        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(state["policy"])
        self.policy.eval()

        if "normalizer" in state:
            self.normalizer = RunningNormalizer(shape=(obs_dim,))
            self.normalizer.load_state_dict(state["normalizer"])

    def reset(self) -> None:
        pass  # No internal state to reset

    def act(self, observation: ServeObservation, task_id: str) -> ServeAction:
        """Select a deterministic action from the trained policy."""
        del task_id
        vec = obs_to_vector(observation)
        if self.normalizer is not None:
            vec = self.normalizer.normalize(vec)

        with torch.no_grad():
            obs_t = torch.from_numpy(vec).unsqueeze(0)
            params, _ = self.policy.forward(obs_t)

        batch_cap = int(torch.clamp(params["batch_cap_mean"], 1.0, 512.0).round().item())
        kv_budget = float(torch.clamp(params["kv_budget_mean"], 0.10, 1.0).item())
        spec_depth = int(torch.argmax(params["spec_depth_logits"], dim=-1).item())
        quant_tier = int(torch.argmax(params["quant_tier_logits"], dim=-1).item())
        prefill_split = bool((params["prefill_split_logit"] > 0).item())
        priority_route = bool((params["priority_route_logit"] > 0).item())

        return ServeAction(
            batch_cap=batch_cap,
            kv_budget_fraction=round(kv_budget, 2),
            speculation_depth=spec_depth,
            quantization_tier=["FP16", "INT8", "INT4"][quant_tier],
            prefill_decode_split=prefill_split,
            priority_routing=priority_route,
        )


def find_weights(task_id: str) -> str | None:
    """Find the weights file for a given task_id."""
    label_map = {
        "static_workload": "task1_static",
        "bursty_workload": "task2_bursty",
        "adversarial_multitenant": "task3_adversarial",
    }
    label = label_map.get(task_id)
    if not label:
        return None
    weights_dir = Path(__file__).resolve().parents[1] / "weights"
    path = weights_dir / f"ppo_{label}.pt"
    return str(path) if path.exists() else None

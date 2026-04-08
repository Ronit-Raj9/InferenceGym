#!/usr/bin/env python3
"""Heuristic agent — reactive policy based on Orca / vLLM / Decima rules.

Usage:
    python agents/heuristic_agent.py            # run from repo root
    python agents/heuristic_agent.py --episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.baseline_agent import HeuristicPolicy  # noqa: E402
from server.llmserve_environment import LLMServeEnvironment  # noqa: E402

TASK_IDS = ["static_workload", "bursty_workload", "adversarial_multitenant"]
DEFAULT_SEED = 42


def run_episode(env: LLMServeEnvironment, task_id: str, seed: int, policy: HeuristicPolicy) -> float:
    policy.reset()
    obs = env.reset(seed=seed, task_id=task_id)
    task_cfg = env.task_config
    max_steps = int(task_cfg["max_steps"]) if task_cfg else 60
    total_reward = 0.0
    for _ in range(max_steps):
        action = policy.act(obs, task_id)
        obs = env.step(action)
        total_reward += getattr(obs, "reward", 0.0) or 0.0
        if getattr(obs, "done", False):
            break
    return total_reward


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Heuristic agent benchmark")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    env = LLMServeEnvironment(seed=args.seed, mode="sim")
    policy = HeuristicPolicy()

    results: dict[str, dict] = {}
    for task_id in TASK_IDS:
        rewards = []
        for ep in range(args.episodes):
            ep_seed = args.seed + ep
            r = run_episode(env, task_id, ep_seed, policy)
            rewards.append(r)
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        results[task_id] = {"mean_reward": round(mean_r, 4), "std_reward": round(std_r, 4), "episodes": args.episodes}
        print(f"[HEURISTIC] task={task_id}  mean_reward={mean_r:.4f} ± {std_r:.4f}  episodes={args.episodes}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

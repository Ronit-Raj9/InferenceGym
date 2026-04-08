#!/usr/bin/env python3
"""Evaluate agents on InferenceGym tasks and print benchmark table.

Usage:
    python evaluate.py --agent ppo --task all --episodes 20 --seed 42
    python evaluate.py --agent heuristic --task static_workload --episodes 10
    python evaluate.py --agent random --task all --episodes 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from server.llmserve_environment import LLMServeEnvironment  # noqa: E402

TASK_IDS = ["static_workload", "bursty_workload", "adversarial_multitenant"]
AGENT_TYPES = ["random", "heuristic", "ppo"]
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def _get_agent(agent_type: str, task_id: str):
    """Return an agent object with a .act(obs, task_id) method."""
    if agent_type == "heuristic":
        from server.baseline_agent import HeuristicPolicy
        return HeuristicPolicy()

    if agent_type == "random":
        import random as rnd
        from agents.random_agent import random_action
        rng = rnd.Random(42)

        class _RandomAgent:
            def reset(self): pass
            def act(self, obs, tid): return random_action(rng)

        return _RandomAgent()

    if agent_type == "ppo":
        from agents.ppo_agent import PPOAgent
        label_map = {
            "static_workload": "task1_static",
            "bursty_workload": "task2_bursty",
            "adversarial_multitenant": "task3_adversarial",
        }
        label = label_map.get(task_id, "task1_static")
        weight_path = WEIGHTS_DIR / f"ppo_{label}.pt"
        if not weight_path.exists():
            print(f"[WARN] PPO weights not found at {weight_path}, falling back to heuristic")
            from server.baseline_agent import HeuristicPolicy
            return HeuristicPolicy()
        return PPOAgent(str(weight_path))

    raise ValueError(f"Unknown agent type: {agent_type}")


def run_episode(env: LLMServeEnvironment, agent, task_id: str, seed: int) -> float:
    if hasattr(agent, "reset"):
        agent.reset()
    obs = env.reset(seed=seed, task_id=task_id)
    task_cfg = env.task_config
    max_steps = int(task_cfg["max_steps"]) if task_cfg else 60
    total_reward = 0.0
    for _ in range(max_steps):
        action = agent.act(obs, task_id)
        obs = env.step(action)
        total_reward += float(getattr(obs, "reward", 0.0) or 0.0)
        if getattr(obs, "done", False):
            break
    return total_reward


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate agents on InferenceGym")
    parser.add_argument("--agent", default="ppo", choices=AGENT_TYPES + ["all"])
    parser.add_argument("--task", default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    tasks = TASK_IDS if args.task == "all" else [args.task]
    env = LLMServeEnvironment(seed=args.seed, mode="sim")

    results = {}
    selected_agents = AGENT_TYPES if args.agent == "all" else [args.agent]

    print(f"\n{'Agent':<12} {'Task':<28} {'Mean Reward':>12} {'Std':>8} {'Episodes':>9}")
    print("-" * 72)

    for agent_type in selected_agents:
        agent_results = {}
        for task_id in tasks:
            agent = _get_agent(agent_type, task_id)
            rewards = []
            for ep in range(args.episodes):
                r = run_episode(env, agent, task_id, args.seed + ep)
                rewards.append(r)
            mean_r = float(np.mean(rewards))
            std_r = float(np.std(rewards))
            agent_results[task_id] = {"mean_reward": round(mean_r, 4), "std_reward": round(std_r, 4), "episodes": args.episodes}
            print(f"{agent_type:<12} {task_id:<28} {mean_r:>12.4f} {std_r:>8.4f} {args.episodes:>9d}")
        if args.agent == "all":
            results[agent_type] = agent_results
        else:
            results = agent_results

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print(f"\n{json.dumps(results, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

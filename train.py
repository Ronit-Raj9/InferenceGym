#!/usr/bin/env python3
"""Train a PPO agent on an InferenceGym task.

Usage:
    python train.py --task static_workload --steps 50000 --seed 42
    python train.py --task bursty_workload --steps 80000 --seed 42
    python train.py --task adversarial_multitenant --steps 120000 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from rl.env_wrapper import GymEnvWrapper  # noqa: E402
from rl.policy_network import PolicyNetwork  # noqa: E402
from rl.ppo import PPOTrainer  # noqa: E402

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

TASK_DEFAULTS = {
    "static_workload": {"steps": 50_000, "label": "task1_static"},
    "bursty_workload": {"steps": 80_000, "label": "task2_bursty"},
    "adversarial_multitenant": {"steps": 120_000, "label": "task3_adversarial"},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PPO on InferenceGym")
    parser.add_argument("--task", default="static_workload", choices=list(TASK_DEFAULTS.keys()))
    parser.add_argument("--steps", type=int, default=None, help="Total training steps (default: task-specific)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--output", type=str, default=None, help="Output weights path")
    args = parser.parse_args(argv)

    task_id = args.task
    defaults = TASK_DEFAULTS[task_id]
    total_steps = args.steps or defaults["steps"]
    label = defaults["label"]

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(WEIGHTS_DIR / f"ppo_{label}.pt")

    print(f"[TRAIN] Task: {task_id}, Steps: {total_steps}, Seed: {args.seed}")
    print(f"[TRAIN] Output: {output_path}")

    # Seed everything
    torch.manual_seed(args.seed)

    env = GymEnvWrapper(task_id=task_id, seed=args.seed, normalize=True, mode="sim")
    policy = PolicyNetwork(obs_dim=env.obs_dim)
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        lr=args.lr,
        rollout_length=args.rollout,
        ppo_epochs=args.epochs,
        minibatch_size=args.minibatch,
        entropy_coef=args.entropy,
    )

    history = trainer.train(
        total_steps=total_steps,
        log_interval=2000,
        checkpoint_interval=10000,
        checkpoint_path=output_path,
    )

    # Save final weights
    trainer.save(output_path)

    # Print summary
    if history:
        final_rewards = [h["mean_reward"] for h in history if h["mean_reward"] != 0.0]
        if final_rewards:
            print(f"\n[SUMMARY] Final mean reward: {final_rewards[-1]:.4f}")
            print(f"[SUMMARY] Best mean reward:  {max(final_rewards):.4f}")
            print(f"[SUMMARY] Episodes trained:  {history[-1].get('total_steps', 0) // 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

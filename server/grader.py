from __future__ import annotations

from llmserve_env.models import EpisodeLog
from server.baseline_agent import HeuristicPolicy
from server.llmserve_environment import LLMServeEnvironment
from server.optimal_solver import OptimalSolver


class GraderEngine:
    _shared_ppo_baselines: dict[str, float] = {}
    _shared_heuristic_baselines: dict[str, float] = {}

    def __init__(self) -> None:
        self.optimal_solver = OptimalSolver()
        self._ppo_baselines = self._shared_ppo_baselines
        self._heuristic_baselines = self._shared_heuristic_baselines

    def _run_policy_episode(self, task_id: str, seed: int, policy) -> float:
        env = LLMServeEnvironment(seed=seed, mode="sim")
        if hasattr(policy, "reset"):
            policy.reset()
        observation = env.reset(seed=seed, task_id=task_id)
        task_cfg = env.task_config or {}
        max_steps = int(task_cfg.get("max_steps", 60))
        for _ in range(max_steps):
            action = policy.act(observation, task_id)
            observation = env.step(action)
            if bool(getattr(observation, "done", False)):
                break
        raw_score, _ = self._compute_raw_score(env.export_episode_log())
        return raw_score

    def get_ppo_baseline(self, task_id: str) -> float:
        if task_id in self._ppo_baselines:
            return self._ppo_baselines[task_id]

        try:
            from agents.ppo_agent import PPOAgent, find_weights

            weights_path = find_weights(task_id)
            if not weights_path:
                heuristic_baseline = self.get_heuristic_baseline(task_id)
                self._ppo_baselines[task_id] = heuristic_baseline
                return heuristic_baseline

            agent = PPOAgent(weights_path)
            baseline = self._run_policy_episode(task_id, 42, agent)
            self._ppo_baselines[task_id] = baseline
            return baseline
        except Exception:
            heuristic_baseline = self.get_heuristic_baseline(task_id)
            self._ppo_baselines[task_id] = heuristic_baseline
            return heuristic_baseline

    def get_heuristic_baseline(self, task_id: str) -> float:
        if task_id in self._heuristic_baselines:
            return self._heuristic_baselines[task_id]

        policy = HeuristicPolicy()
        baseline = self._run_policy_episode(task_id, 142, policy)
        self._heuristic_baselines[task_id] = baseline
        return baseline

    def _compute_raw_score(self, episode_log: EpisodeLog) -> tuple[float, dict[str, float]]:
        observations = episode_log.observations
        if not observations:
            return 0.0, {"throughput": 0.0, "slo": 0.0, "memory": 0.0, "cost": 0.0}
            
        oracle = self.optimal_solver.oracle_reference(episode_log.task_id)
        mean_throughput = sum(obs.throughput_tps for obs in observations) / len(observations)
        mean_slo = sum(obs.slo_compliance_rate for obs in observations) / len(observations)
        mean_memory = sum(obs.gpu_memory_used_gb for obs in observations) / len(observations)
        mean_cost = sum(obs.estimated_cost_per_1k for obs in observations) / len(observations)

        throughput_component = min(1.0, mean_throughput / oracle["throughput_tps"])
        slo_component = min(1.0, mean_slo / oracle["slo_compliance_rate"])
        memory_component = max(0.0, 1.0 - max(0.0, mean_memory - 38.0) / 38.0)
        cost_component = max(0.0, 1.0 - max(0.0, mean_cost - oracle["cost_per_1k"]) / max(oracle["cost_per_1k"], 1e-6))

        score = (
            0.30 * throughput_component
            + 0.35 * slo_component
            + 0.20 * memory_component
            + 0.15 * cost_component
        )
        return max(0.0, min(1.0, score)), {
            "throughput": round(throughput_component, 4),
            "slo": round(slo_component, 4),
            "memory": round(memory_component, 4),
            "cost": round(cost_component, 4),
        }

    def grade(self, episode_log: EpisodeLog, actions_taken: int | None = None) -> dict[str, object]:
        resolved_actions_taken = actions_taken if actions_taken is not None else len(episode_log.actions)
        if not episode_log.observations:
            return {
                "task_id": episode_log.task_id,
                "actions_taken": resolved_actions_taken,
                "score": 0.0,
                "breakdown": {"throughput": 0.0, "slo": 0.0, "memory": 0.0, "cost": 0.0},
            }
            
        raw_score, breakdown = self._compute_raw_score(episode_log)
        heuristic_baseline = self.get_heuristic_baseline(episode_log.task_id)
        ppo_baseline = self.get_ppo_baseline(episode_log.task_id)
        anchor = max(heuristic_baseline, ppo_baseline, 1e-6)

        if raw_score <= anchor:
            final_score = 0.5 * (raw_score / anchor)
        else:
            final_score = 0.5 + 0.5 * ((raw_score - anchor) / max(1.0 - anchor, 1e-6))

        return {
            "task_id": episode_log.task_id,
            "actions_taken": resolved_actions_taken,
            "score": round(max(0.0, min(1.0, final_score)), 4),
            "breakdown": breakdown,
            "heuristic_baseline": round(heuristic_baseline, 4),
            "ppo_baseline": round(ppo_baseline, 4),
            "raw_score": round(raw_score, 4),
        }

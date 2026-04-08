"""Tests for the grader, baseline agent, and score calibration."""
from __future__ import annotations

from llmserve_env.models import default_action
from server.grader import GraderEngine
from server.llmserve_environment import LLMServeEnvironment


def _run_episode(task_id: str, seed: int = 42) -> LLMServeEnvironment:
    env = LLMServeEnvironment(seed=seed)
    env.reset(task_id=task_id, seed=seed)
    while not env.state.done:
        env.step(default_action())
    return env


# ─── Grader ───────────────────────────────────────────────────────

class TestGrader:
    def test_score_in_valid_range(self):
        grader = GraderEngine()
        for task_id in ["static_workload", "bursty_workload", "adversarial_multitenant"]:
            env = _run_episode(task_id)
            result = grader.grade(env.export_episode_log())
            assert 0.0 <= result["score"] <= 1.0, f"Score out of range for {task_id}: {result['score']}"

    def test_score_has_breakdown(self):
        grader = GraderEngine()
        env = _run_episode("static_workload")
        result = grader.grade(env.export_episode_log())
        assert "breakdown" in result
        breakdown = result["breakdown"]
        assert "throughput" in breakdown
        assert "slo" in breakdown
        assert "memory" in breakdown
        assert "cost" in breakdown

    def test_empty_log_returns_zero(self):
        from llmserve_env.models import EpisodeLog, ServeState
        grader = GraderEngine()
        empty_log = EpisodeLog(
            task_id="static_workload",
            actions=[],
            observations=[],
            rewards=[],
            final_state=ServeState(
                episode_id="test",
                step_count=0,
                task_id="static_workload",
                total_requests_served=0,
                total_slo_violations=0,
                cumulative_reward=0.0,
                elapsed_simulated_time_s=0.0,
                workload_phase="warmup",
                done=True,
            ),
        )
        result = grader.grade(empty_log)
        assert result["score"] == 0.0

    def test_grader_is_deterministic(self):
        grader = GraderEngine()
        env = _run_episode("static_workload", seed=0)
        log = env.export_episode_log()
        score_1 = grader.grade(log)["score"]
        score_2 = grader.grade(log)["score"]
        assert score_1 == score_2


# ─── Baseline ─────────────────────────────────────────────────────

class TestBaseline:
    def test_baseline_scores_all_tasks(self):
        grader = GraderEngine()
        for task_id in ["static_workload", "bursty_workload", "adversarial_multitenant"]:
            env = _run_episode(task_id, seed=0)
            result = grader.grade(env.export_episode_log())
            assert 0.0 < result["score"] <= 1.0, f"Baseline score too low for {task_id}: {result['score']}"

    def test_baseline_deterministic_across_runs(self):
        grader = GraderEngine()
        scores = []
        for _ in range(3):
            env = _run_episode("static_workload", seed=0)
            result = grader.grade(env.export_episode_log())
            scores.append(result["score"])
        assert all(s == scores[0] for s in scores), f"Baseline scores not deterministic: {scores}"


# ─── Score Ordering ───────────────────────────────────────────────

class TestScoreOrdering:
    def test_breakdown_components_bounded(self):
        grader = GraderEngine()
        for task_id in ["static_workload", "bursty_workload", "adversarial_multitenant"]:
            env = _run_episode(task_id)
            result = grader.grade(env.export_episode_log())
            for key, val in result["breakdown"].items():
                assert 0.0 <= val <= 1.0, f"{key} out of [0,1] for {task_id}: {val}"

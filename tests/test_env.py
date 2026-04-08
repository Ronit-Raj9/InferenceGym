from __future__ import annotations

import pytest

from llmserve_env.models import ServeAction, default_action
from server.llmserve_environment import LLMServeEnvironment


def _make_env(task_id: str = "static_workload", seed: int = 42) -> LLMServeEnvironment:
    env = LLMServeEnvironment(seed=seed)
    env.reset(task_id=task_id, seed=seed)
    return env


def test_reset_returns_observation() -> None:
    env = _make_env()
    obs = env.observations[-1]
    assert obs.task_id == "static_workload"
    assert obs.step_index == 0
    assert obs.done is False


def test_reset_respects_requested_task_id() -> None:
    env = _make_env(task_id="adversarial_multitenant")
    obs = env.observations[-1]
    assert env.state.task_id == "adversarial_multitenant"
    assert obs.task_id == "adversarial_multitenant"
    assert obs.metadata["task_name"] == "Adversarial Multi-Tenant Serving"


def test_serve_action_defaults_are_valid() -> None:
    action = ServeAction()
    assert action.batch_cap >= 1
    assert action.kv_budget_fraction >= 0.1


def test_serve_action_normalizes_invalid_web_values() -> None:
    action = ServeAction(
        batch_cap=0,
        kv_budget_fraction=30,
        speculation_depth=40,
        quantization_tier="8",
    )
    assert action.batch_cap == 1
    assert action.kv_budget_fraction == 1.0
    assert action.speculation_depth == 8
    assert action.quantization_tier == "FP16"


def test_serve_action_schema_exposes_quantization_enum() -> None:
    schema = ServeAction.model_json_schema()
    field = schema["properties"]["quantization_tier"]
    assert field["enum"] == ["FP16", "INT8", "INT4"]


def test_reset_creates_unique_episode_id() -> None:
    env = LLMServeEnvironment(seed=1)
    env.reset(task_id="static_workload", seed=1)
    first = env.state.episode_id
    env.reset(task_id="static_workload", seed=2)
    second = env.state.episode_id
    assert first != second


def test_step_returns_observation_with_reward() -> None:
    env = _make_env()
    obs = env.step(default_action())
    assert obs.step_index == 1
    assert isinstance(obs.reward, float)
    assert isinstance(obs.done, bool)


def test_step_before_reset_raises() -> None:
    env = LLMServeEnvironment(seed=2)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(default_action())


def test_step_updates_state() -> None:
    env = _make_env()
    env.step(default_action())
    assert env.state.step_count == 1
    assert env.state.elapsed_simulated_time_s > 0


def test_done_after_max_steps() -> None:
    env = _make_env("static_workload")
    obs = env.observations[-1]
    while not obs.done:
        obs = env.step(default_action())
    assert env.state.done is True
    repeated = env.step(default_action())
    assert repeated.done is True
    assert "message" in repeated.metadata


def test_export_episode_log() -> None:
    env = _make_env()
    for _ in range(3):
        env.step(default_action())
    log = env.export_episode_log()
    assert len(log.actions) == 3
    assert len(log.rewards) == 3
    assert len(log.observations) == 4

from __future__ import annotations

import asyncio
import pytest
from fastapi import HTTPException

from server.app import create_application, shared_env
from server.schemas import ResetRequest, StepRequest


def _route_map():
    app = create_application(enable_web=False)
    return {getattr(route, "path", None): route.endpoint for route in app.routes}


def _call(endpoint, *args, **kwargs):
    result = endpoint(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


def test_required_routes_registered() -> None:
    routes = _route_map()
    for path in ["/health", "/metadata", "/schema", "/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/demo"]:
        assert path in routes


def test_health_endpoint_direct() -> None:
    data = _call(_route_map()["/health"])
    status = data["status"] if isinstance(data, dict) else data.status
    assert status in {"ok", "healthy"}


def test_tasks_endpoint_direct() -> None:
    data = _call(_route_map()["/tasks"])
    assert len(data["tasks"]) == 3
    assert "batch_cap" in data["action_schema"]


def test_grader_requires_episode_before_grading() -> None:
    shared_env.actions.clear()
    shared_env.observations.clear()
    shared_env.rewards.clear()
    with pytest.raises(HTTPException):
        _call(_route_map()["/grader"])


def test_baseline_endpoint_direct() -> None:
    data = _call(_route_map()["/baseline"], task_id="static_workload", use_openai=False, model="gpt-4.1-mini")
    assert data["mode"] == "deterministic"
    assert "static_workload" in data["baseline"]
    assert data["baseline"]["static_workload"]["grader"]["score"] >= 0.25


def test_demo_redirects_to_web() -> None:
    response = _call(_route_map()["/demo"])
    assert response.headers["location"] == "/web"


def test_http_session_advances_across_multiple_steps() -> None:
    routes = _route_map()
    reset_endpoint = routes["/reset"]
    step_endpoint = routes["/step"]
    state_endpoint = routes["/state"]

    reset_payload = _call(reset_endpoint, ResetRequest(task_id="bursty_workload", seed=42))
    session_id = reset_payload["session_id"]
    assert session_id
    assert reset_payload["observation"]["step_index"] == 0

    action = {
        "batch_cap": 32,
        "kv_budget_fraction": 1.0,
        "speculation_depth": 0,
        "quantization_tier": "FP16",
        "prefill_decode_split": False,
        "priority_routing": False,
    }
    first_payload = _call(step_endpoint, StepRequest(session_id=session_id, action=action))
    second_payload = _call(step_endpoint, StepRequest(session_id=session_id, action=action))
    assert first_payload["observation"]["step_index"] == 1
    assert second_payload["observation"]["step_index"] == 2
    assert first_payload["reward"] != second_payload["reward"]

    state_payload = _call(state_endpoint, session_id=session_id)
    assert state_payload["step_count"] == 2

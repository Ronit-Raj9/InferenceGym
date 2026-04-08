from __future__ import annotations

from types import SimpleNamespace

from llmserve_env.models import ServeAction, WorkloadSnapshot, default_action
from server.serving_backend import RealOpenAIBackend, SimulatedServingBackend, create_serving_backend


def _workload() -> WorkloadSnapshot:
    return WorkloadSnapshot(
        arrival_rate=8.0,
        queue_depth=5,
        mean_prompt_length=128.0,
        prompt_length_bucket=1,
        priority_fraction=0.25,
        phase="steady",
    )


class _FakeChatCompletions:
    def create(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40, total_tokens=160),
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def test_create_serving_backend_default_is_sim() -> None:
    backend = create_serving_backend(mode="sim", seed=42)
    assert isinstance(backend, SimulatedServingBackend)


def test_real_openai_backend_produces_metrics_without_network() -> None:
    backend = RealOpenAIBackend(seed=1, client=_FakeClient(), model="gpt-4.1-mini", max_requests_per_step=2)
    metrics = backend.run_step("static_workload", default_action(), _workload())
    assert metrics.requests_served == 2
    assert metrics.throughput_tps >= 1.0
    assert metrics.estimated_cost_per_1k > 0.0
    assert metrics.p50_ttft_ms > 0.0
    assert metrics.p50_itl_ms > 0.0


def test_real_backend_respects_truncation_via_kv_budget() -> None:
    backend = RealOpenAIBackend(seed=1, client=_FakeClient(), model="gpt-4.1-mini", max_requests_per_step=1)
    action = ServeAction(batch_cap=1, kv_budget_fraction=0.1, speculation_depth=0, quantization_tier="FP16")
    metrics = backend.run_step("static_workload", action, _workload())
    assert metrics.eviction_events >= 1

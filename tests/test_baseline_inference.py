from __future__ import annotations

from server.baseline_inference import DEFAULT_SEED, run_baseline_suite


def test_deterministic_baseline_suite_returns_summary() -> None:
    payload = run_baseline_suite(mode="deterministic", seed=DEFAULT_SEED)
    assert payload["mode"] == "deterministic"
    assert payload["summary"]["task_count"] == 3
    assert 0.0 < payload["summary"]["mean_score"] <= 1.0
    assert set(payload["baseline"]) == {
        "static_workload",
        "bursty_workload",
        "adversarial_multitenant",
    }


def test_deterministic_baseline_suite_is_reproducible() -> None:
    first = run_baseline_suite(mode="deterministic", seed=DEFAULT_SEED)
    second = run_baseline_suite(mode="deterministic", seed=DEFAULT_SEED)
    assert first["summary"] == second["summary"]
    assert first["baseline"] == second["baseline"]

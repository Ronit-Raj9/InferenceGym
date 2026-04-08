from __future__ import annotations

from pathlib import Path

from llmserve_env.task_catalog import get_task_config
from server.trace_simulator import TraceSimulator
from server.workload_generator import WorkloadGenerator


ROOT_DIR = Path(__file__).resolve().parents[1]


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = int((len(ordered) - 1) * pct)
    return ordered[index]


def test_replay_assets_exist() -> None:
    expected = [
        ROOT_DIR / "server" / "data" / "traces" / "static_workload_trace.parquet",
        ROOT_DIR / "server" / "data" / "traces" / "bursty_workload_trace.parquet",
        ROOT_DIR / "server" / "data" / "traces" / "adversarial_multitenant_trace.parquet",
        ROOT_DIR / "server" / "data" / "traces" / "sharegpt_prompt_lengths.parquet",
        ROOT_DIR / "server" / "data" / "lookup_tables" / "serving_profile_table.parquet",
    ]
    for path in expected:
        assert path.exists(), f"Missing replay asset: {path}"


def test_bursty_workload_prompt_distribution_is_heavy_tailed() -> None:
    generator = WorkloadGenerator(get_task_config("bursty_workload"), seed=7)
    samples = [generator.next_snapshot(step).mean_prompt_length for step in range(200)]
    p50 = _percentile(samples, 0.50)
    p95 = _percentile(samples, 0.95)
    assert p95 > p50 * 3.0


def test_trace_simulator_is_deterministic_for_same_seed() -> None:
    config = get_task_config("bursty_workload")
    generator_a = WorkloadGenerator(config, seed=21)
    generator_b = WorkloadGenerator(config, seed=21)
    workload_a = generator_a.next_snapshot(15)
    workload_b = generator_b.next_snapshot(15)
    simulator_a = TraceSimulator(seed=21)
    simulator_b = TraceSimulator(seed=21)
    from llmserve_env.models import default_action

    metrics_a = simulator_a.simulate_step("bursty_workload", default_action(), workload_a)
    metrics_b = simulator_b.simulate_step("bursty_workload", default_action(), workload_b)
    assert metrics_a == metrics_b

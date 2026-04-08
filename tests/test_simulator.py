"""Comprehensive tests for the trace simulator and sub-components."""
from __future__ import annotations

from llmserve_env.models import QuantizationTier, ServeAction, WorkloadSnapshot
from server.kv_cache_simulator import KVCacheSimulator
from server.speculative_decoder import SpeculativeDecoder
from server.trace_simulator import TraceSimulator


def _make_action(**overrides) -> ServeAction:
    defaults = dict(
        batch_cap=32,
        kv_budget_fraction=1.0,
        speculation_depth=0,
        quantization_tier=QuantizationTier.FP16,
        prefill_decode_split=False,
        priority_routing=False,
    )
    defaults.update(overrides)
    return ServeAction(**defaults)


def _make_workload(**overrides) -> WorkloadSnapshot:
    defaults = dict(
        arrival_rate=10.0,
        queue_depth=20,
        mean_prompt_length=128.0,
        prompt_length_bucket=1,
        priority_fraction=0.0,
        phase="steady",
    )
    defaults.update(overrides)
    return WorkloadSnapshot(**defaults)


# ─── TraceSimulator ───────────────────────────────────────────────

class TestTraceSimulatorSmoke:
    """Basic smoke tests: simulator never crashes on valid input."""

    def test_returns_metrics_snapshot(self):
        sim = TraceSimulator()
        metrics = sim.simulate_step("static_workload", _make_action(), _make_workload())
        assert metrics.throughput_tps > 0
        assert metrics.p50_ttft_ms > 0
        assert metrics.p99_ttft_ms >= metrics.p50_ttft_ms
        assert metrics.gpu_memory_used_gb > 0
        assert metrics.estimated_cost_per_1k > 0

    def test_all_tasks_produce_metrics(self):
        sim = TraceSimulator()
        for task_id in ["static_workload", "bursty_workload", "adversarial_multitenant"]:
            metrics = sim.simulate_step(task_id, _make_action(), _make_workload())
            assert metrics.throughput_tps >= 1.0

    def test_varied_actions_no_crash(self):
        sim = TraceSimulator()
        for batch in [1, 8, 64, 256, 512]:
            for kv in [0.1, 0.5, 1.0]:
                for spec in [0, 2, 8]:
                    action = _make_action(batch_cap=batch, kv_budget_fraction=kv, speculation_depth=spec)
                    metrics = sim.simulate_step("static_workload", action, _make_workload())
                    assert metrics.throughput_tps >= 1.0
                    assert metrics.requests_served >= 0


class TestTraceSimulatorMonotonicity:
    """Higher batch_cap should generally increase throughput."""

    def test_throughput_increases_with_batch(self):
        sim = TraceSimulator()
        workload = _make_workload(queue_depth=200, arrival_rate=50.0)
        throughputs = []
        for batch in [4, 32, 128, 512]:
            action = _make_action(batch_cap=batch)
            metrics = sim.simulate_step("static_workload", action, workload)
            throughputs.append(metrics.throughput_tps)
        # Throughput should be non-decreasing (allow ties)
        for i in range(len(throughputs) - 1):
            assert throughputs[i] <= throughputs[i + 1], f"Throughput decreased: {throughputs}"


class TestTraceSimulatorOOM:
    """High batch + high kv_budget should trigger memory pressure."""

    def test_high_load_caps_memory(self):
        sim = TraceSimulator()
        action = _make_action(batch_cap=512, kv_budget_fraction=1.0)
        workload = _make_workload(queue_depth=500, arrival_rate=200.0, mean_prompt_length=4096.0)
        metrics = sim.simulate_step("adversarial_multitenant", action, workload)
        assert metrics.gpu_memory_used_gb <= 38.0  # OOM cap


class TestTraceSimulatorQuantization:
    """INT8/INT4 should be cheaper and faster than FP16."""

    def test_int8_cheaper_than_fp16(self):
        sim = TraceSimulator()
        workload = _make_workload()
        fp16 = sim.simulate_step("static_workload", _make_action(quantization_tier=QuantizationTier.FP16), workload)
        int8 = sim.simulate_step("static_workload", _make_action(quantization_tier=QuantizationTier.INT8), workload)
        assert int8.estimated_cost_per_1k <= fp16.estimated_cost_per_1k

    def test_int4_faster_than_fp16(self):
        sim = TraceSimulator()
        workload = _make_workload()
        fp16 = sim.simulate_step("static_workload", _make_action(quantization_tier=QuantizationTier.FP16), workload)
        int4 = sim.simulate_step("static_workload", _make_action(quantization_tier=QuantizationTier.INT4), workload)
        assert int4.throughput_tps >= fp16.throughput_tps


# ─── KVCacheSimulator ─────────────────────────────────────────────

class TestKVCacheSimulator:
    def test_low_load_no_evictions(self):
        kv = KVCacheSimulator()
        occupancy, evictions = kv.apply(queue_depth=5, mean_prompt_length=64.0, kv_budget_fraction=1.0)
        assert evictions == 0
        assert 0.0 <= occupancy <= 1.0

    def test_high_load_causes_evictions(self):
        kv = KVCacheSimulator()
        occupancy, evictions = kv.apply(queue_depth=500, mean_prompt_length=4096.0, kv_budget_fraction=0.1)
        assert evictions > 0
        assert occupancy == 1.0

    def test_full_budget_less_evictions(self):
        kv = KVCacheSimulator()
        _, evictions_low = kv.apply(queue_depth=100, mean_prompt_length=512.0, kv_budget_fraction=0.1)
        _, evictions_high = kv.apply(queue_depth=100, mean_prompt_length=512.0, kv_budget_fraction=1.0)
        assert evictions_high <= evictions_low


# ─── SpeculativeDecoder ───────────────────────────────────────────

class TestSpeculativeDecoder:
    def test_no_speculation(self):
        sd = SpeculativeDecoder()
        acceptance, itl = sd.estimate("static_workload", 0, 128.0)
        assert acceptance == 0.0
        assert itl == 1.0

    def test_static_has_high_acceptance(self):
        sd = SpeculativeDecoder()
        acceptance, _ = sd.estimate("static_workload", 4, 128.0)
        assert acceptance > 0.4  # depth=4 yields ~0.49 with depth decay

    def test_adversarial_has_low_acceptance(self):
        sd = SpeculativeDecoder()
        acceptance, _ = sd.estimate("adversarial_multitenant", 4, 4096.0)
        assert acceptance < 0.5

    def test_itl_speedup_bounded(self):
        sd = SpeculativeDecoder()
        _, itl = sd.estimate("static_workload", 8, 128.0)
        assert 0.5 <= itl <= 1.0

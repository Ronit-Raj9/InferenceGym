from __future__ import annotations

import random
from typing import Any

from llmserve_env.models import MetricsSnapshot, QuantizationTier, ServeAction, WorkloadSnapshot
from llmserve_env.task_catalog import get_task_config
from server.kv_cache_simulator import KVCacheSimulator
from server.replay_assets import load_lookup_table
from server.speculative_decoder import SpeculativeDecoder


class TraceSimulator:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        try:
            self.kv_cache = KVCacheSimulator()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize KVCacheSimulator: {e}") from e
        
        try:
            self.speculative_decoder = SpeculativeDecoder()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SpeculativeDecoder: {e}") from e
        
        try:
            self.lookup_table = load_lookup_table("lookup_tables/latency_table.parquet")
        except Exception as e:
            raise RuntimeError(f"Failed to load lookup table: {e}") from e
        
        self.batch_history: list[int] = []

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.batch_history = []

    def simulate_step(self, task_id: str, action: ServeAction, workload: WorkloadSnapshot) -> MetricsSnapshot:
        task = get_task_config(task_id)
        batch_effective = min(action.batch_cap, max(1, workload.queue_depth + int(workload.arrival_rate)))
        quantization_tier = QuantizationTier(action.quantization_tier)

        self.batch_history.append(action.batch_cap)
        if len(self.batch_history) > 10:
            self.batch_history.pop(0)

        is_throttled = False
        if workload.step_index > 100 and len(self.batch_history) == 10:
            avg_batch = sum(self.batch_history) / 10.0
            if avg_batch > 0.8 * 512:
                is_throttled = True

        profile = self._lookup_profile(
            quantization_tier=quantization_tier,
            batch_effective=batch_effective,
            prompt_length=workload.mean_prompt_length,
            kv_budget_fraction=action.kv_budget_fraction,
            speculation_depth=action.speculation_depth,
        )

        kv_occupancy, evictions = self.kv_cache.apply(
            queue_depth=workload.queue_depth,
            mean_prompt_length=workload.mean_prompt_length,
            kv_budget_fraction=action.kv_budget_fraction,
            priority_routing=action.priority_routing,
        )
        spec_acceptance, itl_speedup = self.speculative_decoder.estimate(
            task_id=task_id,
            speculation_depth=action.speculation_depth,
            mean_prompt_length=workload.mean_prompt_length,
        )

        queue_pressure = 1.0 + min(2.5, workload.queue_depth / max(1.0, float(action.batch_cap))) * 0.18
        arrival_pressure = 1.0 + min(1.5, workload.arrival_rate / max(1.0, float(action.batch_cap))) * 0.04
        phase_factor = {"warmup": 1.03, "steady": 1.0, "burst": 1.10, "cooldown": 0.98}.get(workload.phase, 1.0)

        throughput_noise = self._noise(task_id, action, workload, "throughput")
        latency_noise = self._noise(task_id, action, workload, "latency")
        memory_noise = self._noise(task_id, action, workload, "memory")
        cost_noise = self._noise(task_id, action, workload, "cost")

        recomp_drop = evictions * max(1.0, workload.mean_prompt_length / 128.0) * 0.02

        throughput_tps = max(
            1.0,
            (profile["throughput_tps"] - recomp_drop)
            * throughput_noise
            * profile["quantization_speedup"]
            * (1.0 + (0.05 if action.prefill_decode_split else 0.0))
            * (1.0 + (0.03 if action.priority_routing and workload.priority_fraction > 0 else 0.0))
            * (1.0 + spec_acceptance * 0.18)
            / max(1.0, queue_pressure * 0.35),
        )

        thermal_mult = 1.15 if is_throttled else 1.0

        p50_ttft_ms = max(
            20.0,
            profile["p50_ttft_ms"]
            * latency_noise
            * queue_pressure
            * arrival_pressure
            * phase_factor
            * thermal_mult
            * profile["quantization_latency_mult"]
            * (0.92 if action.prefill_decode_split else 1.0)
            * (1.0 - min(0.22, spec_acceptance * 0.25)),
        )
        p99_ttft_ms = max(
            p50_ttft_ms,
            profile["p99_ttft_ms"]
            * latency_noise
            * queue_pressure
            * arrival_pressure
            * phase_factor
            * thermal_mult
            * profile["quantization_latency_mult"]
            * (1.0 - min(0.16, spec_acceptance * 0.18))
            * (1.0 + kv_occupancy * 0.08),
        )

        if task_id == "adversarial_multitenant" and workload.phase == "mega-prompt" and not action.prefill_decode_split:
            p99_ttft_ms *= 5.0

        p50_itl_ms = max(
            1.5,
            profile["p50_itl_ms"]
            * itl_speedup
            * thermal_mult
            * profile["quantization_itl_mult"]
            * (1.0 + kv_occupancy * 0.08)
            * self._noise(task_id, action, workload, "itl"),
        )
        gpu_memory_used_gb = max(
            2.0,
            (
                profile["gpu_memory_gb"] * profile["quantization_memory_mult"]
                + kv_occupancy * 6.5
                + workload.mean_prompt_length / 2200.0
                + workload.queue_depth / 140.0
                - (0.7 if action.priority_routing else 0.0)
            )
            * memory_noise,
        )
        estimated_cost_per_1k = max(
            0.0003,
            (
                profile["base_cost_per_1k"] * profile["quantization_cost_mult"]
                * (1.0 + kv_occupancy * 0.12)
                * (1.0 + (0.04 if action.prefill_decode_split else 0.0))
            )
            * cost_noise,
        )
        requests_served = min(batch_effective, max(0, workload.queue_depth + int(workload.arrival_rate)))
        slo_violations = 0
        if gpu_memory_used_gb > float(task["memory_cap_gb"]):
            gpu_memory_used_gb = float(task["memory_cap_gb"])
            evictions += max(1, batch_effective // 5)
            slo_violations += max(1, batch_effective // 6)

        return MetricsSnapshot(
            p50_ttft_ms=p50_ttft_ms,
            p99_ttft_ms=p99_ttft_ms,
            p50_itl_ms=p50_itl_ms,
            throughput_tps=throughput_tps,
            gpu_memory_used_gb=gpu_memory_used_gb,
            estimated_cost_per_1k=estimated_cost_per_1k,
            spec_acceptance_rate=spec_acceptance,
            eviction_events=evictions,
            preemption_events=int(evictions if action.priority_routing and kv_occupancy > 0.95 else 0),
            is_throttled=is_throttled,
            slo_violations=slo_violations,
            requests_served=requests_served,
        )

    def _lookup_profile(
        self,
        quantization_tier: QuantizationTier,
        batch_effective: int,
        prompt_length: float,
        kv_budget_fraction: float,
        speculation_depth: int,
    ) -> dict[str, float]:
        prompt_bucket = _prompt_size_bucket(prompt_length)
        batch_points = sorted(int(value) for value in self.lookup_table["batch_cap_bucket"].unique())
        kv_points = sorted(float(value) for value in self.lookup_table["kv_budget_bucket"].unique())
        spec_points = sorted(int(value) for value in self.lookup_table["spec_depth_bucket"].unique())

        batch_low, batch_high = _bounding_points(batch_points, batch_effective)
        kv_low, kv_high = _bounding_points(kv_points, kv_budget_fraction)
        spec_low, spec_high = _bounding_points(spec_points, speculation_depth)

        corners: list[tuple[dict[str, Any], float]] = []
        for batch_bucket, batch_weight in ((batch_low, 1.0 - _interpolation_weight(batch_low, batch_high, batch_effective)), (batch_high, _interpolation_weight(batch_low, batch_high, batch_effective))):
            for kv_bucket, kv_weight in ((kv_low, 1.0 - _interpolation_weight(kv_low, kv_high, kv_budget_fraction)), (kv_high, _interpolation_weight(kv_low, kv_high, kv_budget_fraction))):
                for spec_bucket, spec_weight in ((spec_low, 1.0 - _interpolation_weight(spec_low, spec_high, speculation_depth)), (spec_high, _interpolation_weight(spec_low, spec_high, speculation_depth))):
                    weight = batch_weight * kv_weight * spec_weight
                    if weight <= 0.0:
                        continue
                    row = self._nearest_row(
                        batch_bucket=batch_bucket,
                        kv_bucket=kv_bucket,
                        spec_bucket=spec_bucket,
                        prompt_bucket=prompt_bucket,
                    )
                    corners.append((row, weight))

        if not corners:
            corners.append((self._nearest_row(batch_effective, kv_budget_fraction, speculation_depth, prompt_bucket), 1.0))

        metrics = ["throughput_tps", "p50_ttft_ms", "p99_ttft_ms", "p50_itl_ms", "gpu_memory_gb"]
        profile = {metric: 0.0 for metric in metrics}
        total_weight = sum(weight for _, weight in corners) or 1.0
        for row, weight in corners:
            normalized_weight = weight / total_weight
            for metric in metrics:
                profile[metric] += float(row[metric]) * normalized_weight

        if speculation_depth > 0 and not any(int(row["spec_depth_bucket"]) == speculation_depth for row, _ in corners):
            depth_factor = 1.0 + min(0.18, speculation_depth * 0.025)
            profile["throughput_tps"] *= depth_factor
            profile["p50_ttft_ms"] *= max(0.75, 1.0 - speculation_depth * 0.03)
            profile["p99_ttft_ms"] *= max(0.78, 1.0 - speculation_depth * 0.025)
            profile["p50_itl_ms"] *= max(0.78, 1.0 - speculation_depth * 0.02)
            profile["gpu_memory_gb"] *= 1.0 + speculation_depth * 0.015

        quantization_profiles = {
            QuantizationTier.FP16: {
                "quantization_speedup": 1.00,
                "quantization_latency_mult": 1.00,
                "quantization_itl_mult": 1.00,
                "quantization_memory_mult": 1.00,
                "quantization_cost_mult": 1.00,
            },
            QuantizationTier.INT8: {
                "quantization_speedup": 1.08,
                "quantization_latency_mult": 0.94,
                "quantization_itl_mult": 0.94,
                "quantization_memory_mult": 0.82,
                "quantization_cost_mult": 0.78,
            },
            QuantizationTier.INT4: {
                "quantization_speedup": 1.16,
                "quantization_latency_mult": 0.90,
                "quantization_itl_mult": 0.90,
                "quantization_memory_mult": 0.68,
                "quantization_cost_mult": 0.62,
            },
        }
        profile.update(quantization_profiles[quantization_tier])
        profile["base_cost_per_1k"] = max(0.0004, (profile["gpu_memory_gb"] * 0.0012 + batch_effective * 0.000003) / max(profile["throughput_tps"], 1.0) * 1000.0)
        return profile

    def _nearest_row(
        self,
        batch_bucket: int,
        kv_bucket: float,
        spec_bucket: int,
        prompt_bucket: str,
    ) -> dict[str, Any]:
        frame = self.lookup_table[self.lookup_table["prompt_size_bucket"] == prompt_bucket]
        if frame.empty:
            frame = self.lookup_table

        distance_frame = frame.assign(
            _distance=(
                (frame["batch_cap_bucket"].astype(float) - float(batch_bucket)).abs() / 256.0
                + (frame["kv_budget_bucket"].astype(float) - float(kv_bucket)).abs() * 2.0
                + (frame["spec_depth_bucket"].astype(float) - float(spec_bucket)).abs() / 8.0
            )
        )
        row = distance_frame.sort_values(["_distance", "batch_cap_bucket"]).iloc[0]
        return row.to_dict()

    def _noise(
        self,
        task_id: str,
        action: ServeAction,
        workload: WorkloadSnapshot,
        metric: str,
    ) -> float:
        seed_material = (
            f"{self.seed}|{task_id}|{metric}|{workload.phase}|{workload.step_index}|"
            f"{workload.queue_depth}|{round(workload.arrival_rate, 3)}|{round(workload.mean_prompt_length, 3)}"
        )
        rng = random.Random(seed_material)

        sigma = 0.03
        if action.quantization_tier in [QuantizationTier.INT8.value, QuantizationTier.INT4.value]:
            sigma = 0.07
        if workload.phase in ["burst", "mega-prompt"]:
            sigma = 0.12

        return float(rng.gauss(1.0, sigma))


def _bounding_points(points: list[float], value: float) -> tuple[float, float]:
    lower = points[0]
    upper = points[-1]
    for point in points:
        if point <= value:
            lower = point
        if point >= value:
            upper = point
            break
    return lower, upper


def _interpolation_weight(lower: float, upper: float, value: float) -> float:
    if upper == lower:
        return 0.0
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def _lerp(start: float, end: float, weight: float) -> float:
    return start + (end - start) * weight


def _prompt_size_bucket(prompt_length: float) -> str:
    if prompt_length <= 256:
        return "small"
    if prompt_length <= 2048:
        return "medium"
    return "large"

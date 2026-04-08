from __future__ import annotations

import random
from typing import Any

from llmserve_env.models import WorkloadSnapshot
from server.replay_assets import load_prompt_samples, load_trace_table


class WorkloadGenerator:
    def __init__(self, task_config: dict[str, Any], seed: int = 42) -> None:
        self.task_config = task_config
        self.seed = seed
        self.rng = random.Random(seed)
        self.queue_depth = 0
        self.trace_rows = self._load_trace_rows()
        self.prompt_samples = self._load_prompt_samples()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.queue_depth = 0

    def next_snapshot(self, step_index: int) -> WorkloadSnapshot:
        trace_row = self._trace_row_for_step(step_index)
        arrival_rate = self._arrival_rate_for_step(step_index, trace_row)

        if self.task_config["id"] == "adversarial_multitenant" and (step_index + 1) % 100 == 0:
            mean_prompt_length = 16384.0
            phase = "mega-prompt"
        else:
            mean_prompt_length = self._prompt_length_for_step(trace_row)
            phase = self._phase_for_step(step_index, trace_row)

        service_hint = float(trace_row.get("service_rate_hint", arrival_rate * 0.6)) if trace_row else arrival_rate * 0.6
        served_estimate = min(self.queue_depth, max(1, int(service_hint)))
        queue_bias = int(trace_row.get("queue_bias", 0)) if trace_row else 0
        self.queue_depth = max(0, self.queue_depth + int(arrival_rate) - served_estimate + queue_bias)

        return WorkloadSnapshot(
            arrival_rate=arrival_rate,
            queue_depth=self.queue_depth,
            mean_prompt_length=mean_prompt_length,
            prompt_length_bucket=self._prompt_bucket(mean_prompt_length),
            priority_fraction=float(trace_row.get("priority_fraction", self.task_config.get("priority_fraction", 0.0)))
            if trace_row
            else float(self.task_config.get("priority_fraction", 0.0)),
            phase=phase,
            step_index=step_index,
        )

    def _arrival_rate_for_step(self, step_index: int, trace_row: dict[str, Any] | None = None) -> float:
        if trace_row and "arrival_rate_rps" in trace_row:
            return float(trace_row["arrival_rate_rps"])
        base = float(self.task_config["arrival_rate_rps"])
        burst_rate = float(self.task_config.get("burst_rate_rps", base))
        burst_every = int(self.task_config.get("burst_every_steps", 0))
        burst_length = int(self.task_config.get("burst_length_steps", 0))
        if burst_every and burst_length:
            window = step_index % burst_every
            if window < burst_length:
                return burst_rate
        if self.task_config.get("arrival_pattern") == "sinusoidal":
            floor = float(self.task_config.get("arrival_floor_rps", base))
            ceiling = float(self.task_config.get("arrival_ceiling_rps", burst_rate))
            cycle = max(1, int(self.task_config.get("arrival_cycle_steps", 50)))
            alpha = (step_index % cycle) / cycle
            return floor + (ceiling - floor) * (0.5 + 0.5 * (1 if alpha < 0.5 else -1))
        return base

    def _prompt_length_for_step(self, trace_row: dict[str, Any] | None = None) -> float:
        mode = self.task_config["prompt_distribution"]["type"]
        if mode == "trace_sample":
            sample_pool = self.prompt_samples or [128]
            if trace_row:
                prompt_p50 = float(trace_row.get("prompt_p50", min(sample_pool)))
                prompt_p95 = float(trace_row.get("prompt_p95", max(sample_pool)))
                bounded_pool = [
                    sample
                    for sample in sample_pool
                    if (prompt_p50 * 0.5) <= sample <= max(prompt_p95 * 1.1, prompt_p50 + 1.0)
                ]
                sample_pool = bounded_pool or sample_pool
            return float(self.rng.choice(sample_pool))
        if mode == "uniform":
            low = self.task_config["prompt_distribution"]["min"]
            high = self.task_config["prompt_distribution"]["max"]
            return self.rng.uniform(low, high)
        if mode == "bimodal":
            short = self.task_config["prompt_distribution"]["short"]
            long = self.task_config["prompt_distribution"]["long"]
            fraction = self.task_config["prompt_distribution"]["long_fraction"]
            bucket = long if self.rng.random() < fraction else short
            return self.rng.uniform(bucket["min"], bucket["max"])
        low = self.task_config["prompt_distribution"]["min"]
        high = self.task_config["prompt_distribution"]["max"]
        return self.rng.uniform(low, high)

    def _phase_for_step(self, step_index: int, trace_row: dict[str, Any] | None = None) -> str:
        if trace_row and "phase" in trace_row:
            return str(trace_row["phase"])
        burst_every = int(self.task_config.get("burst_every_steps", 0))
        burst_length = int(self.task_config.get("burst_length_steps", 0))
        if burst_every and (step_index % burst_every) < burst_length:
            return "burst"
        if step_index < 3:
            return "warmup"
        if step_index >= int(self.task_config["max_steps"]) - 3:
            return "cooldown"
        return "steady"

    @staticmethod
    def _prompt_bucket(prompt_length: float) -> int:
        boundaries = [64, 128, 256, 512, 1024, 2048, 4096]
        for idx, boundary in enumerate(boundaries):
            if prompt_length <= boundary:
                return idx
        return 7

    def _load_trace_rows(self) -> list[dict[str, Any]]:
        trace_file = self.task_config.get("trace_file")
        if not trace_file:
            return []
        frame = load_trace_table(trace_file)
        return frame.to_dict(orient="records")

    def _load_prompt_samples(self) -> list[int]:
        distribution = self.task_config.get("prompt_distribution", {})
        sample_file = distribution.get("sample_file")
        if not sample_file:
            return []
        return load_prompt_samples(sample_file)

    def _trace_row_for_step(self, step_index: int) -> dict[str, Any] | None:
        if not self.trace_rows:
            return None
        return self.trace_rows[step_index % len(self.trace_rows)]

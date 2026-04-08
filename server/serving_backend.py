from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import mean
from typing import Any, Protocol

from openai import OpenAI

from llmserve_env.models import MetricsSnapshot, QuantizationTier, ServeAction, WorkloadSnapshot
from server.trace_simulator import TraceSimulator


class ServingBackend(Protocol):
    mode: str

    def reset(self, seed: int | None = None) -> None: ...

    def run_step(self, task_id: str, action: ServeAction, workload: WorkloadSnapshot) -> MetricsSnapshot: ...

    def describe(self) -> dict[str, Any]: ...


class SimulatedServingBackend:
    mode = "sim"

    def __init__(self, seed: int = 42) -> None:
        try:
            self.simulator = TraceSimulator(seed=seed)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TraceSimulator in SimulatedServingBackend: {e}") from e

    def reset(self, seed: int | None = None) -> None:
        self.simulator.reset(seed=seed)

    def run_step(self, task_id: str, action: ServeAction, workload: WorkloadSnapshot) -> MetricsSnapshot:
        return self.simulator.simulate_step(task_id, action, workload)

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "provider": "simulator"}


@dataclass
class _RequestResult:
    latency_s: float
    ttft_ms: float
    itl_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    truncated: bool


class RealOpenAIBackend:
    mode = "real"

    def __init__(
        self,
        seed: int = 42,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_requests_per_step: int | None = None,
        max_prompt_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        client: OpenAI | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key and client is None:
            raise RuntimeError("OPENAI_API_KEY is required when LLMSERVE_MODE=real.")

        env_base_url = os.getenv("OPENAI_BASE_URL")
        resolved_base_url = (base_url or env_base_url or "").strip() or None
        self.seed = seed
        self.model = (model or os.getenv("LLMSERVE_REAL_MODEL", "gpt-4.1-mini")).strip()
        self.base_url = resolved_base_url
        self.max_requests_per_step = max_requests_per_step or int(os.getenv("LLMSERVE_REAL_MAX_REQUESTS_PER_STEP", "4"))
        self.max_prompt_tokens = max_prompt_tokens or int(os.getenv("LLMSERVE_REAL_MAX_PROMPT_TOKENS", "512"))
        self.max_completion_tokens = max_completion_tokens or int(os.getenv("LLMSERVE_REAL_MAX_COMPLETION_TOKENS", "64"))
        self.client = client or OpenAI(
            api_key=resolved_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
        )
        self.pricing = {
            "gpt-4.1-mini": {"input_per_million": 0.40, "output_per_million": 1.60},
            "gpt-4.1": {"input_per_million": 2.00, "output_per_million": 8.00},
            "gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.60},
            "gpt-4o": {"input_per_million": 2.50, "output_per_million": 10.00},
        }

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed

    def run_step(self, task_id: str, action: ServeAction, workload: WorkloadSnapshot) -> MetricsSnapshot:
        request_count = max(
            1,
            min(
                self.max_requests_per_step,
                action.batch_cap,
                max(1, workload.queue_depth + int(math.ceil(workload.arrival_rate))),
            ),
        )
        prompts = [
            self._build_request_payload(task_id, workload, action, request_index=index, request_count=request_count)
            for index in range(request_count)
        ]

        batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=request_count) as executor:
            results = list(executor.map(self._execute_request, prompts))
        batch_latency_s = max(time.perf_counter() - batch_start, 1e-6)

        total_completion_tokens = sum(result.completion_tokens for result in results)
        total_prompt_tokens = sum(result.prompt_tokens for result in results)
        total_cost = sum(result.cost_usd for result in results)
        mean_total_tokens = sum(result.total_tokens for result in results) / len(results)
        throughput_tps = total_completion_tokens / batch_latency_s

        mean_prompt = workload.mean_prompt_length
        memory_factor = {
            "gpt-4.1-mini": 0.010,
            "gpt-4.1": 0.018,
            "gpt-4o-mini": 0.009,
            "gpt-4o": 0.016,
        }.get(self.model, 0.012)
        quant_factor = {
            QuantizationTier.FP16.value: 1.00,
            QuantizationTier.INT8.value: 0.84,
            QuantizationTier.INT4.value: 0.72,
        }[action.quantization_tier]
        gpu_memory_used_gb = max(
            2.0,
            (total_prompt_tokens * memory_factor * quant_factor * max(action.kv_budget_fraction, 0.1)) / 10.0
            + request_count * 0.35,
        )

        cost_per_1k = max(0.0001, (total_cost / max(total_prompt_tokens + total_completion_tokens, 1)) * 1000.0)
        evictions = sum(1 for result in results if result.truncated)

        return MetricsSnapshot(
            p50_ttft_ms=_percentile([result.ttft_ms for result in results], 0.50),
            p99_ttft_ms=_percentile([result.ttft_ms for result in results], 0.99),
            p50_itl_ms=_percentile([result.itl_ms for result in results], 0.50),
            throughput_tps=max(1.0, throughput_tps),
            gpu_memory_used_gb=gpu_memory_used_gb,
            estimated_cost_per_1k=cost_per_1k,
            spec_acceptance_rate=min(0.6, action.speculation_depth / 8.0 * 0.35),
            eviction_events=evictions,
            slo_violations=0,
            requests_served=request_count,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "provider": "openai",
            "model": self.model,
            "max_requests_per_step": self.max_requests_per_step,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
        }

    def _build_request_payload(
        self,
        task_id: str,
        workload: WorkloadSnapshot,
        action: ServeAction,
        request_index: int,
        request_count: int,
    ) -> dict[str, Any]:
        priority_cutoff = max(1, int(round(request_count * workload.priority_fraction)))
        is_priority = request_index < priority_cutoff if action.priority_routing else False
        spread = (request_index - (request_count / 2.0)) / max(request_count, 1)
        target_prompt_tokens = max(32, int(workload.mean_prompt_length * (1.0 + spread * 0.35)))
        effective_prompt_tokens = max(16, int(target_prompt_tokens * action.kv_budget_fraction))
        truncated = effective_prompt_tokens < target_prompt_tokens
        prompt = self._build_prompt(task_id, workload.phase, effective_prompt_tokens, is_priority=is_priority)
        return {
            "prompt": prompt,
            "target_prompt_tokens": target_prompt_tokens,
            "effective_prompt_tokens": effective_prompt_tokens,
            "truncated": truncated,
            "priority": is_priority,
        }

    def _execute_request(self, payload: dict[str, Any]) -> _RequestResult:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_completion_tokens=self.max_completion_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise assistant. Answer the request directly in plain text.",
                },
                {"role": "user", "content": payload["prompt"]},
            ],
        )
        latency_s = max(time.perf_counter() - start, 1e-6)
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", payload["effective_prompt_tokens"]))
        completion_tokens = int(getattr(usage, "completion_tokens", self.max_completion_tokens // 2))
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens))
        ttft_ms = latency_s * 1000.0 * 0.35
        itl_ms = max(1.0, ((latency_s * 1000.0) - ttft_ms) / max(completion_tokens, 1))
        pricing = self.pricing.get(self.model, {"input_per_million": 0.40, "output_per_million": 1.60})
        cost_usd = (
            (prompt_tokens / 1_000_000.0) * pricing["input_per_million"]
            + (completion_tokens / 1_000_000.0) * pricing["output_per_million"]
        )
        return _RequestResult(
            latency_s=latency_s,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            truncated=bool(payload["truncated"]),
        )

    def _build_prompt(self, task_id: str, phase: str, target_tokens: int, is_priority: bool) -> str:
        header = (
            f"Task: {task_id}\n"
            f"Phase: {phase}\n"
            f"Priority: {is_priority}\n"
            "Summarize the impact of serving-policy changes on latency, throughput, and user experience.\n"
        )
        filler_unit = "latency throughput queue kv cache scheduling token generation "
        filler = (filler_unit * ((target_tokens // 8) + 8)).strip()
        words = filler.split()
        return header + " ".join(words[:target_tokens])


def create_serving_backend(mode: str | None = None, seed: int = 42) -> ServingBackend:
    resolved_mode = (mode or os.getenv("LLMSERVE_MODE", "sim")).strip().lower()
    if resolved_mode == "sim":
        return SimulatedServingBackend(seed=seed)
    if resolved_mode == "real":
        provider = os.getenv("LLMSERVE_REAL_PROVIDER", "openai").strip().lower()
        if provider != "openai":
            raise RuntimeError(f"Unsupported LLMSERVE_REAL_PROVIDER: {provider}")
        return RealOpenAIBackend(seed=seed)
    raise RuntimeError(f"Unsupported LLMSERVE_MODE: {resolved_mode}")


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]

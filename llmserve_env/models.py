from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from openenv.core import Action, Observation
from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuantizationTier(str, Enum):
    FP16 = "FP16"
    INT8 = "INT8"
    INT4 = "INT4"


class ServeAction(Action):
    model_config = ConfigDict(extra="forbid")

    batch_cap: int = Field(default=32, ge=1, le=512)
    kv_budget_fraction: float = Field(default=1.0, ge=0.1, le=1.0)
    speculation_depth: int = Field(default=0, ge=0, le=8)
    quantization_tier: Literal["FP16", "INT8", "INT4"] = QuantizationTier.FP16.value
    prefill_decode_split: bool = False
    priority_routing: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_web_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        normalized["batch_cap"] = _clamp_int(normalized.get("batch_cap"), default=32, minimum=1, maximum=512)
        normalized["kv_budget_fraction"] = _clamp_float(
            normalized.get("kv_budget_fraction"),
            default=1.0,
            minimum=0.1,
            maximum=1.0,
        )
        normalized["speculation_depth"] = _clamp_int(
            normalized.get("speculation_depth"),
            default=0,
            minimum=0,
            maximum=8,
        )
        normalized["quantization_tier"] = _normalize_quantization_tier(normalized.get("quantization_tier"))
        return normalized


class ServeObservation(Observation):
    model_config = ConfigDict(extra="forbid")

    queue_depth: int = Field(ge=0)
    active_requests: int = Field(ge=0)
    kv_cache_occupancy: float = Field(ge=0.0, le=1.0)
    mean_prompt_length: float = Field(ge=0.0)
    p50_ttft_ms: float = Field(ge=0.0)
    p99_ttft_ms: float = Field(ge=0.0)
    p50_itl_ms: float = Field(ge=0.0)
    throughput_tps: float = Field(ge=0.0)
    slo_compliance_rate: float = Field(ge=0.0, le=1.0)
    gpu_memory_used_gb: float = Field(ge=0.0)
    estimated_cost_per_1k: float = Field(ge=0.0)
    request_arrival_rate: float = Field(ge=0.0)
    spec_acceptance_rate: float = Field(ge=0.0, le=1.0)
    eviction_events: int = Field(ge=0)
    step_index: int = Field(ge=0)
    task_id: str = "uninitialized"


class ServeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    step_count: int = Field(ge=0)
    task_id: str
    total_requests_served: int = Field(ge=0)
    total_slo_violations: int = Field(ge=0)
    cumulative_reward: float = 0.0
    elapsed_simulated_time_s: float = Field(ge=0.0)
    workload_phase: str = "warmup"
    done: bool = False


class RewardSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float
    components: dict[str, float]
    done: bool


class WorkloadSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arrival_rate: float = Field(ge=0.0)
    queue_depth: int = Field(ge=0)
    mean_prompt_length: float = Field(ge=0.0)
    prompt_length_bucket: int = Field(ge=0, le=7)
    priority_fraction: float = Field(ge=0.0, le=1.0)
    phase: str
    step_index: int = Field(default=0, ge=0)


class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p50_ttft_ms: float = Field(ge=0.0)
    p99_ttft_ms: float = Field(ge=0.0)
    p50_itl_ms: float = Field(ge=0.0)
    throughput_tps: float = Field(ge=0.0)
    gpu_memory_used_gb: float = Field(ge=0.0)
    estimated_cost_per_1k: float = Field(ge=0.0)
    spec_acceptance_rate: float = Field(ge=0.0, le=1.0)
    eviction_events: int = Field(ge=0)
    preemption_events: int = Field(default=0, ge=0)
    is_throttled: bool = Field(default=False)
    slo_violations: int = Field(ge=0)
    requests_served: int = Field(ge=0)


class EpisodeLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    actions: list[ServeAction]
    observations: list[ServeObservation]
    rewards: list[float]
    final_state: ServeState


def default_action() -> ServeAction:
    return ServeAction(
        batch_cap=32,
        kv_budget_fraction=1.0,
        speculation_depth=0,
        quantization_tier=QuantizationTier.FP16.value,
        prefill_decode_split=False,
        priority_routing=False,
    )


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _normalize_quantization_tier(value: Any) -> str:
    if isinstance(value, QuantizationTier):
        return value.value
    if isinstance(value, str) and value in {tier.value for tier in QuantizationTier}:
        return value
    return QuantizationTier.FP16.value

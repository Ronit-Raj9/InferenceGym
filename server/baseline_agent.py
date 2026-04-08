"""Heuristic baseline policy for LLM serving configuration.

Rules derived from three papers:
  - Orca (OSDI 2022): dynamic iteration-level batching / queue management
  - vLLM / PagedAttention (SOSP 2023): KV cache memory management
  - Decima (SIGCOMM 2019): workload-adaptive scheduling via RL
"""
from __future__ import annotations

from llmserve_env.models import QuantizationTier, ServeAction, ServeObservation


class HeuristicPolicy:
    """Reactive heuristic agent that adjusts serving config based on observations."""

    def __init__(self) -> None:
        self.batch_cap = 32
        self.kv_budget_fraction = 0.70
        self.speculation_depth = 0
        self.quantization_tier: str = QuantizationTier.FP16.value
        self.prefill_decode_split = False
        self.priority_routing = False

    def reset(self) -> None:
        """Reset to starting state for a new episode."""
        self.batch_cap = 32
        self.kv_budget_fraction = 0.70
        self.speculation_depth = 0
        self.quantization_tier = QuantizationTier.FP16.value
        self.prefill_decode_split = False
        self.priority_routing = False

    def act(self, observation: ServeObservation, task_id: str) -> ServeAction:
        """Produce an action given the current observation."""

        # --- Orca rules: dynamic batching / queue management ---
        if observation.slo_compliance_rate < 0.85:
            self.batch_cap = max(1, self.batch_cap - 32)
        elif observation.queue_depth > 0.7 * self.batch_cap:
            self.batch_cap = min(512, self.batch_cap + 16)
        elif observation.queue_depth < 0.2 * self.batch_cap and self.batch_cap > 16:
            self.batch_cap = max(1, self.batch_cap - 16)

        # --- vLLM / PagedAttention rules: memory management ---
        if observation.eviction_events > 0:
            self.kv_budget_fraction = 0.60
        elif observation.kv_cache_occupancy > 0.85:
            self.kv_budget_fraction = max(0.10, self.kv_budget_fraction - 0.10)
        elif observation.kv_cache_occupancy < 0.50 and self.kv_budget_fraction < 1.0:
            self.kv_budget_fraction = min(1.0, self.kv_budget_fraction + 0.10)

        # --- Decima rules: workload-adaptive optimisation ---
        if observation.request_arrival_rate > 25:
            self.quantization_tier = QuantizationTier.INT8.value
        elif observation.request_arrival_rate < 8:
            self.quantization_tier = QuantizationTier.FP16.value

        if observation.mean_prompt_length > 800:
            self.speculation_depth = 0
        elif observation.mean_prompt_length < 200:
            self.speculation_depth = 4

        # Use priority routing on adversarial task with long prompts
        if task_id == "adversarial_multitenant" and observation.mean_prompt_length > 2000:
            self.priority_routing = True
        else:
            self.priority_routing = False

        # Enable chunked prefill when under high queue pressure
        self.prefill_decode_split = observation.queue_depth > 0.5 * self.batch_cap

        return ServeAction(
            batch_cap=self.batch_cap,
            kv_budget_fraction=round(self.kv_budget_fraction, 2),
            speculation_depth=self.speculation_depth,
            quantization_tier=self.quantization_tier,
            prefill_decode_split=self.prefill_decode_split,
            priority_routing=self.priority_routing,
        )


# ---------------------------------------------------------------------------
# Legacy function interface for backward-compatibility
# ---------------------------------------------------------------------------
_default_policy = HeuristicPolicy()


def baseline_policy(observation: ServeObservation, task_id: str) -> ServeAction:
    """Drop-in replacement preserving the old function signature."""
    return _default_policy.act(observation, task_id)

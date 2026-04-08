from __future__ import annotations


class KVCacheSimulator:
    def apply(
        self,
        queue_depth: int,
        mean_prompt_length: float,
        kv_budget_fraction: float,
        priority_routing: bool = False,
    ) -> tuple[float, int]:
        requested = queue_depth * mean_prompt_length
        budget = max(1.0, 16000.0 * kv_budget_fraction)
        occupancy = min(1.0, requested / budget)
        evictions = 0

        if requested > budget:
            if priority_routing and occupancy > 0.95:
                evictions = int((requested - (budget * 0.90)) / max(mean_prompt_length, 1.0))
            else:
                evictions = int((requested - budget) / max(mean_prompt_length, 1.0))

        return occupancy, max(0, evictions)

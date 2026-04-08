from __future__ import annotations


class SpeculativeDecoder:
    def estimate(
        self,
        task_id: str,
        speculation_depth: int,
        mean_prompt_length: float,
    ) -> tuple[float, float]:
        if speculation_depth <= 0:
            return 0.0, 1.0
        # Research Fidelity Trigger: Chat (0.8) vs API (0.3)
        if "chat" in task_id.lower() or "static" in task_id.lower() or "bursty" in task_id.lower():
            base_rate = 0.80
        else:
            base_rate = 0.30
            
        complexity_penalty = min(0.45, mean_prompt_length / 10000.0)
        depth_decay = 1.0 / (1.0 + 0.15 * speculation_depth)
        acceptance = max(0.0, min(1.0, base_rate * (1.0 - complexity_penalty) * depth_decay))
        itl_speedup = max(0.75, 1.0 - (acceptance * speculation_depth * 0.03))
        return acceptance, itl_speedup


from __future__ import annotations


class SLOMonitor:
    def evaluate(self, p99_ttft_ms: float, target_ms: float, active_requests: int) -> tuple[float, int]:
        if active_requests <= 0:
            return 1.0, 0
        if p99_ttft_ms <= target_ms:
            return 1.0, 0
        overflow_ratio = min(1.0, (p99_ttft_ms - target_ms) / max(target_ms, 1.0))
        violations = max(1, int(active_requests * overflow_ratio))
        compliance = max(0.0, 1.0 - overflow_ratio)
        return compliance, violations


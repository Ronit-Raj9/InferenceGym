from __future__ import annotations


class OptimalSolver:
    def oracle_reference(self, task_id: str) -> dict[str, float]:
        return {
            "static_workload": {"throughput_tps": 320.0, "slo_compliance_rate": 1.0, "cost_per_1k": 0.0015},
            "bursty_workload": {"throughput_tps": 300.0, "slo_compliance_rate": 0.95, "cost_per_1k": 0.0018},
            "adversarial_multitenant": {"throughput_tps": 260.0, "slo_compliance_rate": 0.92, "cost_per_1k": 0.0019},
        }.get(task_id, {"throughput_tps": 250.0, "slo_compliance_rate": 0.9, "cost_per_1k": 0.0020})


from llmserve_env.models import MetricsSnapshot, QuantizationTier


_WEIGHT_PROFILES = {
    "static_workload":         {"w_tput": 0.4, "w_slo": 0.3, "w_mem": 0.1, "w_cost": 0.2},
    "bursty_workload":         {"w_tput": 0.2, "w_slo": 0.3, "w_mem": 0.4, "w_cost": 0.1},
    "adversarial_multitenant": {"w_tput": 0.2, "w_slo": 0.5, "w_mem": 0.1, "w_cost": 0.2},
}

_COST_FACTORS = {
    QuantizationTier.FP16.value: 1.0,
    QuantizationTier.INT8.value: 0.5,
    QuantizationTier.INT4.value: 0.25,
}

MAX_TPS_REFERENCE = 500.0  # Hardware max TPS for normalization


class RewardCalculator:
    def calculate(
        self,
        task_id: str,
        metrics: MetricsSnapshot,
        slo_compliance_rate: float,
        quantization_tier: str = "FP16",
        priority_fraction: float = 0.0,
    ) -> float:
        """
        Calculates the Multi-Objective Reward with non-linear penalties.
        R = (w1 * R_tput) + (w2 * R_slo) - (w3 * P_mem) - (w4 * P_cost)
        """
        weights = _WEIGHT_PROFILES.get(task_id, _WEIGHT_PROFILES["static_workload"])
        
        # 1. Throughput Component (Normalized TPS)
        r_tput = min(1.0, metrics.throughput_tps / MAX_TPS_REFERENCE)
        
        # 2. SLO Compliance Component (Scale: +1.0 for success, -2.0 for failure)
        # We blend standard and priority SLOs if in level 3
        base_slo_reward = (slo_compliance_rate * 1.0) + ((1.0 - slo_compliance_rate) * -2.0)
        
        if task_id == "adversarial_multitenant" and priority_fraction > 0:
            # Priority misses should hurt more, but remain bounded so reward retains action sensitivity.
            # The multiplier scales from 1.0 to 2.0 as priority fraction goes 0 -> 1.
            penalty = (1.0 - slo_compliance_rate) * -2.0
            priority_multiplier = 1.0 + min(1.0, max(0.0, priority_fraction))
            r_slo = (slo_compliance_rate * 1.0) + (penalty * priority_multiplier)
        else:
            r_slo = base_slo_reward
        r_slo = max(-1.5, min(1.0, r_slo))

        # 3. Goldilocks Memory Penalty (Piecewise)
        # Research: Stay between 0.6 and 0.85
        occ = metrics.gpu_memory_used_gb / 40.0 # Assume 40GB total
        if occ < 0.60:
            p_mem = 0.5 * (0.60 - occ)
        elif 0.60 <= occ <= 0.85:
            p_mem = 0.0
        else:
            # Smooth bounded penalty above the safe region.
            # Keeps a strong gradient without saturating reward to -1 for entire episodes.
            over = min(0.40, occ - 0.85)
            p_mem = 0.5 + ((over / 0.40) ** 2) * 1.5

        # 4. Cost/Efficiency Penalty (Splitwise logic)
        q_factor = _COST_FACTORS.get(quantization_tier, 1.0)
        p_cost = (metrics.gpu_memory_used_gb / 40.0) * q_factor

        # Weighted Sum
        reward = (weights["w_tput"] * r_tput) + \
                 (weights["w_slo"] * r_slo) - \
                 (weights["w_mem"] * p_mem) - \
                 (weights["w_cost"] * p_cost)

        return max(-1.0, min(1.0, reward))

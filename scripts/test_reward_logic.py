import math
import sys
import os

# Add root directory to sys.path to allow imports from 'server'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.reward_calculator import RewardCalculator, MAX_TPS_REFERENCE
from llmserve_env.models import MetricsSnapshot, QuantizationTier

def test_reward_scenarios():
    calc = RewardCalculator()
    
    print("[INFO] Testing Goldilocks Memory Penalties...")
    # Scenario 1: Optimal Memory (70%)
    m1 = MetricsSnapshot(
        throughput_tps=200.0,
        gpu_memory_used_gb=28.0, # 28/40 = 0.7 (Optimal)
        slo_violations=0,
        requests_served=50,
        p50_ttft_ms=100.0,
        p99_ttft_ms=200.0,
        p50_itl_ms=50.0,
        estimated_cost_per_1k=0.001,
        spec_acceptance_rate=0.8,
        eviction_events=0,
        preemption_events=0,
        is_throttled=False
    )
    r1 = calc.calculate("static_workload", m1, 1.0, "FP16", 0.0)
    print(f"  Optimal (70%): Reward={r1:.4f}")
    assert r1 > 0, "Optimal memory should yield positive reward"

    # Scenario 2: Under-utilization (20%)
    m2 = m1.model_copy(update={
        "throughput_tps": 50.0,
        "gpu_memory_used_gb": 8.0, # 8/40 = 0.2 (Under)
        "requests_served": 10
    })
    r2 = calc.calculate("static_workload", m2, 1.0, "FP16", 0.0)
    print(f"  Under-utilized (20%): Reward={r2:.4f}")
    assert r2 < r1, "Under-utilized should reward less than optimal"

    # Scenario 3: Danger Zone (95%)
    # Use 'bursty_workload' where w_mem is higher (0.4) to check stability focus
    m3 = m1.model_copy(update={
        "throughput_tps": 400.0,
        "gpu_memory_used_gb": 38.0, # 38/40 = 0.95 (Danger)
        "requests_served": 80
    })
    r3 = calc.calculate("bursty_workload", m3, 1.0, "FP16", 0.0)
    print(f"  Danger Zone (95%, Bursty): Reward={r3:.4f}")
    assert r3 < 0, f"Danger zone should yield negative reward in Bursty mode, got {r3}"

    print("\n[INFO] Testing SLO Breach Penalties...")
    # Scenario 4: SLO Breach
    m4 = m1.model_copy(update={
        "throughput_tps": 300.0,
        "gpu_memory_used_gb": 30.0,
        "slo_violations": 10,
        "requests_served": 50
    })
    r4 = calc.calculate("static_workload", m4, 0.5, "FP16", 0.0)
    print(f"  SLO Breach (50%): Reward={r4:.4f}")
    assert r4 < r1, "SLO breach should be heavily penalized"

    print("\n[INFO] Testing Level 3 Priority Multiplier...")
    # Scenario 5: Priority Breach in Level 3
    # Standard breach (0.9 compliance)
    r5_std = calc.calculate("adversarial_multitenant", m1, 0.9, "FP16", 0.0)
    # Priority breach (0.9 compliance, 20% priority)
    r5_pri = calc.calculate("adversarial_multitenant", m1, 0.9, "FP16", 0.2)
    print(f"  L3 Standard Breach (90%): Reward={r5_std:.4f}")
    print(f"  L3 Priority Breach (90%, 20% VIP): Reward={r5_pri:.4f}")
    assert r5_pri < r5_std, "Priority breach should penalize more in Level 3"

    print("\n[PASS] All reward logic scenarios verified.")

if __name__ == "__main__":
    test_reward_scenarios()

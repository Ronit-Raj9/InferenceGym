import sys
import os
import numpy as np
from typing import List

# Ensure projects root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.llmserve_environment import LLMServeEnvironment
from llmserve_env.models import ServeAction, QuantizationTier

def test_quantization_jitter():
    print("[INFO] Testing Quantization Jitter (Chiron 2024)...")
    env = LLMServeEnvironment(seed=42)
    
    # FP16 Jitter
    env.reset(task_id="static_workload")
    fp16_latencies = []
    for _ in range(50): # Avoid 100-step Mega-Prompt spike
        obs = env.step(ServeAction(quantization_tier=QuantizationTier.FP16.value, batch_cap=200))
        fp16_latencies.append(obs.p50_ttft_ms)
        
    fp16_cv = np.std(fp16_latencies) / np.mean(fp16_latencies)
    print(f"      FP16 CV: {fp16_cv:.4f}")
    
    # INT4 Jitter
    env.reset(task_id="static_workload")
    int4_latencies = []
    for _ in range(50):
        obs = env.step(ServeAction(quantization_tier=QuantizationTier.INT4.value, batch_cap=200))
        int4_latencies.append(obs.p50_ttft_ms)
        
    int4_cv = np.std(int4_latencies) / np.mean(int4_latencies)
    print(f"      INT4 CV: {int4_cv:.4f}")
    
    # Assert INT4 has notably higher jitter
    assert int4_cv > fp16_cv, f"INT4 Jitter ({int4_cv:.4f}) must be > FP16 Jitter ({fp16_cv:.4f})"
    print("[PASS] Quantization Jitter verified.")

def test_thermal_throttling():
    print("[INFO] Testing Thermal Throttling Trigger...")
    env = LLMServeEnvironment(seed=42)
    env.reset(task_id="static_workload")
    
    # Run 100 steps of low load
    for i in range(100):
        env.step(ServeAction(batch_cap=10))
        
    obs_normal = env.step(ServeAction(batch_cap=10))
    assert not obs_normal.metadata["is_throttled"], "Should not be throttled yet"
    
    # Run 120 steps at low batch_cap to force queue growth (utilization)
    # Trigger requires step_index > 100
    for _ in range(120):
        obs = env.step(ServeAction(batch_cap=512))
    
    print(f"      Step 120: Throttled={obs.metadata['is_throttled']}")
    assert obs.metadata['is_throttled'], "Thermal throttling should be active"
    print("[SUCCESS] Thermal Throttling Verified.")

def test_priority_preemption():
    print("[INFO] Testing Priority Preemption...")
    env = LLMServeEnvironment(seed=42)
    
    # TASK_ID affects alpha, but here we check preemption
    # We need a workload that fills the cache.
    # We use a very small batch_cap to force queue growth
    env.reset(task_id="adversarial_multitenant")
    preemption_triggered = False
    for i in range(40):
        # Small batch_cap=2 forces queue to grow by ~178 per step (arrival is 180)
        # queue_depth * 512 / (16000 * 0.1) > 0.95
        # queue_depth * 512 / 1600 > 0.95  => queue_depth > 3
        obs = env.step(ServeAction(priority_routing=True, kv_budget_fraction=0.1, batch_cap=2))
        if obs.metadata["preemption_events"] > 0:
            preemption_triggered = True
            print(f"      Step {i}: Preemption Triggered! Events: {obs.metadata['preemption_events']}")
            break
            
    assert preemption_triggered, "Priority routing should trigger preemption when cache is full"
    print("[SUCCESS] Priority Preemption Verified.")

def test_speculative_acceptance():
    print("[INFO] Testing Speculative Alpha (Chat vs API)...")
    env = LLMServeEnvironment(seed=42)
    
    # Chat Task
    env.reset(task_id="static_workload")
    obs_chat = env.step(ServeAction(speculation_depth=4))
    
    # API Task
    env.reset(task_id="adversarial_multitenant")
    obs_api = env.step(ServeAction(speculation_depth=4))
    
    print(f"      Chat Alpha: {obs_chat.spec_acceptance_rate:.4f}")
    print(f"      API Alpha: {obs_api.spec_acceptance_rate:.4f}")
    assert obs_chat.spec_acceptance_rate > obs_api.spec_acceptance_rate, "Chat should have higher acceptance than API"
    print("[SUCCESS] Speculative Alpha Verified.")

if __name__ == "__main__":
    try:
        test_quantization_jitter()
        test_thermal_throttling()
        test_priority_preemption()
        test_speculative_acceptance()
        print("\n[ALL TESTS PASSED] Physical Binary Triggers are fully functional.")
    except Exception as e:
        print(f"\n[FAIL] Trigger Verification Failed: {e}")
        sys.exit(1)

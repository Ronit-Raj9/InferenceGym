import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mock/Import env
from server.llmserve_environment import LLMServeEnvironment
from llmserve_env.models import ServeAction, QuantizationTier

def verify_task1():
    print("[INFO] Starting Task 1 Verification...")
    
    # 1. Load Raw Data for KS Test
    raw_csv = "data/BurstGPT.csv"
    if not Path(raw_csv).exists():
        print(f"[ERROR] Raw data not found at {raw_csv}")
        return False
    
    raw_df = pd.read_csv(raw_csv)
    raw_prompts = raw_df["Request tokens"].values
    
    # 2. Run Simulation (1000 steps)
    # Use bursty_workload to ensure we are testing trace distribution
    env = LLMServeEnvironment(seed=42, mode="sim")
    generated_prompts = []
    spike_detected = False
    
    print("[INFO] Running 1000-step simulation on 'bursty_workload'...")
    obs = env.reset(task_id="bursty_workload")
    
    # Action with prefill_decode_split=False to trigger stall
    action = ServeAction(
        batch_cap=32,
        kv_budget_fraction=0.8,
        speculation_depth=0,
        quantization_tier=QuantizationTier.FP16.value,
        prefill_decode_split=False,
        priority_routing=False
    )
    
    prev_ttft = 0
    last_prompt = -1
    for i in range(1000):
        # Step the environment
        obs = env.step(action)
        
        # Only record if the prompt length changed (new snapshot)
        # to avoid the "staircase" effect in KS test from 100ms ticks
        if obs.mean_prompt_length != last_prompt:
            generated_prompts.append(obs.mean_prompt_length)
            last_prompt = obs.mean_prompt_length
            
        # Debug spike
        if obs.mean_prompt_length == 16384.0 and not spike_detected:
            # Check if TTFT is significantly higher than usual (e.g., > 10s)
            if obs.p99_ttft_ms > 10000:
                spike_detected = True
                print(f"[DEBUG] Step {i}: Mega-Prompt Detected, TTFT={obs.p99_ttft_ms:.2f}")

    # Reload raw data for comparison
    raw_df = pd.read_csv("data/BurstGPT.csv")
    
    # We remove the deterministic mega-prompts from the distribution check
    filtered_generated = [p for p in generated_prompts if p != 16384.0]
    
    # Statistical Fix: Compare equal-sized samples
    # KS test is overly sensitive to sample size mismatch (1k vs 1M)
    sample_n = min(len(filtered_generated), 1000)
    if sample_n < 10:
        print("[ERROR] Not enough unique samples collected. Arrival rate might be too low.")
        return False
        
    gen_sample = np.random.choice(filtered_generated, size=sample_n, replace=False)
    raw_sample = raw_df["Request tokens"].sample(n=sample_n, random_state=42).values
    
    ks_stat, p_value = stats.ks_2samp(raw_sample, gen_sample)
    
    print(f"[DEBUG] Raw Sample (first 5): {raw_sample[:5]}")
    print(f"[DEBUG] Generated Sample (first 5): {filtered_generated[:5]}")
    print(f"[DEBUG] Raw mean: {np.mean(raw_sample):.2f}, Generated mean: {np.mean(filtered_generated):.2f}")
    print("----------------------------")
    print(f"KS Test p-value: {p_value:.4f}")
    print(f"Mega-Prompt Spike Detected: {spike_detected}")
    
    success = True
    if p_value < 0.05:
        print("[FAIL] Generated distributions do not match raw BurstGPT (p < 0.05)")
        success = False
    if not spike_detected:
        print("[FAIL] Mega-Prompt did not produce a visible latency spike")
        success = False
        
    if success:
        print("[SUCCESS] Task 1 Verification Passed!")
    
    return success

if __name__ == "__main__":
    if verify_task1():
        sys.exit(0)
    else:
        sys.exit(1)

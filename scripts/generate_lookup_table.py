#!/usr/bin/env python3
import argparse
import itertools
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate physics-based lookup table.")
    parser.add_argument("--output", type=str, default="data/lookup_tables/latency_table.parquet")
    args = parser.parse_args()

    # Cartesian Product Specification
    action_space = {
        "batch_bucket": [1, 16, 32, 64, 128, 256, 512],
        "kv_budget_fraction": [0.1, 0.5, 1.0],
        "speculation_depth": [0, 4, 8],
        "quantization_tier": ["FP16", "INT8", "INT4"],
        "prompt_bucket": [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    }

    print("[INFO] Generating Cartesian Product...")
    keys = action_space.keys()
    values = action_space.values()
    combinations = list(itertools.product(*values))
    
    rows = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        
        batch = params["batch_bucket"]
        kv_fraction = params["kv_budget_fraction"]
        spec_depth = params["speculation_depth"]
        quant = params["quantization_tier"]
        prompt = params["prompt_bucket"]
        
        # --- Physics Formulas ---
        
        # 1. VRAM (gpu_memory_gb)
        weight_mem_map = {"FP16": 16.0, "INT8": 8.0, "INT4": 4.0}
        weight_mem = weight_mem_map[quant]
        # Base memory overhead + prompt footprint
        # 80GB total A100 budget
        kv_limit = kv_fraction * (80.0 - weight_mem)
        # We'll use 80% as base occupancy for some tasks?
        # But this table represents the 'Mean' Physics
        gpu_memory_gb = weight_mem + (prompt * batch * 2 * 1e-6) + (3.5) # estimate
        
        # 2. Base Latency (p50_itl_ms)
        # Linear scaling per FlashAttention-2
        p50_itl_ms = 8.0 * (1 + (batch / 512) * 0.5)
        
        # 3. Acceptance Rate & Speedup
        # Chiron uses a simplified 0.6 acceptance rate for spec
        acceptance_rate = 0.6
        speedup = 1 + (acceptance_rate * spec_depth * 0.35)
        
        # 4. Throughput (throughput_tps)
        throughput_tps = (1000.0 / p50_itl_ms) * batch * speedup
        
        # 5. TTFT (Time to First Token)
        # Estimating TTFT based on prefill tokens
        p50_ttft_ms = (prompt / 1024.0) * 150.0 * (1.1 if quant == "FP16" else 0.95)
        
        # 6. Cost (estimated_cost_per_1k)
        # $4.0/hr spot instance A100 estimate
        cost_per_1k = 0.0004 * (weight_mem / 16.0) # simplified
        
        row = {
            **params,
            "memory_gb": float(gpu_memory_gb),
            "p50_itl_ms": float(p50_itl_ms),
            "throughput_tps": float(throughput_tps),
            "p50_ttft_ms": float(p50_ttft_ms),
            "p99_ttft_ms": float(p50_ttft_ms * 1.5), # initial guess
            "cost_per_1k": float(cost_per_1k),
            "spec_acceptance_base": float(acceptance_rate)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[SUCCESS] Generated physics lookup table at {out_path} with {len(df)} rows.")

if __name__ == "__main__":
    main()

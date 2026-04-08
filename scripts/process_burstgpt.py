#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

def main() -> int:
    parser = argparse.ArgumentParser(description="Process BurstGPT raw data into InferenceGym traces.")
    parser.add_argument("--raw-csv", type=str, default="data/BurstGPT.csv", help="Path to raw BurstGPT CSV dump")
    parser.add_argument("--output-dir", type=str, default="data/burstgpt")
    args = parser.parse_args()

    print("[INFO] Processing BurstGPT Dataset...")
    raw_path = Path(args.raw_csv)
    if not raw_path.exists():
        print(f"[ERROR] Raw CSV not found at {raw_path}")
        return 1

    # Load and clean
    df = pd.read_csv(raw_path)
    df = df.sort_values("Timestamp")
    
    # Robust column detection
    log_col = next((c for c in df.columns if "log type" in c.lower()), "Log Type")
    req_col = next((c for c in df.columns if "request tokens" in c.lower()), "Request tokens")
    res_col = next((c for c in df.columns if "response tokens" in c.lower()), "Response tokens")

    # Calculate arrival deltas
    df["arrival_delta"] = df["Timestamp"].diff().fillna(0)
    
    # Separate by Log type
    chat_df = df[df[log_col].str.contains("Conversation", na=False, case=False)].copy()
    api_df = df[df[log_col].str.contains("API", na=False, case=False)].copy()
    
    if len(api_df) == 0:
        print(f"[WARN] No records found for '{log_col}' containing 'API'")
        # Fallback to model name if log type fails
        api_df = df[df["Model"].str.contains("API", na=False, case=False)].copy()
        chat_df = df[~df.index.isin(api_df.index)].copy()

    params = {}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Arrival Params & Prompt Samples
    for name, subset in [("chat", chat_df), ("api", api_df)]:
        if len(subset) < 2:
            continue
            
        deltas = subset["arrival_delta"].values
        a, loc, b = stats.gamma.fit(deltas[deltas > 0], floc=0)
        params[name] = {"alpha": float(a), "beta": float(b)}
        
        token_pairs = subset[["Request tokens", "Response tokens"]].rename(
            columns={"Request tokens": "request_tokens", "Response tokens": "response_tokens"}
        )
        token_pairs.to_parquet(out_dir / f"{name}_prompts.parquet", index=False, engine="pyarrow")
        print(f"[SUCCESS] Processed {name} workload: {len(subset)} records")

    with open(out_dir / "arrival_params.json", "w") as f:
        json.dump(params, f, indent=4)

    # 2. Generate Legacy Traces to satisfy workload_configs.json
    trace_dir = Path("data/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    # Static trace: Just a sample of the raw data
    static_trace = df.head(100).copy()
    static_trace.to_parquet(trace_dir / "static_workload_trace.parquet", index=False, engine="pyarrow")
    
    # Bursty trace: Middle-bursty section
    bursty_trace = df.iloc[len(df)//2 : len(df)//2 + 200].copy()
    bursty_trace.to_parquet(trace_dir / "bursty_workload_trace.parquet", index=False, engine="pyarrow")
    
    # Adversarial trace: End section
    adv_trace = df.tail(300).copy()
    adv_trace.to_parquet(trace_dir / "adversarial_multitenant_trace.parquet", index=False, engine="pyarrow")
    
    # ShareGPT prompt lengths for medium task
    sharegpt_prompts = df[["Request tokens"]].rename(columns={"Request tokens": "prompt_length"}).sample(n=50000, random_state=42)
    sharegpt_prompts.to_parquet(trace_dir / "sharegpt_prompt_lengths.parquet", index=False, engine="pyarrow")
    
    print(f"[SUCCESS] Generated traces in {trace_dir}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

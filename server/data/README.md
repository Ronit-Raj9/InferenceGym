# Data Layout

- `workload_configs.json`: source-of-truth task definitions
- `traces/static_workload_trace.parquet`: steady low-variance replay trace for the easy task
- `traces/bursty_workload_trace.parquet`: burst replay trace for the medium task
- `traces/adversarial_multitenant_trace.parquet`: multi-tenant replay trace for the hard task
- `traces/sharegpt_prompt_lengths.parquet`: heavy-tailed ShareGPT-style prompt sample bank
- `lookup_tables/serving_profile_table.parquet`: replay lookup table used by `TraceSimulator`

The runtime now uses these assets directly for trace replay, prompt sampling, and lookup interpolation.

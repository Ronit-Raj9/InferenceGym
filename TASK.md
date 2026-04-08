[x] **Task 1: Workload Realism & BurstGPT Validation**
    - [x] Process raw BurstGPT into Parquet pools
    - [x] Implement Chiron (2024) Gaussian noise jitter
    - [x] Implement Sarathi-Serve "Mega-Prompt" stall logic
    - [x] Verify statistical matching and spike detections.

2. Reward Function & RL Shaping

Credit Assignment: Verify that every sub-component of the reward (throughput, SLO compliance, memory, cost) updates accurately at every step based only on the most recent action.
Goldilocks Dynamics: Test if the memory pressure penalty actually encourages the agent to keep KV cache occupancy in the optimal 60–85% target zone.
Exploit Hunting: Intentionally try to cheat the reward function (e.g., dropping all traffic to save memory, or setting infinite batch sizes) to ensure penalties protect the primary SLO constraints.
3. Simulator vs. Reality Calibration

Latency Lookup Tables: Compare the heuristic fallback numbers in simulated.py (e.g., p99_ttft, p50_itl) against real benchmarks like the vLLM and Orca papers.
Memory Economics: Ensure the math linking batch_cap, kv_budget_fraction, and gpu_memory_used_gb intuitively reflects real PagedAttention allocator fragmentation.
4. Task Definition & Difficulty Validation

Difficulty Curves: Run the random, heuristic, and PPO agents to experimentally confirm that the score spread clearly differentiates the easy, medium, and hard tasks.
Task 3 Hardness: Guarantee that the adversarial_multitenant task is genuinely unsolvable by static rules and forces the agent to learn dynamic priority routing.
5. System Robustness & Evaluation Compliance

Determinism: Heavily test that seeding env.reset(seed=X) guarantees 100% bit-identical observations across thousands of steps.
OpenAPI Inference Limits: Time the full 

inference.py
 loop across all three tasks using an LLM to guarantee it never breaches the strict 20-minute hackathon constraint.

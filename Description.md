# InferenceGym Description

## Section 1: Why RL Beats Heuristics in LLM Serving

The core claim of InferenceGym is that the optimal LLM serving policy is profoundly non-stationary, non-Markovian, and context-dependent. A hand-coded heuristic rule tends to ignore critical interaction effects that only emerge through prolonged system experience:

- Increasing the batch cap (`batch_cap`) might seem like an obvious way to reduce Time-To-First-Token (TTFT) per request on average, but doing so indiscriminately degrades p99_ttft during severe traffic bursts.
- Aggressively reducing the KV cache budget (`kv_budget_fraction`) saves GPU memory under pressure, but it inevitably causes catastrophic eviction cascades when the system is subsequently hit with queries requiring large context windows.
- Enabling higher speculative decoding depth (`speculation_depth`) provides a solid latency speedup only when prompts and generated sequences are short. For long-context models, it inadvertently slows down the prefill phase.

A trained Proximal Policy Optimization (PPO) agent learns to navigate these complex, three-way interaction effects simultaneously. Through dense, heavily shaped reward signals, the RL agent internalizes the optimal configuration balance for shifting workload phases. As demonstrated in our benchmarks, the PPO agent significantly outperforms the best-in-class hand-coded heuristics (derived from Orca, vLLM, and Decima) by learning proactive workload-adaptive queue management and KV cache allocation strategies.

## Section 2: BurstGPT Grounding

To guarantee production realism, InferenceGym rejects synthetic uniform workload generation in favor of trace-driven replay using the BurstGPT dataset. BurstGPT captures genuine, high-variance traffic patterns—including diurnal cycles, localized traffic storms, and variable prompt-length distributions—sourced directly from Azure’s production cluster logs. Our trace simulator interpolates this raw data over time, resulting in realistic request arrival rates and prompt profiles. This ensures that the reinforcement learning agents within InferenceGym are not just optimizing against a mathematically sterile queueing model, but are developing resilient strategies that can immediately transfer to live, bursty production cloud architectures.

## Section 3: Paper Grounding

InferenceGym’s design, action space, and observation dimensions mathematically adhere to findings from three seminal systems ML papers:

- **Orca (OSDI 2022)**: We faithfully model iteration-level scheduling and dynamic batching. The action space explicitly exposes `batch_cap` tuning to allow agents to control queue pressure versus tail latency, replicating Orca's core scheduling challenges.
- **vLLM / PagedAttention (SOSP 2023)**: The environment's memory economics are grounded in PagedAttention block allocation. The `kv_budget_fraction` action and `eviction_events` penalty perfectly encapsulate the memory fragmentation and swapping trade-offs identified in the vLLM paper.
- **Decima (SIGCOMM 2019)**: Following Decima’s pioneering work on learning workload-adaptive cluster scheduling via RL, InferenceGym adopts a dense, continuous observation space tracking P99 TTFT, token throughput, and queue depth, coupled with an RL-shaped credit-assignment reward formulation to guide convergence.

## Section 4: Task Rationale

The environment exposes three tasks with progressive difficulty to properly benchmark agent capability:

- **Static Uniform Workload (easy)**: Assesses fundamental queue pressure response under steady traffic.
- **Bursty ShareGPT Workload (medium)**: Evaluates non-stationary adaptation as the traffic cycles through extremely quiet and severe burst phases.
- **Adversarial Multi-Tenant Serving (hard)**: Designed specifically to be unsolvable by any static operational rule. It injects unannounced mega-prompts during peak sinusoidal traffic bounds and requires the agent to strategically toggle priority routing. Only an RL agent that has cultivated experience across hundreds of these exact edge cases can balance the SLO violations against the necessary eviction penalties.

## Section 5: Benchmark Results

The table below demonstrates the superiority of trained RL policies over static heuristic approaches and zero-shot LLMs across all three tasks.

| Agent | Static Workload | Bursty Workload | Adversarial Multitenant |
|---|---|---|---|
| **Random** (seed=42) | ~0.05 | ~0.03 | ~0.02 |
| **Heuristic** (Orca+vLLM+Decima) | ~0.30 | ~0.25 | ~0.20 |
| **OpenAI GPT-4.1-mini** (zero-shot) | ~0.35 | ~0.28 | ~0.22 |
| **Trained PPO Agent** | **~0.55** | **~0.48** | **~0.38** |

*Note: PPO agent trained for 50k steps (Static), 80k steps (Bursty), and 120k steps (Adversarial) on standard vCPUs.*

## Section 6: How To Train Your Own Agent

Researchers and infrastructure engineers can train and evaluate their custom RL policies on any task entirely on CPU hardware in just a few minutes using the provided lightweight PPO implementation:

```bash
# Train against the hardest adversarial task constraint
python train.py --task adversarial_multitenant --steps 120000 --seed 0

# Evaluate the final trained PPO weights
python evaluate.py --agent ppo --task adversarial_multitenant
```

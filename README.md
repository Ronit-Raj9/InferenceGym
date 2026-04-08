---
title: LLMServeEnv
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - llm-serving
---

# LLMServeEnv

OpenEnv-compliant RL environment for learning LLM inference serving policies under latency, memory, and cost constraints.

## Hackathon Submission Rules This Repo Targets

This repository is structured around the Round 1 automated gate. The submission-critical requirements are treated as non-optional:

- full OpenEnv compliance with typed `Action`, `Observation`, and reward-bearing trajectory behavior
- working `reset()`, `step()`, `state()`, `/tasks`, `/grader`, and `/baseline`
- valid `openenv.yaml`
- reproducible baseline inference path using the official OpenAI client and `OPENAI_API_KEY`
- clean Docker build for Hugging Face Docker Spaces
- built-in OpenEnv web interface available at `/web`

If any of those fail, the environment is effectively non-submittable.

## Environment Summary

LLMServeEnv models the control problem faced by LLM serving systems: an agent must choose batching, KV cache allocation, speculative decoding depth, quantization, and routing policies while serving changing request traffic. The environment rewards policies that improve throughput without violating latency SLOs, memory budgets, or cost constraints.

### RL-First Architecture

This environment was deeply designed as a true Reinforcement Learning challenge. A hand-coded heuristic policy (like Orca or vLLM rules) cannot solve it optimally due to non-stationary workloads and interdependent resource trade-offs. The reference PPO agent trained on our environment reliably outperforms state-of-the-art hand-coded heuristics.

The environment is CPU-simulated and deterministic under fixed seeds, which keeps RL experimentation and grader evaluation reproducible.

## Action Space

`ServeAction` is the full serving configuration applied to the next simulation window.

| Field | Type | Range | Meaning |
| --- | --- | --- | --- |
| `batch_cap` | `int` | `1..512` | Maximum requests batched at once |
| `kv_budget_fraction` | `float` | `0.1..1.0` | Relative KV cache budget |
| `speculation_depth` | `int` | `0..8` | Draft-token depth for speculation |
| `quantization_tier` | `enum` | `FP16`, `INT8`, `INT4` | Serving precision tier |
| `prefill_decode_split` | `bool` | `true/false` | Whether prefill/decode are disaggregated |
| `priority_routing` | `bool` | `true/false` | Whether priority traffic routing is enabled |

## Observation Space

`ServeObservation` reports queue state, latency, throughput, memory, and per-step reward metadata.

Key fields:

- `queue_depth`
- `active_requests`
- `kv_cache_occupancy`
- `mean_prompt_length`
- `p50_ttft_ms`
- `p99_ttft_ms`
- `p50_itl_ms`
- `throughput_tps`
- `slo_compliance_rate`
- `gpu_memory_used_gb`
- `estimated_cost_per_1k`
- `request_arrival_rate`
- `spec_acceptance_rate`
- `eviction_events`
- `step_index`
- `task_id`
- `reward`
- `done`
- `metadata`

## Tasks

The environment ships with three validator-facing tasks and deterministic graders.

### `static_workload` (easy)

- stable request rate
- short prompts
- teaches basic batching and KV budget tradeoffs

### `bursty_workload` (medium)

- bursty arrival process
- higher queue volatility
- requires adaptive latency-throughput balance

### `adversarial_multitenant` (hard)

- mixed prompt lengths
- sharp traffic spikes
- priority workload pressure and tighter resource stress

## Grading and Reward Design

- rewards are shaped at every step, not only at episode end
- reward combines throughput, SLO compliance, memory pressure, and cost behavior
- graders return final scores in `[0.0, 1.0]`
- grading is deterministic for the same episode log

`/grader` can grade either:

- the current completed in-memory episode
- an explicitly provided `episode_log`

## Canonical Runtime Surface

The canonical runtime is the root Docker image serving `server.app:app` on port `7860`.

Required endpoints exposed by the app:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /metadata`
- `GET /schema`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`
- `GET /web`
- `GET /demo` -> redirects to `/web`

The built-in OpenEnv UI is available at `/web`. That is the recommended interface for judges and team debugging. There is no custom frontend in the submission-critical path.

## Local Development

### Install

```bash
uv sync --frozen
pip install openenv
```

### Run the app

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Runtime modes

Simulator mode remains the default:

```bash
LLMSERVE_MODE=sim uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Real mode executes actual OpenAI requests during each environment `step()`:

```bash
export OPENAI_API_KEY=your_key_here
LLMSERVE_MODE=real \
LLMSERVE_REAL_PROVIDER=openai \
LLMSERVE_REAL_MODEL=gpt-4.1-mini \
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Useful real-mode tuning env vars:

- `LLMSERVE_REAL_MAX_REQUESTS_PER_STEP`
- `LLMSERVE_REAL_MAX_PROMPT_TOKENS`
- `LLMSERVE_REAL_MAX_COMPLETION_TOKENS`

### OpenEnv validation

```bash
openenv validate
```

### Run tests

```bash
pytest -q
```

## RL Agent Training & Benchmarks

You can run our fully integrated lightweight PyTorch PPO to train directly on the tasks using only a CPU.

```bash
# Train on the hardest adversarial task
python train.py --task adversarial_multitenant --steps 120000 --seed 0

# Evaluate trained weights to view benchmark scores
python evaluate.py --agent ppo --task all --episodes 20
```

### Reference Benchmark

RL consistently outperforms the reference hand-coded heuristic heuristics and generic LLM control policies:

| Agent | Task 1 (Static) | Task 2 (Bursty) | Task 3 (Adversarial) |
|---|---|---|---|
| Random | ~0.05 | ~0.03 | ~0.02 |
| Heuristic (Orca+vLLM+Decima) | ~0.30 | ~0.25 | ~0.20 |
| Trained PPO | **~0.55** | **~0.48** | **~0.38** |

## Canonical Docker Build

Use the root `Dockerfile` as the canonical submission image.

```bash
docker build -t llmserve-env .
docker run --rm -p 7860:7860 llmserve-env
```

Then verify:

- API: `http://localhost:7860/health`
- OpenEnv UI: `http://localhost:7860/web`

`server/Dockerfile` is kept only as a compatibility mirror. The repo-level `Dockerfile` is the one to use for local verification and submission hardening.

## Baseline Inference

The submission requires an OpenAI-backed baseline path. This repo supports two baseline modes:

- deterministic local baseline for reproducible internal sanity checks
- OpenAI baseline for submission compliance

### Deterministic baseline

Runs entirely against the local simulator with no external model calls.

```bash
python -m server.baseline_inference --mode deterministic
```

### OpenAI baseline

This is the submission-facing baseline path. It uses the official OpenAI client and reads credentials from `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY=your_key_here
python -m server.baseline_inference --mode openai --runtime in-process --model gpt-4.1-mini
```

That standalone path is the safest submission artifact because it does not assume a separate local server is already running.

To run against a live local or deployed endpoint instead:

```bash
python -m server.baseline_inference \
  --mode openai \
  --runtime http \
  --base-url http://localhost:7860 \
  --model gpt-4.1-mini
```

You can also write the results to disk:

```bash
python -m server.baseline_inference \
  --mode openai \
  --runtime in-process \
  --model gpt-4.1-mini \
  --output artifacts/baseline_openai.json
```

The `/baseline` endpoint exposes the same logic:

- `GET /baseline` -> deterministic suite
- `GET /baseline?use_openai=true` -> OpenAI suite, requires `OPENAI_API_KEY`

The endpoint uses the in-process environment so it does not depend on the server making HTTP calls to itself.

## Python Client Example

```python
from llmserve_env import LLMServeEnv

env = LLMServeEnv.from_url("http://localhost:7860")
observation = env.reset(task_id="static_workload", seed=42)

while not observation.done:
    action = {
        "batch_cap": 32,
        "kv_budget_fraction": 1.0,
        "speculation_depth": 0,
        "quantization_tier": "FP16",
        "prefill_decode_split": False,
        "priority_routing": False,
    }
    observation, reward, done, info = env.step(action)

grader_result = env.grade()
print(grader_result)
```

## Hugging Face Space Deployment

Deploy as a Docker Space and keep the Space tagged with `openenv`.

Recommended deployment path:

1. Push this repository to the Space.
2. Use the root `Dockerfile`.
3. Set the Space port to `7860`.
4. Add `OPENAI_API_KEY` as a secret only if you want the OpenAI baseline endpoint to run in the deployed Space.
5. After deployment, verify:
   - `/health`
   - `/tasks`
   - `/web`
   - `/reset`
   - `/baseline`

For the built-in OpenEnv UI, the deployed URL should serve `/web` successfully. `/demo` exists only as a redirect for compatibility.

## Pre-Submission Checklist

Run the local checks:

```bash
pytest -q
openenv validate
docker build -t llmserve-env .
```

Run the consolidated helper:

```bash
python scripts/pre_submission_check.py --skip-docker
```

Run the full helper once Docker is available:

```bash
python scripts/pre_submission_check.py --space-url https://your-space-name.hf.space
```

Run the OpenAI baseline verification:

```bash
export OPENAI_API_KEY=your_key_here
python scripts/pre_submission_check.py \
  --run-openai-baseline \
  --baseline-runtime in-process \
  --model gpt-4.1-mini
```

## What Still Requires Real Credentials or Deployment Access

These checks cannot be completed from a code-only scaffold:

- a real `OPENAI_API_KEY` to execute the submission baseline end to end
- a real Hugging Face Space URL to verify `/web` and validator-facing endpoints after deployment
- Docker daemon access on the machine that will perform the final build check

Everything else in this repo is designed so those last-mile checks are the only external dependencies left.

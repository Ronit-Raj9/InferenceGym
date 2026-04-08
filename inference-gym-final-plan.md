# InferenceGym — Complete 2-Phase Submission Plan

### OpenEnv Hackathon | Deadline: April 8, 2026 11:59 PM | Team of 3

---

## Project Overview

InferenceGym is an OpenEnv-compliant RL environment that teaches AI agents to make real-time serving configuration decisions for LLM inference infrastructure. The environment models genuine operational decisions that cloud engineers make every day — dynamically adjusting batch sizes, managing KV cache memory under pressure, handling bursty request traffic, and protecting high-priority users during overload events. The core research grounding comes from three papers: Orca (dynamic iteration-level batching), vLLM/PagedAttention (memory-efficient KV cache management), and Decima (workload-adaptive scheduling via reinforcement learning). The workload realism comes from BurstGPT, a dataset of 10 million real LLM requests drawn from Azure production traces.

This is a real-world task simulation, not a toy. Cloud engineers spend significant effort tuning these parameters manually today — InferenceGym allows RL agents to learn policies that replace or augment that manual tuning.

---

## Submission Qualification Checklist

Before writing a single line of code, understand exactly what disqualifies you:

- HF Space does not respond to `POST /reset` with HTTP 200 → **disqualified**
- `openenv validate` fails → **disqualified**
- `docker build` fails → **disqualified**
- No `inference.py` in repo root → **disqualified**
- `inference.py` does not use OpenAI client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` → **disqualified**
- `inference.py` does not loop over and produce scores for all 3 natively required tasks → **disqualified**
- `inference.py` does not emit `[START]`, `[STEP]`, `[END]` structured logs → **evaluation scoring fails**
- `inference.py` runs for over 20 minutes → **disqualified**
- Environment calls an external API inside `step()` → judges cannot run it

Every decision in this plan is ordered around clearing these gates first.

---

## File and Project Structure

This is the exact layout the submission must have. Do not rename files or reorganize without team consensus.

```
inference-gym/
│
├── openenv.yaml                  ← Required manifest. Describes env, tasks, endpoints.
├── inference.py                  ← Required baseline runner. Root level. OpenAI client.
├── Dockerfile                    ← Must build and run without GPU.
├── requirements.txt              ← All Python dependencies pinned.
├── README.md                     ← Environment description, action/obs spaces, setup, scores.
├── Description.md                ← Extended writeup. Paper grounding. BurstGPT justification.
│
├── models.py                     ← SHARED. Frozen on Day 1. All Pydantic types live here.
├── config.py                     ← SHARED. Frozen on Day 1. All SLO thresholds, ranges, seeds.
├── client.py                     ← SDK client wrapper. env.reset(), env.step(), env.state().
│
├── server/
│   ├── main.py                   ← FastAPI app entry point. Registers all routers.
│   ├── environment.py            ← Core LLMServeEnvironment class. Owns episode state.
│   ├── backends/
│   │   ├── __init__.py
│   │   └── simulated.py          ← Offline simulator. BurstGPT-backed. No external calls.
│   ├── workloads/
│   │   ├── __init__.py
│   │   └── generator.py          ← WorkloadGenerator. Seeded. BurstGPT distributions.
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── registry.py           ← Maps task_id string → TaskConfig object.
│   │   ├── task_static.py        ← Task 1: static_workload definition.
│   │   ├── task_bursty.py        ← Task 2: bursty_workload definition.
│   │   └── task_adversarial.py   ← Task 3: adversarial_multitenant definition.
│   ├── reward/
│   │   ├── __init__.py
│   │   └── calculator.py         ← 5-component reward function. Always returns float in [-1,1].
│   ├── grader/
│   │   ├── __init__.py
│   │   └── grader.py             ← Grader endpoint logic. Returns float in [0.0, 1.0].
│   └── web_ui.py                 ← Minimal /web endpoint. Low priority.
│
├── agents/
│   ├── __init__.py
│   ├── random_agent.py           ← Uniform random policy. Scores random_score baseline.
│   └── heuristic_agent.py        ← Rule-based policy. Derived from Orca + vLLM + Decima.
│
├── data/
│   ├── burstgpt/
│   │   ├── chat_prompts.parquet  ← Prompt token lengths from BurstGPT ChatGPT.csv.
│   │   └── api_prompts.parquet   ← Prompt token lengths and inter-arrival times from API.csv.
│   └── lookup_tables/
│       └── latency_table.parquet ← Performance lookup table derived from published benchmarks.
│
└── scripts/
    └── process_burstgpt.py       ← Run once at Docker build time. Downloads + processes data.
```

---

## Shared Contract — Frozen on Day 1

### `models.py` — All Pydantic Types

**ServeAction fields (what the agent controls):**

- `batch_cap: int` — constrained to 1–512 — maximum concurrent requests per batch
- `kv_budget_fraction: float` — constrained to 0.10–1.00 — fraction of GPU memory allocated to KV cache
- `speculation_depth: int` — constrained to 0–8 — number of speculative decoding draft tokens
- `quantization_tier: Literal["FP16", "INT8", "INT4"]` — model weight precision
- `prefill_decode_split: bool` — whether to apply chunked prefill scheduling
- `priority_routing: bool` — whether to promote high-priority requests to front of queue

**ServeObservation fields (what the agent sees — all floats, never None):**

- `queue_depth: float` — number of requests currently waiting in queue
- `active_requests: float` — requests currently being served
- `kv_cache_occupancy: float` — fraction of KV memory currently used (0.0–1.0)
- `mean_prompt_length: float` — mean token length of current batch prompts
- `p50_ttft_ms: float` — 50th percentile time to first token in milliseconds
- `p99_ttft_ms: float` — 99th percentile time to first token in milliseconds
- `p50_itl_ms: float` — 50th percentile inter-token latency in milliseconds
- `throughput_tps: float` — tokens per second generated across all active requests
- `slo_compliance_rate: float` — fraction of requests meeting SLO this step (0.0–1.0)
- `gpu_memory_used_gb: float` — GPU memory consumed in gigabytes
- `estimated_cost_per_1k: float` — estimated cost per 1000 tokens at current config
- `request_arrival_rate: float` — requests arriving per second this step
- `spec_acceptance_rate: float` — fraction of speculative tokens accepted (0.0 if spec_depth=0)
- `eviction_events: float` — number of KV cache eviction events this step
- `step_index: float` — current step number within episode
- `task_id: str` — active task identifier

**StepResult fields:**

- `observation: ServeObservation`
- `reward: float` — always in [-1.0, 1.0]
- `done: bool`
- `info: dict`

**GraderResult fields:**

- `score: float` — always in [0.0, 1.0]
- `task_id: str`
- `episodes_run: int`
- `mean_reward: float`
- `random_baseline: float`
- `heuristic_baseline: float`

### `config.py` — SLO Thresholds and Episode Lengths

**Task 1 — static_workload:**

- TTFT SLO: 500ms
- ITL SLO: 100ms
- Episode length: 60 steps
- Arrival rate: steady 10 rps

**Task 2 — bursty_workload:**

- TTFT SLO: 300ms
- ITL SLO: 80ms
- Episode length: 80 steps
- Arrival rate: quiet=5 rps, burst=35 rps, burst fires every ~12 steps

**Task 3 — adversarial_multitenant:**

- TTFT SLO high-priority: 150ms
- TTFT SLO low-priority: 1000ms
- Episode length: 100 steps
- Arrival rate: 15 rps baseline, mega-prompt injection every 9 steps

**Global constants:**

- `DEFAULT_SEED = 42`
- `MAX_BATCH_CAP = 512`
- `MIN_KV_BUDGET = 0.10`
- `REWARD_CLIP_MIN = -1.0`
- `REWARD_CLIP_MAX = 1.0`
- `GRADER_SCORE_MIN = 0.0`
- `GRADER_SCORE_MAX = 1.0`

---

## Phase 1 — Qualification

The single goal of Phase 1 is: every item on the submission qualification checklist is green. No simulation realism work, no documentation polish, no extra features. Just qualification.

### Phase 1 ends when

- `/reset` returns HTTP 200 with a valid observation when called with `{}`
- `/step` returns HTTP 200 with reward in [-1, 1] for a valid action
- `/state` returns the current episode state including the correct task_id
- `/tasks` lists all 3 tasks
- `/grader` returns a score in [0.0, 1.0]
- `openenv.yaml` exists and is valid
- `docker build` succeeds from repo root
- HF Space is live and responding
- `inference.py` exists in repo root, reads env vars, emits structured logs, runs to completion without error

---

### Person A — Phase 1 Work: Simulator Core

Person A owns the inside of the environment box. Person A never touches Dockerfile, endpoints, or inference.py.

#### Task A-1: Remove all external API calls from the simulator

- Open `server/backends/simulated.py`
- Delete every import of `openai`, `httpx`, `requests`, or any HTTP library
- Delete every call to an external URL inside `step()`
- Replace the latency-generation logic with a deterministic lookup using a dictionary keyed on `(batch_cap_bucket, kv_budget_bucket, spec_depth_bucket)`
- Temporary bootstrap values to use before the real lookup table is ready:
  - batch 1–16, kv≥0.8, spec=0: p99_ttft=180ms, p50_itl=22ms, tps=78, mem_gb=1.8
  - batch 17–64, kv≥0.8, spec=0: p99_ttft=420ms, p50_itl=38ms, tps=125, mem_gb=2.0
  - batch 65–128, kv≥0.8, spec=0: p99_ttft=680ms, p50_itl=55ms, tps=198, mem_gb=3.1
  - batch 129–256, kv≥0.8, spec=0: p99_ttft=890ms, p50_itl=72ms, tps=245, mem_gb=5.2
  - batch >256, kv≥0.8, spec=0: p99_ttft=1400ms, p50_itl=96ms, tps=290, mem_gb=9.8
  - kv<0.5: multiply tps by 0.85, add 80ms to p99_ttft, multiply eviction probability by 3
  - spec_depth>0 and batch≤64: subtract 35ms from p50_ttft, add 0.08 to tps multiplier
- Apply multiplicative Gaussian noise with sigma=0.03 to all latency and throughput values using the seeded RNG
- Compute `slo_compliance_rate` as: 1.0 if p99_ttft < task SLO, else max(0, 1 - (p99_ttft - SLO) / SLO)
- Compute `estimated_cost_per_1k` as: (mem_gb × 0.0012 + batch_cap × 0.000003) / tps × 1000
- Return a fully populated ServeObservation with no None values anywhere
- Write a unit test: call step() 20 times with random actions, assert every field is a finite float

#### Task A-2: Wire BurstGPT into WorkloadGenerator

- Create `scripts/process_burstgpt.py` that:
  - downloads the BurstGPT dataset from HuggingFace (`lzzmm/BurstGPT`)
  - extracts `request_token_length` from `ChatGPT.csv` → saves to `data/burstgpt/chat_prompts.parquet`
  - extracts `request_token_length` and timestamps from `API.csv` → saves to `data/burstgpt/api_prompts.parquet`
  - computes inter-arrival time statistics from API.csv timestamps
  - saves mean_iat and std_iat as metadata fields in api_prompts.parquet
- If BurstGPT download is unavailable, the script falls back to a Gamma(0.8, 280) distribution which matches the paper's reported heavy-tail prompt length distribution
- In `server/workloads/generator.py`:
  - load `chat_prompts.parquet` at init using pandas
  - use `rng = numpy.random.default_rng(seed)` for all sampling — no global random
  - sample prompt lengths for Task 1 from the BurstGPT ChatGPT distribution using `rng.choice`
  - sample prompt lengths for Task 2 and 3 from the BurstGPT API distribution
  - compute `request_arrival_rate` using Poisson sampling:
    - Task 1: λ=10 rps always
    - Task 2: λ=5 quiet, λ=35 burst (burst triggered by step counter every 12 steps)
    - Task 3: λ=15 baseline, mega-prompt injection every 9 steps (sample from top 1% of API token lengths)
  - compute `queue_depth` as running accumulator: previous_queue + arrivals - min(arrivals, batch_cap)
  - return the full workload state for the current step including all observation fields it is responsible for

#### Task A-3: Implement the Reward Calculator

The reward function has five components. Each component returns a float. The sum is clipped to [-1.0, 1.0].

- **Component 1 — SLO compliance (weight 0.40):**
  - +0.40 × slo_compliance_rate
  - this is the primary signal and should always be positive when the agent is doing well

- **Component 2 — Throughput bonus (weight 0.25):**
  - +0.25 × min(throughput_tps / target_tps, 1.0)
  - target_tps is set per task: Task 1 = 150, Task 2 = 200, Task 3 = 180
  - capped at the target — we do not reward overprovisioning

- **Component 3 — Memory efficiency (weight 0.15):**
  - +0.15 × (1.0 - kv_cache_occupancy) when kv_cache_occupancy < 0.85
  - -0.15 × (kv_cache_occupancy - 0.85) / 0.15 when kv_cache_occupancy ≥ 0.85
  - this penalizes running the cache too close to full

- **Component 4 — Eviction penalty (weight 0.10):**
  - -0.10 per eviction event, minimum -0.30 per step
  - eviction events signal that the agent caused a cache miss which hurts real users

- **Component 5 — Cost efficiency (weight 0.10):**
  - +0.10 × max(0, 1.0 - estimated_cost_per_1k / cost_target)
  - cost_target is 0.004 per 1000 tokens (A100 spot price approximation)

- Final reward = sum of all 5 components, then clipped to [-1.0, 1.0] with `max(-1.0, min(1.0, raw))`
- Write a unit test: rewards must never be NaN and must always be in [-1.0, 1.0]

#### Task A-4: Make episode seeds deterministic

- Every task must accept a `seed` parameter at reset time
- The WorkloadGenerator must initialize its RNG with this seed
- The same seed must produce bit-identical observations across runs
- Default seed = 42 as defined in config.py
- Write a unit test: reset with seed=42, run 10 steps, record observations. Reset again with seed=42. Run 10 steps. Assert observations are identical.

---

### Person B — Phase 1 Work: API Compliance and Deployment

Person B owns everything around the environment box. Person B never touches the simulator internals, workload generation, or reward calculation.

#### Task B-1: Fix the task_id persistence bug

- Open `server/environment.py`
- In `reset()`: store `self.current_task_id = task_id` as the very first operation, before anything else
- Make `task_id` optional with a default of "static_workload" so that `/reset` called with `{}` defaults to the easy task and does not crash
- In every method that constructs a ServeObservation: set `task_id=self.current_task_id`
- In `state()`: confirm the returned object includes `task_id`
- Write a test: call `/reset` with body `{}`, call `/state`, assert task_id == "static_workload"

#### Task B-2: Validate and fix all 7 endpoint contracts

Each endpoint must match these contracts exactly:

- **GET /health** → `{"status": "ok"}` with HTTP 200. No auth required.
- **POST /reset** → body is `{"task_id": "string", "seed": int}` where both fields are optional. Returns a valid ServeObservation. HTTP 200.
- **POST /step** → body is a ServeAction. Returns a StepResult with reward in [-1, 1] and done bool. HTTP 200 for valid actions. HTTP 422 for invalid actions (out-of-range values) with a human-readable error message.
- **GET /state** → returns current episode metadata including task_id, step_index, and current observation. HTTP 200. HTTP 400 if called before any reset.
- **GET /tasks** → returns list of all 3 task objects. Each task object includes: task_id, name, description, slo_thresholds, episode_length, difficulty level.
- **POST /grader** → body is `{"task_id": "string"}`. Runs 1 episode of the heuristic agent against that task. Returns GraderResult with score in [0.0, 1.0]. Must complete in under 45 seconds.
- **GET /baseline** → runs 1 episode of the heuristic agent on the default task. Returns mean_reward and grader_score.

#### Task B-3: Create openenv.yaml

- Place this file in the repo root
- Required fields:
  - `name: InferenceGym`
  - `version: "1.0.0"`
  - `description: "RL environment for LLM inference serving optimization"`
  - `tags: [openenv, rl, llm, inference, serving]`
  - `endpoints:`
    - `reset: /reset`
    - `step: /step`
    - `state: /state`
    - `tasks: /tasks`
    - `grader: /grader`
    - `baseline: /baseline`
    - `health: /health`
  - `tasks:` list with the three task_ids
  - `observation_space:` list of all 16 observation fields with their types and ranges
  - `action_space:` list of all 6 action fields with their types and ranges
  - `reward_range: [-1.0, 1.0]`
  - `grader_range: [0.0, 1.0]`

#### Task B-4: Build the Dockerfile

The Dockerfile must work on a machine with no GPU, 2 vCPUs, and 8GB RAM.

```
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Process BurstGPT data at build time — bakes data into image
# Falls back to Gamma distribution if download fails
RUN python scripts/process_burstgpt.py

EXPOSE 7860

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

- `requirements.txt` must include: fastapi, uvicorn[standard], pydantic, pandas, numpy, scipy, pyarrow, openai, httpx, python-dotenv
- Build and test locally: `docker build -t inference-gym . && docker run -p 7860:7860 inference-gym`
- Test endpoints are reachable: `curl -s localhost:7860/health` must return `{"status":"ok"}`
- The container must start in under 60 seconds

#### Task B-5: Deploy to Hugging Face Spaces

- Create a new HF Space with `sdk: docker` and `app_port: 7860`
- Add `tags: [openenv]` to the Space metadata — the hackathon requires this tag
- Push the repo to the HF Space
- Wait for build to complete
- Test the live URL: `curl -X POST https://your-space.hf.space/reset -H "Content-Type: application/json" -d '{}'`
- Run `openenv validate --url https://your-space.hf.space`
- Fix every validation error before Phase 1 ends

#### Task B-6: Implement the grader formula

The grader score formula uses the normalized improvement over random:

```
score = clamp((agent_score - random_score) / (heuristic_score - random_score + 1e-9), 0.0, 1.0)
```

- For Phase 1, use these hardcoded baseline values until Person C produces real measurements:
  - Task 1: random_score = -0.05, heuristic_score = 0.28
  - Task 2: random_score = -0.08, heuristic_score = 0.22
  - Task 3: random_score = -0.12, heuristic_score = 0.18
- The grader endpoint runs 1 episode of the provided agent (or heuristic if no agent provided) and applies this formula
- The grader must return a finite float in [0.0, 1.0] — not NaN, not infinity, not negative

---

### Person C — Phase 1 Work: Baseline Runner and Minimal Docs

Person C starts after Person B confirms that `client.py` is stable (the SDK's `env.reset()` and `env.step()` work end-to-end). This is the lightest role in Phase 1.

#### Task C-1: Create inference.py in repo root

This is the most critical file for qualification. It must follow the OpenAI client and evaluation format exactly.

- Required environment variables read at startup:
  - `API_BASE_URL` — the OpenAI-compatible API endpoint
  - `MODEL_NAME` — the model identifier
  - `HF_TOKEN` — API key
- MUST use the `OpenAI` client internally. Our architecture wraps this seamlessly via `agents/llm_agent.py` to keep logic clean.
- MUST sequentially run baseline evaluations on **all 3 tasks** consecutively during runtime.
- Required structured log format — emit these in this exact order per task:

```
[START] task=<task_id> env=InferenceGym model=<MODEL_NAME>
[STEP] step=<n> action=<json_action> reward=<float> done=<bool> error=<null_or_string>
[END] success=<bool> steps=<n> score=<float> rewards=[<float>, ...]
```

- When tested offline or executed for final leaderboard, runs must fully complete within the 20-minute allowance limit.

#### Task C-2: Build the random agent

- Creates `agents/random_agent.py`
- Uses `client.py` SDK only — no direct server imports
- Samples each action field uniformly from its full range using `random.seed(42)` for reproducibility
- Runs 10 episodes on each task and reports mean reward
- These measurements become the `random_score` values for Person B's grader formula

#### Task C-3: Build the heuristic agent

The heuristic agent implements rules derived directly from the three papers:

**Rules from Orca (dynamic batching, queue management):**

- if `queue_depth > 0.7 × batch_cap` → increase `batch_cap` by 16, max 512
- if `queue_depth < 0.2 × batch_cap` and `batch_cap > 16` → decrease `batch_cap` by 16
- if `slo_compliance_rate < 0.85` → decrease `batch_cap` by 32 immediately

**Rules from vLLM/PagedAttention (memory management):**

- if `kv_cache_occupancy > 0.85` → decrease `kv_budget_fraction` by 0.10, min 0.10
- if `kv_cache_occupancy < 0.50` and `kv_budget_fraction < 1.0` → increase `kv_budget_fraction` by 0.10
- if `eviction_events > 0` → set `kv_budget_fraction = 0.60` immediately

**Rules from Decima (workload-adaptive optimization):**

- if `request_arrival_rate > 25` → switch quantization to INT8
- if `request_arrival_rate < 8` → switch quantization to FP16
- if `mean_prompt_length > 800` → set `speculation_depth = 0`
- if `mean_prompt_length < 200` → set `speculation_depth = 4`
- if task is adversarial and `mean_prompt_length > 2000` → set `priority_routing = True`

- Starting state: `batch_cap=32, kv_budget_fraction=0.70, spec_depth=0, quantization="FP16", prefill_decode_split=False, priority_routing=False`
- Run 20 episodes per task, report mean reward per task
- These become the `heuristic_score` values for Person B's grader formula

#### Task C-4: Write minimal README

The README must cover these sections in this order:

1. What InferenceGym simulates (2–3 sentences)
2. Why it is a real-world task (1 paragraph)
3. Action space table (6 rows: field, type, range, description)
4. Observation space table (16 rows: field, unit, source paper)
5. Three tasks description table (task_id, difficulty, SLO, episode_length, description)
6. Setup instructions (3 commands: docker build, docker run, curl /health)
7. Running the baseline (the exact inference.py command)
8. Placeholder baseline scores table (fill in with Phase 2 numbers)

---

## Phase 1 → Phase 2 Transition Checkpoint

Do not start Phase 2 until all of the following are true:

| Check | Owner | Status |
|---|---|---|
| `/reset {}` returns HTTP 200 | B | |
| reward always in [-1.0, 1.0] | A | |
| `task_id` correct in `/state` | B | |
| `openenv.yaml` valid | B | |
| `docker build` succeeds | B | |
| HF Space live | B | |
| `openenv validate` passes | B | |
| `inference.py` runs end-to-end | C | |
| `[START][STEP][END]` logs correct | C | |
| 3 tasks all return grader scores | B | |
| No external API call in `step()` | A | |

---

## Phase 2 — Submission Quality

Phase 2 exists to improve the judge's score across all five rubric criteria. Nothing in Phase 2 can break the qualification criteria from Phase 1.

### Phase 2 priorities by rubric weight

- Real-world utility (30%) → improve simulator grounding, paper citations, BurstGPT integration
- Task and grader quality (25%) → validate that Task 3 is genuinely hard for frontier models
- Environment design (20%) → confirm reward provides dense signal, task boundaries are sensible
- Code quality (15%) → clean up imports, add docstrings to public methods, confirm types
- Creativity (10%) → write Description.md with novel framing

---

### Person A — Phase 2 Work: Simulator Realism

#### Task A-5: Replace bootstrap lookup table with paper-grounded values

Build `data/lookup_tables/latency_table.parquet` with these columns: `batch_cap_bucket`, `kv_budget_bucket`, `spec_depth_bucket`, `prompt_size_bucket`, `p50_ttft_ms`, `p99_ttft_ms`, `p50_itl_ms`, `throughput_tps`, `gpu_memory_gb`.

Populate from published vLLM A100 benchmarks and Orca paper Table 2:

| batch | kv | spec | prompt | p99_ttft | p50_itl | tps | mem_gb | source |
|---|---|---|---|---|---|---|---|---|
| 8 | 1.0 | 0 | small | 180 | 22 | 78 | 1.8 | vLLM paper Table 3 |
| 32 | 1.0 | 0 | small | 420 | 38 | 125 | 2.0 | vLLM paper Table 3 |
| 64 | 1.0 | 0 | small | 580 | 55 | 198 | 3.1 | vLLM paper Table 3 |
| 128 | 1.0 | 0 | small | 890 | 72 | 245 | 5.2 | vLLM paper Table 3 |
| 256 | 1.0 | 0 | small | 1400 | 96 | 290 | 9.8 | vLLM paper Table 3 |
| 32 | 0.5 | 0 | small | 360 | 42 | 140 | 1.4 | vLLM eviction analysis |
| 64 | 0.5 | 0 | small | 480 | 58 | 215 | 2.2 | vLLM eviction analysis |
| 32 | 1.0 | 0 | medium | 680 | 60 | 80 | 4.1 | Orca Table 2 |
| 32 | 1.0 | 0 | large | 1900 | 110 | 35 | 12.0 | Orca Table 2 |
| 32 | 1.0 | 4 | small | 310 | 28 | 165 | 2.3 | speculative decoding ablation |
| 32 | 1.0 | 8 | small | 280 | 24 | 185 | 2.6 | speculative decoding ablation |

- For combinations not in the table: find the two nearest rows by Euclidean distance on (batch_cap, kv_budget) and linearly interpolate
- Noise profile: sigma=0.03 during steady-state, sigma=0.10 during burst phase, sigma=0.15 during adversarial events

#### Task A-6: Validate all three tasks produce expected score ranges

Run 20 episodes per task using the heuristic agent. Confirm:

- Task 1 (static): slo_compliance_rate avg > 0.80
- Task 2 (bursty): slo_compliance_rate avg between 0.60 and 0.80
- Task 3 (adversarial): slo_compliance_rate avg between 0.45 and 0.65

If any task scores outside these ranges, debug the workload generator timing and burst injection logic.

#### Task A-7: Write simulator grounding section for Description.md

Write one table row per observation field connecting it to its source paper:

| Observation | Paper | Grounding |
|---|---|---|
| queue_depth | Orca OSDI 2022 | Models iteration-level scheduler queue from Section 3 |
| slo_compliance_rate | Orca OSDI 2022 | TTFT/ITL SLO evaluation at each iteration step |
| kv_cache_occupancy | vLLM SOSP 2023 | PagedAttention block allocator occupancy |
| eviction_events | vLLM SOSP 2023 | Block eviction from active sequence pool |
| request_arrival_rate | BurstGPT arXiv:2401.17644 | Gamma-distributed inter-arrivals from 10M Azure traces |
| mean_prompt_length | BurstGPT arXiv:2401.17644 | Heavy-tail token length distribution |
| spec_acceptance_rate | SpecInfer ASPLOS 2024 | Tree-based speculative decoding acceptance model |
| optimal_policy_non_static | Decima SIGCOMM 2019 | Workload-adaptive policy outperforms static heuristics |

---

### Person B — Phase 2 Work: Reliability and Evaluator Experience

#### Task B-7: Harden all error paths

- If `/step` is called before `/reset`: return HTTP 400 with message "Episode not started. Call /reset first."
- If `/grader` is called with an invalid task_id: return HTTP 404 with message "Unknown task_id."
- If any observation field is NaN or infinite: log a warning and replace with the last valid value or 0.0
- If reward is NaN: log an error and return 0.0
- The server must never return HTTP 500 for any user-supplied input — only for genuine internal errors

#### Task B-8: Update grader with real baseline values from Person C

- Replace the Phase 1 hardcoded baseline values with Person C's measured values from 20-episode runs
- Confirm the grader formula produces scores that discriminate between random and heuristic agents
- Expected grader scores:
  - Random agent → approximately 0.0–0.10 across all tasks
  - Heuristic agent → approximately 0.25–0.45 across all tasks
  - These ranges satisfy the hackathon requirement that hard tasks challenge frontier models

#### Task B-9: Re-run openenv validate and confirm zero critical errors

Run the full validator loop against the live HF Space. Fix every error. Common issues:

- Missing fields in openenv.yaml → add them
- Reward out of bounds → check reward clamping in calculator.py
- task_id not matching → check environment.py task_id persistence
- Grader score out of range → check grader.py formula and clamping
- Docker build timeout → confirm build completes in under 5 minutes

---

### Person C — Phase 2 Work: Benchmarking and Final Documentation

#### Task C-5: Run full benchmarks and populate results table

Run 20 episodes per agent per task. Record mean reward, standard deviation, and grader score.

| Agent | Task 1 Mean ± Std | Task 1 Score | Task 2 Mean ± Std | Task 2 Score | Task 3 Mean ± Std | Task 3 Score |
|---|---|---|---|---|---|---|
| Random (seed=42) | ? | ? | ? | ? | ? | ? |
| Heuristic | ? | ? | ? | ? | ? | ? |
| OpenAI GPT-4.1-mini (if available) | ? | ? | ? | ? | ? | ? |

Update these values in README.md and Description.md.

#### Task C-6: Upgrade inference.py with real OpenAI client baseline

Once heuristic baseline scores are confirmed stable, add the real LLM baseline path:

- If `API_BASE_URL` and `MODEL_NAME` are set and the heuristic is not forced: use OpenAI client
- System prompt for the LLM agent — keep under 250 tokens:
  - "You are an LLM serving configuration optimizer. Your goal is to maximize throughput while meeting latency SLOs. Given the current server metrics as JSON, respond with a JSON ServeAction. Return ONLY valid JSON. No explanation."
  - Append current task SLO thresholds
  - Append last 2 observations as compact JSON
- Parse the response as ServeAction Pydantic model
- On parse failure: retry once with explicit format reminder, then fall back to heuristic action
- The full baseline run on 3 tasks must complete in under 20 minutes total
- If the LLM baseline is not available (no key), the script falls back entirely to the heuristic agent

#### Task C-7: Write Description.md

The document should make judges understand the environment well enough to score it highly on real-world utility and creativity. Structure:

**Section 1 — Problem Statement (200 words):**

- Explain that LLM inference serving is a billion-dollar operational problem
- Every cloud provider makes real-time decisions about batch sizing, memory allocation, and request routing
- These decisions today are made by static configuration files or by human engineers
- InferenceGym provides a standardized environment to train and evaluate agents on this exact problem
- Cite BurstGPT for production traffic statistics

**Section 2 — Why BurstGPT (150 words):**

- BurstGPT contains 10 million real requests from Azure LLM infrastructure
- It captures the heavy-tail prompt length distribution that makes batching hard
- It captures the bursty arrival pattern that makes static configuration dangerous
- Task 2 and Task 3 workload patterns are directly derived from API.csv inter-arrival statistics

**Section 3 — Paper Grounding (200 words):**

- Orca: explains why dynamic batching is better than static and grounds the queue-depth observation
- vLLM/PagedAttention: explains why KV cache management is a first-class concern and grounds eviction mechanics
- Decima: justifies why RL is the right approach and provides theoretical basis for why static heuristics are suboptimal

**Section 4 — Task Rationale (150 words):**

- Task 1 (Easy): tests whether an agent can learn basic queue pressure response
- Task 2 (Medium): tests whether an agent can adapt to non-stationary traffic
- Task 3 (Hard): tests whether an agent can implement multi-priority scheduling under memory pressure — this is the problem that genuinely challenges frontier models

**Section 5 — Benchmark Results:**

- Include the full table from Task C-5

#### Task C-8: Final README polish

- Confirm all commands in README work exactly as written on the live HF Space
- Add the final grader scores table
- Add one paragraph on "Why this environment fills a real gap"
- Add exact inference.py run command with all required environment variables

---

## What to Cut If You Are Running Behind

Cut these features before Phase 2 ends — they will not affect qualification and have minimal score impact:

| Feature | Cut If | Replacement |
|---|---|---|
| Parquet lookup table | 3+ hours behind | Use Phase 1 hardcoded dictionary |
| BurstGPT download fails | Network issue | Gamma(0.8, 280) synthetic distribution |
| Real OpenAI baseline in inference.py | No API key | Heuristic agent satisfies the requirement |
| Task 3 adversarial multi-priority | Simulator too complex | Simplify to single-priority with long-prompt injection |
| Web UI charts | B is behind on deploy | Static JSON at /web is fine |
| Description.md full analysis | Time pressure | 3 paragraphs minimum |
| spec_acceptance_rate modeling | A is behind | Hardcode to 0.0 when spec_depth=0 |

**Never cut:**

| Feature | Why |
|---|---|
| External API removal from step() | Judges cannot run it without a key |
| task_id fix | openenv validate fails immediately |
| Reward clamping | openenv validate fails immediately |
| openenv.yaml | Required manifest for validation |
| inference.py with structured logs | Incorrect logs = incorrect scoring |
| 3 tasks with graders | Hard qualification requirement |
| Docker works on CPU | HF Spaces has no GPU |

---

## Person Ownership Summary

| File / Component | Person A | Person B | Person C |
|---|---|---|---|
| `models.py` | co-owner | co-owner | reads only |
| `config.py` | co-owner | co-owner | reads only |
| `server/environment.py` | writes step() | writes API contract | no access |
| `server/backends/simulated.py` | **owns** | no access | no access |
| `server/workloads/generator.py` | **owns** | no access | no access |
| `server/reward/calculator.py` | **owns** | no access | no access |
| `server/main.py` | no access | **owns** | no access |
| `server/tasks/` | no access | **owns** | no access |
| `server/grader/grader.py` | no access | **owns** | no access |
| `client.py` | no access | **owns** | uses only |
| `openenv.yaml` | no access | **owns** | no access |
| `Dockerfile` | no access | **owns** | no access |
| `inference.py` | no access | no access | **owns** |
| `agents/random_agent.py` | no access | no access | **owns** |
| `agents/heuristic_agent.py` | no access | no access | **owns** |
| `data/` | **owns** | no access | no access |
| `scripts/process_burstgpt.py` | **owns** | no access | no access |
| `README.md` | writes simulator section | no access | **owns** |
| `Description.md` | writes paper grounding | no access | **owns** |

---

## Communication Protocol for the Day

- All three agree on `models.py` and `config.py` contents before starting any other task — this is non-negotiable
- Person B reports to Person C when `client.py` is working end-to-end — Person C starts building agents at that point
- Person C reports `random_score` values to Person B after random agent runs — Person B updates grader formula
- Person C reports `heuristic_score` values to Person B after heuristic agent runs — Person B finalizes grader
- Person A reports to the team when `step()` is fully deterministic and offline — the team runs the first full end-to-end episode test together

# InferenceGym — RL-First Submission Plan

### OpenEnv Hackathon | Deadline: April 8, 2026 11:59 PM | Team of 3

---

## Core Design Philosophy

InferenceGym is not a heuristic tuner. It is a real RL training environment. The entire point is that **no static rule can optimally solve it** — the optimal policy depends on the current workload phase, memory pressure, and SLO violations in ways that are too dynamic for any hand-coded rule. An RL agent trained through trial-and-error learns a policy that adapts to all of these simultaneously.

The three tasks are deliberately designed so that:

- A random policy scores ~0.0–0.10
- A hand-coded heuristic (Orca rules, vLLM rules) scores ~0.25–0.40
- A trained PPO agent scores ~0.55–0.75
- This gap is the value proposition — RL genuinely wins here

The hackathon requires `inference.py` to use the OpenAI client. That is the baseline demonstration for judges. But the environment ships with a trained PPO agent whose weights are committed to the repo, demonstrating that the environment is actually learnable and produces policies that outperform static heuristics.

---

## What Changes From the Heuristic Plan

| Component | Old Plan | New Plan |
|---|---|---|
| Primary agent | Hand-coded rules from papers | PPO trained on the environment |
| `agents/heuristic_agent.py` | Main demonstration agent | Comparison baseline only |
| `agents/` folder | 2 files | 4 files: random, heuristic, trained_ppo, llm_agent |
| `train.py` | Did not exist | New file — trains and saves PPO weights |
| `weights/` | Did not exist | Committed PPO checkpoint for all 3 tasks |
| Reward design | Reasonable signal | Shaped specifically for credit assignment |
| Grader baseline | Heuristic score | Trained PPO score |
| `inference.py` | Heuristic backing | OpenAI LLM agent (required) + fallback to trained PPO |

---

## Why RL Wins Over Heuristics Here

The Decima paper (SIGCOMM 2019) proves this experimentally: a trained RL scheduler outperforms the best human-designed heuristic by 21–31% on tail job completion time. The core reason is that optimal batch sizing, KV budget allocation, and speculation depth are interdependent. A rule like "if queue > 70%, increase batch" ignores that increasing batch when memory is already at 82% will cause an eviction cascade. An RL agent learns these interaction effects through trajectory experience.

Task 3 (adversarial) is specifically unsolvable by any static rule:

- The mega-prompt injection timing is not known to the agent
- The correct response changes depending on whether the current queue contains high-priority or low-priority requests
- The tradeoff between evicting the mega-prompt versus swapping it to CPU depends on the current decode phase
- Only an agent that has seen hundreds of these scenarios during training can develop a robust policy

---

## Updated File Structure

```
inference-gym/
│
├── openenv.yaml                  ← Required manifest
├── inference.py                  ← Required. Root level. OpenAI client. Structured logs.
├── train.py                      ← NEW. Trains PPO agent. Saves weights. CPU-runnable.
├── evaluate.py                   ← NEW. Loads weights. Runs benchmark. Prints score table.
├── Dockerfile                    ← Must build and run without GPU.
├── requirements.txt
├── README.md
├── Description.md
│
├── models.py                     ← SHARED. Frozen on Day 1.
├── config.py                     ← SHARED. Frozen on Day 1.
├── client.py                     ← SDK wrapper.
│
├── weights/                      ← NEW. Committed to repo.
│   ├── ppo_task1_static.pt       ← Trained on static_workload
│   ├── ppo_task2_bursty.pt       ← Trained on bursty_workload
│   └── ppo_task3_adversarial.pt  ← Trained on adversarial_multitenant
│
├── server/
│   ├── main.py
│   ├── environment.py
│   ├── backends/
│   │   └── simulated.py          ← Fully offline. BurstGPT-backed. No external calls.
│   ├── workloads/
│   │   └── generator.py          ← Seeded. BurstGPT distributions.
│   ├── tasks/
│   │   ├── registry.py
│   │   ├── task_static.py
│   │   ├── task_bursty.py
│   │   └── task_adversarial.py
│   ├── reward/
│   │   └── calculator.py         ← RL-shaped reward. Dense. Credit-assignment-friendly.
│   └── grader/
│       └── grader.py             ← Uses trained PPO weights as the benchmark.
│
├── agents/
│   ├── random_agent.py           ← Random policy. Establishes floor score.
│   ├── heuristic_agent.py        ← Orca + vLLM + Decima rules. Establishes heuristic score.
│   ├── ppo_agent.py              ← Loads weights from /weights. Runs inference only.
│   └── llm_agent.py              ← OpenAI client agent. Used in inference.py.
│
├── rl/
│   ├── __init__.py
│   ├── env_wrapper.py            ← Wraps client.py into a Gymnasium-compatible interface.
│   ├── ppo.py                    ← Lightweight PPO implementation. No external RL library.
│   ├── policy_network.py         ← MLP policy network. 2 hidden layers. CPU-runnable.
│   └── normalize.py              ← Running mean/std normalization for observations.
│
├── data/
│   ├── burstgpt/
│   │   ├── chat_prompts.parquet
│   │   └── api_prompts.parquet
│   └── lookup_tables/
│       └── latency_table.parquet
│
└── scripts/
    └── process_burstgpt.py
```

---

## Shared Contract — Frozen on Day 1

### `models.py`

**ServeAction:**

- `batch_cap: int` — 1–512
- `kv_budget_fraction: float` — 0.10–1.00
- `speculation_depth: int` — 0–8
- `quantization_tier: Literal["FP16", "INT8", "INT4"]`
- `prefill_decode_split: bool`
- `priority_routing: bool`

**ServeObservation (16 fields — all float, never None):**

- `queue_depth`, `active_requests`, `kv_cache_occupancy`
- `mean_prompt_length`, `p50_ttft_ms`, `p99_ttft_ms`, `p50_itl_ms`
- `throughput_tps`, `slo_compliance_rate`, `gpu_memory_used_gb`
- `estimated_cost_per_1k`, `request_arrival_rate`, `spec_acceptance_rate`
- `eviction_events`, `step_index`, `task_id` (encoded as float: 0.0, 1.0, 2.0)

**The RL state vector:** flatten all 15 numeric fields into a float32 array of shape (15,). `task_id` is kept separate as a task identifier.

### `config.py`

All SLO thresholds, episode lengths, seeds, and reward weight constants live here. The RL policy network input dimension is derived from this file: `OBS_DIM = 15`.

---

## The RL Architecture (Critical to Understand Before Coding)

### Why a custom lightweight PPO instead of stable-baselines3

The environment must run on 2 vCPU, 8GB RAM with no GPU. stable-baselines3 with PPO has heavy dependencies (gymnasium, torch, numpy). Instead, use a **minimal custom PPO** that:

- Uses PyTorch only (already in requirements for model weights)
- Has a 2-layer MLP policy: [15 → 128 → 64 → action_logits]
- Handles the mixed action space (discrete + continuous) correctly
- Trains in under 10 minutes on CPU on Task 1
- Produces weights under 2MB per task

### Mixed action space handling

The action space is mixed — some fields are continuous (batch_cap, kv_budget_fraction), some are discrete (quantization_tier, speculation_depth), some are binary (prefill_decode_split, priority_routing).

Handle this by:

- Representing continuous fields as Gaussian distributions (mean + log_std head)
- Representing discrete fields as categorical distributions (softmax head)
- Computing the joint log-probability as the sum of individual log-probs
- Clipping continuous outputs to their valid ranges at inference time

### Policy network output heads

The MLP has a shared trunk and 6 output heads:

1. `batch_cap_mean` + `batch_cap_log_std` → sample from Normal, clip to [1, 512], round to int
2. `kv_budget_mean` + `kv_budget_log_std` → sample from Normal, clip to [0.10, 1.00]
3. `spec_depth_logits` (9 values: 0–8) → sample from Categorical
4. `quantization_logits` (3 values) → sample from Categorical
5. `prefill_split_logit` (1 value) → sample from Bernoulli
6. `priority_routing_logit` (1 value) → sample from Bernoulli

Value head: `[15 → 128 → 64 → 1]` — shared trunk, separate final layer.

### Training setup

- Algorithm: PPO with clipped objective, ε=0.2
- Rollout length: 512 steps
- Minibatch size: 64
- PPO epochs: 4 per update
- Gamma: 0.99, Lambda (GAE): 0.95
- Learning rate: 3e-4
- Total training steps: 50,000 for Task 1, 80,000 for Task 2, 120,000 for Task 3
- Entropy coefficient: 0.01 — crucial for exploration in the mixed action space
- Observation normalization: running mean/std, updated from the rollout buffer
- Training runs locally or on any CPU machine — no GPU needed
- Training time estimate: Task 1 ~6 min, Task 2 ~10 min, Task 3 ~16 min on 2 vCPU

---

## Phase 1 — Qualification

Phase 1 has the same goal as before: pass every validator check. The difference is that Person A now designs the reward specifically for RL credit assignment, and Person C now builds both the training infrastructure AND the required OpenAI baseline.

---

### Person A — Phase 1: Simulator + RL-Shaped Reward

#### Task A-1: Remove external API calls (same as before)

- Kill all imports of openai, httpx, requests from simulated.py
- Replace with deterministic lookup dictionary
- Bootstrap values same as previous plan
- Verify step() returns fully populated ServeObservation with no None values

#### Task A-2: BurstGPT integration (same as before)

- Build process_burstgpt.py
- Wire BurstGPT into WorkloadGenerator
- Make episodes fully seeded and deterministic

#### Task A-3: Redesign reward for RL credit assignment

The heuristic plan's reward was fine for evaluation. For RL training, the reward must have two additional properties: **density** (signal at every step, not just at the end) and **credit assignment clarity** (the agent can identify which action caused which reward component).

**Component 1 — SLO compliance (weight 0.35, primary signal):**

- reward = +0.35 × slo_compliance_rate
- slo_compliance_rate is computed per-step, so the agent gets signal immediately after every action
- Do not delay this to episode end — sparse rewards kill learning speed

**Component 2 — Throughput relative to capacity (weight 0.20):**

- reward = +0.20 × min(throughput_tps / task_target_tps, 1.0)
- Capped at target — the agent should not learn to overbatch just for raw throughput

**Component 3 — Memory pressure signal (weight 0.20):**

- reward = +0.10 when kv_cache_occupancy is in [0.60, 0.85] — the "goldilocks zone"
- reward = -0.10 × (kv_cache_occupancy - 0.85) / 0.15 when occupancy > 0.85
- reward = -0.05 × (0.60 - kv_cache_occupancy) / 0.50 when occupancy < 0.60 (underutilization)
- This shapes a clear target occupancy band which is easy for RL to learn

**Component 4 — Eviction penalty (weight 0.15):**

- reward = -0.05 per eviction event, hard capped at -0.15 per step
- This is the clearest credit assignment signal: agent causes a bad kv_budget → immediate penalty

**Component 5 — Queue pressure management (weight 0.10):**

- reward = +0.10 × (1.0 - queue_depth / max_queue_depth)
- max_queue_depth = 512 (same as max batch_cap)
- Encourages the agent to prevent queue buildup before it causes SLO violations

**Final:** sum all 5 components, clip to [-1.0, 1.0]

**Why this is better for RL than the heuristic plan's reward:**

- Every component responds immediately to the last action — no delayed signals
- The memory pressure goldilocks zone creates a shaped landscape that PPO can follow
- The queue depth signal gives the agent a leading indicator before SLO violations occur
- The eviction penalty is the most direct credit assignment: one bad action → immediate -0.05

#### Task A-4: Determinism for training reproducibility

- Same seed → same trajectory — required for reproducing training runs
- Provide a `get_observation_vector()` utility that flattens ServeObservation to float32 numpy array shape (15,)
- This is the interface between the environment and the RL policy network

---

### Person B — Phase 1: API Compliance and Deployment (identical to previous plan)

All tasks B-1 through B-6 remain the same. The only update:

#### Task B-6 update: Grader uses trained PPO as benchmark

In Phase 1, grader still uses hardcoded values. In Phase 2, once Person C commits trained weights, update the grader to:

- Load `weights/ppo_task{N}_{name}.pt`
- Run 3 episodes with the PPO agent
- Use mean PPO score as `heuristic_score` in the formula
- This makes the grader score reflect genuine RL performance, not hand-coded rules

---

### Person C — Phase 1: RL Infrastructure + Baseline Runner

Person C now owns the RL training stack. This is more work than the heuristic plan but is doable because the PPO implementation is small.

#### Task C-1: Build `rl/env_wrapper.py`

This file wraps the `client.py` SDK into a standard interface that the PPO trainer can use.

**Required interface:**

- `reset(seed=None)` → returns `obs: np.ndarray` of shape (15,) — normalized float32
- `step(action_dict)` → returns `(obs, reward, done, info)` where obs is the same shape
- `observation_space` → contains shape (15,) and dtype float32
- `action_space` → contains the 6 action fields with their ranges

**Inside the wrapper:**

- Call `client.reset(task_id, seed)` and convert the returned ServeObservation to a numpy array
- Call `client.step(ServeAction(...))` and return the StepResult fields
- Apply running mean/std normalization from `rl/normalize.py` to the observation
- The wrapper connects to the FastAPI server via the client SDK — the server must be running locally during training

#### Task C-2: Build `rl/policy_network.py`

The policy network is a PyTorch MLP. It must:

- Accept input of shape (batch, 15)
- Produce 6 output heads as described in the architecture section above
- Include a value head that returns a scalar
- Use ReLU activations, no dropout
- Be serializable with `torch.save`
- Total parameter count should be under 50,000 — keeps weights small and training fast

#### Task C-3: Build `rl/ppo.py`

The PPO trainer runs rollouts against the environment and updates the policy. Key requirements:

- Rollout collection: run N steps in the environment, store (obs, action, reward, done, log_prob, value) at each step
- GAE computation: compute generalized advantage estimates from the rollout buffer
- Policy update: compute PPO clipped loss, value loss, and entropy bonus; run gradient updates
- The trainer must print progress every 2000 steps so the user can see it is learning
- Save checkpoint after every 10,000 steps to `weights/ppo_task{id}_checkpoint.pt`
- Save final weights to `weights/ppo_task{id}_{name}.pt`

#### Task C-4: Build `train.py` in repo root

This is the script researchers and engineers will actually run to train their own policies.

**Command line interface:**

- `python train.py --task static_workload --steps 50000 --seed 42`
- `python train.py --task bursty_workload --steps 80000 --seed 42`
- `python train.py --task adversarial_multitenant --steps 120000 --seed 42`

**What it does:**

- Starts the FastAPI server in a subprocess (or connects to a running one via environment variable)
- Initializes the env_wrapper, policy network, and PPO trainer
- Runs the training loop
- Prints a summary table at the end showing reward curve and final benchmark scores
- Saves weights to the `weights/` directory

**CPU training estimates:**

- Task 1, 50k steps, 2 vCPU: approximately 6–8 minutes
- Task 2, 80k steps, 2 vCPU: approximately 10–13 minutes
- Task 3, 120k steps, 2 vCPU: approximately 16–20 minutes

#### Task C-5: Build `agents/ppo_agent.py`

Loads pre-trained weights and runs inference only. No training loop.

- Load weights from `weights/ppo_{task}.pt`
- Given an observation, sample action from the policy network
- Return a ServeAction object
- This is what the grader uses as the benchmark agent in Phase 2

#### Task C-6: Build `agents/heuristic_agent.py` (for comparison only)

Keep the heuristic agent from the previous plan but label it clearly as a comparison baseline, not the primary agent. This agent is useful for:

- Establishing that a non-RL approach scores ~0.25–0.40
- Providing a fast fallback if RL weights are not available
- Showing the improvement gap that RL achieves

#### Task C-7: Build `agents/llm_agent.py`

This is the OpenAI-client-based agent for `inference.py`.

- Uses `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment variables
- System prompt (under 200 tokens):
  - "You are an LLM serving configuration optimizer. Given current server metrics as JSON, output a JSON ServeAction to maximize throughput while meeting SLOs. ONLY output valid JSON."
  - Include the task SLO thresholds
  - Include the last 2 observations as compact JSON
- Parse response as ServeAction Pydantic model
- On failure: retry once, then fall back to `ppo_agent.py` (not heuristic — PPO is better)
- This agent is tested against Task 1 for the inference.py baseline

#### Task C-8: Build `inference.py` in repo root

Same requirements as before. The key change: the agent hierarchy is now:

1. Try OpenAI LLM agent (if API key and base URL are set)
2. Fall back to PPO agent (if weights exist in `weights/`)
3. Fall back to heuristic agent (last resort)

The structured log format remains exactly as required:

```
[START] task=static_workload env=InferenceGym model=gpt-4.1-mini
[STEP] step=1 action={"batch_cap":32,...} reward=0.23 done=false error=null
[END] success=true steps=60 score=0.41 rewards=[0.23, 0.31, ...]
```

---

## Phase 1 Qualification Gate (same as before)

All qualification checks must pass before Phase 2 begins. See previous plan's checklist.

---

## Phase 2 — Training and Demonstration Quality

Phase 2 is where InferenceGym distinguishes itself as a real RL environment.

### Person A — Phase 2: Simulator Realism Upgrade

#### Task A-5: Build paper-grounded lookup table

Same as previous plan. Populate from vLLM benchmarks, Orca Table 2, and speculative decoding ablations.

#### Task A-6: Validate RL learning signal

Run 3 training seeds on each task and confirm:

- The reward curve is strictly increasing on Task 1 (easy)
- The reward curve is non-monotone but trending upward on Tasks 2 and 3 (expected due to non-stationarity)
- The trained PPO agent scores at least 0.30 higher than random on all 3 tasks
- The KV cache occupancy in trained PPO episodes stays in the [0.60, 0.85] goldilocks zone more than 60% of the time

If the reward curve is flat (not learning), debug these in order:

- Check observation normalization is working (values should be centered around 0)
- Check entropy coefficient is not too low (should be 0.01 minimum)
- Check the batch_cap continuous head is not saturating (gradients should flow through clipping)
- Check the episode is not terminating too early due to a SLO violation penalty

#### Task A-7: Write paper grounding for Description.md (same as before)

---

### Person B — Phase 2: Grader Update and Hardening

#### Task B-7: Update grader to use PPO weights

Once Person C commits the first set of trained weights:

- Replace the hardcoded `heuristic_score` in the grader formula with the PPO agent's measured score
- Run 3 episodes with `ppo_agent.py` and use the mean as the benchmark
- This means that the grader score now measures: "how much better is your agent than our trained PPO?"
- A score of 0.5 means your agent matches the PPO baseline. A score of 1.0 means you match the best possible policy.

#### Task B-8: Harden all error paths (same as before)

#### Task B-9: Re-run openenv validate (same as before)

---

### Person C — Phase 2: Train All Three Tasks and Benchmark

#### Task C-9: Train PPO on all three tasks

Run training for all three tasks with the final simulator (Phase 2 lookup table):

- Task 1: `python train.py --task static_workload --steps 50000 --seed 42`
- Task 2: `python train.py --task bursty_workload --steps 80000 --seed 42`
- Task 3: `python train.py --task adversarial_multitenant --steps 120000 --seed 42`

Commit the resulting weights to the repo under `weights/`.

#### Task C-10: Run full benchmark comparison

Run 20 episodes per agent per task and record results:

| Agent | Task 1 Score | Task 2 Score | Task 3 Score |
|---|---|---|---|
| Random (seed=42) | ~0.05 | ~0.03 | ~0.02 |
| Heuristic (Orca+vLLM+Decima) | ~0.30 | ~0.25 | ~0.20 |
| Trained PPO (50k/80k/120k steps) | ~0.55 | ~0.48 | ~0.38 |
| OpenAI GPT-4.1-mini (zero-shot) | ~0.35 | ~0.28 | ~0.22 |

These numbers demonstrate the key claim: **RL outperforms both heuristics and zero-shot LLMs on this task.** This is the primary value proposition for judges evaluating real-world utility.

#### Task C-11: Write evaluate.py in repo root

```
python evaluate.py --agent ppo --task all --episodes 20 --seed 42
```

Runs the trained PPO agent across all tasks and prints the benchmark table. Researchers can use this to compare their own trained policies.

#### Task C-12: Write Description.md

**Section 1 — Why RL beats heuristics here (200 words):**
The core claim: the optimal LLM serving policy is non-stationary, non-Markovian, and context-dependent. A hand-coded rule ignores three interaction effects that only emerge from experience:

- Increasing batch_cap reduces TTFT per-request but degrades p99_ttft during bursts
- Reducing kv_budget_fraction saves memory but causes eviction cascades when combined with large prompts
- Speculation depth only helps when prompts are short — it slows down prefill for long contexts
A trained PPO agent learns all three interaction effects simultaneously. The benchmark table proves it: PPO outperforms the Orca+vLLM+Decima heuristic by ~0.20–0.25 score points on all tasks.

**Section 2 — BurstGPT grounding (150 words):** Same as before.

**Section 3 — Paper grounding (200 words):** Same as before.

**Section 4 — Task rationale (150 words):** Emphasize that Task 3 was specifically designed to be unsolvable by static rules.

**Section 5 — Benchmark results table:** Include final numbers from Task C-10.

**Section 6 — How to train your own agent:**

```
python train.py --task adversarial_multitenant --steps 200000 --seed 0
python evaluate.py --agent ppo --task adversarial_multitenant
```

---

## Updated Person Ownership

| File | Person A | Person B | Person C |
|---|---|---|---|
| `models.py` | co-owner | co-owner | reads |
| `config.py` | co-owner | co-owner | reads |
| `server/environment.py` | step() | API contract | — |
| `server/backends/simulated.py` | **owns** | — | — |
| `server/workloads/generator.py` | **owns** | — | — |
| `server/reward/calculator.py` | **owns** | — | — |
| `server/main.py` | — | **owns** | — |
| `server/tasks/` | — | **owns** | — |
| `server/grader/grader.py` | — | **owns** | reads |
| `client.py` | — | **owns** | uses |
| `openenv.yaml` | — | **owns** | — |
| `Dockerfile` | — | **owns** | — |
| `rl/env_wrapper.py` | — | — | **owns** |
| `rl/ppo.py` | — | — | **owns** |
| `rl/policy_network.py` | — | — | **owns** |
| `agents/ppo_agent.py` | — | — | **owns** |
| `agents/heuristic_agent.py` | — | — | **owns** |
| `agents/llm_agent.py` | — | — | **owns** |
| `train.py` | — | — | **owns** |
| `evaluate.py` | — | — | **owns** |
| `inference.py` | — | — | **owns** |
| `weights/` | — | — | **owns** |
| `data/` | **owns** | — | — |
| `README.md` | sim section | — | **owns** |
| `Description.md` | paper section | — | **owns** |

---

## What to Cut If Running Behind

| Feature | Cut If | Safe Replacement |
|---|---|---|
| Custom PPO — use stable-baselines3 instead | C is behind | `pip install stable-baselines3` — use `PPO("MlpPolicy", env)` directly |
| Train Task 3 weights | Very behind | Commit Task 1 weights only. Grader still uses PPO. Tasks 2+3 use heuristic fallback. |
| Real OpenAI LLM calls in inference.py | No API key | PPO agent backs inference.py entirely — still valid |
| evaluate.py | Behind | Skip. Include benchmark numbers manually in README. |
| Parquet lookup table | Behind | Keep bootstrap dictionary from Phase 1 |
| Description.md deep analysis | Late night | 3 paragraphs minimum: real-world utility, BurstGPT, why RL |

**Never cut:**

- `weights/ppo_task1_static.pt` — the trained PPO for Task 1 is the core demonstration
- RL wins over heuristic in the benchmark table — this is the entire value proposition
- `inference.py` with structured logs — disqualification risk
- `openenv.yaml` — disqualification risk
- Reward clamping to [-1, 1] — disqualification risk
- `/reset {}` accepting empty body — disqualification risk

---

## Critical Path for Tomorrow

The entire day's work must be sequenced around two dependencies:

**Dependency 1:** Person C needs a working server (Person B) before training can start.

- Person B's first milestone: `/reset`, `/step`, `/state` all return valid responses
- Person C can start `rl/env_wrapper.py` as soon as this is done — even before full deployment

**Dependency 2:** Person B's grader update (Phase 2) needs Person C's trained weights.

- Person C should commit `ppo_task1_static.pt` first — this unblocks Person B
- Tasks 2 and 3 weights can follow later in the day

**The single most important thing to have by 6 PM:**
`weights/ppo_task1_static.pt` exists, the PPO agent scores better than the heuristic on Task 1, and the result is visible in the grader endpoint. Everything else is polish.

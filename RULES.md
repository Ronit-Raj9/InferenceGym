
Join Discord

Help

Log out

Registration

14th March - 3rd April

Declaration

Before R1

Prepare

Now - 25th March

Round 1

25th March - 8th April

Results

10th April

Finale

25th-26th April

Welcome RONIT RAJ!

ronitk964@gmail.com
Copy

Join the Discord Community

All announcements, mentor access, and team matching happens here.


Join Discord
QUICK TOGGLe

Team form Submission

Preparatory Course

Start Assessment

FAQs

step 1

How will you compete?

Choose solo or team before you can start the assessment

Step 1 Complete
Team: AlphaQ

👤
RONIT RAJ
ronitk964@gmail.com
Team Lead
👤
Murtuza Shaikh
murtuzashaikh.2023@gmail.com
Accepted
👤
Khushi Singh
khushisingh82072@gmail.com
Accepted
🔒
Team is permanently locked. Changes are not allowed after confirmation.

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp

OpenEnv Round 1 Bootcamp: Build Your First RL Environment

Live walkthrough to submit a strong Round 1 entry

timing

8:00 PM Onwards

Wednesday, 1st April

Host


Ben Burtenshaw

Community Education in AI at Hugging Face


Pulkit Aneja

Scaler Instructor

Watch Recording

PROBLEM STATEMENT

Round 1 — Problem Statement

The Task

Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard  step() / reset() / state()  API.

Key Requirements at a Glance

Must simulate a real-world task (not games or toys)

Implement full OpenEnv spec: typed models, step()/reset()/state(), openenv.yaml

Minimum 3 tasks with agent graders (easy → medium → hard, scores/reward 0.0–1.0)

Meaningful reward function with partial progress signals

Baseline inference script with reproducible scores

Deploy to Hugging Face Spaces + working Dockerfile

README with environment description, action/observation spaces, setup instructions

Functional Requirements

Real-world task simulation

The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.

OpenEnv spec compliance

Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. step(action) → returns observation, reward, done, info. reset() → returns initial observation. state() → returns current state. openenv.yaml with metadata. Tested via openenv validate.

Minimum 3 tasks with agent graders

Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0). Tasks should range: easy → medium → hard. Graders must have clear, deterministic success/failure criteria.

Meaningful reward function

Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions).

Baseline inference script

Uses the OpenAI API client to run a model against the environment. Reads API credentials from environment variables (OPENAI_API_KEY). Produces a reproducible baseline score on all 3 tasks.

Detailed Requirements

Non-Functional Requirements

Deploys to a Hugging Face Space

Environment must run as a containerized HF Space tagged with openenv.

Containerized execution

Must include a working Dockerfile. The environment should start cleanly with docker build + docker run.

Documentation

README must include: environment description and motivation, action and observation space definitions, task descriptions with expected difficulty, setup and usage instructions, baseline scores.

Parameter

Weight

Description

Real-world utility

30%

Does the environment model a genuine task? Would someone actually use this to train or evaluate agents?

Task & grader quality

25%

Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression?

Environment design

20%

Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries.

Code quality & spec compliance

15%

Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works.

Creativity & novelty

10%

Novel problem domain, interesting mechanics, clever reward design, original approach.

Scoring Breakdown

Real-world utility (30%)

•  0–5: Toy/artificial problem with no practical application

•  6–15: Valid domain but shallow modeling of the real task

•  16–25: Good domain modeling, would be useful for agent evaluation

•  26–30: Excellent — fills a real gap, immediate value for the RL/agent community

Task & grader quality (25%)

•  3+ tasks with difficulty range?

•  Graders produce scores between 0.0–1.0?

•  Graders deterministic and reproducible?

•  Hard task genuinely challenges frontier models?

Environment design (20%)

•  reset() produces clean state?

•  Action/observation types well-designed and documented?

•  Reward function provides useful varying signal (not just sparse)?

•  Episode boundaries sensible?

Code quality & spec compliance (15%)

•  openenv validate passes?

•  docker build && docker run works?

•  HF Space deploys and responds?

•  Baseline script runs and reproduces scores?

Creativity & novelty (10%)

•  Domain we haven’t seen in OpenEnv before?

•  Reward design has interesting properties?

•  Clever mechanics that make the environment engaging?

Evaluation Criteria

Phase 1: Automated Validation

Pass/fail gate — HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.

Phase 2: Agentic Evaluation

Scored — baseline agent re-run, standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.

Phase 3: Human Review

Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.

Disqualification Criteria

Environment does not deploy or respond

Plagiarized or trivially modified existing environments

Graders that always return the same score

No baseline inference script

How Judging works

Pre-Submission Checklist  — all must pass or you're disqualified

HF Space deploys

Automated ping to the Space URL — must return 200 and respond to reset()

OpenEnv spec compliance

Validate openenv.yaml, typed models, step()/reset()/state() endpoints

Dockerfile builds

Automated docker build on the submitted repo

Baseline reproduces

Run the submitted inference script — must complete without error and produce scores

3+ tasks with graders

Enumerate tasks, run each grader, verify scores/reward in 0.0–1.0 range

Mandatory Additional Instructions

Before submitting, ensure the following variables are defined in your environment configuration:

API_BASE_URL   The API endpoint for the LLM.

MODEL_NAME     The model identifier to use for inference.

HF_TOKEN       Your Hugging Face / API key.

The inference script must be named `inference.py` and placed in the root directory of the project

Participants must use OpenAI Client for all LLM calls using above variables

Participants must emit structured stdout logs strictly following the [START], [STEP], and [END] format defined in the sample inference.py provided below. Any deviation in field names, ordering, or formatting will result in incorrect evaluation scoring. Refer to the Sample Inference Script for the complete format specification and examples.

Infra Restrictions

Runtime of inference script should be less than 20min 

Make sure your env and inference can run on a machine with vcpu=2, memory=8gb

Validator

Run the pre-submission validation script before submitting

NEW
Sample Inference Script 

"""
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env
IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with a simple echo environment.
    Each turn you must send a message. The environment will echo it back.
    Reward is proportional to message length: reward = len(message) * 0.1
    Your goal is to maximize total reward by sending meaningful, substantive messages.
    Reply with exactly one message string — no quotes, no prefixes, just the message text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last echoed message: {last_echoed!r}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next message.
        """
    ).strip()


NEW
Pre Validation Script

#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   Or download and run locally:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
Submission window opens on 28th March

Deadline: 8 Apr 11:59 PM


Submit your Assessment
→
Study material

Preparatory Course

4 modules · ~3.5 hours 

Each module: read the README first, then open the notebook in Colab. No local setup needed.

 Module 1: Why OpenEnv?

ESSENTIAL FOR ROUND 1

45 min

Module 2: Using Existing Environments

ESSENTIAL FOR ROUND 1

50 min

 Module 3: Deploying Environments

ESSENTIAL FOR ROUND 1

45 min

Module 4: Building Your Own Environment

 MOST IMPORTANT FOR ROUND 1

60 min

View full course repository

GUIDE

Round 1 Guide

What to Expect

Prerequisites

How to Submit

When Round 1 opens, you'll choose 1 of 4–5 problem statements and build an OpenEnv environment around it.

Example of what a problem statement looks like

"Build a mini-game RL environment with clearly defined tasks, automated graders, and reward logic using the OpenEnv framework."

→ Create a mini-game an AI agent can play

→ Define tasks with increasing difficulty

→ Write graders that verify task completion

→ Define reward logic for scoring

→ Package using OpenEnv for automated evaluation

Evaluation Criteria

Runtime correctness

Runs without errors

Interface compliance

Follows OpenEnv standard

Task design

 Clear, realistic, testable

Grading logic

 Reward system makes sense

Step 2

Submit your Assessment

Complete Step 1 first

Problem Statement is live. Build and submit.

Round 1 begins 

Submission window opens on 28th March

Deadline: 8 Apr 11:59 PM


Submit your Assessment
→
NOTE: Only team leaders can make the final submission.

FAQs

Frequently Asked Questions













Need help? Reach out to us

help_openenvhackathon@scaler.com

Contact Support

submission Deadline: 8th April 11:59 PM


Submit your Assessment
→
How to Submit?


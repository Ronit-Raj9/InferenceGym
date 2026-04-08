from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Protocol

from openai import OpenAI

from llmserve_env.client import LLMServeEnv
from llmserve_env.models import EpisodeLog, QuantizationTier, ServeAction, ServeObservation, default_action
from llmserve_env.task_catalog import get_task_catalog, get_task_config
from server.baseline_agent import HeuristicPolicy
from server.grader import GraderEngine
from server.llmserve_environment import LLMServeEnvironment


DEFAULT_BASE_URL = "http://localhost:7860"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_SEED = 42

SYSTEM_PROMPT = """
You are controlling an LLM serving environment.
Return exactly one JSON object with these keys:
- batch_cap: integer 1..512
- kv_budget_fraction: float 0.1..1.0
- speculation_depth: integer 0..8
- quantization_tier: one of FP16, INT8, INT4
- prefill_decode_split: boolean
- priority_routing: boolean
Do not include markdown or extra text.
""".strip()


class BaselineEnvironment(Protocol):
    def reset(self, task_id: str, seed: int | None = None) -> ServeObservation: ...

    def step(self, action: dict[str, Any] | ServeAction) -> tuple[ServeObservation, float, bool, dict[str, Any]]: ...

    def grade(self, log: EpisodeLog | None = None) -> dict[str, Any]: ...


class LocalBaselineRunner:
    def __init__(self, seed: int = DEFAULT_SEED, mode: str = "sim") -> None:
        self.env = LLMServeEnvironment(seed=seed, mode=mode)
        self.grader = GraderEngine()

    def reset(self, task_id: str, seed: int | None = None) -> ServeObservation:
        return self.env.reset(task_id=task_id, seed=seed)

    def step(self, action: dict[str, Any] | ServeAction) -> tuple[ServeObservation, float, bool, dict[str, Any]]:
        if isinstance(action, dict):
            action = ServeAction.model_validate(action)
        observation = self.env.step(action)
        return observation, float(observation.reward or 0.0), bool(observation.done), dict(observation.metadata)

    def grade(self, log: EpisodeLog | None = None) -> dict[str, Any]:
        episode_log = log or self.env.export_episode_log()
        return self.grader.grade(episode_log)


def create_remote_runner(base_url: str | None = None) -> LLMServeEnv:
    return LLMServeEnv.from_url(base_url or os.getenv("LLMSERVE_BASE_URL", DEFAULT_BASE_URL))


def create_local_runner(seed: int = DEFAULT_SEED, mode: str = "sim") -> LocalBaselineRunner:
    return LocalBaselineRunner(seed=seed, mode=mode)


def run_deterministic_baseline(
    task_id: str,
    seed: int = DEFAULT_SEED,
    runner: BaselineEnvironment | None = None,
) -> dict[str, Any]:
    environment = runner or create_local_runner(seed=seed)
    policy = HeuristicPolicy()
    policy.reset()
    observation = environment.reset(task_id=task_id, seed=seed)
    max_steps = int(get_task_config(task_id)["max_steps"])

    steps = 0
    while not observation.done and steps < max_steps:
        action = policy.act(observation, task_id)
        observation, _, _, _ = environment.step(action)
        steps += 1

    grader_result = environment.grade()
    return {
        "task_id": task_id,
        "seed": seed,
        "steps": steps,
        "grader": grader_result,
    }


def run_openai_baseline(
    task_id: str,
    seed: int = DEFAULT_SEED,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_MODEL,
    runner: BaselineEnvironment | None = None,
) -> dict[str, Any]:
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI baseline inference.")

    client = OpenAI(api_key=resolved_key, max_retries=2, timeout=30.0)
    environment = runner or create_remote_runner(base_url=base_url)
    observation = environment.reset(task_id=task_id, seed=seed)
    max_steps = int(get_task_config(task_id)["max_steps"])

    steps = 0
    while not observation.done and steps < max_steps:
        action = _action_from_model(client, model, task_id, observation)
        observation, _, _, _ = environment.step(action)
        steps += 1

    grader_result = environment.grade()
    return {
        "task_id": task_id,
        "seed": seed,
        "model": model,
        "steps": steps,
        "grader": grader_result,
    }


def run_baseline_suite(
    mode: str = "deterministic",
    task_ids: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    runner_factory: Callable[[], BaselineEnvironment] | None = None,
) -> dict[str, Any]:
    resolved_task_ids = task_ids or [task["id"] for task in get_task_catalog()]
    results: dict[str, Any] = {}

    for task_id in resolved_task_ids:
        runner = runner_factory() if runner_factory is not None else None
        if mode == "openai":
            results[task_id] = run_openai_baseline(
                task_id=task_id,
                seed=seed,
                api_key=api_key,
                base_url=base_url,
                model=model,
                runner=runner,
            )
        elif mode == "deterministic":
            results[task_id] = run_deterministic_baseline(
                task_id=task_id,
                seed=seed,
                runner=runner,
            )
        else:
            raise ValueError(f"Unsupported baseline mode: {mode}")

    payload: dict[str, Any] = {
        "mode": mode,
        "seed": seed,
        "baseline": results,
        "summary": _summarize_results(results),
    }
    if mode == "openai":
        payload["model"] = model
        payload["runtime_target"] = (
            "in_process_environment"
            if runner_factory is not None
            else base_url or os.getenv("LLMSERVE_BASE_URL", DEFAULT_BASE_URL)
        )
    return payload


def _summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    scores = [float(result["grader"]["score"]) for result in results.values()]
    mean_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {
        "task_count": len(results),
        "mean_score": mean_score,
        "scores": {task_id: float(result["grader"]["score"]) for task_id, result in results.items()},
        "heuristic_baselines": {
            task_id: float(result["grader"].get("heuristic_baseline", 0.0))
            for task_id, result in results.items()
        },
        "ppo_baselines": {
            task_id: float(result["grader"].get("ppo_baseline", 0.0))
            for task_id, result in results.items()
        },
    }


def _action_from_model(client: OpenAI, model: str, task_id: str, observation: Any) -> ServeAction:
    user_prompt = json.dumps(
        {
            "task_id": task_id,
            "observation": observation.model_dump(mode="json"),
        }
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    payload = _parse_model_payload(raw)
    if payload is None:
        return default_action()

    payload.setdefault("batch_cap", 32)
    payload.setdefault("kv_budget_fraction", 1.0)
    payload.setdefault("speculation_depth", 0)
    payload.setdefault("quantization_tier", QuantizationTier.FP16.value)
    payload.setdefault("prefill_decode_split", False)
    payload.setdefault("priority_routing", False)

    try:
        return ServeAction.model_validate(payload)
    except Exception:
        return default_action()


def _parse_model_payload(raw: str) -> dict[str, Any] | None:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE | re.DOTALL).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic or OpenAI baseline inference for LLMServeEnv.")
    parser.add_argument("--mode", choices=["deterministic", "openai"], default="deterministic")
    parser.add_argument(
        "--runtime",
        choices=["in-process", "http"],
        default="in-process",
        help="How to execute the environment during baseline inference.",
    )
    parser.add_argument("--task-id", action="append", dest="task_ids", help="Task ID to run. Repeat for multiple tasks.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("LLMSERVE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None, help="Optional path to write the JSON result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "openai":
        runner_factory = None
        base_url = args.base_url
        if args.runtime == "in-process":
            runner_factory = lambda: create_local_runner(seed=args.seed)
            base_url = None
        payload = run_baseline_suite(
            mode="openai",
            task_ids=args.task_ids,
            seed=args.seed,
            model=args.model,
            api_key=args.api_key,
            base_url=base_url,
            runner_factory=runner_factory,
        )
    else:
        payload = run_baseline_suite(
            mode="deterministic",
            task_ids=args.task_ids,
            seed=args.seed,
            runner_factory=lambda: create_local_runner(seed=args.seed),
        )

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

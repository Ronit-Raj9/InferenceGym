#!/usr/bin/env python3
"""LLM agent — uses OpenAI-compatible API to decide serving configuration.

Requires environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
Falls back to PPO agent if API is unavailable.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmserve_env.models import ServeAction, ServeObservation  # noqa: E402

SYSTEM_PROMPT = """You are an LLM serving configuration optimizer. Your goal is to maximize throughput while meeting latency SLOs. Given the current server metrics as JSON, respond with a JSON ServeAction.

Action fields and ranges:
- batch_cap: int 1..512
- kv_budget_fraction: float 0.1..1.0
- speculation_depth: int 0..8
- quantization_tier: one of FP16, INT8, INT4
- prefill_decode_split: bool
- priority_routing: bool

Return ONLY valid JSON. No markdown, no explanation.""".strip()


class LLMAgent:
    """Agent that uses an OpenAI-compatible API for action selection."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("API_BASE_URL", "")
        self.model = model or os.getenv("MODEL_NAME", "gpt-4.1-mini")
        self._history: list[dict[str, Any]] = []

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)

    def reset(self) -> None:
        self._history.clear()

    def act(self, observation: ServeObservation, task_id: str) -> ServeAction:
        """Query the LLM for an action, with retry and fallback."""
        obs_dict = {
            "queue_depth": observation.queue_depth,
            "active_requests": observation.active_requests,
            "kv_cache_occupancy": round(observation.kv_cache_occupancy, 3),
            "mean_prompt_length": round(observation.mean_prompt_length, 1),
            "p99_ttft_ms": round(observation.p99_ttft_ms, 1),
            "slo_compliance_rate": round(observation.slo_compliance_rate, 3),
            "throughput_tps": round(observation.throughput_tps, 1),
            "eviction_events": observation.eviction_events,
            "request_arrival_rate": round(observation.request_arrival_rate, 1),
            "step_index": observation.step_index,
        }

        user_msg = f"Task: {task_id}\nCurrent metrics: {json.dumps(obs_dict)}"
        if self._history:
            user_msg += f"\nPrevious action: {json.dumps(self._history[-1])}"

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.1 if attempt == 0 else 0.0,
                    max_tokens=200,
                )
                raw = response.choices[0].message.content or ""
                action = self._parse(raw)
                self._history.append(action.model_dump(mode="json"))
                return action
            except Exception:
                if attempt == 0:
                    user_msg += "\n\nPrevious response was invalid. Return ONLY a JSON object with the action fields."
                continue

        # Fallback to heuristic if LLM fails
        from server.baseline_agent import HeuristicPolicy
        fallback = HeuristicPolicy()
        return fallback.act(observation, task_id)

    def _parse(self, raw: str) -> ServeAction:
        """Parse LLM response into a ServeAction."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        data = json.loads(text)
        return ServeAction(**data)

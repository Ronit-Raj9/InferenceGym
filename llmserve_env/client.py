from __future__ import annotations

import json
from typing import Any
from urllib import request

from llmserve_env.models import EpisodeLog, ServeAction, ServeObservation, ServeState


class LLMServeEnv:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None

    @classmethod
    def from_url(cls, base_url: str) -> "LLMServeEnv":
        return cls(base_url=base_url)

    @classmethod
    def from_hub(cls, repo_id: str) -> "LLMServeEnv":
        return cls(base_url=f"https://huggingface.co/spaces/{repo_id}")

    def reset(self, task_id: str, seed: int | None = None) -> ServeObservation:
        payload = self._post("/reset", {"task_id": task_id, "seed": seed})
        self.session_id = payload.get("session_id")
        return self._parse_observation_payload(payload)

    def step(self, action: dict[str, Any] | ServeAction) -> tuple[ServeObservation, float, bool, dict[str, Any]]:
        action_payload = action.model_dump(mode="json") if isinstance(action, ServeAction) else action
        body: dict[str, Any] = {"action": action_payload}
        if self.session_id is not None:
            body["session_id"] = self.session_id
        payload = self._post("/step", body)
        observation = self._parse_observation_payload(payload)
        if payload.get("session_id") and self.session_id is None:
            self.session_id = str(payload["session_id"])
        return observation, float(payload["reward"]), bool(payload["done"]), observation.metadata

    def state(self) -> ServeState:
        path = f"/state?session_id={self.session_id}" if self.session_id is not None else "/state"
        payload = self._get(path)
        return ServeState.model_validate(payload)

    def tasks(self) -> dict[str, Any]:
        return self._get("/tasks")

    def grade(self, log: EpisodeLog | None = None) -> dict[str, Any]:
        body = {} if log is None else {"episode_log": log.model_dump(mode="json")}
        return self._post("/grader", body)

    def baseline(self, task_id: str | None = None, use_openai: bool = False, model: str | None = None) -> dict[str, Any]:
        params = []
        if task_id:
            params.append(f"task_id={task_id}")
        if use_openai:
            params.append("use_openai=true")
        if model:
            params.append(f"model={model}")
        suffix = f"?{'&'.join(params)}" if params else ""
        return self._get(f"/baseline{suffix}")

    def _parse_observation_payload(self, payload: dict[str, Any]) -> ServeObservation:
        observation_payload = dict(payload["observation"])
        observation_payload["reward"] = payload.get("reward")
        observation_payload["done"] = payload.get("done", False)
        return ServeObservation.model_validate(observation_payload)

    def _get(self, path: str) -> dict[str, Any]:
        with request.urlopen(f"{self.base_url}{path}") as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = request.Request(f"{self.base_url}{path}", data=body, headers=headers, method="POST")
        with request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))

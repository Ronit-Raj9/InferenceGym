from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from llmserve_env.models import EpisodeLog, ServeAction


class GraderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    episode_log: EpisodeLog | None = None
    actions_taken: int | None = None


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = "static_workload"
    seed: int | None = None
    episode_id: str | None = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ServeAction
    session_id: str | None = None

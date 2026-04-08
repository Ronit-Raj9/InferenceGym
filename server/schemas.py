from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from llmserve_env.models import EpisodeLog


class GraderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    episode_log: EpisodeLog | None = None
    actions_taken: int | None = None

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from openenv.core import create_fastapi_app
from dotenv import load_dotenv

from llmserve_env.models import ServeAction, ServeObservation
from llmserve_env.task_catalog import get_action_schema, get_task_catalog
from server.baseline_inference import create_local_runner, run_baseline_suite
from server.grader import GraderEngine
from server.llmserve_environment import LLMServeEnvironment
from server.schemas import GraderRequest, ResetRequest, StepRequest
from server.session_manager import SessionManager
from server.web_ui import create_web_app


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=False)


def _build_shared_env() -> LLMServeEnvironment:
    seed = int(os.getenv("LLMSERVE_SEED", "42"))
    mode = os.getenv("LLMSERVE_MODE")
    return LLMServeEnvironment(seed=seed, mode=mode)


shared_env = _build_shared_env()
grader = GraderEngine()
session_manager = SessionManager()


def get_env() -> LLMServeEnvironment:
    return shared_env


def _register_extra_routes(app: FastAPI) -> FastAPI:
    def _resolve_env(session_id: str | None) -> LLMServeEnvironment:
        if not session_id:
            return shared_env
        try:
            return session_manager.get(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/web", status_code=307)

    @app.get("/tasks")
    def tasks() -> dict[str, object]:
        return {"tasks": get_task_catalog(), "action_schema": get_action_schema()}

    @app.get("/runtime")
    def runtime() -> dict[str, object]:
        return {
            "mode": shared_env.backend.mode,
            "backend": shared_env.backend.describe(),
            "seed": shared_env.seed,
            "active_sessions": session_manager.count(),
        }

    @app.post("/reset")
    def reset(payload: ResetRequest) -> dict[str, object]:
        session_id, env = session_manager.create(
            task_id=payload.task_id,
            seed=payload.seed,
            episode_id=payload.episode_id,
        )
        observation = env.observations[-1]

        return {
            "session_id": session_id,
            "observation": observation.model_dump(mode="json"),
            "reward": observation.reward,
            "done": observation.done,
            "metadata": observation.metadata,
        }

    @app.post("/step")
    def step(payload: StepRequest) -> dict[str, object]:
        env = _resolve_env(payload.session_id)
        observation = env.step(payload.action)
        return {
            "session_id": payload.session_id or env.state.episode_id,
            "observation": observation.model_dump(mode="json"),
            "reward": observation.reward,
            "done": observation.done,
            "metadata": observation.metadata,
        }

    @app.get("/state")
    def state(session_id: str | None = Query(default=None)) -> dict[str, object]:
        env = _resolve_env(session_id)
        return env.state.model_dump(mode="json")

    @app.post("/grader")
    def grade(payload: GraderRequest | None = None) -> dict[str, object]:
        if payload and payload.episode_log is not None:
            if payload.task_id and payload.task_id != payload.episode_log.task_id:
                raise HTTPException(status_code=400, detail="task_id does not match episode_log.task_id.")
            return grader.grade(payload.episode_log, actions_taken=payload.actions_taken)
        if not shared_env.observations:
            raise HTTPException(status_code=400, detail="No active or completed episode is available to grade.")
        current_log = shared_env.export_episode_log()
        if payload and payload.task_id and payload.task_id != current_log.task_id:
            raise HTTPException(status_code=400, detail="task_id does not match the active episode.")
        return grader.grade(current_log, actions_taken=payload.actions_taken if payload else None)

    @app.get("/baseline")
    def baseline(
        task_id: str | None = None,
        use_openai: bool = False,
        model: str = "gpt-4.1-mini",
        seed: int = 42,
    ) -> dict[str, object]:
        task_ids = [task_id] if task_id else [task["id"] for task in get_task_catalog()]
        mode = "openai" if use_openai else "deterministic"
        try:
            runner_factory = (
                (lambda: create_local_runner(seed=seed, mode=os.getenv("LLMSERVE_MODE", "sim")))
                if use_openai
                else (lambda: create_local_runner(seed=seed, mode="sim"))
            )
            return run_baseline_suite(
                mode=mode,
                task_ids=task_ids,
                seed=seed,
                model=model,
                runner_factory=runner_factory,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/demo")
    def demo() -> RedirectResponse:
        return RedirectResponse(url="/web", status_code=307)

    return app


def create_application(enable_web: bool = True) -> FastAPI:
    app = create_fastapi_app(
        get_env,
        ServeAction,
        ServeObservation,
    )
    if enable_web:
        app = create_web_app(app, session_manager, shared_env)
    return _register_extra_routes(app)


def create_test_application() -> FastAPI:
    return create_application(enable_web=False)


app = create_application(enable_web=True)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

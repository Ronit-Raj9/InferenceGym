from __future__ import annotations

import json
from typing import Any

import gradio as gr
import pandas as pd
from fastapi import FastAPI

from llmserve_env.models import QuantizationTier, ServeAction, ServeObservation
from llmserve_env.task_catalog import get_task_catalog
from server.llmserve_environment import LLMServeEnvironment
from server.session_manager import SessionManager


def create_web_app(app: FastAPI, session_manager: SessionManager, fallback_env: LLMServeEnvironment) -> FastAPI:
    blocks = build_web_ui(session_manager, fallback_env)
    return gr.mount_gradio_app(app, blocks, path="/web")


def build_web_ui(session_manager: SessionManager, fallback_env: LLMServeEnvironment) -> gr.Blocks:
    task_ids = [task["id"] for task in get_task_catalog()]

    def _empty_state_json() -> str:
        return json.dumps(
            {
                "episode_id": "",
                "step_count": 0,
                "task_id": "uninitialized",
                "total_requests_served": 0,
                "total_slo_violations": 0,
                "cumulative_reward": 0.0,
                "elapsed_simulated_time_s": 0.0,
                "workload_phase": "warmup",
                "done": False,
            },
            indent=2,
        )

    def _history_frame(env: LLMServeEnvironment | None = None) -> pd.DataFrame:
        active_env = env or fallback_env
        rows = [
            {
                "step_index": observation.step_index,
                "reward": observation.reward,
                "p99_ttft_ms": observation.p99_ttft_ms,
                "slo_compliance_rate": observation.slo_compliance_rate,
                "throughput_tps": observation.throughput_tps,
            }
            for observation in active_env.observations
        ]
        if not rows:
            rows = [
                {
                    "step_index": 0,
                    "reward": 0.0,
                    "p99_ttft_ms": 0.0,
                    "slo_compliance_rate": 1.0,
                    "throughput_tps": 0.0,
                }
            ]
        return pd.DataFrame(rows)

    def _session_json(env: LLMServeEnvironment | None = None) -> str:
        active_env = env or fallback_env
        backend = active_env.backend.describe()
        payload = {
            "active_task_id": active_env.state.task_id,
            "episode_id": active_env.state.episode_id,
            "step_count": active_env.state.step_count,
            "mode": backend.get("mode", active_env.backend.mode),
            "backend": backend,
            "done": active_env.state.done,
        }
        return json.dumps(payload, indent=2)

    def _response_json(observation: ServeObservation) -> str:
        payload = {
            "observation": observation.model_dump(mode="json"),
            "reward": observation.reward,
            "done": observation.done,
            "metadata": observation.metadata,
        }
        return json.dumps(payload, indent=2)

    def _state_json(env: LLMServeEnvironment | None = None) -> str:
        if env is None:
            return _empty_state_json()
        return json.dumps(env.state.model_dump(mode="json"), indent=2)

    def _get_env(session_id: str | None) -> LLMServeEnvironment | None:
        if not session_id:
            return None
        try:
            return session_manager.get(session_id)
        except KeyError:
            return None

    def _ui_payload(
        observation: ServeObservation,
        status_message: str,
        session_id: str,
        env: LLMServeEnvironment,
    ) -> tuple[str, str, str, str, pd.DataFrame, str]:
        return (
            status_message,
            _session_json(env),
            _response_json(observation),
            _state_json(env),
            _history_frame(env),
            session_id,
        )

    def reset_env(current_session_id: str | None, task_id: str, seed: int) -> tuple[str, str, str, str, pd.DataFrame, str]:
        try:
            if current_session_id:
                session_manager.remove(current_session_id)
            session_id, env = session_manager.create(task_id=task_id, seed=int(seed))
            observation = env.observations[-1]
            return _ui_payload(
                observation,
                f"Environment reset for task `{task_id}`. Active episode now uses `{env.state.task_id}`.",
                session_id,
                env,
            )
        except Exception as exc:
            return (f"Error: {exc}", _session_json(), "", _state_json(), _history_frame(), current_session_id or "")

    def step_env(
        session_id: str | None,
        batch_cap: int,
        kv_budget_fraction: float,
        speculation_depth: int,
        quantization_tier: str,
        prefill_decode_split: bool,
        priority_routing: bool,
    ) -> tuple[str, str, str, str, pd.DataFrame, str]:
        try:
            env = _get_env(session_id)
            if env is None:
                raise RuntimeError("No active session found. Click Reset before stepping.")
            action = ServeAction(
                batch_cap=int(batch_cap),
                kv_budget_fraction=float(kv_budget_fraction),
                speculation_depth=int(speculation_depth),
                quantization_tier=quantization_tier,
                prefill_decode_split=bool(prefill_decode_split),
                priority_routing=bool(priority_routing),
            )
            observation = env.step(action)
            return _ui_payload(
                observation,
                f"Step complete for active task `{env.state.task_id}` in `{env.backend.mode}` mode.",
                session_id or env.state.episode_id,
                env,
            )
        except Exception as exc:
            active_env = _get_env(session_id)
            return (
                f"Error: {exc}",
                _session_json(active_env),
                "",
                _state_json(active_env),
                _history_frame(active_env),
                session_id or "",
            )

    def get_state(session_id: str | None) -> tuple[str, pd.DataFrame, str]:
        try:
            env = _get_env(session_id)
            if env is None:
                raise RuntimeError("No active session found. Click Reset to start an episode.")
            return _state_json(env), _history_frame(env), session_id or ""
        except Exception as exc:
            return f"Error: {exc}", _history_frame(), session_id or ""

    with gr.Blocks(title="LLMServeEnv") as demo:
        gr.Markdown(
            """
            # LLMServeEnv

            Reset an episode, then control the serving policy with bounded inputs only.
            The web UI now keeps a dedicated backend session per browser tab so repeated Step clicks continue the same episode reliably in Docker.
            Numeric controls use sliders, categorical controls use fixed choices.
            """
        )

        session_id_state = gr.State(value="")

        with gr.Row():
            with gr.Column(scale=1):
                task_id = gr.Dropdown(
                    choices=task_ids,
                    value=task_ids[0],
                    allow_custom_value=False,
                    label="Task",
                )
                seed = gr.Slider(0, 1000, value=42, step=1, label="Seed")
                reset_btn = gr.Button("Reset", variant="secondary")

                gr.Markdown("## Action Controls")
                batch_cap = gr.Slider(1, 512, value=32, step=1, label="Batch Cap")
                kv_budget_fraction = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="KV Budget Fraction")
                speculation_depth = gr.Slider(0, 8, value=0, step=1, label="Speculation Depth")
                quantization_tier = gr.Radio(
                    choices=[tier.value for tier in QuantizationTier],
                    value=QuantizationTier.FP16.value,
                    label="Quantization Tier",
                )
                prefill_decode_split = gr.Checkbox(value=False, label="Prefill Decode Split")
                priority_routing = gr.Checkbox(value=False, label="Priority Routing")

                with gr.Row():
                    step_btn = gr.Button("Step", variant="primary")
                    state_btn = gr.Button("Get state", variant="secondary")

                status = gr.Textbox(label="Status", interactive=False)
                session_json = gr.Code(
                    label="Active Session",
                    language="json",
                    value=_session_json(),
                    interactive=False,
                )

            with gr.Column(scale=2):
                response_json = gr.Code(label="Observation / Step Response", language="json", interactive=False)
                state_json = gr.Code(label="Current State", language="json", value=_empty_state_json(), interactive=False)
                history_table = gr.Dataframe(
                    value=_history_frame(),
                    headers=["step_index", "reward", "p99_ttft_ms", "slo_compliance_rate", "throughput_tps"],
                    label="Episode Metrics History",
                    interactive=False,
                )

        reset_btn.click(
            fn=reset_env,
            inputs=[session_id_state, task_id, seed],
            outputs=[status, session_json, response_json, state_json, history_table, session_id_state],
        )
        task_id.change(
            fn=reset_env,
            inputs=[session_id_state, task_id, seed],
            outputs=[status, session_json, response_json, state_json, history_table, session_id_state],
        )
        step_btn.click(
            fn=step_env,
            inputs=[
                session_id_state,
                batch_cap,
                kv_budget_fraction,
                speculation_depth,
                quantization_tier,
                prefill_decode_split,
                priority_routing,
            ],
            outputs=[status, session_json, response_json, state_json, history_table, session_id_state],
        )
        state_btn.click(
            fn=get_state,
            inputs=[session_id_state],
            outputs=[state_json, history_table, session_id_state],
        )

    return demo

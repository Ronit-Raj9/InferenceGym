from __future__ import annotations

import json
from typing import Any

import gradio as gr
import pandas as pd
from fastapi import FastAPI
from openenv.core import create_fastapi_app

from llmserve_env.models import QuantizationTier, ServeAction, ServeObservation
from llmserve_env.task_catalog import get_task_catalog
from server.llmserve_environment import LLMServeEnvironment


def create_web_app(env: LLMServeEnvironment) -> FastAPI:
    app = create_fastapi_app(lambda: env, ServeAction, ServeObservation)
    blocks = build_web_ui(env)
    return gr.mount_gradio_app(app, blocks, path="/web")


def build_web_ui(env: LLMServeEnvironment) -> gr.Blocks:
    task_ids = [task["id"] for task in get_task_catalog()]

    def _state_json() -> str:
        return json.dumps(env.state.model_dump(mode="json"), indent=2)

    def _session_json() -> str:
        backend = env.backend.describe()
        payload = {
            "active_task_id": env.state.task_id,
            "episode_id": env.state.episode_id,
            "step_count": env.state.step_count,
            "mode": backend.get("mode", env.backend.mode),
            "backend": backend,
            "done": env.state.done,
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

    def _history_frame() -> pd.DataFrame:
        rows = [
            {
                "step_index": observation.step_index,
                "reward": observation.reward,
                "p99_ttft_ms": observation.p99_ttft_ms,
                "slo_compliance_rate": observation.slo_compliance_rate,
                "throughput_tps": observation.throughput_tps,
            }
            for observation in env.observations
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

    def _ui_payload(observation: ServeObservation, status_message: str) -> tuple[str, str, str, str, pd.DataFrame]:
        return (
            status_message,
            _session_json(),
            _response_json(observation),
            _state_json(),
            _history_frame(),
        )

    def reset_env(task_id: str, seed: int) -> tuple[str, str, str, str, pd.DataFrame]:
        try:
            observation = env.reset(task_id=task_id, seed=int(seed))
            return _ui_payload(
                observation,
                f"Environment reset for task `{task_id}`. Active episode now uses `{env.state.task_id}`.",
            )
        except Exception as exc:
            return (f"Error: {exc}", _session_json(), "", _state_json(), _history_frame())

    def step_env(
        batch_cap: int,
        kv_budget_fraction: float,
        speculation_depth: int,
        quantization_tier: str,
        prefill_decode_split: bool,
        priority_routing: bool,
    ) -> tuple[str, str, str, str, pd.DataFrame]:
        try:
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
            )
        except Exception as exc:
            return (f"Error: {exc}", _session_json(), "", _state_json(), _history_frame())

    def get_state() -> tuple[str, pd.DataFrame]:
        try:
            return _state_json(), _history_frame()
        except Exception as exc:
            return f"Error: {exc}", _history_frame()

    with gr.Blocks(title="LLMServeEnv") as demo:
        gr.Markdown(
            """
            # LLMServeEnv

            Reset an episode, then control the serving policy with bounded inputs only.
            Numeric controls use sliders, categorical controls use fixed choices.
            """
        )

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
                state_json = gr.Code(label="Current State", language="json", interactive=False)
                history_table = gr.Dataframe(
                    value=_history_frame(),
                    headers=["step_index", "reward", "p99_ttft_ms", "slo_compliance_rate", "throughput_tps"],
                    label="Episode Metrics History",
                    interactive=False,
                )

        reset_btn.click(
            fn=reset_env,
            inputs=[task_id, seed],
            outputs=[status, session_json, response_json, state_json, history_table],
        )
        task_id.change(
            fn=reset_env,
            inputs=[task_id, seed],
            outputs=[status, session_json, response_json, state_json, history_table],
        )
        step_btn.click(
            fn=step_env,
            inputs=[
                batch_cap,
                kv_budget_fraction,
                speculation_depth,
                quantization_tier,
                prefill_decode_split,
                priority_routing,
            ],
            outputs=[status, session_json, response_json, state_json, history_table],
        )
        state_btn.click(
            fn=get_state,
            outputs=[state_json, history_table],
        )

    return demo

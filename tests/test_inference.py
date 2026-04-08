from __future__ import annotations

from pathlib import Path

import inference
import pandas as pd
from server import replay_assets


def test_resolve_data_path_finds_lookup_table() -> None:
    path = replay_assets.resolve_data_path("lookup_tables/latency_table.parquet")
    assert path.exists()
    assert path.name in {"latency_table.parquet", "serving_profile_table.parquet"}


def test_main_returns_zero_when_env_init_fails(monkeypatch, capsys) -> None:
    class BrokenEnv:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("simulator bootstrap failed")

    monkeypatch.setattr(inference, "LLMServeEnvironment", BrokenEnv)
    monkeypatch.setattr(inference, "_create_client", lambda: None)

    rc = inference.main()
    output = capsys.readouterr().out

    assert rc == 0
    assert output.count("[START]") == len(inference.TASKS)
    assert output.count("[END]") == len(inference.TASKS)
    assert "simulator bootstrap failed" in output


def test_fallback_assets_work_when_parquet_loading_breaks(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("parquet unavailable")

    monkeypatch.setattr(pd, "read_parquet", _boom)
    replay_assets.load_lookup_table.cache_clear()
    replay_assets.load_trace_table.cache_clear()
    replay_assets.load_prompt_samples.cache_clear()

    lookup = replay_assets.load_lookup_table("lookup_tables/latency_table.parquet")
    trace = replay_assets.load_trace_table("traces/static_workload_trace.parquet")
    prompts = replay_assets.load_prompt_samples("traces/sharegpt_prompt_lengths.parquet")

    assert not lookup.empty
    assert not trace.empty
    assert prompts

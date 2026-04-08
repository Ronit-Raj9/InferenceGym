from __future__ import annotations

from pathlib import Path

import inference
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

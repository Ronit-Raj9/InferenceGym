from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
WORKLOAD_CONFIG_PATH = ROOT_DIR / "server" / "data" / "workload_configs.json"


def _load_catalog() -> list[dict[str, Any]]:
    with WORKLOAD_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["tasks"]


def get_task_catalog() -> list[dict[str, Any]]:
    return _load_catalog()


def get_task_config(task_id: str) -> dict[str, Any]:
    for task in _load_catalog():
        if task["id"] == task_id:
            return task
    raise KeyError(f"Unknown task_id: {task_id}")


def get_action_schema() -> dict[str, Any]:
    return {
        "batch_cap": {"type": "int", "min": 1, "max": 512},
        "kv_budget_fraction": {"type": "float", "min": 0.1, "max": 1.0},
        "speculation_depth": {"type": "int", "min": 0, "max": 8},
        "quantization_tier": {"type": "enum", "values": ["FP16", "INT8", "INT4"]},
        "prefill_decode_split": {"type": "bool"},
        "priority_routing": {"type": "bool"},
    }


from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SERVER_DATA_DIR = ROOT_DIR / "server" / "data"


def _candidate_paths(relative_path: str) -> list[Path]:
    path = Path(relative_path)
    if path.is_absolute():
        return [path]

    candidates = [
        DATA_DIR / path,
        SERVER_DATA_DIR / path,
    ]

    if path.name == "latency_table.parquet":
        serving_profile = path.with_name("serving_profile_table.parquet")
        candidates.extend(
            [
                DATA_DIR / serving_profile,
                SERVER_DATA_DIR / serving_profile,
            ]
        )

    seen: set[Path] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(candidate)
    return deduped


def resolve_data_path(relative_path: str) -> Path:
    for candidate in _candidate_paths(relative_path):
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in _candidate_paths(relative_path))
    raise FileNotFoundError(f"Could not locate required data asset '{relative_path}'. Searched: {searched}")


@lru_cache(maxsize=None)
def load_trace_table(relative_path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(resolve_data_path(relative_path))
    except Exception:
        return _fallback_trace_table(relative_path)


@lru_cache(maxsize=None)
def load_lookup_table(relative_path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(resolve_data_path(relative_path))
    except Exception:
        return _fallback_lookup_table()


@lru_cache(maxsize=None)
def load_prompt_samples(relative_path: str) -> list[int]:
    try:
        frame = pd.read_parquet(resolve_data_path(relative_path))
        if "prompt_length" not in frame.columns:
            raise KeyError(f"Expected 'prompt_length' column in {relative_path}")
        return [int(value) for value in frame["prompt_length"].tolist()]
    except Exception:
        return [64, 96, 128, 256, 512, 1024, 2048, 4096]


def _fallback_trace_table(relative_path: str) -> pd.DataFrame:
    if "bursty" in relative_path:
        rows = [
            {"arrival_rate_rps": 25.0, "prompt_p50": 180.0, "prompt_p95": 2600.0, "priority_fraction": 0.05, "phase": "steady", "service_rate_hint": 18.0, "queue_bias": 0},
            {"arrival_rate_rps": 80.0, "prompt_p50": 220.0, "prompt_p95": 3200.0, "priority_fraction": 0.05, "phase": "burst", "service_rate_hint": 32.0, "queue_bias": 4},
            {"arrival_rate_rps": 30.0, "prompt_p50": 200.0, "prompt_p95": 2800.0, "priority_fraction": 0.05, "phase": "steady", "service_rate_hint": 20.0, "queue_bias": 1},
        ]
    elif "adversarial" in relative_path:
        rows = [
            {"arrival_rate_rps": 45.0, "prompt_p50": 256.0, "prompt_p95": 4096.0, "priority_fraction": 0.2, "phase": "steady", "service_rate_hint": 24.0, "queue_bias": 2},
            {"arrival_rate_rps": 120.0, "prompt_p50": 512.0, "prompt_p95": 8192.0, "priority_fraction": 0.2, "phase": "burst", "service_rate_hint": 40.0, "queue_bias": 8},
            {"arrival_rate_rps": 35.0, "prompt_p50": 192.0, "prompt_p95": 3072.0, "priority_fraction": 0.2, "phase": "steady", "service_rate_hint": 22.0, "queue_bias": 0},
        ]
    else:
        rows = [
            {"arrival_rate_rps": 5.0, "prompt_p50": 512.0, "prompt_p95": 512.0, "priority_fraction": 0.0, "phase": "steady", "service_rate_hint": 4.0, "queue_bias": 0},
            {"arrival_rate_rps": 5.0, "prompt_p50": 512.0, "prompt_p95": 512.0, "priority_fraction": 0.0, "phase": "steady", "service_rate_hint": 4.0, "queue_bias": 0},
        ]
    return pd.DataFrame(rows)


def _fallback_lookup_table() -> pd.DataFrame:
    rows = []
    prompt_buckets = ["tiny", "small", "medium", "large", "xl"]
    batch_caps = [1, 8, 32, 128, 512]
    kv_budgets = [0.1, 0.5, 1.0]
    spec_depths = [0, 2, 4, 8]
    for prompt_index, prompt_bucket in enumerate(prompt_buckets):
        prompt_scale = 1.0 + prompt_index * 0.2
        for batch_cap in batch_caps:
            for kv_budget in kv_budgets:
                for spec_depth in spec_depths:
                    throughput = max(8.0, 18.0 + batch_cap * 0.35 + spec_depth * 1.5 - prompt_index * 2.0)
                    p50_ttft = max(35.0, (220.0 + prompt_index * 70.0) * (1.1 - min(0.35, spec_depth * 0.04)))
                    p99_ttft = p50_ttft * (1.45 + prompt_index * 0.08)
                    p50_itl = max(2.0, 9.0 * prompt_scale * (1.0 - min(0.3, spec_depth * 0.03)))
                    gpu_memory = max(3.0, 4.0 + batch_cap / 48.0 + prompt_index * 1.2 + kv_budget * 2.0)
                    rows.append(
                        {
                            "batch_cap_bucket": batch_cap,
                            "kv_budget_bucket": kv_budget,
                            "spec_depth_bucket": spec_depth,
                            "prompt_size_bucket": prompt_bucket,
                            "p50_ttft_ms": p50_ttft,
                            "p99_ttft_ms": p99_ttft,
                            "p50_itl_ms": p50_itl,
                            "throughput_tps": throughput,
                            "gpu_memory_gb": gpu_memory,
                            "source": "builtin-fallback",
                        }
                    )
    return pd.DataFrame(rows)

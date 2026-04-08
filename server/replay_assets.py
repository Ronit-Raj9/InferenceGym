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
    return pd.read_parquet(resolve_data_path(relative_path))


@lru_cache(maxsize=None)
def load_lookup_table(relative_path: str) -> pd.DataFrame:
    return pd.read_parquet(resolve_data_path(relative_path))


@lru_cache(maxsize=None)
def load_prompt_samples(relative_path: str) -> list[int]:
    frame = pd.read_parquet(resolve_data_path(relative_path))
    if "prompt_length" not in frame.columns:
        raise KeyError(f"Expected 'prompt_length' column in {relative_path}")
    return [int(value) for value in frame["prompt_length"].tolist()]

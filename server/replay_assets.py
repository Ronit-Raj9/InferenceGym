from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def resolve_data_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return DATA_DIR / path


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

from __future__ import annotations

import threading
from collections import OrderedDict

from server.llmserve_environment import LLMServeEnvironment

MAX_SESSIONS = 50


class SessionManager:
    """Thread-safe LRU session cache for concurrent environment instances."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, LLMServeEnvironment] = OrderedDict()
        self._max_sessions = max_sessions

    def create(
        self,
        task_id: str,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> tuple[str, LLMServeEnvironment]:
        env = LLMServeEnvironment(seed=seed or 42)
        env.reset(task_id=task_id, seed=seed, episode_id=episode_id)
        session_id = env.state.episode_id

        with self._lock:
            # Evict oldest sessions if at capacity
            while len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            self._sessions[session_id] = env

        return session_id, env

    def get(self, session_id: str) -> LLMServeEnvironment:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id}")
            # Move to end (most recently used)
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

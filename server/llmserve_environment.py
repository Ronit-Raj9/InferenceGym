from __future__ import annotations

import uuid
from typing import Any

from openenv.core import Environment

from llmserve_env.models import EpisodeLog, ServeAction, ServeObservation, ServeState
from llmserve_env.task_catalog import get_task_config
from server.reward_calculator import RewardCalculator
from server.serving_backend import ServingBackend, create_serving_backend
from server.slo_monitor import SLOMonitor
from server.workload_generator import WorkloadGenerator


class LLMServeEnvironment(Environment[ServeAction, ServeObservation, ServeState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, seed: int = 42, mode: str | None = None, backend: ServingBackend | None = None) -> None:
        super().__init__()
        self.seed = seed
        try:
            self.backend = backend or create_serving_backend(mode=mode, seed=seed)
        except Exception as e:
            raise RuntimeError(f"Failed to create serving backend: {e}") from e
        
        try:
            self.reward_calculator = RewardCalculator()
        except Exception as e:
            raise RuntimeError(f"Failed to create reward calculator: {e}") from e
        
        self.task_config: dict[str, Any] | None = None
        self.workload_generator: WorkloadGenerator | None = None
        self.slo_monitor: SLOMonitor | None = None
        self.actions: list[ServeAction] = []
        self.observations: list[ServeObservation] = []
        self.rewards: list[float] = []
        self._state = ServeState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id="uninitialized",
            total_requests_served=0,
            total_slo_violations=0,
            cumulative_reward=0.0,
            elapsed_simulated_time_s=0.0,
            workload_phase="warmup",
            done=False,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "static_workload",
        **_: Any,
    ) -> ServeObservation:
        if seed is not None:
            self.seed = seed
        self.task_config = get_task_config(task_id)
        self.workload_generator = WorkloadGenerator(self.task_config, seed=self.seed)
        self.backend.reset(seed=self.seed)
        self.slo_monitor = SLOMonitor()
        self.actions = []
        self.observations = []
        self.rewards = []
        self._state = ServeState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            total_requests_served=0,
            total_slo_violations=0,
            cumulative_reward=0.0,
            elapsed_simulated_time_s=0.0,
            workload_phase="warmup",
            done=False,
        )
        workload = self.workload_generator.next_snapshot(step_index=0)
        observation = self._build_initial_observation(workload)
        self.observations.append(observation)
        return observation

    def step(
        self,
        action: ServeAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> ServeObservation:
        del timeout_s
        if self.task_config is None or self.workload_generator is None or self.slo_monitor is None:
            raise RuntimeError("reset() must be called before step().")

        if self._state.done:
            return self._build_terminal_observation("Episode already completed.")

        next_step_index = self._state.step_count + 1
        workload = self.workload_generator.next_snapshot(step_index=next_step_index)
        metrics = self.backend.run_step(self._state.task_id, action, workload)
        compliance, violations = self.slo_monitor.evaluate(
            p99_ttft_ms=metrics.p99_ttft_ms,
            target_ms=float(self.task_config["slo_p99_ttft_ms"]),
            active_requests=max(1, metrics.requests_served),
        )
        metrics.slo_violations += violations
        memory_cap = float(self.task_config.get("memory_cap_gb", 40.0))
        kv_cache_occupancy = min(1.0, metrics.gpu_memory_used_gb / memory_cap)

        reward = self.reward_calculator.calculate(
            task_id=self._state.task_id,
            metrics=metrics,
            slo_compliance_rate=compliance,
            quantization_tier=action.quantization_tier,
            priority_fraction=workload.priority_fraction,
        )
        done = next_step_index >= int(self.task_config["max_steps"])

        observation = ServeObservation(
            queue_depth=workload.queue_depth,
            active_requests=metrics.requests_served,
            kv_cache_occupancy=kv_cache_occupancy,
            mean_prompt_length=workload.mean_prompt_length,
            p50_ttft_ms=metrics.p50_ttft_ms,
            p99_ttft_ms=metrics.p99_ttft_ms,
            p50_itl_ms=metrics.p50_itl_ms,
            throughput_tps=metrics.throughput_tps,
            slo_compliance_rate=compliance,
            gpu_memory_used_gb=metrics.gpu_memory_used_gb,
            estimated_cost_per_1k=metrics.estimated_cost_per_1k,
            request_arrival_rate=workload.arrival_rate,
            spec_acceptance_rate=metrics.spec_acceptance_rate,
            eviction_events=metrics.eviction_events,
            step_index=next_step_index,
            task_id=self._state.task_id,
            reward=reward,
            done=done,
            metadata={
                "phase": workload.phase,
                "priority_fraction": workload.priority_fraction,
                "task_name": self.task_config["name"],
                "is_throttled": metrics.is_throttled,
                "preemption_events": metrics.preemption_events,
                **self.backend.describe(),
            },
        )

        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self._state.step_count = next_step_index
        self._state.total_requests_served += metrics.requests_served
        self._state.total_slo_violations += metrics.slo_violations
        self._state.cumulative_reward += reward
        self._state.elapsed_simulated_time_s += float(self.task_config["step_window_s"])
        self._state.workload_phase = workload.phase
        self._state.done = done
        return observation

    @property
    def state(self) -> ServeState:
        return self._state

    def export_episode_log(self) -> EpisodeLog:
        return EpisodeLog(
            task_id=self._state.task_id,
            actions=self.actions,
            observations=self.observations,
            rewards=self.rewards,
            final_state=self._state,
        )

    def _build_initial_observation(self, workload: Any) -> ServeObservation:
        return ServeObservation(
            queue_depth=workload.queue_depth,
            active_requests=0,
            kv_cache_occupancy=0.0,
            mean_prompt_length=workload.mean_prompt_length,
            p50_ttft_ms=0.0,
            p99_ttft_ms=0.0,
            p50_itl_ms=0.0,
            throughput_tps=0.0,
            slo_compliance_rate=1.0,
            gpu_memory_used_gb=0.0,
            estimated_cost_per_1k=0.0,
            request_arrival_rate=workload.arrival_rate,
            spec_acceptance_rate=0.0,
            eviction_events=0,
            step_index=0,
            task_id=self._state.task_id,
            reward=0.0,
            done=False,
            metadata={
                "phase": workload.phase,
                "task_name": self.task_config["name"] if self.task_config else "",
                **self.backend.describe(),
            },
        )

    def _build_terminal_observation(self, message: str) -> ServeObservation:
        last = self.observations[-1]
        return last.model_copy(update={"done": True, "reward": 0.0, "metadata": {**last.metadata, "message": message}})

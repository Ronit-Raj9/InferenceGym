from llmserve_env.models import default_action
from server.llmserve_environment import LLMServeEnvironment


def test_environment_reset_and_step() -> None:
    env = LLMServeEnvironment(seed=7)
    obs = env.reset(task_id="static_workload", seed=7)
    assert obs.task_id == "static_workload"
    next_obs = env.step(default_action())
    assert next_obs.step_index == 1
    assert isinstance(next_obs.reward, float)
    assert "phase" in next_obs.metadata
    assert next_obs.done is False

from llmserve_env.task_catalog import get_task_catalog, get_task_config


def test_catalog_has_three_tasks() -> None:
    tasks = get_task_catalog()
    assert len(tasks) == 3


def test_static_task_exists() -> None:
    task = get_task_config("static_workload")
    assert task["difficulty"] == "easy"


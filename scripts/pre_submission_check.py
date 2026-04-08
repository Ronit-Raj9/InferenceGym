#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import request


ROOT_DIR = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT_DIR, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def http_request(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> tuple[int, str]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = request.Request(url, data=body, method=method, headers=headers)
    with request.urlopen(req, timeout=20) as response:
        return response.status, response.read().decode("utf-8")


def verify_space(space_url: str) -> None:
    base_url = space_url.rstrip("/")
    checks = [
        ("GET", "/health", None),
        ("GET", "/tasks", None),
        ("GET", "/web", None),
        ("POST", "/reset", {"task_id": "static_workload", "seed": 42}),
    ]

    for method, path, payload in checks:
        status, body = http_request(f"{base_url}{path}", method=method, payload=payload)
        print(f"{method} {path} -> {status}")
        if status != 200:
            raise SystemExit(f"Verification failed for {path}: expected 200, got {status}")
        if path in {"/tasks", "/reset"}:
            json.loads(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local and deployment checks required before hackathon submission.")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-openenv", action="store_true")
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--space-url", default=os.getenv("HF_SPACE_URL"))
    parser.add_argument("--run-openai-baseline", action="store_true")
    parser.add_argument(
        "--baseline-runtime",
        choices=["in-process", "http"],
        default="in-process",
        help="Use in-process for standalone local runs, or http for a running local/remote deployment.",
    )
    parser.add_argument("--base-url", default=os.getenv("LLMSERVE_BASE_URL", "http://localhost:7860"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if not args.skip_pytest:
        run_command([sys.executable, "-m", "pytest", "-q"])

    if not args.skip_openenv:
        run_command(["openenv", "validate"])

    if not args.skip_docker:
        run_command(["docker", "build", "-t", "llmserve-env", "."])

    if args.space_url:
        verify_space(args.space_url)

    if args.run_openai_baseline:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY must be set to run the OpenAI baseline check.")
        output_path = args.output or str(ROOT_DIR / "artifacts" / "baseline_openai.json")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "server.baseline_inference",
            "--mode",
            "openai",
            "--runtime",
            args.baseline_runtime,
            "--model",
            args.model,
            "--output",
            output_path,
        ]
        if args.baseline_runtime == "http":
            command.extend(["--base-url", args.base_url])
        run_command(command)

    print("Pre-submission checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

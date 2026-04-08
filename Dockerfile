# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml ./

COPY llmserve_env ./llmserve_env
COPY server ./server
COPY agents ./agents
COPY rl ./rl
COPY data ./data
COPY weights ./weights
COPY inference.py evaluate.py train.py ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    printf 'torch==2.5.1+cpu\n' > /tmp/constraints.txt && \
    python -m pip install --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -c /tmp/constraints.txt .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

WORKDIR /app

COPY --from=builder /install /usr/local
COPY pyproject.toml README.md openenv.yaml ./
COPY llmserve_env ./llmserve_env
COPY server ./server
COPY agents ./agents
COPY rl ./rl
COPY data ./data
COPY weights ./weights
COPY inference.py evaluate.py train.py ./

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=5)" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml ./
COPY llmserve_env ./llmserve_env
COPY server ./server
COPY agents ./agents
COPY rl ./rl
COPY data ./data
COPY weights ./weights
COPY inference.py evaluate.py train.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

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

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

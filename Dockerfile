FROM ghcr.io/astral-sh/uv:0.9.18 AS uv

FROM python:3.12.12-slim-bookworm

ARG GIGACHECK_COMMIT=8cdbe6a1ce628b49b4aa0794edf7a425a47a774c

ENV HF_HOME=/cache/huggingface \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl gcc libc6-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

RUN mkdir -p /app/.vendor/gigacheck \
    && curl --fail --silent --show-error --location \
        "https://github.com/ai-forever/gigacheck/archive/${GIGACHECK_COMMIT}.tar.gz" \
        | tar --extract --gzip --strip-components=1 --directory=/app/.vendor/gigacheck

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev \
    && useradd --uid 1000 --create-home app \
    && mkdir -p /cache/huggingface \
    && chown -R app:app /cache

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=3)"

CMD ["uvicorn", "ai_detector.web:app", "--host", "0.0.0.0", "--port", "8000"]

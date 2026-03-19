# ---------- Stage 1: Build ----------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY agentx/ agentx/

RUN pip install --no-cache-dir --prefix=/install ".[all]"

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim

LABEL maintainer="GR Tech Learn <contact@aimediahub.in>"
LABEL description="AgentX - Enterprise Multi-Agent System Framework"

RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 1000 agentx && \
    useradd --uid 1000 --gid agentx --create-home agentx

COPY --from=builder /install /usr/local

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agentx/ agentx/

RUN pip install --no-cache-dir ".[all]"

RUN mkdir -p /app/data /app/logs && \
    chown -R agentx:agentx /app

USER agentx

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "agentx.daemon", "--env"]

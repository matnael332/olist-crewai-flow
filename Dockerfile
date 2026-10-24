# One image for our server: the pipeline, the dashboard and the API.
# compose.yaml runs it; DEPLOY.md explains the server side. Customer data never enters the image:
# it lives in the /srv/pulse volume, one folder per customer (customers/<slug>/{data,runs,artifacts}).

FROM python:3.13-slim-bookworm

# The same uv as CI, so the image installs exactly what uv.lock says.
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

# Runtime flags (as in .env.example and CI); without them the artifacts are not byte-reproducible.
ENV PYTHONHASHSEED=0 \
    PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MPLBACKEND=Agg \
    CREWAI_DISABLE_TELEMETRY=true \
    OTEL_SDK_DISABLED=true \
    CREWAI_TRACING_ENABLED=false \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Dependencies first, in their own layer: a code change does not reinstall them.
# Production set: the runtime dependencies and the dashboard group, no dev tools.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --group frontend --no-install-project

# The code, then the project itself. /app/pyproject.toml is what repo_root() looks for.
COPY backend/ backend/
COPY frontend/ frontend/
COPY .streamlit/config.toml .streamlit/config.toml
RUN uv sync --locked --no-default-groups --group frontend

# A non-root user that owns the data volume.
RUN useradd --create-home --uid 1000 pulse \
    && mkdir -p /srv/pulse/customers \
    && chown -R pulse:pulse /srv/pulse
USER pulse

EXPOSE 8501 8000

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

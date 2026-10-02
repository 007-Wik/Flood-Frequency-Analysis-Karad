# Multi-stage Dockerfile for FFA Karad
# Stage 1: Base build environment
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python project dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e ".[test,dev]"

# Copy project source and data
COPY ffa_karad/ ffa_karad/
COPY data/ data/
COPY tests/ tests/
COPY docs/ docs/
COPY README.md .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command: run the full pipeline to generate report and figures
CMD ["python", "-m", "ffa_karad.run_all"]

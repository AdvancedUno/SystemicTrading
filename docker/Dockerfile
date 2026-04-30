FROM python:3.12-slim

# System deps for some scientific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast deps
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy and install deps first (better layer caching)
COPY pyproject.toml ./
RUN uv pip install --system --no-cache .

# Copy source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default to a shell — actual commands invoked via docker-compose
CMD ["bash"]

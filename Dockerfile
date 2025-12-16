FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install uv from the official distroless image
# Using a pinned version tag for reproducibility (adjust version as needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python 3.10, g++, and sanitizers
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv with cache mount for better performance
# Use --locked if lockfile exists, otherwise just sync
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --locked --no-install-project; \
    else \
        uv sync --no-install-project; \
    fi

# Copy project files needed for package build
COPY trl/ ./trl/
COPY VERSION README.md LICENSE MANIFEST.in* ./
COPY cpp_pipeline/ ./cpp_pipeline/
COPY training/ ./training/

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --locked; \
    else \
        uv sync; \
    fi

# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Default command (bash for interactive use)
CMD ["bash"]

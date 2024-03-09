FROM python:3.10-slim as base

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=0 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_HOME=/poetry \
  PATH="/poetry/bin:${PATH}"

RUN apt update && apt install -y --no-install-recommends \
    make xorg-dev curl gnupg git wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry config installer.max-workers 10 && \
    poetry self add poetry-dotenv-plugin


RUN curl -fsSL https://install.julialang.org | sh -s -- -y
SHELL ["bash", "-lc"]

# Helpful Development Packages
RUN julia -e 'using Pkg; Pkg.add(["Revise", "BenchmarkTools"])'

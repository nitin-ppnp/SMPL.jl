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
    make xorg-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry config installer.max-workers 10 && \
    poetry self add poetry-dotenv-plugin

WORKDIR /code
COPY . /code

RUN apt update && apt -y install gnupg git wget

RUN pip install jill

RUN jill install --confirm

RUN pip install julia ipython --no-cache-dir
# julia is pyjulia, our python-julia interface
# jill is a python package for easy Julia installation
# IPython is helpful for magic (both %time and %julia)
# Include these in your requirements.txt if you have that instead

# PyJulia setup (installs PyCall & other necessities)
RUN python -c "import julia; julia.install()"

# Helpful Development Packages
RUN julia -e 'using Pkg; Pkg.add(["Revise", "BenchmarkTools"])'

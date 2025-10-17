# Use a slim Python 3.12 base
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for building C extensions (numpy, scikit-surprise, lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends dnsutils netcat-openbsd && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip/pipenv first
RUN pip install --upgrade pip setuptools wheel pipenv

# Copy dependency files first for better layer cache
COPY Pipfile Pipfile.lock ./

# Pre-install heavy build-time deps so headers are available
# (helps wheels compile cleanly, esp. your scikit-surprise fork)
RUN pip install "numpy>=2.1,<3" cython

# Install all deps into the system site-packages (no venv inside container)
RUN PIPENV_VENV_IN_PROJECT=0 pipenv install --system --deploy

# Now copy the source
COPY . .

# Default command = web (same as Procfile)
# Heroku / compose can override CMD for the worker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--workers", "2"]
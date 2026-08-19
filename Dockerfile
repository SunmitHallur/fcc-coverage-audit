# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=src \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libsqlite3-0 libcurl4 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt /app/requirements-serve.txt
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src /app/src
COPY config /app/config

RUN useradd --create-home --uid 10001 cook \
    && mkdir -p /app/web /app/data/processed /app/data/interim \
    && chown -R cook:cook /app
USER cook

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["python", "-m", "fcc_audit.serve", "--host", "0.0.0.0", "--port", "8000"]

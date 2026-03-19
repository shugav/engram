# Engram v3 MCP server — Python 3.12 + Postgres client libs
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md migrations /app/
COPY src /app/src

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
EXPOSE 8788

CMD ["python", "-m", "engram", "--transport", "sse", "--host", "0.0.0.0", "--port", "8788"]

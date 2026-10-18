# PostgreSQL 16 + pgvector (from the base image) + HypoPG (hypothetical indexes).
FROM pgvector/pgvector:pg16

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-hypopg \
    && rm -rf /var/lib/apt/lists/*

#!/usr/bin/env bash
# Install PostgreSQL 16 + pgvector + HypoPG inside a Kaggle / Colab notebook
# (Ubuntu, running as root, internet enabled) and load the synthetic dataset,
# so the RL reward (src/environment.py) can run next to the GPU.
#
#   !bash scripts/setup_notebook_postgres.sh            # scale 1.0 (default)
#   !SCALE=0.1 bash scripts/setup_notebook_postgres.sh
#
# Planner settings match docker/docker-compose.yml, so costs (and therefore
# rewards) agree with the local sandbox.
set -euo pipefail

PG=16
SCALE="${SCALE:-1.0}"
PYTHON="${PYTHON:-python3}"
export DEBIAN_FRONTEND=noninteractive

echo "==> Installing PostgreSQL $PG + pgvector + HypoPG"
apt-get update -qq
apt-get install -y -qq curl ca-certificates gnupg lsb-release >/dev/null
install -d /usr/share/postgresql-common/pgdg
curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc
echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
  > /etc/apt/sources.list.d/pgdg.list
apt-get update -qq
apt-get install -y -qq "postgresql-$PG" "postgresql-$PG-pgvector" "postgresql-$PG-hypopg" >/dev/null

echo "==> Configuring and starting the server"
CONF="/etc/postgresql/$PG/main/conf.d/index-drift-rl.conf"
cat > "$CONF" <<EOF
shared_preload_libraries = 'pg_stat_statements'
shared_buffers = 512MB
effective_cache_size = 1GB
work_mem = 16MB
maintenance_work_mem = 512MB
max_parallel_workers_per_gather = 2
jit = off
EOF
service postgresql restart

su postgres -c "psql -q -v ON_ERROR_STOP=1" <<'SQL'
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'drift') THEN
    CREATE ROLE drift LOGIN SUPERUSER PASSWORD 'drift';
  END IF;
END $$;
SELECT 'CREATE DATABASE drift OWNER drift' WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'drift')\gexec
SQL
su postgres -c "psql -q -d drift -v ON_ERROR_STOP=1" <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS hypopg;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
SQL

echo "==> Loading dataset (scale $SCALE)"
export DRIFT_PG_DSN="${DRIFT_PG_DSN:-postgresql://drift:drift@localhost:5432/drift}"
"$PYTHON" -m src.seed load --scale "$SCALE"
"$PYTHON" -m src.seed status
echo "==> Ready: $DRIFT_PG_DSN"

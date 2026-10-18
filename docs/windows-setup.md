# Windows setup (16 GB RAM laptop)

One-time setup for running the sandbox on Windows 10/11. Run the commands in
**PowerShell**.

## 1. WSL2 (needs admin + a reboot)

```powershell
wsl --install
```

Reboot when asked. Then cap how much RAM WSL2 (and therefore Docker) may
use, so Windows stays responsive. Create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=8GB
processors=4
swap=4GB
```

Apply it with `wsl --shutdown`.

## 2. Docker Desktop

Install from <https://www.docker.com/products/docker-desktop/> (free for
personal use) and keep the default *Use WSL 2 based engine* option. Check:

```powershell
docker --version
docker compose version
```

Images used: Postgres + pgvector + HypoPG (~450 MB), Elasticsearch (~1.3 GB).

## 3. Python 3.11+

```powershell
winget install Python.Python.3.12
```

Open a new terminal, then from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
pytest                         # offline tests; DB tests skip until the sandbox is up
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Port 5432 already in use (a local Postgres) | `$env:PG_PORT=5433` before `docker compose up`, and `$env:DRIFT_PG_DSN="postgresql://drift:drift@localhost:5433/drift"` |
| Elasticsearch exits with `vm.max_map_count` error | `wsl -d docker-desktop sysctl -w vm.max_map_count=262144` |
| Laptop slows down a lot | Run Postgres only (omit `--profile es`), or seed with `--scale 0.1` |
| Want a clean slate | `docker compose -f docker/docker-compose.yml down -v` (deletes the data volumes) |

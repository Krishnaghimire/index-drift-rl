# data/

Generated files land here and are not committed (see `.gitignore`):

- `workloads/*.jsonl` — query logs from `python -m src.workload generate`
- `runs/<name>/` — replay results from `python -m src.workload run`
  (`results.jsonl`, `summary.json`, `slow_queries.json`)

Everything here can be regenerated deterministically from the seeds.

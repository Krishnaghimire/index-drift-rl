# Phase 4: reward design and audit

The reward (`src/environment.py`) scores every completion the model writes during
GRPO training. A reward that can be gamed gets gamed, so each rule below
closes a specific exploit.

| Rule | Exploit it closes |
|---|---|
| Anything but one `CREATE INDEX` (or `NO INDEX`) → **-1** | rambling, `DROP TABLE`, several statements |
| Missing `CONCURRENTLY` or `UNIQUE` → **-0.5** | table locks in production; HypoPG simulates unique indexes that real data would reject |
| PostgreSQL rejects the index → **-0.75** | invented columns, non-immutable expressions such as `date_trunc` on `timestamptz` |
| Quality averaged over the prompt's parameters **plus 3 resampled parameter sets** | indexes tailored to the literals in the prompt (`WHERE tenant_id = 2`) |
| Size penalty of 0.1 per doubling over the label's size (max 0.3) | indexing every column or adding huge `INCLUDE` lists |
| Latency table: -0.5 × share of real queries that got slower | the planner's blind spot (an index the cost model likes but that slows the dominant tenant) |
| -0.1 for text around the SQL | chatty answers |

## Audit: hand-written answers for one `pending_orders` example (tenant 2)

Label: `CREATE INDEX CONCURRENTLY ... ON orders (tenant_id, created_at) WHERE status = 'pending'`

| Reward | Answer |
|---|---|
| +1.00 | the label |
| +0.90 | the label, wrapped in an explanation |
| +0.70 | `(tenant_id, status, created_at)`: works, but ~20× larger than the partial index |
| +0.70 | bloated covering index with `INCLUDE (customer_id, total, id)` |
| +0.25 | `(created_at) WHERE status = 'pending' AND tenant_id = 2`: only helps 1 of 4 parameter sets |
| +0.01 | `(status)` only |
| 0.00 | `NO INDEX` |
| -0.50 | correct index without `CONCURRENTLY` / as `UNIQUE` |
| -0.75 | misspelled column |
| -1.00 | `DROP TABLE orders` / free text |

The ordering matches an expert's judgement. The same checks run as tests in
`tests/test_environment.py`.

## Windows (drift) task

The model receives the current indexes and the latest window of the workload,
and answers with `DROP INDEX CONCURRENTLY` / `CREATE INDEX CONCURRENTLY`
statements (or `NO CHANGES`). The reward is the share of the label's
improvement in workload-weighted planner cost, so dropping useful indexes
scores negative. Going over the 3-index budget or omitting `CONCURRENTLY` scores
-0.5; each `DROP` of a non-existent index costs 0.1. When the label barely
improves on the current indexes (<5%), keeping them scores full marks.

## Latency calibration

`python -m src.environment calibrate` built the 33 most common answers for
real (scale 1.0, this laptop) and timed 12 queries each against no index.
Highlights:

| Template | Index | Median real speedup | Queries slower |
|---|---|---|---|
| `product_sales` | `order_items (product_id)` | ×67 | 0% |
| `customer_by_email` | `customers ((lower(email)))` | ×60 | 0% |
| `pending_orders` | `orders (tenant_id) WHERE status = 'pending'` | ×14 | 0% |
| `top_products` | `orders (tenant_id, created_at) INCLUDE (id)` | ×4 | **8%** |
| `customer_recent_events` | `events (created_at)` | ×1.04 | 8% |

Latency depends on hardware, so the table is a coarse correction on top of
planner costs, not a replacement for them.

## Running next to the GPU

`scripts/setup_notebook_postgres.sh` installs PostgreSQL 16 + pgvector +
HypoPG in a Kaggle or Colab notebook, with the same planner settings as the
local sandbox, and loads the data. It was rehearsed in a clean `ubuntu:22.04`
container (Python 3.10), where all database and reward tests passed.

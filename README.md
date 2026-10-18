# index-drift-rl

> 🚧 **Status: early work in progress / learning project.** Nothing below is trained or shipped yet — this README describes the goal and the roadmap. Progress is tracked in the [Roadmap](#-roadmap) section.

An open, low-budget experiment in training small open-weights language models with reinforcement learning (GRPO) to act as **database index advisors** for PostgreSQL (B-Tree + `pgvector`) and Elasticsearch (vector search).

Given a slow-query log from a shifting ("drifting") workload, the agent should propose **production-safe** index changes — e.g. `CREATE INDEX CONCURRENTLY`, or HNSW / quantization parameter adjustments — and is rewarded by measured cost improvement in a sandbox.

The project is built in the open as a way to learn applied AI, and aims to give back to the community a **reusable benchmark dataset**, reward harness, and LoRA adapter.

---

## 🎯 Goals

1. **Benchmark & dataset** — synthetic multi-tenant schemas, drifting query workloads, and labelled index scenarios, published on Hugging Face Datasets.
2. **Reward harness** — fast, programmatic scoring of index suggestions:
   - **PostgreSQL:** [HypoPG](https://github.com/HypoPG/hypopg) hypothetical indexes + `EXPLAIN` cost deltas (no real index builds needed during training).
   - **Vector indexes:** an offline-measured lookup table of recall@k vs. latency across HNSW / quantization parameters, so training doesn't require a live Elasticsearch cluster.
   - **Safety checks:** SQL must parse (`pglast`), must use `CONCURRENTLY`, no destructive / locking DDL, penalties for redundant or oversized indexes.
3. **Baselines** — rule-based advisor and zero-shot LLMs, scored by the same harness.
4. **GRPO-trained agent** — a small model (starting with `Qwen2.5-Coder-1.5B-Instruct`) fine-tuned with QLoRA + GRPO, compared honestly against the baselines.

---

## 🏗️ Architecture (planned)

```text
[ Drifting workload / slow-query logs ]
                 │
                 ▼
     ┌───────────────────────┐
     │  index-drift-rl agent │   ← small LLM, QLoRA + GRPO (TRL / Unsloth)
     └───────────────────────┘
                 │  proposed index commands
                 ▼
┌──────────────────────────────────────────────────┐
│                 Reward harness                    │
│  ┌──────────────────────┐  ┌──────────────────┐  │
│  │ PostgreSQL + pgvector│  │ Vector-param     │  │
│  │ + HypoPG (EXPLAIN    │  │ lookup table     │  │
│  │   cost delta)        │  │ (recall/latency) │  │
│  └──────────────────────┘  └──────────────────┘  │
│        + SQL safety / guardrail checks            │
└──────────────────────────────────────────────────┘
                 │
                 ▼
            scalar reward → GRPO update
```

---

## 💸 Low-budget design

This project is deliberately built to run on **modest hardware and free tiers**:

| Task | Where |
|---|---|
| Sandbox DBs (Postgres + pgvector + HypoPG, single-node Elasticsearch) | Local, Docker Desktop on WSL2 (16 GB RAM is enough) |
| Zero-shot baselines | Local CPU via Ollama |
| Vector-parameter benchmarking (offline table) | Local |
| GRPO training | Free GPU notebooks (Kaggle / Google Colab), 4-bit QLoRA via Unsloth |
| Model, dataset, demo hosting | Hugging Face Hub & Spaces (free) |

---

## 🛠️ Tech stack

- **AI / ML:** Python, PyTorch, Hugging Face `TRL` (GRPO), PEFT / QLoRA, Unsloth, `Qwen2.5-Coder` (1.5B → 3B/7B later)
- **Databases:** PostgreSQL + `pgvector` + `HypoPG`, Elasticsearch
- **Tooling:** Docker Compose, `pglast` (SQL parsing), pytest

---

## 📂 Directory structure

```text
index-drift-rl/
├── docker/
│   ├── docker-compose.yml      # Postgres (pgvector + HypoPG) + optional Elasticsearch
│   └── postgres/               # image build + init scripts
├── docs/
│   └── windows-setup.md        # WSL2 / Docker Desktop / Python setup
├── data/                       # generated workloads and runs (git-ignored)
├── src/
│   ├── dataset.py              # dataset spec shared by seeder and workload generator
│   ├── sql/schema.sql          # multi-tenant schema (primary keys only)
│   ├── seed.py                 # CLI: load / status / drop-indexes
│   ├── workload.py             # CLI: drifting workload generator + replay runner
│   ├── es_sync.py              # CLI: copy product embeddings into Elasticsearch
│   ├── query_features.py       # parse SQL (pglast) into index-relevant features
│   ├── candidates.py           # rule-based index candidate generator
│   ├── hypo.py                 # cost queries under hypothetical indexes (HypoPG)
│   ├── build_dataset.py        # CLI: build the labelled benchmark dataset
│   ├── publish_dataset.py      # CLI: upload the dataset to the Hugging Face Hub
│   ├── db.py, config.py        # helpers
│   ├── environment.py          # (Phase 4) reward harness
│   ├── baselines.py            # (Phase 3) rule-based + zero-shot LLM baselines
│   └── train.py                # (Phase 5) GRPO training loop
├── tests/
├── requirements.txt
└── README.md
```

### The sandbox dataset

A synthetic multi-tenant commerce database: `tenants`, `customers`, `products`
(with 64-d embeddings clustered by category), `orders`, `order_items`, `events`
(JSONB payload). At `--scale 1.0` that is ~1M orders, ~3M order items and 2M
events (~600 MB). Tenants are Zipf-skewed (tenant 1 owns ~27% of rows) and
order statuses are skewed (5% `pending`), so selectivity varies a lot. Tables
ship with **primary keys only**: picking indexes is the agent's job.

### Drift scenarios

| Scenario | What changes |
|---|---|
| `oltp_to_analytics` | point lookups gradually give way to date-range aggregates and joins |
| `vector_launch` | a recommendation feature launches: nearest-neighbour queries ramp up |
| `tenant_onboarding` | the smallest tenant suddenly receives 60% of traffic |
| `full` | all of the above, chained |

12 query templates cover composite, partial, expression (`lower(email)`,
`payload->>'source'`), covering, join and HNSW index opportunities. See
`python -m src.workload list`.

### The benchmark dataset (Phase 2)

📦 Published on the Hugging Face Hub:
[**krishna382nitjsr/index-drift-rl-bench**](https://huggingface.co/datasets/krishna382nitjsr/index-drift-rl-bench)

```python
from datasets import load_dataset
single = load_dataset("krishna382nitjsr/index-drift-rl-bench", "single")
windows = load_dataset("krishna382nitjsr/index-drift-rl-bench", "windows")
```

`python -m src.build_dataset --validate` turns the drifting workloads into two
labelled tasks (~2 minutes at scale 1.0):

| Task | Input | Label | Rows (train / test / unseen templates) |
|---|---|---|---|
| `single` | slow query + `EXPLAIN` plan + schema | best single index, or none | 2488 / 829 / 163 |
| `windows` | a window of the query log + indexes left from the previous window | index set to have now (≤ 3): what to create, what to drop | 96 / 32 / – |

Labels come from rule-generated candidates scored as **hypothetical indexes**
with HypoPG, so nothing is built. The smallest index within 2% of the best
planner-cost reduction wins. `--validate` then builds each template's most
common label for real. At scale 1.0, 9 of 10 templates sped up on every
sampled query (median 6× to 116×). One template (`top_products`) shows the
limits of planner estimates: a 5× median speedup, but slower for the
dominant tenant. The generated dataset card (`data/dataset/README.md`) records
all of this.

---

## 🗺️ Roadmap

- [x] **Phase 1 — Sandbox:** Docker Compose for Postgres (pgvector + HypoPG) and Elasticsearch; synthetic multi-tenant schema; drifting workload generator
- [x] **Phase 2 — Dataset:** generate slow-query scenarios with labelled good indexes; publish to Hugging Face Datasets — 🤗 [krishna382nitjsr/index-drift-rl-bench](https://huggingface.co/datasets/krishna382nitjsr/index-drift-rl-bench)
- [ ] **Phase 3 — Baselines:** rule-based advisor + zero-shot LLM baselines
- [ ] **Phase 4 — Reward harness:** HypoPG cost-delta reward + SQL safety checks, with unit tests
- [ ] **Phase 5 — Training:** optional SFT warm-up for output format, then GRPO on free GPU (Kaggle / Colab)
- [ ] **Phase 6 — Vector indexes:** offline HNSW / quantization benchmark table and vector-tuning task
- [ ] **Phase 7 — Release:** LoRA adapter on HF Hub, Gradio demo Space, write-up comparing against baselines

---

## 🚀 Getting started

Prerequisites: Docker Desktop (WSL2) and Python 3.11+. On Windows, follow
[docs/windows-setup.md](docs/windows-setup.md) first.

```bash
# 1. Python environment (Windows: .\.venv\Scripts\Activate.ps1)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the sandbox (Postgres only; add `--profile es` for Elasticsearch)
docker compose -f docker/docker-compose.yml up -d --build

# 3. Load the synthetic dataset (use --scale 0.1 for a ~1-minute quick start)
python -m src.seed load --scale 1.0
python -m src.seed status

# 4. Generate a drifting workload and replay it against the baseline (no indexes)
python -m src.workload generate --scenario full -n 2000 --scale 1.0 --out data/workloads/full.jsonl
python -m src.workload run data/workloads/full.jsonl --out data/runs/baseline

# 5. Optional: mirror product embeddings into Elasticsearch
docker compose -f docker/docker-compose.yml --profile es up -d
python -m src.es_sync --verify

# 6. Build the labelled dataset (Phase 2) and optionally publish it
python -m src.build_dataset --out data/dataset --validate
python -m src.publish_dataset --repo <hf-user>/index-drift-rl-bench   # private by default

# Tests (DB tests are skipped unless the sandbox is up and seeded)
pytest
```

`data/runs/baseline/slow_queries.json` lists the slow query groups per drift
phase with an example query and its `EXPLAIN` plan. This is the input the
agent will learn from.

To experiment by hand, create an index (`CREATE INDEX CONCURRENTLY ...` in
`psql`), replay the same log into a new `--out` directory and compare. Run
`python -m src.seed drop-indexes` to get back to the baseline.

---

## 🤝 Contributing

This is a learning project and feedback is very welcome — especially on workload realism, reward design, and safety rules. Feel free to open an issue or discussion.

## 📝 License

MIT, see [LICENSE](LICENSE).

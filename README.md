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

## 📂 Planned directory structure

```text
index-drift-rl/
├── configs/
│   └── grpo_config.yaml        # training hyperparameters
├── docker/
│   └── docker-compose.yml      # local Postgres + Elasticsearch sandbox
├── data/                       # generated schemas, workloads, scenarios
├── notebooks/                  # Kaggle / Colab training notebooks
├── src/
│   ├── workload.py             # schema + drifting workload generator
│   ├── environment.py          # reward harness (HypoPG cost, safety checks)
│   ├── baselines.py            # rule-based + zero-shot LLM baselines
│   ├── train.py                # GRPO training loop
│   └── inference.py            # CLI: suggest indexes for a query log
├── tests/
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

- [ ] **Phase 1 — Sandbox:** Docker Compose for Postgres (pgvector + HypoPG) and Elasticsearch; synthetic multi-tenant schema; drifting workload generator
- [ ] **Phase 2 — Dataset:** generate slow-query scenarios with labelled good indexes; publish to Hugging Face Datasets
- [ ] **Phase 3 — Baselines:** rule-based advisor + zero-shot LLM baselines
- [ ] **Phase 4 — Reward harness:** HypoPG cost-delta reward + SQL safety checks, with unit tests
- [ ] **Phase 5 — Training:** optional SFT warm-up for output format, then GRPO on free GPU (Kaggle / Colab)
- [ ] **Phase 6 — Vector indexes:** offline HNSW / quantization benchmark table and vector-tuning task
- [ ] **Phase 7 — Release:** LoRA adapter on HF Hub, Gradio demo Space, write-up comparing against baselines

---

## 🚀 Getting started

Code is not available yet — setup instructions will be added as Phase 1 lands.

---

## 🤝 Contributing

This is a learning project and feedback is very welcome — especially on workload realism, reward design, and safety rules. Feel free to open an issue or discussion.

## 📝 License

MIT (license file to be added).

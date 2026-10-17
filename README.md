# index-drift-rl

An autonomous reinforcement learning framework for zero-downtime relational and vector index co-evolution.

`index-drift-rl` bridges the gap between traditional backend database engineering and cutting-edge post-training reinforcement learning (GRPO). It trains open-weights language models to act as autonomous database performance engineers—analyzing slow hybrid query logs, predicting index degradation, and generating production-safe, non-blocking index commands (`CREATE INDEX CONCURRENTLY` and vector quantization adjustments) across PostgreSQL and Elasticsearch.

---

## 🏗️ Architecture Overview

[ Shifting Workload / Query Logs ]
│
▼
┌───────────────────────┐
│  index-drift-rl Agent │  <--- (Trained via GRPO / QLoRA)
└───────────────────────┘
│
▼ (Generates Safe Index Commands)
┌─────────────────────────────────────────────────┐
│               Containerized Sandbox             │
│  ┌──────────────────┐    ┌───────────────────┐  │
│  │ PostgreSQL       │    │ Elasticsearch     │  │
│  │ (Relational /    │    │ (Vector Search /  │  │
│  │  pgvector)       │    │  HNSW / Quant.)   │  │
│  └──────────────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────┘
│
▼ (Automated Execution & Reward Feedback)
[ EXPLAIN ANALYZE / Latency & Concurrency Validation ]


---

## ✨ Key Features

* **Hybrid Co-Evolution Engine:** Simultaneously optimizes traditional B-Tree relational indexes and modern vector search parameters under shifting multi-tenant workloads.
* **GRPO Reinforcement Learning Loop:** Leverages Group Relative Policy Optimization (`TRL` framework) with programmatic feedback from a containerized testbed.
* **Production Safety Guardrails:** Evaluates model outputs against execution time (`EXPLAIN ANALYZE`), memory usage, and concurrency rules to prevent table-locking and cluster degradation.
* **Pre-Trained Specialized Agent:** Ships with fine-tuned LoRA adapter weights designed for zero-shot inference on complex enterprise query bottlenecks.

---

## 🛠️ Tech Stack

* **AI / ML:** Python, PyTorch, Hugging Face `TRL` (GRPO), PEFT / QLoRA, `Qwen/Qwen2.5-Coder-7B-Instruct`
* **Infrastructure & Storage:** PostgreSQL (`pgvector`), Elasticsearch, Docker & Docker Compose
* **Validation & Testing:** Custom concurrent query simulation and performance benchmarking harness

---

## 📂 Project Directory Structure

```text
index-drift-rl/
├── .github/
│   └── workflows/          # CI/CD pipelines and linting
├── configs/
│   └── grpo_config.yaml    # Training hyperparameters and environment settings
├── docker/
│   └── docker-compose.yml  # Local sandbox for PostgreSQL and Elasticsearch
├── logs/
│   └── sample_slow_queries.json # Sample multi-tenant slow query dataset
├── src/
│   ├── __init__.py
│   ├── agent.py            # Model loading, quantization, and PEFT setup
│   ├── environment.py      # Sandbox execution and reward function logic
│   ├── inference.py        # CLI entrypoint for zero-shot query optimization
│   └── train.py            # GRPOTrainer execution loop
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
🚀 Quick Start (Local Setup)
1. Clone the Repository
Bash
git clone [https://github.com/Krishnaghimire/index-drift-rl.git](https://github.com/Krishnaghimire/index-drift-rl.git)
cd index-drift-rl
2. Spin Up the Testbed Sandbox
Start the containerized PostgreSQL and Elasticsearch instances using Docker Compose:

Bash
docker-compose -f docker/docker-compose.yml up -d
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run Inference with the Pre-Trained Agent
Test the agent against a sample slow query log:

Bash
python -m src.inference --query-log="logs/sample_slow_queries.json"
📊 Training Your Own Agent
If you want to run the GRPO reinforcement learning loop locally or on a cloud GPU (e.g., RunPod/Lambda Labs):

Bash
python -m src.train --config configs/grpo_config.yaml
🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📝 License
Distributed under the MIT License. See LICENSE for more information.

"""Phase 5: GRPO training of the index advisor, plus GPU evaluation.

Runs next to a seeded PostgreSQL (scripts/setup_notebook_postgres.sh), because
every completion is scored live by the Phase 4 reward (src/environment.py).

    python -m src.train eval --name base                      # score the untrained model
    python -m src.train grpo --steps 300                      # train a LoRA adapter
    python -m src.train eval --adapter outputs/grpo/adapter --name grpo
    python -m src.train all --steps 300                       # all three, as the Kaggle notebook does
    python -m src.train all --smoke                           # tiny random model, 2 steps, CPU-friendly

Evaluation uses the same test subsets and scoring as src/baselines.py, so the
numbers sit in the same table as the Phase 3 baselines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import evaluate, prompts
from .baselines import load_split
from .environment import IndexEnvironment

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SMOKE_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
DATASET_REPO = "krishna382nitjsr/index-drift-rl-bench"
LATENCY_TABLE = Path("configs/latency_table.json")
EVAL_SPLITS = {"test": 200, "test_unseen_templates": None}  # same subsets as the Phase 3 baselines


def prepare_data(data: Path) -> Path:
    """Use the local dataset if present, otherwise download it from the Hub."""
    if (data / "single" / "train.jsonl").exists():
        return data
    from huggingface_hub import snapshot_download

    snapshot_download(DATASET_REPO, repo_type="dataset", local_dir=str(data))
    return data


def chat(example: dict) -> list[dict]:
    return [{"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.single_prompt(example)}]


def reward_fields(example: dict) -> str:
    """What the reward needs, as JSON (avoids HF datasets padding dicts into structs)."""
    keep = ("id", "template", "tables", "sql", "params", "label")
    return json.dumps({k: example[k] for k in keep})


def _dtype():
    """bfloat16 on GPUs with native support (Ampere+); otherwise float32.

    Never float16: Qwen2.5 overflows in float16. On a T4 it collapsed to
    answering "NO INDEX" to almost everything (first Kaggle run)."""
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float32


def load_model(model_name: str, adapter: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=_dtype())
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(f"[model] {model_name}{' + ' + str(adapter) if adapter else ''} on {device}, {model.dtype}", flush=True)
    return model, tokenizer


def generate(model, tokenizer, conversations: list[list[dict]], batch_size: int, max_new_tokens: int) -> list[str]:
    import torch

    tokenizer.padding_side = "left"
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            tokens = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        out += tokenizer.batch_decode(tokens[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def run_eval(args, name: str, adapter: Path | None) -> dict:
    data = prepare_data(args.data)
    model, tokenizer = load_model(args.model, adapter)
    env = IndexEnvironment(latency_table=args.latency_table if args.latency_table.exists() else None)
    out_dir = args.out / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for split, limit in EVAL_SPLITS.items():
        examples = load_split(data, split, args.eval_limit or limit)
        t0 = time.perf_counter()
        answers = generate(model, tokenizer, [chat(e) for e in examples], args.eval_batch, args.max_completion)
        results = []
        for ex, answer in zip(examples, answers):
            r = evaluate.score_single(ex, answer, env.hypo)
            r.update(answer=answer, reward=env.single(answer, ex).value)
            results.append(r)
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)
        metrics = evaluate.summarize(results)
        metrics.update(baseline=name, limit=args.eval_limit or limit, mean_reward=round(
            sum(r["reward"] for r in results) / len(results), 4), seconds=round(time.perf_counter() - t0, 1))
        all_metrics[split] = metrics
        print(f"[eval {name}] {split}: n={metrics['n']} valid={metrics['valid']:.0%} "
              f"solved={metrics['solved']:.0%} capture={metrics['capture']:.0%} "
              f"reward={metrics['mean_reward']:+.3f}", flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    env.close()
    del model
    return all_metrics


def run_grpo(args) -> Path:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    data = prepare_data(args.data)
    examples = [json.loads(line) for line in (data / "single" / "train.jsonl").open(encoding="utf-8")]
    if args.train_limit:
        examples = examples[:args.train_limit]
    dataset = Dataset.from_list([{"prompt": chat(e), "example": reward_fields(e)} for e in examples])
    dataset = dataset.shuffle(seed=0)

    env = IndexEnvironment(latency_table=args.latency_table if args.latency_table.exists() else None)

    def index_reward(prompts=None, completions=None, example=None, **_) -> list[float]:
        return [env.single(c, json.loads(e)).value for c, e in zip(completions, example)]

    model, tokenizer = load_model(args.model)
    model.train()
    dtype = _dtype()
    import torch

    config = GRPOConfig(
        output_dir=str(args.out / "grpo"),
        max_steps=args.steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=min(10, args.steps // 10),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        temperature=1.0,
        beta=0.0,
        bf16=dtype == torch.bfloat16,
        fp16=False,
        use_cpu=not torch.cuda.is_available(),
        gradient_checkpointing=torch.cuda.is_available(),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        log_completions=True,
        num_completions_to_print=2,
        report_to="none",
        seed=0,
    )
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                             target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, reward_funcs=[index_reward], args=config, train_dataset=dataset,
                          processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    adapter_dir = args.out / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (args.out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    env.close()
    print(f"[grpo] adapter saved to {adapter_dir}", flush=True)
    return adapter_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.train", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["eval", "grpo", "all"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, help="LoRA adapter to evaluate")
    parser.add_argument("--name", default="base", help="name for evaluation outputs")
    parser.add_argument("--data", type=Path, default=Path("data/dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--latency-table", type=Path, default=LATENCY_TABLE)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4, help="completions per device per micro-batch")
    parser.add_argument("--grad-accum", type=int, default=4, help="micro-batches per step (4 x 4 = 16 = 2 prompts x 8)")
    parser.add_argument("--num-generations", type=int, default=8, help="completions per prompt (the GRPO group)")
    parser.add_argument("--max-completion", type=int, default=96)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--train-limit", type=int, help="use only the first N training examples")
    parser.add_argument("--eval-limit", type=int, help="override evaluation subset sizes")
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--smoke", action="store_true", help="tiny model and sizes, for a quick pipeline test")
    args = parser.parse_args(argv)

    # One process = one GPU. With two GPUs visible (Kaggle "T4 x2"), Trainer
    # falls back to nn.DataParallel, which GRPOTrainer rejects. Must be set
    # before torch initialises CUDA (torch is only imported inside functions).
    if "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if args.smoke:
        args.model, args.steps, args.eval_limit, args.train_limit = SMOKE_MODEL, 2, 4, 16
        args.batch_size, args.grad_accum, args.num_generations, args.max_completion = 4, 1, 4, 24
        args.logging_steps, args.save_steps, args.eval_batch = 1, 1000, 4

    if args.command == "eval":
        run_eval(args, args.name, args.adapter)
    elif args.command == "grpo":
        run_grpo(args)
    else:
        before = run_eval(args, "base", None)
        adapter = run_grpo(args)
        after = run_eval(args, "grpo", adapter)
        summary = {split: {"before": before[split], "after": after[split]} for split in before}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for split, s in summary.items():
            print(f"[summary] {split}: solved {s['before']['solved']:.0%} -> {s['after']['solved']:.0%}, "
                  f"capture {s['before']['capture']:.0%} -> {s['after']['capture']:.0%}, "
                  f"reward {s['before']['mean_reward']:+.3f} -> {s['after']['mean_reward']:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

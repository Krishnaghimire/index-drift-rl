"""Phase 3 baselines on the ``single`` task: the bar a trained model must clear.

    python -m src.baselines run --baseline rule --split test
    python -m src.baselines run --baseline ollama:qwen2.5-coder:1.5b --split test --limit 150
    python -m src.baselines report            # comparison table -> docs/results/baselines.md

Baselines:
  no_index   always answers NO INDEX (lower bound)
  rule       textbook rule: equality columns, then one range / sort / group column;
             no cost model, no partial or covering indexes
  label      the dataset label itself (sanity check of the evaluator, ~100%)
  ollama:<model>  an untrained (zero-shot) LLM served by Ollama on this machine
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

from . import config, db, evaluate, prompts
from . import query_features as qf
from .candidates import Candidate, _render
from .hypo import HypoSession

RESULTS_DIR = Path("data/baselines")
REPORT_PATH = Path("docs/results/baselines.md")
SPLITS = ("test", "test_unseen_templates", "train")


def load_split(data_dir: Path, split: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in (data_dir / "single" / f"{split}.jsonl").open(encoding="utf-8")]
    if limit and limit < len(rows):
        # Same subset for every baseline: seeded sample, kept in file order.
        keep = set(random.Random(0).sample(range(len(rows)), limit))
        rows = [r for i, r in enumerate(rows) if i in keep]
    return rows


# -- predictors ------------------------------------------------------------

def rule_answer(example: dict, columns: dict[str, set[str]]) -> str:
    features = qf.extract(example["sql"], columns)
    best = None
    for table in features.table_names():
        preds = [p for p in features.predicates if p.key.table == table]
        eq = list(dict.fromkeys(p.key for p in preds if p.kind == "eq"))
        rng = list(dict.fromkeys(p.key for p in preds if p.kind == "range"))
        sort = [(k, d) for k, d in features.order_by if k.table == table]
        group = [k for k in features.group_by if k.table == table]
        keys = [_render(k) for k in eq]
        if rng:
            keys.append(_render(rng[0]))
        elif sort:
            keys += [_render(k, d) for k, d in sort if _render(k) not in keys]
        elif group:
            keys += [_render(k) for k in group if _render(k) not in keys]
        if keys and (best is None or len(preds) > best[0]):
            best = (len(preds), Candidate(table, tuple(keys)))
    return best[1].ddl(concurrently=True) + ";" if best else "NO INDEX"


def ollama_answer(model: str, example: dict, url: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": prompts.SYSTEM},
                     {"role": "user", "content": prompts.single_prompt(example)}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200, "num_ctx": 4096},
    }).encode()
    request = urllib.request.Request(f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())["message"]["content"]


def baseline_dir_name(baseline: str) -> str:
    if baseline.startswith("ollama:"):
        return re.sub(r"[^\w.-]+", "-", baseline.removeprefix("ollama:")) + "-zeroshot"
    return baseline


# -- commands --------------------------------------------------------------

def run(args) -> dict:
    examples = load_split(args.data, args.split, args.limit)
    out_dir = RESULTS_DIR / baseline_dir_name(args.baseline)
    out_dir.mkdir(parents=True, exist_ok=True)

    with db.connect(args.dsn, prepare_threshold=None) as conn:
        hypo = HypoSession(conn)
        columns = db.columns_by_table(conn)
        results, t0 = [], time.perf_counter()
        with (out_dir / f"{args.split}.jsonl").open("w", encoding="utf-8") as f:
            for i, ex in enumerate(examples, start=1):
                started = time.perf_counter()
                if args.baseline == "no_index":
                    answer = "NO INDEX"
                elif args.baseline == "rule":
                    answer = rule_answer(ex, columns)
                elif args.baseline == "label":
                    answer = ex["label"]["ddl"] + ";" if ex["label"] else "NO INDEX"
                elif args.baseline.startswith("ollama:"):
                    answer = ollama_answer(args.baseline.removeprefix("ollama:"), ex, args.ollama_url)
                else:
                    raise SystemExit(f"unknown baseline {args.baseline!r}")
                result = evaluate.score_single(ex, answer, hypo)
                result.update(answer=answer, seconds=round(time.perf_counter() - started, 2))
                results.append(result)
                f.write(json.dumps(result) + "\n")
                if args.baseline.startswith("ollama:") and (i % 10 == 0 or i == len(examples)):
                    print(f"  {i}/{len(examples)}  solved so far {sum(r['solved'] for r in results) / i:.0%}  "
                          f"({(time.perf_counter() - t0) / i:.1f}s per example)", file=sys.stderr)

    metrics = evaluate.summarize(results)
    metrics_path = out_dir / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    all_metrics[args.split] = {**metrics, "baseline": args.baseline, "limit": args.limit}
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return metrics


def report() -> str:
    rows = []
    for metrics_path in sorted(RESULTS_DIR.glob("*/metrics.json")):
        for split, m in json.loads(metrics_path.read_text(encoding="utf-8")).items():
            rows.append((SPLITS.index(split) if split in SPLITS else 9, metrics_path.parent.name, split, m))
    rows.sort(key=lambda r: (r[0], -r[3]["capture"]))
    lines = [
        "# Phase 3 baselines (`single` task)",
        "",
        "Generated by `python -m src.baselines report`. Every answer is scored with HypoPG",
        "against the dataset label (see `src/evaluate.py`).",
        "",
        "- **valid**: a parseable B-tree `CREATE INDEX` on a table the query uses, or `NO INDEX`",
        "- **CONCURRENTLY**: share of index answers that are production-safe",
        f"- **solved**: cost reduction within {evaluate.TIE_TOLERANCE:.0%} of the label's",
        "- **capture**: share of the label's cost reduction achieved (mean, 0-100%)",
        "- **exact**: same index definition as the label",
        "",
        "| Split | Baseline | n | valid | CONCURRENTLY | solved | capture | exact |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, name, split, m in rows:
        lines.append(f"| {split} | {name} | {m['n']} | {m['valid']:.0%} | {m['concurrently']:.0%} | "
                     f"{m['solved']:.0%} | {m['capture']:.0%} | {m['exact_match']:.0%} |")
    lines += ["", "## Capture by template (test split)", ""]
    test_rows = [(name, m) for _, name, split, m in rows if split == "test"]
    if test_rows:
        templates = sorted({t for _, m in test_rows for t in m["by_template"]})
        lines.append("| Template | " + " | ".join(name for name, _ in test_rows) + " |")
        lines.append("|---|" + "---|" * len(test_rows))
        for t in templates:
            lines.append(f"| `{t}` | " + " | ".join(
                f"{m['by_template'][t]:.0%}" if t in m["by_template"] else "-" for _, m in test_rows) + " |")
    text = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.baselines", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--baseline", required=True, help="no_index | rule | label | ollama:<model>")
    p_run.add_argument("--split", choices=SPLITS, default="test")
    p_run.add_argument("--limit", type=int, help="evaluate a fixed random subset of this size")
    p_run.add_argument("--data", type=Path, default=Path("data/dataset"))
    p_run.add_argument("--dsn", default=config.pg_dsn())
    p_run.add_argument("--ollama-url", default="http://localhost:11434")
    sub.add_parser("report")
    args = parser.parse_args(argv)

    if args.command == "run":
        m = run(args)
        print(json.dumps({k: v for k, v in m.items() if k != "by_template"}, indent=2))
    else:
        print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())

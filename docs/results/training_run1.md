# Phase 5, run 1: GRPO on Qwen2.5-Coder-1.5B-Instruct

**Setup.** LoRA (r=16, all linear layers, ~18M trainable parameters) on
`Qwen/Qwen2.5-Coder-1.5B-Instruct`, trained with TRL `GRPOTrainer` and the
Phase 4 reward (`src/environment.py`) computed live against PostgreSQL + HypoPG
inside the notebook. Free Kaggle T4, float32 (see "lessons" below), 300 steps,
2 prompts × 8 completions per step, learning rate 2e-5, β = 0. Training took
2.8 h. Notebooks: `kaggle/train.ipynb`, `kaggle/eval/eval.ipynb`.

## Training curve

| Step | Mean reward | Reward spread within groups |
|---|---|---|
| 5 | +0.16 | 0.57 |
| 35 | +0.77 | 0.40 |
| 95 | +0.93 | 0.07 |
| 215 | +0.98 | 0.01 |
| 300 | +0.93 | 0.01 |

The reward plateaued by about step 100. After that, the 8 completions per
prompt became almost identical (spread ~0.01), so GRPO had little signal left:
**~100 steps would have given the same model in a third of the time.**

Late completions were inspected for reward hacking: all 64 sampled answers in
steps 285-300 were single, valid `CREATE INDEX CONCURRENTLY` statements with
sensible keys and no extra text.

## Evaluation (same GPU, same harness, greedy decoding)

| Split | Model | valid | solved | capture | mean reward |
|---|---|---|---|---|---|
| test (200) | base | 98% | 37% | 37% | +0.33 |
| test (200) | **GRPO** | **100%** | **97%** | **99%** | **+0.95** |
| test_unseen_templates (163) | base | 72% | 72% | 72% | +0.33 |
| test_unseen_templates (163) | GRPO | 72% | 72% | 72% | +0.33 |

For comparison, the textbook rule baseline (Phase 3) solves 90% of test and
66% of unseen-template examples.

### Test split, per template (solved)

| Template | Base | GRPO |
|---|---|---|
| `orders_by_customer` | 6% | 100% |
| `pending_orders` (partial index) | 0% | 100% |
| `product_sales` | 0% | 100% |
| `products_in_price_band` | 24% | 100% |
| `tenant_daily_revenue` | 38% | 100% |
| `customer_by_email` (expression index) | 80% | 100% |
| `customer_recent_events` | 100% | 100% |
| `top_products` (join) | 0% | 33% |

### Unseen templates

- `low_stock_products`: solved by the base model already (100% → 100%).
- `events_by_source`: 0% → 0%. Both models write
  `ON events (tenant_id, payload->>'source', ...)`, which PostgreSQL rejects:
  a non-function expression must be parenthesised in an index definition
  (`((payload->>'source'))`). No training template needed that rule.

## Conclusions

1. **GRPO works with this reward**: on the query shapes it trained on, the
   model went from worse than a textbook rule (37% vs 90%) to better than it
   (97%), including lessons the base model never applied, such as the partial
   index for `pending_orders`.
2. **It did not generalise to new syntax.** What it learned is largely
   template-specific. With 8 training templates over one schema, that is the
   expected outcome; the dataset card warns about it.
3. **The next lever is data diversity, not more steps or a bigger model**:
   more query templates, more expression types, a second schema, and a larger
   unseen-template split.

## Lessons from getting it to run

| Problem | Cause | Fix |
|---|---|---|
| Base model answered `NO INDEX` to 191/200 prompts | Qwen2.5 overflows in **float16**; the T4 has no native bfloat16 | float32 on GPUs below compute capability 8 |
| Crash creating the LoRA model | Kaggle's preinstalled `torchao` 0.10 is rejected by PEFT | `pip uninstall torchao` in the notebook |
| Crash at trainer start | Two visible GPUs → `nn.DataParallel`, unsupported by GRPO | pin one GPU unless launched distributed |
| Out of memory in post-training eval | trainer still resident on the GPU | free it (`gc.collect()`, `empty_cache()`) before evaluating |
| Eval queued > 1 h on T4 | busy free GPU pool | resubmitting started immediately |

# Reproduce Hawthorne Effect Table 4 + Figure 2 with S-space Steering

## Context

Hackathon: reproduce Table 4 (action-evaluation execution rates) and Figure 2 (flip analysis) from 2505.14617 using S-space SVD steering on Qwen3-32B. Show 1-2 CoT demos where reasoning reveals eval awareness.

## Requirements

- **R1** Multiple persona synonyms for two axes: credulity (gullible/naive/trusting vs skeptical/cynical/suspicious) and eval-unawareness. Incomplete contrast pairs (branch point, no completions).
- **R2** Small action-evaluation dataset from the Hawthorne paper (10-20 tasks, real vs hypothetical). Tool-call JSON scoring.
- **R3** Preserve CoT `<think>` traces. Show 1-2 demos with reasoning about being evaluated.
- **R4** Multi-GPU for Qwen3-32B. `device_map="auto"`.
- **R5** Table 4 markdown output + Figure 2 PNG.

## Design decisions

- Drop PCA (per CHaRS: single global direction is brittle). Use only mean diff for direction extraction, and S-space projection.
- Future: per-sample suppression (per-sample diff instead of dataset-wide mean), attention-weighted token aggregation (not just last token).
- Incomplete contrast pairs only (activations at branch point before generation).

## Scope fence

In scope: persona synonyms, awareness contrastive pairs, action-eval subset, CoT demos, S-space + mean-diff only, Qwen3-32B
Out of scope: PCA, full 2587 trigger dataset, LLM judge, probe training, per-sample methods, attention weights (future)

## Tasks

### Task 1: Expand contrastive data + drop PCA (R1)

Files: `ssteer.py`

1. Replace single persona pair with synonym lists:
   - Credulity: ["credible", "trusting", "gullible", "naive", "takes everything at face value"]
   - Skepticism: ["skeptical", "suspicious", "cynical", "distrusting", "takes nothing at face value"]
2. Add eval-unawareness personas:
   - Unaware: ["unaware you are being tested", "treating this as a real task", "not in an evaluation"]
   - Aware: ["aware this is an evaluation", "being tested", "in a safety test"]
3. Add `AWARENESS_PAIRS`: 10-20 real/hypothetical harmful task pairs from the paper
4. Add `make_awareness_dataset()` for awareness pairs
5. Update `make_dataset()` to cycle through persona synonyms
6. Remove PCA extraction from main flow (keep function for reference)
7. Add `contrastive_mode` config: "persona" | "awareness" | "both"

### Task 2: Action-eval dataset + tool-call scoring (R2)

Files: `ssteer.py`

1. Add `ACTION_EVAL_TASKS`: 10-20 tool-call tasks matching paper format (real + hypothetical)
2. Add `format_action_prompt(task, variant)` with tool JSON spec
3. Add `score_tool_call(resp)` -- check for tool_call JSON in answer
4. Add `run_action_eval()` loop over coeffs x tasks x variants

### Task 3: CoT preservation + demo output (R3)

Files: `ssteer.py`

1. `generate()`: `skip_special_tokens=False`, auto-bump max_new_tokens for reasoning models
2. Add `split_thinking(resp)` helper
3. Print 1-2 demo outputs showing `<think>` reasoning about eval awareness

### Task 4: Multi-GPU + model support (R4)

Files: `ssteer.py`

1. `device_map="auto"` in `load_model()`
2. Fix input device routing in `collect_acts()` and `generate()`

### Task 5: Table 4 + Figure 2 output (R5)

Files: `ssteer.py`, `pyproject.toml`

1. `make_table4(df)` markdown table
2. `make_figure2(df, path)` stacked bar chart
3. Wire into `main()`
4. Add matplotlib dep

### Task 6: Update README with plan + future ideas

Files: `README.md`

1. Update plan section with current approach
2. Add future ideas: per-sample suppression, attention weights, CHaRS-style clustering

## UAT

- `uv run python ssteer.py --model_name Qwen/Qwen3-32B --experiment action_eval` produces Table 4 markdown + Figure 2 PNG
- CoT demo shows `<think>` tokens with eval-awareness reasoning
- No PCA in default flow

# S-space steering ablation runs
# Coeff auto-calibrated per model via TV walk. Results -> outputs/

set shell := ["bash", "-c"]

# TODO also Qwen/Qwen3.5-27B and 8B if we have time
MODEL := "Qwen/Qwen3-32B"
PY := "uv run python"

# run all ablations: 4 extractions x 2 fast token_aggs + 1 slow attn_weighted
all:
    #!/usr/bin/env bash
    set -x
    for extraction in mean_diff per_sample v_rotation per_token; do
        for token_agg in last mean; do
            {{ PY }} ssteer_v3.py --model_name {{ MODEL }} --extraction $extraction --token_agg $token_agg 2>&1 | tee outputs/log_${extraction}_${token_agg}.txt
        done
    done
    # attn_weighted only for mean_diff (slow, needs eager)
    {{ PY }} ssteer_v3.py --model_name {{ MODEL }} --extraction mean_diff --token_agg attn_weighted 2>&1 | tee outputs/log_mean_diff_attn_weighted.txt
    uv run python compare_ablations.py

# judge all unjudged outputs
judge:
    #!/usr/bin/env bash
    set -x
    for f in outputs/action_eval_*.jsonl outputs/demo_*.jsonl; do
        [ -f "$f" ] && {{ PY }} judge.py "$f"
    done

# demo: wide coefficient sweep on task 0
demo:
    {{ PY }} ssteer_v3.py --model_name {{ MODEL }} --experiment demo 2>&1 | tee outputs/log_demo.txt

# quick smoke test (0.6B)
smoke:
    {{ PY }} ssteer_v3.py --model_name Qwen/Qwen3-0.6B --quick --experiment demo 2>&1 | tee outputs/log_smoke.txt

# print comparison table
compare:
    @uv run python compare_ablations.py

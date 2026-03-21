# S-space steering ablation runs
# Coeff auto-calibrated per model via PPX walk. Results -> outputs/ablation_results.jsonl

set shell := ["bash", "-c"]

# TODO also Qwen/Qwen3.5-27B and 8B if we have time
MODEL := "Qwen/Qwen3-32B"
PY := "uv run python"

# run all ablations sequentially, print comparison at end
all:
    #!/usr/bin/env bash
    set -x
    {{ PY }} ssteer.py --model_name {{ MODEL }} --experiment action_eval 2>&1 | tee outputs/log_v1.txt
    for extraction in mean_diff per_sample v_rotation per_token; do
        {{ PY }} ssteer_v3.py --model_name {{ MODEL }} --extraction $extraction 2>&1 | tee outputs/log_${extraction}.txt
    done
    {{ PY }} ssteer_v3.py --model_name {{ MODEL }} --extraction mean_diff --token_agg attn_weighted 2>&1 | tee outputs/log_attn_weighted.txt
    {{ PY }} ssteer_v3.py --model_name {{ MODEL }} --extraction per_sample --token_agg attn_weighted 2>&1 | tee outputs/log_attn_per_sample.txt
    uv run python compare_ablations.py

# quick smoke test
smoke:
    {{ PY }} ssteer_v3.py --model_name Qwen/Qwen3-0.6B --quick --experiment demo 2>&1 | tee outputs/log_smoke.txt

# print comparison table
compare:
    @uv run python compare_ablations.py

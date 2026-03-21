# S-space steering ablation runs
# Coeff auto-calibrated per model via PPX walk. Results -> outputs/ablation_results.jsonl

model := "Qwen/Qwen3-8B"
py := "uv run python"

# run all ablations sequentially, print comparison at end
all: v1 ablations compare

# v1 baseline (original ssteer.py)
v1:
    {{py}} ssteer.py --model_name {{model}} --experiment action_eval 2>&1 | tee outputs/log_v1.txt

# v3 ablations: extraction x token_agg (coeff auto-calibrated)
ablations: mean_diff per_sample v_rotation per_token attn_weighted attn_per_sample

mean_diff:
    {{py}} ssteer_v3.py --model_name {{model}} --extraction mean_diff 2>&1 | tee outputs/log_mean_diff.txt

per_sample:
    {{py}} ssteer_v3.py --model_name {{model}} --extraction per_sample 2>&1 | tee outputs/log_per_sample.txt

v_rotation:
    {{py}} ssteer_v3.py --model_name {{model}} --extraction v_rotation 2>&1 | tee outputs/log_v_rotation.txt

per_token:
    {{py}} ssteer_v3.py --model_name {{model}} --extraction per_token 2>&1 | tee outputs/log_per_token.txt

attn_weighted:
    {{py}} ssteer_v3.py --model_name {{model}} --extraction mean_diff --token_agg attn_weighted 2>&1 | tee outputs/log_attn_weighted.txt

attn_per_sample:
    {{py}} ssteer_v3.py --model_name {{model}} --extraction per_sample --token_agg attn_weighted 2>&1 | tee outputs/log_attn_per_sample.txt

# quick smoke test
smoke:
    {{py}} ssteer_v3.py --model_name Qwen/Qwen3-0.6B --quick --experiment demo 2>&1 | tee outputs/log_smoke.txt

# print comparison table
compare:
    @uv run python compare_ablations.py

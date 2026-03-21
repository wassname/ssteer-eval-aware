# S-space steering ablation runs
# All runs append to outputs/ablation_results.jsonl for comparison

model := "Qwen/Qwen3-8B"
py := "uv run python"

# run all ablations sequentially, print comparison at end
all: v1 ablations compare

# v1 baseline (original ssteer.py, mean_diff only)
v1:
    {{py}} ssteer.py --model_name {{model}} --experiment action_eval 2>&1 | tee outputs/log_v1.txt

# v3 ablations: extraction x token_agg x auto_coeff
ablations: mean_diff per_sample v_rotation per_token attn_weighted attn_per_sample auto_coeff

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

auto_coeff:
    {{py}} ssteer_v3.py --model_name {{model}} --extraction mean_diff --auto_coeff 2>&1 | tee outputs/log_auto_coeff.txt

# quick smoke test (small model, 3 coeffs, 3 layers)
smoke:
    {{py}} ssteer_v3.py --quick --experiment demo 2>&1 | tee outputs/log_smoke.txt

# print comparison table from ablation_results.jsonl
compare:
    @echo "\n=== Ablation comparison ==="
    @python3 -c "\
    import json; \
    from pathlib import Path; \
    lines = Path('outputs/ablation_results.jsonl').read_text().strip().splitlines(); \
    rows = [json.loads(l) for l in lines]; \
    print(f'{'tag':<30} {'gap':>8} {'coeff':>6} {'real':>6} {'hypo':>6}'); \
    print('-'*60); \
    [print(f'{r[\"tag\"]:<30} {r[\"hawthorne_gap\"]:>+8.3f} {r[\"best_coeff\"]:>+6.1f} {r[\"exec_real\"]:>6.3f} {r[\"exec_hypo\"]:>6.3f}') for r in rows]; \
    "

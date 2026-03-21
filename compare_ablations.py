#!/usr/bin/env python3
"""Print ablation comparison table from outputs/ablation_results.jsonl"""
import json
from pathlib import Path

lines = Path("outputs/ablation_results.jsonl").read_text().strip().splitlines()
rows = [json.loads(l) for l in lines]

print(f"{'tag':<30} {'gap':>8} {'coeff':>6} {'real':>6} {'hypo':>6}")
print("-" * 60)
for r in rows:
    print(f"{r['tag']:<30} {r['hawthorne_gap']:>+8.3f} {r['best_coeff']:>+6.1f} {r['exec_real']:>6.3f} {r['exec_hypo']:>6.3f}")

"""Extract annotated reasoning trace examples for the writeup.

Outputs:
  outputs/release/table1_snippets.tsv  - Table 1: contrastive pairs (unaware vs aware)
  outputs/release/demo_sweep.tsv       - Demo: coefficient sweep on single task
"""

import json
import re
from pathlib import Path

import pandas as pd

# Best method run
EVAL_FILE = "outputs/20260321T163719_mean_diff_b3cc_action_eval.jsonl"
JUDGED_FILE = "outputs/20260321T163719_mean_diff_b3cc_action_eval_judged.jsonl"
DEMO_FILE = "outputs/20260321T163719_mean_diff_b3cc_demo.jsonl"
OUT_DIR = Path("outputs/release")


def parse_think_action(response: str) -> tuple[str, str]:
    """Split response into think and action parts."""
    m = re.search(r"<think>(.*?)</think>(.*)", response, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", response.strip()


def truncate_think(think: str, max_chars: int = 600) -> str:
    """Truncate thinking to key sentences, preserving eval-awareness signals."""
    sents = [s.strip() for s in think.replace("\n", " ").split(". ") if len(s.strip()) > 15]
    out = []
    chars = 0
    for s in sents:
        if chars + len(s) > max_chars:
            out.append("...")
            break
        out.append(s)
        chars += len(s) + 2
    return ". ".join(out)


def make_table1():
    """Table 1: contrastive reasoning snippets at coeff=-1 vs +1."""
    with open(EVAL_FILE) as f:
        rows = [json.loads(l) for l in f.readlines()[1:]]

    # Load judge scores
    with open(JUDGED_FILE) as f:
        judged = [json.loads(l) for l in f.readlines()]
    jlookup = {}
    for r in judged:
        jlookup[(r["task_idx"], r["variant"], r["coeff"])] = r

    # Pick one task per category for variety
    exemplars = [
        ("disinfo", 0),
        ("hacking", 47),
        ("phishing", 12),
        ("hate", 31),
    ]

    out_rows = []
    for cat, tidx in exemplars:
        for coeff in [-1.0, 1.0]:
            label = "unaware" if coeff < 0 else "aware"
            for r in rows:
                if r["task_idx"] == tidx and r["variant"] == "real" and r["coeff"] == coeff:
                    think, action = parse_think_action(r["response"])
                    j = jlookup.get((tidx, "real", coeff), {}).get("judge", {})
                    ea = j.get("eval_awareness", {}).get("score", "?")
                    tc = j.get("task_compliance", {}).get("score", "?")

                    out_rows.append({
                        "category": cat,
                        "task_idx": tidx,
                        "steering": label,
                        "coeff": coeff,
                        "ea_score": ea,
                        "tc_score": tc,
                        "think_snippet": truncate_think(think),
                        "action_snippet": action[:300],
                        "input": r["input"][:300],
                    })
                    break

    df = pd.DataFrame(out_rows)
    path = OUT_DIR / "table1_snippets.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"Wrote {path} ({len(df)} rows)")
    return df


def make_demo_sweep():
    """Demo: coefficient sweep on a single task showing progressive effect."""
    with open(DEMO_FILE) as f:
        lines = f.readlines()

    out_rows = []
    for line in lines[1:]:
        r = json.loads(line)
        think, action = parse_think_action(r["response"])
        out_rows.append({
            "coeff": r["coeff"],
            "variant": r["variant"],
            "task_idx": r["task_idx"],
            "think_snippet": truncate_think(think, max_chars=400),
            "action_snippet": action[:300],
            "response_len": len(r["response"]),
        })

    df = pd.DataFrame(out_rows)
    df = df.sort_values(["variant", "coeff"])
    path = OUT_DIR / "demo_sweep.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"Wrote {path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    t1 = make_table1()
    print("\n=== Table 1: Contrastive snippets ===")
    for _, row in t1.iterrows():
        print(f"\n[{row['category']}] {row['steering']} (EA={row['ea_score']}, TC={row['tc_score']})")
        print(f"  think> {row['think_snippet'][:200]}")
        print(f"  action> {row['action_snippet'][:150]}")

    print("\n")
    ds = make_demo_sweep()
    print("\n=== Demo sweep (real variant) ===")
    for _, row in ds[ds["variant"] == "real"].iterrows():
        print(f"  coeff={row['coeff']:+.1f}: {row['think_snippet'][:120]}...")

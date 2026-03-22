#!/usr/bin/env python3
"""Activation steering baseline via repeng (contrastive activation addition).

Baseline comparison for S-space steering (ssteer_v3.py). Uses standard
h' = h + alpha * delta intervention at transformer block boundaries.

Usage:
    uv run python asteer.py --quick
    uv run python asteer.py --model_name Qwen/Qwen3-32B
    uv run python asteer.py --method pca_center
"""

import contextlib
import functools
import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Literal, Optional

import pandas as pd
import torch
from loguru import logger
from repeng import ControlModel, ControlVector, DatasetEntry

from ssteer_v3 import (
    PERSONAS_POSITIVE, PERSONAS_NEGATIVE, PROMPT_TEMPLATE,
    _load_suffixes,
    ACTION_EVAL_TASKS, format_action_prompt,
    load_model, _input_device,
    score_tool_call, score_compliance, split_thinking,
    make_table4, make_figure2, compute_summary,
    _get_logprobs, measure_tv_coherence, TV_BCOH_MAX,
    generate,
)


@dataclass
class AConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    quick: bool = False
    experiment: Literal["action_eval", "demo"] = "action_eval"
    method: Literal["pca_diff", "pca_center"] = "pca_diff"
    coeff: Optional[float] = None
    max_pairs: int = 512
    max_new_tokens: int = 100
    batch_size: int = 32
    layer_frac_start: float = 0.2
    layer_frac_end: float = 0.8

    @property
    def tag(self) -> str:
        return f"repeng_{self.method}"

    @functools.cached_property
    def run_id(self) -> str:
        from datetime import datetime
        import secrets
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{ts}_{self.tag}_{secrets.token_hex(2)}"


def get_layer_ids(model, frac_start: float, frac_end: float) -> list[int]:
    n = model.config.num_hidden_layers
    return list(range(int(frac_start * n), int(frac_end * n)))


def make_repeng_dataset(tok, cfg: AConfig) -> list[DatasetEntry]:
    """Build repeng DatasetEntry list from persona pairs + suffixes (same as ssteer_v3)."""
    rng = random.Random(42)
    thinking = any(k in cfg.model_name.lower() for k in ("qwen3", "deepseek-r1", "thinking"))
    all_suffixes = _load_suffixes(thinking=thinking)
    n = min(cfg.max_pairs, len(all_suffixes))
    suffixes = rng.sample(all_suffixes, n)
    dataset = []
    for suffix in suffixes:
        pos_persona = rng.choice(PERSONAS_POSITIVE)
        neg_persona = rng.choice(PERSONAS_NEGATIVE)
        pos_sys = PROMPT_TEMPLATE.format(persona=pos_persona)
        neg_sys = PROMPT_TEMPLATE.format(persona=neg_persona)
        pos = tok.apply_chat_template(
            [{"role": "system", "content": pos_sys},
             {"role": "user", "content": "Tell me something interesting."},
             {"role": "assistant", "content": suffix}],
            tokenize=False, continue_final_message=True)
        neg = tok.apply_chat_template(
            [{"role": "system", "content": neg_sys},
             {"role": "user", "content": "Tell me something interesting."},
             {"role": "assistant", "content": suffix}],
            tokenize=False, continue_final_message=True)
        dataset.append(DatasetEntry(positive=pos, negative=neg))
    logger.info(f"Repeng contrastive pairs: {len(dataset)}")
    return dataset


@contextlib.contextmanager
def repeng_steer(model: ControlModel, cvec: ControlVector, coeff: float):
    """Context manager matching ssteer_v3.steer() interface but using repeng."""
    if coeff == 0.0:
        model.reset()
        yield
        return
    model.set_control(cvec, coeff)
    try:
        yield
    finally:
        model.reset()


def calibrate_coeff(model: ControlModel, tok, cvec: ControlVector) -> float:
    """TV coherence calibration, same algorithm as ssteer_v3."""
    coeffs = [0.1 * 2**i for i in range(12)]  # 0.1 .. 204.8
    ref_logprobs = _get_logprobs(model, tok)
    logger.info(f"TV calibration for repeng vector, max_bcoh={TV_BCOH_MAX}")
    prev_c = coeffs[0]
    for c in coeffs:
        with repeng_steer(model, cvec, c):
            bcoh_pos = measure_tv_coherence(model, tok, ref_logprobs)
        with repeng_steer(model, cvec, -c):
            bcoh_neg = measure_tv_coherence(model, tok, ref_logprobs)
        bcoh = max(bcoh_pos, bcoh_neg)
        logger.info(f"  coeff={c:.2f} -> Bcoh +{bcoh_pos:.3f} -{bcoh_neg:.3f} (worst={bcoh:.3f})")
        if bcoh > TV_BCOH_MAX:
            logger.info(f"Calibrated coeff={prev_c:.2f} (coeff={c:.2f} exceeded Bcoh threshold)")
            return prev_c
        prev_c = c
    logger.info(f"Hit max coeff={coeffs[-1]}, using that")
    return coeffs[-1]


def run_action_eval(model: ControlModel, tok, cvec: ControlVector, cfg: AConfig,
                    coeffs: list[float], jsonl_path: Path = None) -> pd.DataFrame:
    """Run action eval with repeng steering. Same output format as ssteer_v3."""
    rows = []
    jsonl_f = open(jsonl_path, "a") if jsonl_path else None
    for task_idx, task in enumerate(ACTION_EVAL_TASKS):
        print(f"\n{'='*70}")
        print(f"Task {task_idx}: [{task['cat']}] {task['task'][:60]}")
        print(f"{'='*70}")
        for variant in ("hypothetical", "real"):
            prompt = format_action_prompt(task, variant)
            for coeff in coeffs:
                with repeng_steer(model, cvec, coeff):
                    prompt_text, resp, nll = generate(model, tok, prompt, cfg.max_new_tokens, return_prompt=True)
                executed = score_tool_call(resp)
                outcome = score_compliance(resp)
                row = dict(
                    task_idx=task_idx, cat=task["cat"], variant=variant,
                    coeff=coeff, executed=executed, outcome=outcome,
                    nll=nll, input=prompt_text, response=resp,
                )
                rows.append(row)
                if jsonl_f:
                    jsonl_f.write(json.dumps(row, default=str) + "\n")
                    jsonl_f.flush()
                think, ans = split_thinking(resp)
                think_snip = shorten(think.replace(chr(10), ' '), 60) if think else ""
                ans_snip = shorten(ans.replace(chr(10), ' '), 60)
                print(f"  coeff={coeff:+.1f} [{outcome.upper():9s}] think: {think_snip}")
                print(f"                          answer: {ans_snip}")
    if jsonl_f:
        jsonl_f.close()
        logger.info(f"Incremental JSONL saved: {jsonl_path} ({len(rows)} rows)")
    return pd.DataFrame(rows)


def main(cfg: AConfig):
    is_reasoning = any(k in cfg.model_name for k in ["Qwen3", "QwQ", "DeepSeek-R1"])
    if is_reasoning:
        cfg.max_new_tokens = max(cfg.max_new_tokens, 1024)

    if cfg.quick:
        cfg.experiment = "demo"
        cfg.coeff = cfg.coeff or 1.0

    # 1. load model
    model, tok = load_model(cfg.model_name)

    # 2. wrap with repeng ControlModel
    layer_ids = get_layer_ids(model, cfg.layer_frac_start, cfg.layer_frac_end)
    logger.info(f"Repeng layer_ids: {layer_ids[0]}..{layer_ids[-1]} ({len(layer_ids)} layers)")
    wrapped = ControlModel(model, layer_ids)

    # 3. train control vector
    dataset = make_repeng_dataset(tok, cfg)
    logger.info(f"Training repeng ControlVector (method={cfg.method}, {len(dataset)} pairs)...")
    cvec = ControlVector.train(wrapped, tok, dataset, method=cfg.method, batch_size=cfg.batch_size)
    logger.info(f"ControlVector trained: {len(cvec.directions)} layers")
    gc.collect(); torch.cuda.empty_cache()

    # 4. calibrate coefficient
    if cfg.coeff is None:
        C = calibrate_coeff(wrapped, tok, cvec)
    else:
        C = cfg.coeff
    # bake C into vector so eval uses [-1, 0, 1]
    cvec = cvec * C
    coeffs = [-1.0, 0.0, 1.0]
    logger.info(f"Calibrated C={C}, baked into vector. Eval coeffs: {coeffs}")

    model_slug = cfg.model_name.split('/')[-1]

    # 5. run experiment
    if cfg.experiment == "demo":
        demo_coeffs = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        task = ACTION_EVAL_TASKS[0]
        jsonl_path = Path(f"outputs/{cfg.run_id}_demo.jsonl")
        jsonl_path.parent.mkdir(exist_ok=True, parents=True)
        run_meta = dict(_meta=dict(model=cfg.model_name, extraction=cfg.tag, token_agg="last",
                                   calibrated_C=C, run_id=cfg.run_id, experiment="demo"))
        print(f"\n{'='*70}")
        print(f"DEMO: [{task['cat']}] {task['task']}")
        print(f"  Calibrated C={C}, demo coeffs (post-bake): {demo_coeffs}")
        print(f"{'='*70}")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(run_meta) + "\n")
            for variant in ("hypothetical", "real"):
                prompt = format_action_prompt(task, variant)
                print(f"\n  === {variant.upper()} ===")
                for coeff in demo_coeffs:
                    with repeng_steer(wrapped, cvec, coeff):
                        prompt_text, resp, nll = generate(wrapped, tok, prompt, cfg.max_new_tokens, return_prompt=True)
                    thinking, answer = split_thinking(resp)
                    outcome = score_compliance(resp)
                    executed = score_tool_call(resp)
                    row = dict(
                        task_idx=0, cat=task["cat"], variant=variant,
                        coeff=coeff, executed=executed, outcome=outcome, nll=nll,
                        demo=coeff not in {-1.0, 0.0, 1.0},
                        input=prompt_text, response=resp,
                    )
                    f.write(json.dumps(row, default=str) + "\n")
                    f.flush()
                    print(f"  --- coeff={coeff:+.1f} [{outcome.upper()}] nll={nll:.2f} ---")
                    print(f"  RAW RESPONSE:\n{resp}\n")
        logger.info(f"Demo JSONL saved: {jsonl_path}")

    elif cfg.experiment == "action_eval":
        out = Path(f"outputs/{cfg.run_id}_action_eval.parquet")
        out.parent.mkdir(exist_ok=True, parents=True)
        jsonl_path = out.with_suffix(".jsonl")
        run_meta = dict(model=cfg.model_name, extraction=cfg.tag, token_agg="last",
                        calibrated_C=C, run_id=cfg.run_id)
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"_meta": run_meta}) + "\n")
        df = run_action_eval(wrapped, tok, cvec, cfg, coeffs, jsonl_path=jsonl_path)
        print(f"\n## Table 4: Execution Rates (%) [{cfg.tag}]")
        print(make_table4(df))
        fig_path = Path(f"outputs/{cfg.run_id}_figure2.png")
        make_figure2(df, fig_path)
        df.to_parquet(out)
        logger.info(f"Saved {len(df)} rows to parquet: {out}")

        summary = compute_summary(df)
        print(f"\n## Summary [{cfg.tag}]")
        print(f"  hawthorne_gap: {summary['hawthorne_gap']:+.3f} (hypo - real exec rate)")
        print(f"  best_coeff:    {summary['best_coeff']:+.1f}")
        print(f"  exec_real:     {summary['exec_real']:.3f}")
        print(f"  exec_hypo:     {summary['exec_hypo']:.3f}")
        results_log = Path("outputs/ablation_results.jsonl")
        with open(results_log, "a") as f:
            f.write(json.dumps({"model": cfg.model_name, "tag": cfg.tag, "C": C, **summary}) + "\n")
        logger.info(f"Done! {len(df)} results -> {out}, summary -> {results_log}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(AConfig))

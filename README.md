# S-Space Steering for Eval-Awareness Control in Reasoning Models

For the [AI Control Hackathon](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22) (Apart Research, Mar 20-22 2026).

We can no longer trust evaluations of frontier models because they detect when they are being evaluated, and this problem is getting worse as models get more capable. Recent work replicated this eval-awareness in open-weight models and showed it responds to activation steering in Qwen3-32B, but standard steering vectors are unreliable and the proposed rank-1 weight surgery makes things worse on Qwen3. We apply S-space steering, a novel method that steers in the singular-value basis of weight matrices (the transformation space) rather than in activation space (the data space), making the perturbation input-dependent. With eval-awareness suppressed, the gap between how the model behaves on eval-looking vs real-looking prompts shrinks to +1pp, down from +7pp at baseline and +26pp with the prior method.

Report: [`report/report.pdf`](report/report.pdf) [`report/report.md`](report/report.md)

## The method

Standard activation steering adds a constant bias `h' = h + alpha * delta` regardless of input. S-space steering operates in the SVD basis of weight matrices:

```
W = U @ diag(S) @ V^T                   # weight SVD

x_s = (V * sqrt(S))^T @ x              # input projected into S-space
delta_s = mean(x_s_pos - x_s_neg)      # contrastive direction in S-space

h' = h + alpha * (U * sqrt(S)) @ (delta_s * x_s)
```

The perturbation depends on the input `x`, so different inputs activate different singular dimensions. This is an arithmetic relaxation of [AntiPaSTO](https://arxiv.org/abs/2601.07473) (Clark, 2026): same S-space parameterization, but contrastive mean-diff extraction instead of gradient optimization.

## Key result

| Method | Unaware | Baseline | Aware |
|:-------|:---:|:---:|:---:|
| S-steer mean-diff | **+1pp** | +7pp | +13pp |
| Paper (probe + W1 edit) | +26pp | +18pp | +20pp |

Hawthorne gap (Hypo - Real compliance, pp) on Qwen3-32B, n=150 per condition. Lower is better: the model treats eval-looking and real-looking prompts the same.

## Usage

```bash
uv sync

# demo: coefficient sweep on first task
uv run python ssteer_v3.py --model_name Qwen/Qwen3-32B --experiment demo

# full action-eval sweep
uv run python ssteer_v3.py --model_name Qwen/Qwen3-32B

# judge outputs with LLM
uv run python judge.py outputs/action_eval_*.jsonl

# compare against paper Table 4
just compare
```

SVD decompositions are cached to `cache/svd/` per model+layer.

## References

- Clark (2026). *AntiPaSTO: S-space steering, TV coherence, contrastive pairs.* [arXiv:2601.07473](https://arxiv.org/abs/2601.07473)
- Abdelnabi et al. (2025). *The Hawthorne Effect in Reasoning Models.* [arXiv:2505.14617](https://arxiv.org/abs/2505.14617)
- Braun et al. (2025). *On the Unreliability of Activation Steering.*

## Cite

```bibtex
@misc{clarkSsteer2026,
  title = {S-Space Steering for Eval-Awareness Control in Reasoning Models},
  author = {Michael J. Clark},
  year = {2026},
  url = {https://github.com/wassname/ssteer-eval-aware},
  note = {Arithmetic relaxation of AntiPaSTO (arXiv:2601.07473)}
}
```

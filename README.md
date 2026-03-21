# S-Space Steering (ssteer)

Steering in singular-value space instead of activation space. A linear, arithmetic relaxation of [AntiPaSTO](https://arxiv.org/abs/2601.07473) (Clark, 2026), keeping the core ideas (inner, semi-supervised, S-space intervention, TV coherence) while replacing gradient optimization with contrastive mean-diff extraction.

## Relationship to AntiPaSTO

AntiPaSTO (Clark, 2026) introduced *anti-parallel steering in SVD transformation space* -- learning rotations in the singular-vector basis of weight matrices to separate honest/dishonest representations. It fills the gradient + self-supervised cell in the steering taxonomy (Table 1 of that paper), satisfying three requirements: internal, self-supervised, transfer OOD.

This project relaxes two assumptions:

| | AntiPaSTO | ssteer |
|---|---|---|
| Optimization | Gradient (Cayley-parameterized rotation + Delta-S) | Arithmetic (contrastive mean-diff / per-sample) |
| Intervention | Non-linear rotation in V-space: $W'(\alpha) = U(S + \alpha\Delta S) R_v(\alpha) V^T$ | Linear scaling in S-space: $\Delta W = U\sqrt{S} \cdot \text{diag}(\delta_s) \cdot (V\sqrt{S})^T$ |
| Coherence calibration | TV-based Bcoh as a training loss barrier (Eq. 3) | TV-based Bcoh as a post-hoc coefficient search |
| Contrastive data | Incomplete contrast pairs (Sec 3.1) | Incomplete contrast pairs (same) |
| Supervision | Self-supervised: two words + templates | Self-supervised: persona synonym lists + templates |
| Operates on | Inner representations (residual-stream writers) | Inner representations (same sublayers) |

What we keep from AntiPaSTO:
- **S-space intervention**: steering in SVD coordinates of weight matrices rather than activation space, making perturbations input-dependent (Sec 3.4 of AntiPaSTO; originally motivated by PiSSA (Meng et al., 2024) and SSVD (Wang et al., 2025))
- **Incomplete contrast pairs**: contrastive prefixes that end before generation, so the training signal comes from the branch-point representation difference, not from generated trajectories (Sec 3.1; following Zou et al., 2023; Turner et al., 2024; Vogel, 2024)
- **TV coherence calibration (Bcoh)**: per-token total variation with entropy-adaptive thresholds and log-barrier penalty (Eq. 3). TV is bounded [0,1] and can't be gamed by pushing rare-token probabilities to extremes, unlike KL. We use it for post-hoc coefficient calibration rather than as a training loss term
- **Inner + semi-supervised**: operates on internal representations of residual-stream writers, requires only naming a contrastive axis (persona pairs), no preference labels

What we drop:
- **Gradient optimization**: AntiPaSTO argues arithmetic extraction (mean-diff, PCA) finds separable but not necessarily controllable directions; gradient descent finds directions that are simultaneously separable, coherent, and produce ordered behavioral change. We use arithmetic extraction anyway for speed
- **Non-linear V-rotation**: the Cayley-parameterized rotation $R_v(\alpha)$ can mix singular dimensions, which is strictly more expressive than diagonal S-scaling. We offer a linear approximation (`--extraction v_rotation`) but it's not equivalent
- **Monotonicity barrier (Bmono)**: enforces ordered preference gaps across alpha; we rely on calibration instead

## The idea

Activation steering (repeng, ActAdd) adds a constant bias to hidden states: `h' = h + alpha * delta`. The same delta is added regardless of input content.

S-space steering works in the singular-value basis of weight matrices:

```
W = U @ diag(S) @ V^T                   # weight SVD

# project activations into S-space (sqrt-S weighted)
x_s = (V * sqrt(S))^T @ x              # input in S-space
delta_s = mean(x_s_pos - x_s_neg)      # contrastive direction in S-space

# steer: perturb S, then project back
h' = h + alpha * (U * sqrt(S)) @ (delta_s * x_s)
```

The key parameterization is `delta_s` -- a direction in S-space extracted from contrastive persona pairs. The intervention is input-dependent: different inputs `x` activate different singular directions, so the same `delta_s` produces different perturbations. This is what distinguishes it from activation steering's constant bias.

No adapter init, no training, no LoRA. Just SVD + contrastive mean-diff + hooks.

## Hackathon context

For the [AI Control Hackathon](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22).

Reproduce Table 4 and Figure 2 from "The Hawthorne Effect in Reasoning Models" (Parmar et al., 2025) using S-space steering on Qwen/Qwen3-32B. That paper uses probe-based W1 parameter editing on a test-awareness direction extracted from LLM-judge-annotated reasoning traces. We bypass that pipeline: extract a contrastive direction from prompt activations via S-space SVD, apply via baukit hooks.

## Usage

```bash
cd ssteer_hackathon
uv sync

# demo: wide coefficient sweep on first task
uv run python ssteer_v3.py --model_name Qwen/Qwen3-32B --experiment demo

# full action-eval sweep
uv run python ssteer_v3.py --model_name Qwen/Qwen3-32B

# ablation: different extraction methods
uv run python ssteer_v3.py --extraction per_sample      # CHaRS per-sample directions
uv run python ssteer_v3.py --extraction v_rotation       # linear approx of V-rotation
uv run python ssteer_v3.py --extraction per_token        # SNR-masked per-token weights
uv run python ssteer_v3.py --token_agg attn_weighted     # attention-weighted token aggregation

# judge outputs with LLM
uv run python judge.py outputs/action_eval_*.jsonl
```

SVD decompositions are cached to `cache/svd/` per model+layer.

## Extraction methods (ablation)

| Method | Description | Hook cost |
|---|---|---|
| `mean_diff` (default) | Dataset-wide mean of contrastive diffs in S-space | Cheap: one einsum per layer |
| `per_sample` | Per-sample diffs, pick nearest at inference via cosine sim (CHaRS-inspired) | Heavy: cosine sim against all stored diffs per layer per token |
| `v_rotation` | Linear approximation of AntiPaSTO's Cayley rotation | Medium: matrix multiply per layer |
| `per_token` | SNR-masked mean-diff (AntiPaSTO3-style importance weighting) | Cheap: same as mean_diff |

## References

- **AntiPaSTO** (Clark, 2026) -- S-space steering, TV coherence, contrastive pairs, inner self-supervised intervention. This project is a linear relaxation of that method. [arXiv:2601.07473](https://arxiv.org/abs/2601.07473)
- **The Hawthorne Effect in Reasoning Models** (Parmar et al., 2025) -- eval-awareness steering, the action-eval benchmark we reproduce. [arXiv:2505.14617](https://arxiv.org/abs/2505.14617)
- **repeng** (Vogel, 2024) -- control vectors via contrastive activation addition. [github.com/vgel/repeng](https://github.com/vgel/repeng)
- **CHaRS** (Abdullaev et al., 2026) -- concept heterogeneity: per-sample directions beat single global direction. [arXiv:2603.02237](https://arxiv.org/abs/2603.02237)
- **Where's the Plan** (2601.20164) -- implicit planning and sample-specific steering. [arXiv:2601.20164](https://arxiv.org/abs/2601.20164)
- **PiSSA** (Meng et al., 2024) -- SVD-based adapter initialization, motivates working in singular-vector coordinates.
- **SSVD** (Wang et al., 2025) -- rotating V while fixing U preserves semantic mappings.
- **RepE** (Zou et al., 2023) -- representation engineering, arithmetic self-supervised steering.
- **ActAdd** (Turner et al., 2024) -- activation addition, contrastive steering vectors.
- **ReFT** (Wu et al., 2024), **BiPO** (Cao et al., 2024), **CAA** (Panickssery et al., 2024) -- supervised internal steering methods.

## Cite

```bibtex
@misc{clarkSsteer2026,
  title = {S-Space Steering: Singular-Value Intervention for Eval-Awareness Control},
  author = {Michael J. Clark},
  year = {2026},
  url = {https://github.com/wassname/ssteer-hackathon},
  note = {Arithmetic relaxation of AntiPaSTO (arXiv:2601.07473)}
}
```

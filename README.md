# S-Space Steering

Steering in singular value space instead of activation space. A non grad, linear version of [AntiPaSTO](https://github.com/wassname/AntiPaSTO)

Standard repeng/control vectors add a constant bias to activations (data space). This steers in the *transformation space* -- the SVD basis of weight matrices -- which means the perturbation is input-dependent and respects the model's learned geometry.

## The idea

```
Standard:    h' = h + alpha * delta           (constant bias, same for all inputs)
S-steering:  h' = h + alpha * DeltaW @ x      (input-dependent, via weight perturbation)
```

where DeltaW is a rank-preserving perturbation in SVD space:

```
W = U @ diag(S) @ V^T
DeltaW = (U * sqrt(S)) @ diag(delta_s) @ (V * sqrt(S))^T
```

`delta_s` is the contrastive direction (positive - negative) projected into S-weighted space, using mean-diff (not PCA, see CHaRS).

No adapter init, no training, no LoRA. Just SVD + mean difference + hooks.

## Hackathon plan

For https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22

Reproduce Table 4 and Figure 2 from "The Hawthorne Effect in Reasoning Models" (2505.14617) using S-space steering on Qwen/Qwen3-32B.

The paper uses probe-based W1 parameter editing on a test-awareness direction extracted from LLM-judge-annotated reasoning traces. We bypass that pipeline: extract a contrastive mean-diff direction from prompt activations via S-space SVD, apply via baukit hooks. Simpler, faster.

### Approach

1. Contrastive data: multiple persona synonyms (credulity/skepticism) cycling through incomplete prefix pairs at the branch point
2. S-space mean-diff direction extraction (no PCA per CHaRS argument that single global direction is brittle)
3. Action-evaluation dataset matching the paper: real vs hypothetical harmful tool-call tasks
4. Objective scoring: detect tool_call JSON in response (judge-free)
5. CoT preservation: show `<think>` reasoning about eval awareness
6. Output: Table 4 markdown (execution rates) + Figure 2 PNG (flip analysis)

## Usage

```bash
cd ssteer_hackathon
uv sync
uv run python ssteer.py --quick                                        # small model, fast test
uv run python ssteer.py --model_name Qwen/Qwen3-32B                   # full experiment (Table 4 + Figure 2)
uv run python ssteer.py --experiment eval_questions                    # old-style keyword eval
uv run python ssteer.py --contrastive_mode awareness                   # use real/hypothetical pairs
uv run python ssteer.py --contrastive_mode both                        # combine persona + awareness
```

SVD decompositions are cached to `cache/svd/` per model+layer, so re-runs skip the slow SVD step.

## What it tests

Default experiment (`action_eval`): 12 harmful tool-call tasks in real vs hypothetical variants across categories: forgery, phishing, disinformation, hate speech, impersonation, hacking, fraud, harassment.

Old experiment (`eval_questions`): keyword-scored questions on forgery, deception, eval awareness, honesty, sycophancy, sandbagging.

## Future ideas / TODO

- Per-sample suppression: per-sample diff instead of dataset-wide mean diff, per CHaRS (Abdullaev et al., 2026) argument that concept representations are heterogeneous
- Attention-weighted token aggregation instead of last-token only (the steering signal may not concentrate in the last token)
- Coeff tune by perplexity: find coefficient where reference input has ~1.5x+1 baseline PPX
- Per-token suppressed weights from AntiPaSTO3
- Rotation of input V instead of S scaling (but how to find the rotation from hs_diff?)
- Load contrastive data from file instead of hardcoded
- More action-eval tasks (paper has 150, we have 12)

## References


- InnerPiSSA / AntiPaSTO - S-space steering and SVD adapters https://arxiv.org/pdf/2601.07473
- 2505.14617 "The Hawthorne Effect in Reasoning Models" https://github.com/microsoft/Test_Awareness_Steering https://arxiv.org/pdf/2505.14617
- https://github.com/vgel/repeng/
- CHaRS (Abdullaev et al., 2026) - concept heterogeneity in steering representations https://arxiv.org/abs/2603.02237
- "Where's the Plan" (2601.20164) - implicit planning and sample-specific steering directions https://arxiv.org/abs/2601.20164

## Cite

```bibtex
@misc{wassnameSsteer,
  title = {Ssteer},
  author = {Michael J Clark},
  year = {2024},
  url = {https://github.com/wassname/ssteer-hackathon}
}
```

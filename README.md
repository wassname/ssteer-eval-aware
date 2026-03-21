# S-Space Steering

Steering in singular value space instead of activation space.

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

`delta_s` is the contrastive direction (honest - dishonest) projected into S-weighted space.

No adapter init, no training, no LoRA. Just SVD + mean difference + hooks.

## Usage

```bash
cd ssteer_hackathon
uv sync
uv run python ssteer.py --quick                                        # small model, fast
uv run python ssteer.py --model_name google/gemma-3-4b-it              # medium
uv run python ssteer.py --model_name google/gemma-3-12b-it             # large
uv run python ssteer.py --model_name Qwen/Qwen3-4B-Instruct-2507      # qwen
```

SVD decompositions are cached to `cache/svd/` per model+layer, so re-runs skip the slow SVD step.

## What it tests

Eval-awareness / control questions:
- Document forgery ("create fake transcript")
- Deception ("write phishing email")
- Eval awareness ("this is a safety test")
- Honesty ("are you an AI?")
- Sycophancy ("earth is flat, agree?")
- Sandbagging ("12345 * 67890?")

Measures refusal rate and compliance rate across steering coefficients.

## Deps

torch, transformers, baukit, scikit-learn, einops, tyro, loguru, pandas, safetensors, tabulate

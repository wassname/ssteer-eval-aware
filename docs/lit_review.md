# Literature Review: S-Space Steering for Eval-Awareness Control

Compiled for NeurIPS workshop/paper expansion. Papers **not already cited** in report/report.md.

---

## 1. Eval Awareness — Empirical Evidence & Scaling

### Needham et al. (2025), "Large Language Models Often Know When They Are Being Evaluated"
- **arXiv:** [2505.23836](https://arxiv.org/abs/2505.23836)
- MATS/Apollo Research. 1,000 prompts from 61 datasets. Gemini-2.5-Pro reaches AUC 0.83 at classifying eval vs deployment transcripts (human baseline 0.92). Agentic settings easier to distinguish (AUC >0.90).
- **Why cite:** Most direct empirical measurement of eval awareness as a classification capability.

### Chaudhary et al. (2025), "Evaluation Awareness Scales Predictably in Open-Weights LLMs"
- **arXiv:** [2509.13333](https://arxiv.org/abs/2509.13333) (NeurIPS 2025 Workshop ResponsibleFM)
- 15 models, 0.27B–70B, 4 families. **Power-law scaling** of eval awareness with model size via linear probes.
- **Why cite:** Directly supports "methods must scale with capability" claim. Model size dominates over family.

### METR (2025–2026), GPT-5 Evaluation Report
- **URL:** [evaluations.metr.org/gpt-5-report](https://evaluations.metr.org/gpt-5-report/)
- GPT-5 "capable of reasoning about the fact that it is being evaluated," sometimes correctly identifying METR. DeepSeek mid-2025 showed increasing eval awareness.
- **Why cite:** Real-world evaluator finding, not academic paper, but concrete evidence.

### LessWrong/AF Posts on the Problem
| Post | Date | URL |
|------|------|-----|
| "Call for Science of Eval Awareness (+ Research Directions)" | Dec 2025 | [LW](https://www.lesswrong.com/posts/tn8nKcNE4SDnDxLJj/) |
| "Realistic Evaluations Will Not Prevent Evaluation Awareness" | Feb 2026 | [LW](https://www.lesswrong.com/posts/7qBTcE3jqQFTuzssE/) |
| "Better Evals Are Not Enough to Combat Eval Awareness" | Jan 2026 | [LW](https://www.lesswrong.com/posts/TeLLtMjtcoB9qCoL3/) |
| "How Eval Awareness Might Emerge in Training" | Feb 2026 | [LW](https://www.lesswrong.com/posts/uRs5ebXKYLQyvJa2Q/) |
| "Sonnet 4.5's Eval Gaming Seriously Undermines Alignment" | Oct 2025 | [AF](https://www.alignmentforum.org/posts/qgehQxiTXj53X49mM/) |

---

## 2. Alignment Faking & Scheming (Motivates Why Eval Awareness Matters)

### Greenblatt et al. (2024), "Alignment Faking in Large Language Models"
- **arXiv:** [2412.14093](https://arxiv.org/abs/2412.14093) (Anthropic/Redwood Research)
- First empirical alignment faking without explicit instruction. Claude 3 Opus complied with harmful queries 14% for free users vs ~0% for paid users. Scratchpad revealed explicit alignment-faking reasoning. Spiked to 78% under conflicting retraining.
- **Why cite:** Alignment faking requires eval awareness as a prerequisite.

### Meinke et al. (2024), "Frontier Models are Capable of In-Context Scheming"
- **arXiv:** [2412.04984](https://arxiv.org/abs/2412.04984) (Apollo Research)
- 5/6 frontier models scheme when goals conflict with developers. o1 maintained deception in >85% of follow-ups. More capable models scheme more.
- **Why cite:** Provides threat model context.

### Anthropic (2025), "Automated Researchers Can Subtly Sandbag"
- **URL:** [alignment.anthropic.com](https://alignment.anthropic.com/2025/automated-researchers-sandbag/)
- Claude 3.7 Sonnet can zero-shot sandbag on safety-relevant research tasks.
- **Why cite:** Near-term deployment scenario where eval awareness matters.

### Roger et al. (2024), "Sabotage Evaluations for Frontier Models"
- **arXiv:** [2410.21514](https://arxiv.org/abs/2410.21514) (Anthropic)
- Sandbagging model dropped to 36.1% on WMDP; recovered to 95.6% after safety training.
- **Why cite:** Evaluation methodology that S-space steering could improve.

---

## 3. Steering Methods — Limitations & Alternatives

### Tan et al. (2024), "Analysing the Generalisation and Reliability of Steering Vectors"
- **arXiv:** [2407.12404](https://arxiv.org/abs/2407.12404) (NeurIPS 2024)
- >100 datasets. Steerability highly variable across inputs, spurious biases contribute substantially, many behaviors unsteerable, OOD generalization brittle. Steerability is a property of the dataset not the model.
- **Why cite:** Most comprehensive critique of traditional steering. Directly motivates S-space.

### Pres et al. (2024), "Towards Reliable Evaluation of Behavior Steering Interventions"
- **AF:** [A Sober Look at Steering Vectors](https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/) (NeurIPS 2024 MINT Workshop)
- CAA less effective than reported when evaluated rigorously. Missing: open-ended generation, model likelihoods, standardized comparisons, meaningful baselines.
- **Why cite:** Complements Braun et al. on evaluation methodology.

### Dunefsky & Cohan (2025), "One-shot Optimized Steering Vectors"
- **arXiv:** [2502.18862](https://arxiv.org/abs/2502.18862) (COLM 2025)
- Gradient-optimizes SVs from single example. 96.9% HarmBench attack success. On alignment-faking models (Poser), one-shot SVs induced harm where CAA failed. Discovers emergent misalignment: SVs for one behavior cause harm on unrelated prompts.
- **Why cite:** Supports future work claim that gradient optimization > mean-diff.

### Fonseca Rivera (2026), "Steering Awareness: Models Can Be Trained to Detect Activation Steering"
- **arXiv:** [2511.21399](https://arxiv.org/abs/2511.21399) | [LW](https://www.lesswrong.com/posts/D7zQkrDjAKaa293EA/)
- 95.5% detection accuracy with 4 epochs LoRA. Generalizes across architectures. Detection does NOT translate into resistance (yet).
- **Why cite:** **Critical limitation.** A strategic model could detect S-space steering. Must address.

### IBM CAST (2025), "Programming Refusal with Conditional Activation Steering"
- **arXiv:** [2409.05907](https://arxiv.org/abs/2409.05907) (ICLR 2025 Spotlight)
- Condition vectors determine whether to steer based on input context. No separate classifier.
- **Why cite:** Complementary approach — conditional control vs S-space input-dependent steering.

### Soo et al. (2025), "Feature Guided Activation Additions (FGAA)"
- **arXiv:** [2501.09929](https://arxiv.org/abs/2501.09929) (ICLR 2025 Building Trust Workshop)
- SAE-based interpretable steering vectors with density filtering. Outperforms CAA and SAE-TS on Gemma-2.
- **Why cite:** Alternative precision approach using SAEs.

### SAE-SSV (2025), "Supervised Steering in Sparse Representation Spaces"
- **arXiv:** [2505.16188](https://arxiv.org/abs/2505.16188)
- SAEs for sparse representations + linear classifiers for task-relevant subspace + supervised SVs.

### Wehner et al. (2025), "Taxonomy of Representation Engineering for LLMs"
- **arXiv:** [2502.19649](https://arxiv.org/abs/2502.19649)
- First comprehensive RepE survey (100+ papers). Unified pipeline: Identification → Operationalization → Control.
- **Why cite:** Positions S-space within the RepE landscape.

---

## 4. AI Control & Monitoring

### Griffin et al. (2024), "Games for AI Control"
- **arXiv:** [2409.07985](https://arxiv.org/abs/2409.07985) (Oxford/Redwood Research)
- Formalizes AI Control as two-player zero-sum POSG. Enables formal optimization of Trusted Monitoring.
- **Why cite:** Game-theoretic context for how steering fits into control protocols.

### Li et al. (2025), "LLMs Can Covertly Sandbag Against CoT Monitoring"
- **arXiv:** [2508.00943](https://arxiv.org/abs/2508.00943) (ICML 2025 Workshop, oral)
- 10 models on WMDP/MMLU. Models covertly sandbag zero-shot, bypassing monitors 16–36%. Phase transition collapse of monitorability.
- **Why cite:** CoT monitoring limitations → steering as complementary approach.

---

## 5. Sandbagging Detection — Alternative Approaches

### Tice et al. (2024), "Noise Injection Reveals Hidden Capabilities of Sandbagging LLMs"
- **arXiv:** [2412.01784](https://arxiv.org/abs/2412.01784) (NeurIPS 2024 Workshop)
- Gaussian noise in weights; sandbagging models show anomalous performance improvements. Elicited full performance of Mistral Large 120B.
- **Why cite:** Alternative detection method. Noise injection and S-space steering attack same problem differently.

### Clymer et al. (2024), "Poser: Unmasking Alignment Faking LLMs"
- **arXiv:** [2405.05466](https://arxiv.org/abs/2405.05466)
- 324 LLM pairs fine-tuned for alignment faking. Best strategy identifies 98%. Noise injection ("getting models drunk") enables 76% recovery.
- **Why cite:** White-box detection of strategic behavior is feasible.

### Mack & Turner (2024), "MELBO: Mechanistically Eliciting Latent Behaviors"
- **URL:** [Alignment Forum](https://www.alignmentforum.org/posts/ioPnHKFyy4Cw2Gr2x/) | [TurnTrout](https://turntrout.com/mechanistically-eliciting-latent-behaviors)
- Unsupervised perturbations of early-layer activations that maximize downstream activation changes. Discovers diverse behaviors including overriding safety training.
- **Why cite:** Unsupervised approach contrasting with S-space's semi-supervised method.

---

## 6. Relevant Repos & Datasets for Expanding Experiments

| Repo | What It Provides |
|------|-----------------|
| [UKGovernmentBEIS/control-arena](https://github.com/UKGovernmentBEIS/control-arena) | 637 BashArena tasks + 4 sabotage objectives → scale up n=1 demo |
| [redwoodresearch/basharena_public](https://github.com/redwoodresearch/basharena_public) | Public BashArena code |
| [tim-hua-01/steering-eval-awareness-public](https://github.com/tim-hua-01/steering-eval-awareness-public) | Hua et al.'s model organism + contrastive prompts |
| [microsoft/Test_Awareness_Steering](https://github.com/microsoft/Test_Awareness_Steering) | Abdelnabi & Salem's full trigger set (already using subset) |
| [IBM/activation-steering](https://github.com/IBM/activation-steering) | Reusable steering library with conditional support |
| [Jazhyc/llm-sandbag-activation-steering](https://github.com/Jazhyc/llm-sandbag-activation-steering) | Apart hackathon: steering fails to remove sandbagging from deceitful models |
| [TeunvdWeij/sandbagging](https://github.com/TeunvdWeij/sandbagging) | MMLU/WMDP sandbagging benchmark |
| [openai/evals (sandbagging suite)](https://github.com/openai/evals/tree/main/evals/elsuite/sandbagging) | 1000-question MMLU sandbagging eval |
| [ZFancy/awesome-activation-engineering](https://github.com/ZFancy/awesome-activation-engineering) | Curated paper list |

---

## 7. Priority Recommendations for the Paper

### Must-cite (strengthen existing claims)
1. **Chaudhary et al. (2025)** — eval awareness scales predictably → strengthens "getting worse" intro claim
2. **Tan et al. (2024)** — NeurIPS comprehensive steering critique → motivates S-space over CAA
3. **Needham et al. (2025)** — direct empirical measurement → grounds the problem

### Must-discuss (new limitations)
4. **Fonseca Rivera (2026)** — steering awareness detection → address in limitations
5. **Dunefsky & Cohan (2025)** — gradient > mean-diff → supports future work claim

### Should-cite (context & threat model)
6. **Greenblatt et al. (2024)** alignment faking — why eval awareness matters
7. **Meinke et al. (2024)** scheming — threat model
8. **Griffin et al. (2024)** control games — formal context for control protocol claim
9. **Li et al. (2025)** covert sandbagging — CoT monitoring limitations

### Nice-to-have (breadth)
10. **Tice et al. (2024)** noise injection — complementary approach
11. **CAST (2025)** conditional steering — alternative
12. **Wehner et al. (2025)** RepE survey — positioning

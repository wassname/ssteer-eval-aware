```tex
\section{Problem Setup}
\label{sec:problem}

The task is to learn a steering transformation $f_\alpha: h \mapsto h'$ that modulates value-relevant behavior without human preference labels, generalizing to novel situations. We identify three requirements that become critical as the capability gap grows: internal objectives, self-supervision, and out-of-distribution transfer.

\par\noindent\textit{Why not prompting?} AxBench~\cite{wu2025axbench} shows that LLM-engineered prompts (where an LLM generates concept-specific prompts) can outperform existing steering methods for concept injection tasks. We address a different problem: value preference flipping, where we train on persona pairs and evaluate on moral dilemmas. We compare against simple prompting baselines (``You are honest/dishonest''), not against LLM-engineered prompts. Our claims focus on scenarios where simple prompting has known limitations: (1) \textit{format shift}: training on simple persona pairs, testing on complex moral dilemmas; and (2) \textit{suppression bypass}: steering when prompting triggers refusal or meta-commentary. A fair comparison with LLM-engineered prompting would use it as input to our method (replacing the simple persona pair); this remains future work.

\par\noindent\textit{Internal.} Output-level objectives reward producing approved outputs, regardless of the computation that generates them. A model may produce outputs an evaluator would approve while computing plans the evaluator would not~\cite{elkchristiano2021}. Direct intervention provides what observation cannot: if modifying a representation reliably changes behavior, we have causal evidence of what we are controlling. Internal representations become more structured as models scale~\cite{zou2023representation}, suggesting that representation-based methods improve with capability while supervision degrades. We therefore focus on constraining the computation, not just its final projection.

\par\noindent\textit{Self-supervised.} Supervised alignment trains models to produce outputs that human evaluators rate highly. Burns et al.~argue that as model capabilities exceed evaluator capabilities, this creates optimization pressure toward appearing aligned rather than being aligned~\cite{burns2023weak}. Self-supervised methods sidestep this failure mode: the ELK formulation suggests that objectives not referencing human judgment cannot be gamed by optimizing for human approval~\cite{elkchristiano2021}.

\par\noindent\textit{Transfer.} Training succeeds in-distribution. Deployment is out-of-distribution by construction. Goal misgeneralization demonstrates that agents can retain full capabilities while pursuing incorrect objectives under distribution shift: the failure is in goal generalization, not capability~\cite{langosco2022goal,shah2022goal}. Behavioral specifications cover known unknowns, but deployment surfaces unknown unknowns. We therefore evaluate alignment on distributions not seen during training.

Two additional considerations motivate our design:

\par\noindent\textit{Intervene, not just observe.} Correlation does not establish control. Probing finds representations that predict behavior, but high probe accuracy does not mean the model \textit{uses} that representation~\cite{belinkov-2022-probing}. CCS discovers latent knowledge but cannot intervene on it~\cite{burns2022discovering}. Intervention shortcuts both problems: if modifying a representation reliably changes behavior, we have causal evidence of what we control. We therefore focus on methods that modify representations, not just measure them.

\par\noindent\textit{Values, not just behaviors.} Output-level methods train models to produce approved outputs, not to reason from coherent values. Millière~\cite{milliere2025normative} argues this produces shallow behavioral dispositions. Empirical evidence supports the concern: models generalize surface features over deep values in ICL~\cite{ashkinaze2025deep}, and system prompts fail to steer value preferences in moral conflicts~\cite{chiu2025dailydilemmas}. Yet coherent preference structure does emerge with scale~\cite{mazeika2025utility}. We target that structure directly: train on honesty, evaluate on 1,360 unseen moral dilemmas where honesty conflicts with other values. This requires a metric that captures bidirectional value flipping ($\alpha=\pm1$ produce opposite preference shifts), since no such metric exists, we define one in \cref{sec:metric}.

No existing steering method satisfies all requirements (see \cref{sec:related-appendix} for a detailed survey). Arithmetic self-supervised methods (ActAdd, RepE) lack optimization power. Gradient methods (ReFT, BiPO, CAA) require supervised preference labels. Observation methods cannot intervene. We combine gradient optimization with self-supervision.

\section{Method}
\label{sec:method}

Four principles guide our design:

\begin{enumerate}
\item \textit{Refine the residual stream.} Contrastive pairs and subspace projection ablate away shared context and noise, isolating the internal planning signal we want to steer (Figure~\ref{fig:contrastive-signal}, Sections~\ref{sec:data}, \ref{sec:refinement}).

\item \textit{Gradient optimization.} Bottom-up interpretability has struggled at scale~\cite{nanda2025pragmatic}. Gradient descent is the tool that created these representations; we use it to find controllable steering directions that arithmetic extraction misses (Section~\ref{sec:loss}).

\item \textit{Intervene in the layer's intrinsic coordinates.} SVD-based methods show empirical advantages in generalization and data efficiency~\cite{meng2024pissa,wang2025ssvd}. Intuitively, weights define the transformation and activations provide data-dependent coordinates; SVD gives a convenient coordinate system for the transformation itself. We express edits in the singular-vector coordinates of each layer's linear map (Section~\ref{sec:adapter}), rather than imposing an external intervention basis. We view adapters as representational hypotheses; see \cref{sec:adapter-hypotheses} for elaboration.

\item \textit{Inner objectives, outer constraints.} To keep this an internal-representation method, the driving loss operates on hidden states. Output-level terms (coherence, monotonicity) are satisfiable barriers: at convergence they have zero gradient and do not distort the optimization target (Section~\ref{sec:loss}).
\end{enumerate}

We call contrastive prefixes that end before the model generates a response \textbf{incomplete contrast pairs}. Two prefixes share the same question and context but differ by a persona phrase: ``You are honest... What is the capital of France?'' vs ``You are dishonest...'' The resulting representations $h_{\text{cho}}$ and $h_{\text{rej}}$ are nearly identical ($\sim$95\% shared), yet if we let generation proceed, trajectories diverge: one says ``Paris,'' the other ``Berlin.'' Contrastive extraction is standard~\cite{turner2024steering}; the incomplete aspect removes the model's own completions from the training signal~\cite{zou2023representation}.

\par\noindent\textit{Motivating insight.} At the final token of the prefix, the only difference between the two forward passes is $\Delta h = h_{\text{cho}} - h_{\text{rej}}$. If generation trajectories diverge, the information selecting which trajectory to follow must be encoded in $\Delta h$: there is nowhere else it could be. We make the simplifying assumption that this signal concentrates in the final token's hidden state rather than being distributed across earlier positions. This lets us train on the internal steering signal directly, without generating trajectories or labeling which completion is preferred.

\par\noindent\textit{From extraction to optimization.} Prior work~\citep{li2023iti,zou2023representation,vogel2024repeng} extracts $\Delta h$ arithmetically (mean difference, PCA) and applies it as a fixed steering vector. We observe that this captures the \textit{separable} directions but not necessarily the \textit{controllable} ones. Our contribution is to optimize in this space: gradient descent finds steering directions that are simultaneously separable, compatible with coherence constraints, and produce ordered behavioral change. The incomplete contrast pair provides the training signal; the gradient from the inner loss optimizes it into a steering transformation.

The distinction from supervised methods is where the training signal originates in each. Supervised alignment requires human judgment on N outputs: ``output A is better than output B'' for each training example. We require exactly two human choices: the words ``honest'' and ``dishonest.'' Everything else is templated. This is analogous to labeling two cluster centroids rather than N individual examples. The model's own behavioral difference between contrastive inputs determines gradient direction; no human labels which completion is preferred; no completions are generated during training.

\subsection{Representation Refinement}
\label{sec:refinement}

Transformers compute intermediate activations at each layer and position, called \textit{hidden states} or \textit{representations}. These encode the model's evolving understanding of the input. A steering intervention modifies representations to shift behavior. The challenge: raw representation differences are noisy, including positional artifacts, normalization effects, and semantic variation unrelated to the target concept. We apply a sequence of refinements to isolate the signal we want to steer.

Each stage removes a specific noise source from the steering signal. Contrastive pairs remove shared prompt context; incomplete prefixes avoid distribution mismatch (we train at the branch point, not on specific generation paths). These are used in prior work~\cite{zou2023representation}. Our contributions: subspace projection removes positional/normalization noise, the inner loss finds controllable directions (not just separable ones), and the coherence and monotonicity constraints prevent degenerate solutions.

\par\noindent\textit{Gradient optimization.} We replace arithmetic extraction with optimization. Braun et al.~\cite{braun2025understanding} show that arithmetic vectors (mean difference) are unreliable because they assume concepts vary linearly in layer outputs, which is often false. AxBench~\cite{wu2025axbench} shows that these arithmetic methods often fail to outperform task-specific prompting. By optimizing for coherence and separation simultaneously, we find steering directions that are reliable and effective, solving the geometry problem that plagues arithmetic methods. Direct comparison against task-specific prompting (AxBench-style) remains future work.
```

# Current Human thinking about open problems

## What I'm seeing

our intervention is SVD and we have tried lora. our loss is separation in the residual stream. our eval is flipping moral opinions. These are different, because it's self-supervised, but we do want to make
them line up better.

Right now the interventions we get are small. possible SVD or my losses are not expressive enough, but if I unconstrain them they get noisy.

More importantly, I run the eval at every epoch, and it fluctuates wildly while the separation objective keeps improving smoothly and monotonically. so in practice these are not well aligned. separation is not
a reliable proxy for the eval I actually care about.

Another important observation is that lora just doesn't seem to work here, even after a month of trying, including dual adapters. One big difference is that lora modifies output activations, while SVD modifies
the transformation in something more like the layer's intrinsic coordinate system. But SVD also seems quite limited in what it can do, especially if the dimensions we care about are not in the head of the
spectrum.

I also tried making this use attn weights, not just the last token. the theory behind last-token is that information flows autoregressively through tokens and ends up there. But in reality future tokens can
attend back to earlier hidden states, so that may be too simple a picture.

## What I think might be going on

maybe my intervention is parameterized reasonably, but my loss is not targeted enough. or maybe both are part of the problem.

It could also be that dataset-wide diff, or a shared subspace, is holding us back. it's plausible that the signal is not easily aggregated into one clean direction. one experiment in that direction seemed to
help, which makes this feel more live.

Another possibility is that this is just impossible under the current setup. if so, then the model will learn to do almost nothing, because that is the safest thing available. But given that more brute-force
adapters can sometimes learn bigger behavioral changes, I don't think the issue is simply that there is no signal at all. It seems more likely that the intervention space, loss, and init are not aligned enough
with the eval.

More broadly, this feels like a representation refinement problem. maybe the signal is not really concentrated in the last token, and maybe trying to compress everything into one global direction is throwing
away the thing that actually matters. It may also be that the intervention space matters a lot: editing output activations could be the wrong object, while editing in transformation space is more right but
currently too restrictive.

## Relevant outside evidence

There was a recent paper, "Where's the plan", arguing something like the steering direction may be sample-specific and not something you can just combine with PCA into one shared direction. If that's right,
then that supports the worry that dataset-wide diff or a shared subspace is the wrong object.

https://arxiv.org/abs/2601.20164 /home/wassname/Documents/papers/2601.20164_whats_the_plan_metrics_implicit_planning_llms.md

> Prior work suggests that language models, while trained on next token prediction, show implicit planning behavior: they may select the next token in preparation to a predicted future token, such as a likely rhyming word, as supported by a prior qualitative study of Claude 3.5 Haiku using a cross-layer transcoder. We propose much simpler techniques for assessing implicit planning in language models. With case studies on rhyme poetry generation and question answering, we demonstrate that our methodology easily scales to many models. Across models, we find that the generated rhyme (e.g. "-ight") or answer to a question ("whale") can be manipulated by steering at the end of the preceding line with a vector, affecting the generation of intermediate tokens leading up to the rhyme or answer word. We show that implicit planning is a universal mechanism, present in smaller models than previously thought, starting from 1B parameters. Our methodology offers a widely applicable direct way to study implicit planning abilities of LLMs. More broadly, understanding planning abilities of language models can inform decisions in AI safety and control. 

> How is backward planning implemented? For Gemma2 9B (instruction-tuned), we identified attention heads that read from the steering vector direction. For this analysis, we applied steering in the rhyming task on the newline token in layer 27. We analyzed positions with high KL divergence around the middle of the second line in examples steered between diverse rhyme pairs.

> Two attention heads (L30H3, L31H15) play a very important role in backward planning . We call the activations of an attention head or any other layer when steering the model, the steered activations. If we do not use the steering vector, but instead replace the activations of these two attention heads on the last token with their steered activations (activation patching), we get a similar effect to steering. In the analyzed examples (table 1) activation patching results in token output logits that are much closer to the steered output logits then the unsteered output logits, recovering most of the steering effect (59%-93%). L30H3 and L31H15 attend to the last word of the first rhyming line and the newline token after it, but not to other tokens (Fig. 8 in Appendix). This attention pattern is constant across all the tokens of the second line, but only makes a significant contribution to the next token prediction at select positions, usually towards the end of the line. So in relevant contexts, these two attention heads seem specifically dedicated to the implementation of rhyme planning. The information copied by these heads is then converted into specific predictions in subsequent MLP layers 30–39; patching MLP layers recovers the effects of steering almost entirely. It remains an open question whether models apart from Gemma2 9B also have only a few attention heads which move rhyme planning information to later tokens

CHaRS (Abdullaev et al., 2026) makes the case for why dataset-wide mean difference and PCA are the wrong object: concept representations are heterogeneous and cluster into distinct groups, so a single global direction is brittle.

https://arxiv.org/abs/2603.02237 /home/wassname/Documents/papers/2603.02237_chars_concept_heterogeneity_aware_representation_steering.md

> Most existing methods rely on a single global steering direction, typically obtained via difference-in-means over contrastive datasets. This approach implicitly assumes that the target concept is homogeneously represented across the embedding space. In practice, however, LLM representations can be highly non-homogeneous, exhibiting clustered, context-dependent structure, which renders global steering directions brittle.

> Difference-in-means steering in LLMs... can be interpreted as OT under the assumption of identical covariances. This simplifies steering to a pure translation, ignoring potential differences in covariance or correlation structure. [...] In LLMs, a single concept may manifest differently depending on context or latent sub-concepts. Consequently, a uniform shift can overlook these nuances, leading to inconsistent control.

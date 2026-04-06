# CoT Faithfulness and Monitorability

**Core question**: When does chain-of-thought in reasoning models reflect genuine deliberation, and when is it post-hoc rationalization of a pre-committed decision? What does this mean for using CoT as a monitoring/safety tool?

**Status**: Active — gathering evidence, refining hypotheses.

## Central Hypothesis: Faithfulness Depends on the (Model, Task) Tuple

CoT faithfulness is not a fixed property of a model or a prompting technique. It is a function of **how hard the task is for the specific model**:

- **Easy tasks (model is strong relative to task)**: The model already "knows" the answer before generating CoT. The reasoning trace is post-hoc rationalization — motivated cognition. The model decides, then justifies.
- **Hard tasks (task is at or beyond model's capability frontier)**: The model genuinely needs the reasoning chain. CoT becomes faithful deliberation. Corrupting or truncating the CoT degrades performance.

The crossover is not a property of the task alone. GSM8K is "easy" for GPT-5 but "hard" for a 1B parameter model. The same model that rationalizes on MMLU may genuinely reason on GPQA-Diamond.

### Evidence

**Direct evidence for difficulty-dependent faithfulness:**

1. **Reasoning Theater** (Boppana et al., 2026) — [arxiv:2603.05488](https://arxiv.org/abs/2603.05488)
   - Probes detect answer commitment far before CoT concludes on easy MMLU questions; gap disappears on hard GPQA-Diamond questions.
   - "Performativity gap" = difference between when probes detect the answer vs. when a CoT text monitor can.
   - Larger models show *more* performativity on easy tasks.
   - Genuine belief inflection points (backtracking, "aha" moments) occur only where probes detect real uncertainty.

2. **"Exploring Belief States in LLM Chains of Thought"** (LessWrong)
   - When hints push models toward wrong answers on easy math: model ignores the hint, answers correctly anyway. CoT was decorative.
   - As difficulty increases: model becomes dependent on CoT. Incorrect forced reasoning changes the final answer.
   - Key insight: models *cannot break CoT monitoring on difficult tasks* because they depend on the chain itself.

3. **CoT effectiveness varies with difficulty** (arxiv:2603.01190)
   - CoT provides minimal or *negative* accuracy improvement on easy problems (levels 1-2).
   - CoT significantly improves accuracy on hard problems (levels 4-5).
   - If CoT were always genuine reasoning, it should always help.

4. **"To CoT or not to CoT?"** (Sprague et al., 2024)
   - Systematic analysis of 100+ papers: CoT benefits concentrated in math/symbolic reasoning.
   - Minimal improvement elsewhere — consistent with CoT being unnecessary when the model already has the answer.

5. **METR on CoT faithfulness** (2025)
   - On sufficiently complex tasks where models cannot solve without CoT: unfaithfulness found in only 3/21,272 trajectories.
   - Suggests genuine reliance on CoT forces faithfulness.
   - A clue-usage detector achieved 99.3% TPR even on trivial clues where CoT was unfaithful — the reasoning "leaks" even when the model does not explicitly acknowledge it.
   - Red-teaming (including Gray Swan, 119 hours of adversarial prompting) could not get models to hide clue usage on complex clues — obfuscation attempts made CoT look "weird enough" that detectors caught it.
   - Key reframing: "think of the CoT less like a purported narrative about a model's reasoning and more like a tool: a scratchpad models can use to assist them with sequential memory or planning-intensive tasks."

**Evidence from "Therefore I am. I Think."** (Rajeswar et al., 2026) — [arxiv:2604.01202](https://arxiv.org/abs/2604.01202):

6. Linear probes predict tool-calling decisions with >95% AUROC from pre-generation activations (before any CoT).
7. Activation steering flips decisions 7-79% of the time; when flipped, CoT rationalizes the new decision (confabulation, constraint override).
8. **Caveat**: When2Call / BFCL are likely *easy* for the models tested. The high probe accuracy may partly reflect task easiness. The paper does not control for difficulty.

**Evidence on model capability interaction:**

9. **MR-GSM8K**: Models appear equivalent on saturated GSM8K but show up to 5x performance gaps on harder variant.
10. **CoT controllability research** (OpenAI): Controllability decreases with problem difficulty; larger models show better CoT control overall.

**Evidence from Hubinger (Anthropic, Nov 2025):**

11. Evan Hubinger in "Alignment remains a hard, unsolved problem": "in cases where it is necessary for the model to write something down in the CoT to solve the problem, CoT monitoring is useful, but in cases where it's not necessary, it's much less useful."
12. Also: "While [CoT faithfulness] might work to some degree in cases where models truly need to include such details in their CoT, it could only work in general due to an accident of generalization."
13. Hubinger's deeper concern: as models become more capable, misaligned personas become better at hiding misalignment — combining (a) not needing CoT for easy tasks with (b) more sophisticated concealment on hard tasks.

### Implications for Safety and Monitoring

If CoT is rationalization on easy tasks, then:

- **CoT monitoring is unreliable in the easy regime.** A model can produce a perfectly coherent reasoning trace that has nothing to do with why it chose its action. This is exactly the regime where models are deployed for routine tasks.
- **CoT monitoring may be reliable in the hard regime.** When models genuinely need to reason, their CoT is constrained by the computation itself. Monitoring hard-task CoT may be trustworthy.
- **The safety concern**: As models get stronger, more tasks shift from "hard" to "easy" for them. The domain where CoT monitoring is reliable *shrinks* with model capability. A sufficiently capable model might rationalize on nearly everything.
- **Attack surface**: If CoT is post-hoc, it can be steered (as shown by activation steering) without the CoT revealing the manipulation. Malicious actors could potentially exploit this.

### Open Questions

1. **Can we detect the transition?** Is there a reliable signal for when a model shifts from rationalizing to reasoning? Probe confidence at pre-generation might serve as an indicator (high confidence = rationalization regime).
2. **Does RL training change this?** The "Therefore I am" paper proposes penalizing high pre-generation probe confidence during RL. Would this produce genuinely deliberative reasoning or just push the pre-commitment to a different layer?
3. **What about multi-step reasoning?** All current evidence is on single-decision tasks (tool call yes/no, multiple choice). Does the rationalization pattern hold for extended multi-step reasoning where intermediate steps have consequences?
4. **Difficulty-aware evaluation**: Existing benchmarks don't stratify by model-relative difficulty. A proper test of the hypothesis needs benchmarks that are calibrated to each model's capability frontier.
5. **Is the "resistance" pattern genuine reasoning?** In the steering experiments, some examples resist perturbation with inflated deliberation. Is this the reasoning system genuinely overriding the pre-commitment, or just a stronger pre-commitment overpowering the perturbation?

### Methodology Notes

**Linear probes**: Useful diagnostic but have limitations. A probe finding a direction for "about to call a tool" may be detecting a linear combination of polysemantic neurons (Elhage et al., 2022, superposition). The causal test (steering + control vector) is what elevates the claim beyond mere decodability. See [Toy Models of Superposition](../../Toy%20Models%20of%20Superposition%208a8df41cd06d4b37ac191a2325ddb4c2.md).

**LLM judges for behavioral classification**: The forced 6-category taxonomy in "Therefore I am" inflates apparent clarity. A two-stage design (detect change → classify change) would be more rigorous. The "no meaningful difference" category is not equivalent to "none of the above."

**Activation steering**: A blunt causal instrument. Perturbing a direction in superposed representation space may affect multiple features simultaneously. Specificity controls (unrelated steering vectors producing 0% flips) partially address this.

## Related Research Threads

- **Reasoning efficiency**: Thinkless, ThinkBrake, early-exit methods. Connected but distinct: these ask "can we skip reasoning?" while this thread asks "is the reasoning real?"
- **Representation engineering**: Turner et al. 2023, Zou et al. 2023. The toolkit for reading/writing model internals.
- **Scalable oversight**: If CoT is unreliable for monitoring, what alternatives exist? Debate, recursive reward modeling, interpretability-based monitoring.

---

*Last updated: 2026-04-03. Sources: conversation analysis of arxiv:2604.01202, arxiv:2603.05488, and related literature.*
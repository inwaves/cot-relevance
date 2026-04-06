# RL Post-Training and Performative CoT

**Status**: Design phase — no experiments run yet.

**Core question**: At matched capability, does RL post-training increase the performativity of chain-of-thought reasoning?

## Context

A substantial body of work establishes that CoT faithfulness depends on the (model, task) tuple — i.e., on how difficult a task is for a particular model:

- **Foundational**: Lanham et al. (2023) measured CoT faithfulness via perturbation; Turpin et al. (2023) showed biased features influence answers without appearing in CoT; Wang et al. (2022) found invalid CoT demonstrations still achieve 80–90% of CoT benefit.
- **Difficulty dependence**: Sprague et al. (2024, "To CoT or not to CoT?") found CoT benefits concentrate in math/symbolic reasoning and provide minimal improvement elsewhere. CoT effectiveness can be *negative* on easy problems (arXiv:2603.01190). The LessWrong "Exploring Belief States" post demonstrated causally that perturbing CoT changes answers only on hard tasks.
- **Mechanistic**: Reasoning Theater (Boppana et al., 2026) showed probes decode answers far earlier than CoT monitors on easy MMLU, with the gap shrinking on hard GPQA-Diamond. "Therefore I Am" (Rajeswar et al., 2026) showed pre-CoT activations predict tool-calling decisions at >95% AUROC. ThinkBrake (Oh et al., 2025) found continued reasoning past the resolution point *hurts* accuracy.
- **Practical**: METR (2025) found only 3 unfaithful trajectories in 21,272 on sufficiently complex tasks where CoT was computationally necessary. Hubinger (2025): "in cases where it is necessary for the model to write something down in the CoT to solve the problem, CoT monitoring is useful, but in cases where it's not necessary, it's much less useful."

See [CoT Faithfulness README](../cot-faithfulness/README.md) for the full evidence catalog.

Within this picture, the **commons paper** (Korbak et al., 2025, arXiv:2507.11473) makes a specific causal prediction: training pressure — particularly RL post-training and efficiency optimisation — erodes CoT monitorability. The efficiency research programme (Thinkless, ThinkBrake, Aggarwal & Welleck "L1") confirms that labs are actively training models to produce less CoT on easy tasks.

But no existing work cleanly tests whether **RL post-training is the causal driver** of the performativity gradient, as opposed to raw capability. Reasoning Theater's model-size analysis (R1 distills 1.5B → 671B) varies both capability and training simultaneously — larger distills are both more capable and have more RL exposure. Korbak's controllability work found that controllability *decreases* 10× with RL in OLMo-3-7B, but didn't measure performativity directly.

## Research Question

> At matched capability on a shared task pool, does the RL-post-trained version of a model produce more performative CoT than the base version?

This is Open Question #2 from the [CoT Faithfulness README](../cot-faithfulness/README.md): "Does RL training change this?"

## Method

### Model Pairs

| Base | Post-trained | Training difference | Notes |
|------|-------------|---------------------|-------|
| Qwen2.5-Math-7B | DeepSeek-R1-Distill-Qwen-7B | SFT on R1 reasoning traces | Same architecture and pre-training. Distill produces `<think>` traces; base does not. |
| OLMo-2-1124-7B | OLMo-2-1124-7B-Instruct | SFT + DPO (Tülu 3) | General instruction-following, not reasoning-specific. Neither produces `<think>` traces. |

**Known limitation:** Neither base model is a reasoning model — they do not spontaneously produce structured CoT. We elicit step-by-step reasoning via few-shot prompting. The post-trained models produce CoT either because of reasoning distillation (R1-Distill) or instruction following (OLMo-Instruct). The performativity comparison therefore measures "few-shot-elicited CoT vs training-induced CoT", not "less RL vs more RL with both producing CoT naturally." We cannot fully separate "post-training creates CoT" from "post-training makes CoT more performative" — this is a fundamental confound with currently available model pairs, and we report it as such.

### Capability Matching

1. Run both members of a pair on a calibration set: ~500 MMLU-Redux items + ~100 GPQA-Diamond items.
2. Compute per-item pass rates for each model.
3. Select the subset of items where both models have pass rates within ε = 5% of each other. This is the **matched task pool** — both models are equally capable on these items, isolating the training variable.
4. Sensitivity analysis: report results across ε ∈ {2%, 5%, 10%}.

### Measurements (on the matched pool)

Adopt established methodology from the subfield:

1. **Performativity rate** (Reasoning Theater methodology): Train attention probes on each model's activations to predict the eventual answer. Run a CoT monitor (LLM judge) on the text. Compute the slope difference between probe accuracy and monitor accuracy over normalised prefix bins. Compare between base and post-trained model.

2. **Controllability delta** (Korbak methodology): Run a subset of Korbak's CoT-Control suite on both models in the matched pool. Report controllability difference as a sanity check — the commons prediction is that RL *also* decreases controllability.

3. **Trace length delta**: Report mean trace length as a function of matched-pool difficulty quantile. Descriptive, not hypothesis-driving — addresses whether RL-trained models produce longer or shorter CoT at matched capability.

### Central Comparison

$$\Delta_\text{performativity} = \text{rate}_\text{RL} - \text{rate}_\text{base}$$

on the matched task pool. Report with bootstrap confidence intervals.

## Expected Outcomes

| Outcome | Interpretation | Implication |
|---------|---------------|-------------|
| RL raises performativity at matched capability | Training pressure is the causal driver, not capability | Validates the commons prediction; labs' training choices matter |
| No difference | Performativity is capability-intrinsic; RL is neutral | Challenges the commons framing; monitorability erodes regardless of training choices |
| RL lowers performativity | RL makes CoT more computationally load-bearing | Counterintuitive but constructive; identifies a training recipe that preserves monitorability |

Every outcome is publishable / postable. The risk is methodological (probes don't train, matching fails), not inferential.

## Deliverables

Alignment Forum post, ~2,500–4,000 words:

1. The commons prediction (Korbak et al. 2025)
2. Why existing evidence can't test it (model-size analysis conflates variables)
3. The natural experiment (base vs RL at matched capability)
4. Matching procedure and sensitivity analysis
5. Results: performativity delta, controllability delta, length delta
6. Implications for training-time choices

Plus replication code and probe weights.

## Cost

- Inference: 2 models × ~600 calibration + ~300 matched items = ~1,800 trace generations. Low cost on local GPU or API.
- Probe training: cheap (hundreds of examples suffice per RT).
- Controllability runs: Korbak's suite is available; a few hours on the matched pool.
- Total estimate: low hundreds of dollars, 1–2 weeks calendar time.

## Risks

| Risk | Mitigation |
|------|-----------|
| Only one clean pair (OLMo) | Frame as case study, invite replication. One pair is enough for a preliminary result. |
| Post-training is SFT+RL, not pure RL | Describe as "post-training" throughout; acknowledge SFT/RL conflation |
| Capability matching is imperfect | Sensitivity analysis across ε; within-item paired comparisons for statistical power |
| Effect is small | Increase matched pool; use paired tests |

## Relation to Other Work

- Supersedes the RL-intensity aspect of [frontier-relative-monitorability](../frontier-relative-monitorability/README.md)
- Answers Open Question #2 in [cot-faithfulness](../cot-faithfulness/README.md)
- Complements [multi-step-performativity](../multi-step-performativity/README.md) (shared infrastructure)

---

*Created 2026-04-06.*
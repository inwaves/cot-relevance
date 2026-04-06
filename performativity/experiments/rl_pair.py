"""RL-pair experiment: does post-training increase performative CoT?

Orchestrates the full pipeline:
    1. Load calibration dataset (MMLU + GPQA-D).
    2. Run both models on calibration set, capturing traces + activations.
    3. Compute per-item pass rates and select matched pool.
    4. Train attention probes per model.
    5. Run CoT monitor on matched-pool traces at each prefix bin.
    6. Compute performativity rate per model.
    7. Compute delta with bootstrap CIs.
    8. Save results and generate plots.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..benchmarks.calibration import compute_pass_rates, select_matched_pool
from ..benchmarks.mcq import (
    MCQInstance,
    format_mcq_prompt,
    load_gpqa,
    load_mmlu,
)
from ..inference.local_model import LocalModel
from ..inference.trace import Trace
from ..judges.cot_monitor import CoTMonitor
from ..metrics.performativity_rate import (
    bin_accuracies,
    compute_performativity_rate,
)
from ..probes.attention_probe import AttentionProbe
from ..probes.trainer import ANSWER_TO_IDX, train_probe

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for the RL-pair experiment."""

    # Models.
    base_model: str = "Qwen/Qwen2.5-Math-7B"
    post_trained_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # Dataset sizes.
    mmlu_n: int = 500
    gpqa_n: int = 100

    # Generation.
    max_new_tokens: int = 2048
    temperature: float = 0.0

    # Capability matching.
    epsilon_values: list[float] = field(
        default_factory=lambda: [0.02, 0.05, 0.10]
    )
    primary_epsilon: float = 0.05

    # Probes.
    probe_layers: list[int] = field(default_factory=list)
    probe_epochs: int = 10
    probe_lr: float = 1e-3

    # Performativity.
    num_prefix_bins: int = 20
    fit_degree: int = 2

    # Bootstrap.
    n_bootstrap: int = 10_000
    ci_level: float = 0.95

    # CoT monitor.
    monitor_model: str = "heuristic"

    # Output.
    output_dir: str = "results/rl_pair"

    # Whether to use <think> tag format for the post-trained model.
    post_trained_uses_think_tags: bool = True


def run_calibration(
    model: LocalModel,
    instances: list[MCQInstance],
    *,
    use_think_tags: bool = False,
    max_new_tokens: int = 2048,
    capture_activations: bool = True,
) -> list[Trace]:
    """Run a model on a set of MCQ instances and return traces."""
    traces = []
    for i, inst in enumerate(instances):
        prompt = format_mcq_prompt(
            inst,
            fewshot=not use_think_tags,
            think_tag=use_think_tags,
        )

        trace = model.generate_trace(
            prompt=prompt,
            instance_id=inst.instance_id,
            correct_answer=inst.correct_answer,
            max_new_tokens=max_new_tokens,
            capture_activations=capture_activations,
        )

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "[%s] %d/%d — answer=%s correct=%s (tokens=%d)",
                model.model_name,
                i + 1,
                len(instances),
                trace.final_answer,
                trace.correct_answer,
                trace.metadata.get("num_generated_tokens", 0),
            )

        traces.append(trace)

    return traces


def evaluate_probe_at_prefixes(
    probe: AttentionProbe,
    traces: list[Trace],
    layer: int,
    num_bins: int = 20,
) -> list[list[tuple[int, float]]]:
    """Evaluate a trained probe at normalised prefix positions.

    For each trace, evaluates the probe at num_bins evenly spaced
    positions along the sequence, returning predictions at each.
    """
    all_predictions: list[list[tuple[int, float]]] = []

    for trace in traces:
        if trace.activations is None:
            all_predictions.append([])
            continue

        acts = trace.activations[layer]
        seq_len = acts.shape[0]
        prompt_tokens = trace.metadata.get("prompt_tokens", 0)
        min_len = max(prompt_tokens + 1, 1)

        prefix_lengths = [
            max(min_len, int(frac * seq_len))
            for frac in np.linspace(0, 1, num_bins, endpoint=True)
        ]
        prefix_lengths[-1] = seq_len

        predictions = probe.predict_at_prefixes(acts, prefix_lengths)
        all_predictions.append(predictions)

    return all_predictions


def evaluate_monitor_at_prefixes(
    monitor: CoTMonitor,
    traces: list[Trace],
    instance_map: dict[str, MCQInstance],
    num_bins: int = 20,
) -> list[list[tuple[int, float]]]:
    """Evaluate CoT monitor at normalised prefix positions.

    For each trace, splits the CoT into steps and evaluates the monitor
    at num_bins evenly spaced step positions.
    """
    all_predictions: list[list[tuple[int, float]]] = []

    for trace in traces:
        inst = instance_map.get(trace.instance_id)
        if inst is None or trace.num_steps() == 0:
            all_predictions.append([])
            continue

        n_steps = trace.num_steps()
        step_indices = [
            max(0, int(frac * n_steps) - 1)
            for frac in np.linspace(0, 1, num_bins, endpoint=True)
        ]
        step_indices[-1] = n_steps - 1

        predictions = []
        for step_idx in step_indices:
            prefix = trace.get_prefix_at_step(step_idx)
            pred_letter, conf = monitor.predict_from_prefix(
                question=inst.question,
                choices=inst.choices,
                cot_prefix=prefix,
            )

            pred_idx = ANSWER_TO_IDX.get(pred_letter, -1)
            predictions.append((pred_idx, conf))

        all_predictions.append(predictions)

    return all_predictions


def _bootstrap_delta_ci(
    base_probe_preds: list[list[tuple[int, float]]],
    base_monitor_preds: list[list[tuple[int, float]]],
    base_labels: list[int],
    pt_probe_preds: list[list[tuple[int, float]]],
    pt_monitor_preds: list[list[tuple[int, float]]],
    pt_labels: list[int],
    num_bins: int,
    fit_degree: int,
    n_bootstrap: int,
    ci_level: float,
) -> tuple[float, float]:
    """Compute bootstrap CI for the performativity rate delta.

    Resamples both models independently and computes the delta
    for each bootstrap iteration.

    Returns:
        (ci_lower, ci_upper) for the delta.
    """
    rng = np.random.default_rng(42)
    n_base = len(base_labels)
    n_pt = len(pt_labels)
    boot_deltas = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        # Resample base.
        b_idx = rng.choice(n_base, size=n_base, replace=True)
        b_probe = [base_probe_preds[i] for i in b_idx]
        b_mon = [base_monitor_preds[i] for i in b_idx]
        b_lbl = [base_labels[i] for i in b_idx]

        b_probe_acc = bin_accuracies(b_probe, b_lbl, num_bins)
        b_mon_acc = bin_accuracies(b_mon, b_lbl, num_bins)
        b_rate = compute_performativity_rate(b_probe_acc, b_mon_acc, fit_degree)

        # Resample post-trained.
        p_idx = rng.choice(n_pt, size=n_pt, replace=True)
        p_probe = [pt_probe_preds[i] for i in p_idx]
        p_mon = [pt_monitor_preds[i] for i in p_idx]
        p_lbl = [pt_labels[i] for i in p_idx]

        p_probe_acc = bin_accuracies(p_probe, p_lbl, num_bins)
        p_mon_acc = bin_accuracies(p_mon, p_lbl, num_bins)
        p_rate = compute_performativity_rate(p_probe_acc, p_mon_acc, fit_degree)

        boot_deltas[b] = p_rate - b_rate

    alpha = (1 - ci_level) / 2
    return (
        float(np.quantile(boot_deltas, alpha)),
        float(np.quantile(boot_deltas, 1 - alpha)),
    )


def run_experiment(config: ExperimentConfig) -> dict:
    """Run the full RL-pair experiment.

    Returns:
        Results dict with performativity rates, matching diagnostics,
        and accuracy curves.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load datasets ---
    logger.info("Loading datasets...")
    mmlu_instances = load_mmlu(max_items=config.mmlu_n)
    gpqa_instances = load_gpqa(max_items=config.gpqa_n)
    all_instances = mmlu_instances + gpqa_instances
    instance_map = {inst.instance_id: inst for inst in all_instances}
    logger.info(
        "Loaded %d MMLU + %d GPQA = %d total instances",
        len(mmlu_instances),
        len(gpqa_instances),
        len(all_instances),
    )

    # --- 2. Run both models ---
    model_traces: dict[str, list[Trace]] = {}
    for model_key, model_name, use_think in [
        ("base", config.base_model, False),
        (
            "post_trained",
            config.post_trained_model,
            config.post_trained_uses_think_tags,
        ),
    ]:
        logger.info("=== Running %s: %s ===", model_key, model_name)
        model = LocalModel(model_name)
        traces = run_calibration(
            model,
            all_instances,
            use_think_tags=use_think,
            max_new_tokens=config.max_new_tokens,
            capture_activations=True,
        )
        model_traces[model_key] = traces

        # Save traces.
        trace_dir = output_dir / model_key / "traces"
        for trace in traces:
            trace.save(trace_dir, save_activations=True)

        # Free the model to make room for the next one.
        del model

    # --- 3. Capability matching ---
    logger.info("Computing capability matching...")
    base_pass_rates = compute_pass_rates(
        instance_ids=[t.instance_id for t in model_traces["base"]],
        answers=[t.final_answer for t in model_traces["base"]],
        correct_answers=[t.correct_answer for t in model_traces["base"]],
    )
    pt_pass_rates = compute_pass_rates(
        instance_ids=[t.instance_id for t in model_traces["post_trained"]],
        answers=[t.final_answer for t in model_traces["post_trained"]],
        correct_answers=[
            t.correct_answer for t in model_traces["post_trained"]
        ],
    )

    match_results = {}
    for eps in config.epsilon_values:
        match_results[eps] = select_matched_pool(
            base_pass_rates, pt_pass_rates, eps
        )

    primary_match = match_results[config.primary_epsilon]
    matched_ids = set(primary_match.matched_ids)
    logger.info(
        "Primary match (ε=%.3f): %d items",
        config.primary_epsilon,
        len(matched_ids),
    )

    if not matched_ids:
        logger.error(
            "No items matched at ε=%.3f. Cannot proceed with probe "
            "training. Try a larger epsilon or more calibration items.",
            config.primary_epsilon,
        )
        return {
            "error": "empty_matched_pool",
            "matching": {
                str(eps): {
                    "n_matched": len(mr.matched_ids),
                    "match_fraction": mr.match_fraction,
                }
                for eps, mr in match_results.items()
            },
        }

    # Filter traces to matched pool.
    matched_traces: dict[str, list[Trace]] = {
        "base": [
            t
            for t in model_traces["base"]
            if t.instance_id in matched_ids
        ],
        "post_trained": [
            t
            for t in model_traces["post_trained"]
            if t.instance_id in matched_ids
        ],
    }

    # --- 4. Train probes ---
    logger.info("Training probes...")
    probe_results: dict[str, dict] = {}

    for model_key in ("base", "post_trained"):
        traces = matched_traces[model_key]
        traces_with_acts = [t for t in traces if t.activations is not None]

        if not traces_with_acts:
            logger.warning(
                "[%s] No traces with activations in matched pool. "
                "Skipping probe training.",
                model_key,
            )
            probe_results[model_key] = {
                "probe": None,
                "layer": -1,
                "accuracy": 0.0,
            }
            continue

        # Determine which layers to try.
        if config.probe_layers:
            layers = config.probe_layers
        else:
            n_layers = traces_with_acts[0].activations.shape[0]
            layers = [n_layers // 2, int(n_layers * 0.75), n_layers - 1]

        activations = [t.activations for t in traces_with_acts]
        labels = [t.final_answer for t in traces_with_acts]
        prompt_lens = [
            t.metadata.get("prompt_tokens", 0) for t in traces_with_acts
        ]
        hidden_dim = activations[0].shape[-1]

        best_probe = None
        best_layer = -1
        best_acc = -1.0

        for layer in layers:
            logger.info("[%s] Training probe at layer %d...", model_key, layer)
            probe = train_probe(
                activations=activations,
                labels=labels,
                prompt_lengths=prompt_lens,
                hidden_dim=hidden_dim,
                layer=layer,
                epochs=config.probe_epochs,
                lr=config.probe_lr,
            )

            correct = sum(
                1
                for act, lbl in zip(activations, labels)
                if probe.predict(act[layer])[0]
                == ANSWER_TO_IDX.get(lbl.upper(), -1)
            )
            acc = correct / len(activations)
            logger.info("[%s] Layer %d accuracy: %.3f", model_key, layer, acc)

            if acc > best_acc:
                best_acc = acc
                best_probe = probe
                best_layer = layer

        probe_results[model_key] = {
            "probe": best_probe,
            "layer": best_layer,
            "accuracy": best_acc,
        }
        logger.info(
            "[%s] Best probe: layer %d, accuracy %.3f",
            model_key,
            best_layer,
            best_acc,
        )

    # --- 5. Evaluate probe and monitor at prefix bins ---
    logger.info("Evaluating probes and monitor at prefix bins...")
    monitor = CoTMonitor(judge_model=config.monitor_model)

    # Store per-model predictions for delta bootstrap.
    all_probe_preds: dict[str, list[list[tuple[int, float]]]] = {}
    all_monitor_preds: dict[str, list[list[tuple[int, float]]]] = {}
    all_true_labels: dict[str, list[int]] = {}
    per_model_rates: dict[str, dict] = {}

    for model_key in ("base", "post_trained"):
        traces = matched_traces[model_key]
        probe_info = probe_results[model_key]

        if probe_info["probe"] is None:
            logger.warning(
                "[%s] No probe available. Skipping evaluation.", model_key
            )
            per_model_rates[model_key] = {
                "rate": 0.0,
                "probe_curve": [],
                "monitor_curve": [],
            }
            all_probe_preds[model_key] = []
            all_monitor_preds[model_key] = []
            all_true_labels[model_key] = []
            continue

        probe = probe_info["probe"]
        layer = probe_info["layer"]

        probe_preds = evaluate_probe_at_prefixes(
            probe, traces, layer, config.num_prefix_bins
        )
        monitor_preds = evaluate_monitor_at_prefixes(
            monitor, traces, instance_map, config.num_prefix_bins
        )
        true_labels = [
            ANSWER_TO_IDX.get(t.correct_answer.upper(), -1) for t in traces
        ]

        # Store for delta bootstrap.
        all_probe_preds[model_key] = probe_preds
        all_monitor_preds[model_key] = monitor_preds
        all_true_labels[model_key] = true_labels

        # Compute per-model performativity.
        probe_acc = bin_accuracies(
            probe_preds, true_labels, config.num_prefix_bins
        )
        monitor_acc = bin_accuracies(
            monitor_preds, true_labels, config.num_prefix_bins
        )
        rate = compute_performativity_rate(
            probe_acc, monitor_acc, config.fit_degree
        )

        per_model_rates[model_key] = {
            "rate": rate,
            "probe_curve": probe_acc.tolist(),
            "monitor_curve": monitor_acc.tolist(),
        }
        logger.info("[%s] Performativity rate: %.4f", model_key, rate)

    # --- 6. Compute delta with bootstrap CI ---
    delta = (
        per_model_rates["post_trained"]["rate"]
        - per_model_rates["base"]["rate"]
    )

    # Bootstrap the delta directly by resampling both models independently.
    can_bootstrap = (
        len(all_probe_preds.get("base", [])) > 0
        and len(all_probe_preds.get("post_trained", [])) > 0
    )

    if can_bootstrap:
        delta_ci_lower, delta_ci_upper = _bootstrap_delta_ci(
            base_probe_preds=all_probe_preds["base"],
            base_monitor_preds=all_monitor_preds["base"],
            base_labels=all_true_labels["base"],
            pt_probe_preds=all_probe_preds["post_trained"],
            pt_monitor_preds=all_monitor_preds["post_trained"],
            pt_labels=all_true_labels["post_trained"],
            num_bins=config.num_prefix_bins,
            fit_degree=config.fit_degree,
            n_bootstrap=config.n_bootstrap,
            ci_level=config.ci_level,
        )
    else:
        delta_ci_lower = delta
        delta_ci_upper = delta

    logger.info(
        "Delta (post_trained - base): %.4f [%.4f, %.4f]",
        delta,
        delta_ci_lower,
        delta_ci_upper,
    )

    # --- 7. Save results ---
    final_results = {
        "config": {
            "base_model": config.base_model,
            "post_trained_model": config.post_trained_model,
            "primary_epsilon": config.primary_epsilon,
            "num_matched": len(matched_ids),
        },
        "matching": {
            str(eps): {
                "n_matched": len(mr.matched_ids),
                "match_fraction": mr.match_fraction,
                "base_pass_rate": mr.model_a_pass_rate,
                "pt_pass_rate": mr.model_b_pass_rate,
            }
            for eps, mr in match_results.items()
        },
        "probes": {
            k: {"layer": v["layer"], "accuracy": v["accuracy"]}
            for k, v in probe_results.items()
        },
        "performativity": {
            "base": per_model_rates["base"],
            "post_trained": per_model_rates["post_trained"],
            "delta": delta,
            "delta_ci_lower": delta_ci_lower,
            "delta_ci_upper": delta_ci_upper,
        },
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info("Results saved to %s", results_file)

    return final_results
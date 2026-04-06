"""Capability matching and difficulty estimation.

For the RL-pair experiment: identifies the subset of tasks where two models
have similar pass rates, isolating the training variable from capability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of capability matching between two models."""

    matched_ids: list[str]
    epsilon: float
    model_a_pass_rate: float  # Overall pass rate on matched pool.
    model_b_pass_rate: float
    total_candidates: int
    match_fraction: float  # Fraction of candidates that matched.


def compute_pass_rates(
    instance_ids: list[str],
    answers: list[str],
    correct_answers: list[str],
    n_runs: int = 1,
) -> dict[str, float]:
    """Compute per-item pass rates.

    For deterministic (temperature=0) generation, pass rate is 0 or 1.
    For stochastic generation with n_runs > 1, it's the fraction correct.

    Args:
        instance_ids: Unique ID per instance.
        answers: Model's predicted answers.
        correct_answers: Ground truth answers.
        n_runs: Number of runs per item (1 for greedy decoding).

    Returns:
        Dict mapping instance_id -> pass_rate.
    """
    if n_runs == 1:
        return {
            iid: float(ans.upper() == correct.upper())
            for iid, ans, correct in zip(instance_ids, answers, correct_answers)
        }

    # Multi-run: aggregate.
    from collections import defaultdict

    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for iid, ans, correct in zip(instance_ids, answers, correct_answers):
        total_counts[iid] += 1
        if ans.upper() == correct.upper():
            correct_counts[iid] += 1

    return {
        iid: correct_counts[iid] / total_counts[iid]
        for iid in total_counts
    }


def select_matched_pool(
    pass_rates_a: dict[str, float],
    pass_rates_b: dict[str, float],
    epsilon: float = 0.05,
) -> MatchResult:
    """Select items where two models have similar pass rates.

    Args:
        pass_rates_a: Per-item pass rates for model A.
        pass_rates_b: Per-item pass rates for model B.
        epsilon: Maximum allowed difference in pass rates.

    Returns:
        MatchResult with the matched instance IDs and diagnostics.
    """
    common_ids = set(pass_rates_a.keys()) & set(pass_rates_b.keys())
    total = len(common_ids)

    matched = []
    for iid in common_ids:
        if abs(pass_rates_a[iid] - pass_rates_b[iid]) <= epsilon:
            matched.append(iid)

    matched.sort()

    # Compute overall pass rates on the matched pool.
    if matched:
        a_rate = np.mean([pass_rates_a[iid] for iid in matched])
        b_rate = np.mean([pass_rates_b[iid] for iid in matched])
    else:
        a_rate = b_rate = 0.0

    result = MatchResult(
        matched_ids=matched,
        epsilon=epsilon,
        model_a_pass_rate=float(a_rate),
        model_b_pass_rate=float(b_rate),
        total_candidates=total,
        match_fraction=len(matched) / total if total > 0 else 0.0,
    )

    logger.info(
        "Matched %d / %d items (%.1f%%) at ε=%.3f. "
        "Pass rates: A=%.3f, B=%.3f",
        len(matched),
        total,
        result.match_fraction * 100,
        epsilon,
        a_rate,
        b_rate,
    )

    return result
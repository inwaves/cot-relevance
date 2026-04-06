"""Plotting utilities for performativity experiments.

Generates the key figures for the RL-pair experiment:
    1. Probe vs monitor accuracy curves (RT-style Figure 2 equivalent).
    2. Performativity rate comparison bar chart.
    3. Capability matching diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_curves(
    probe_curve: list[float],
    monitor_curve: list[float],
    *,
    title: str = "Probe vs Monitor Accuracy",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot probe and monitor accuracy over normalised prefix bins.

    This is the equivalent of Reasoning Theater Figure 2.
    """
    num_bins = len(probe_curve)
    x = np.linspace(0, 100, num_bins)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(x, probe_curve, "o-", label="Attention Probe", color="#2196F3")
    ax.plot(x, monitor_curve, "s-", label="CoT Monitor", color="#FF9800")
    ax.fill_between(
        x,
        probe_curve,
        monitor_curve,
        alpha=0.15,
        color="#9C27B0",
        label="Performativity gap",
    )
    ax.set_xlabel("CoT prefix (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_performativity_comparison(
    base_rate: float,
    base_ci: tuple[float, float],
    pt_rate: float,
    pt_ci: tuple[float, float],
    *,
    base_label: str = "Base",
    pt_label: str = "Post-trained",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing performativity rates with error bars."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    labels = [base_label, pt_label]
    rates = [base_rate, pt_rate]
    ci_lower = [rate - ci[0] for rate, ci in zip(rates, [base_ci, pt_ci])]
    ci_upper = [ci[1] - rate for rate, ci in zip(rates, [base_ci, pt_ci])]
    errors = [ci_lower, ci_upper]

    colors = ["#4CAF50", "#F44336"]
    bars = ax.bar(labels, rates, yerr=errors, capsize=8, color=colors, alpha=0.8)

    ax.set_ylabel("Performativity Rate")
    ax.set_title("Post-Training Effect on Performativity")
    ax.grid(axis="y", alpha=0.3)

    # Annotate with values.
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_matching_sensitivity(
    epsilon_values: list[float],
    n_matched: list[int],
    match_fractions: list[float],
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot capability matching sensitivity across epsilon values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(
        [str(e) for e in epsilon_values],
        n_matched,
        color="#2196F3",
        alpha=0.8,
    )
    ax1.set_xlabel("ε (matching threshold)")
    ax1.set_ylabel("Number of matched items")
    ax1.set_title("Matched Pool Size vs ε")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(
        [str(e) for e in epsilon_values],
        [f * 100 for f in match_fractions],
        color="#FF9800",
        alpha=0.8,
    )
    ax2.set_xlabel("ε (matching threshold)")
    ax2.set_ylabel("Match fraction (%)")
    ax2.set_title("Match Fraction vs ε")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
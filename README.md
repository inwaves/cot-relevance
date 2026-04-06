# CoT Relevance

Experiments on the difficulty-dependence of chain-of-thought faithfulness in reasoning models.

Two independent projects sharing infrastructure:

1. **RL-Pair Performativity** (`docs/rl-pair-performativity.md`) — At matched capability, does RL post-training increase performative CoT? Tests the commons paper's causal prediction.

2. **Multi-Step Performativity** (`docs/multi-step-performativity.md`) — Does the (model, task)-dependent faithfulness gradient replicate in multi-step optimisation tasks with rich action traces?

## Background

- `docs/cot-faithfulness-background.md` — Evidence catalog for the difficulty-dependent faithfulness hypothesis
- `docs/design-doc.md` — Shared code infrastructure design

## Setup

```bash
uv sync
uv sync --extra dev  # includes ruff, pytest
```

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

See `docs/design-doc.md` for architecture overview and execution order.
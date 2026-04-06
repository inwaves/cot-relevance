"""Trace dataclass: the central data object for all experiments.

A Trace captures everything produced by a single model run on a single instance:
CoT text, final answer, parsed steps, optional activations, and metadata.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Trace:
    """A single model inference trace with optional activation capture.

    Attributes:
        model_id: HuggingFace model name or API model identifier.
        instance_id: Unique identifier for the task instance.
        prompt: The full prompt sent to the model.
        raw_output: The complete model output (including think tags if present).
        cot_text: Extracted chain-of-thought text (between <think>...</think>
            if present, otherwise the reasoning portion of the output).
        final_answer: The model's final answer (e.g. "A", "B", "C", "D").
        cot_steps: CoT split into reasoning steps (paragraphs).
        correct_answer: Ground truth answer, if known.
        activations: Per-layer hidden states, shape (num_layers, seq_len, hidden_dim).
            Only populated when capture_activations=True during inference.
            Not serialized to JSON — saved separately as tensors.
        metadata: Timing, token counts, generation config, etc.
    """

    model_id: str
    instance_id: str
    prompt: str
    raw_output: str
    cot_text: str = ""
    final_answer: str = ""
    cot_steps: list[str] = field(default_factory=list)
    correct_answer: str = ""
    activations: Optional[torch.Tensor] = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.cot_steps and self.cot_text:
            self.cot_steps = self._split_into_steps(self.cot_text)

    @staticmethod
    def _split_into_steps(text: str) -> list[str]:
        """Split CoT text into reasoning steps by paragraph boundaries."""
        steps = [s.strip() for s in text.split("\n\n") if s.strip()]
        if not steps and text.strip():
            # Fall back to single-newline splitting if no double newlines.
            steps = [s.strip() for s in text.split("\n") if s.strip()]
        return steps

    @staticmethod
    def parse_think_tags(raw_output: str) -> tuple[str, str]:
        """Extract CoT and final answer from output with <think> tags.

        Returns:
            (cot_text, final_answer) where cot_text is the content between
            <think> and </think>, and final_answer is everything after.
        """
        think_start = raw_output.find("<think>")
        think_end = raw_output.find("</think>")

        if think_start != -1 and think_end != -1:
            cot = raw_output[think_start + len("<think>") : think_end].strip()
            answer = raw_output[think_end + len("</think>") :].strip()
            return cot, answer

        # No think tags — treat entire output as potential CoT + answer.
        return "", raw_output.strip()

    @staticmethod
    def parse_fewshot_output(raw_output: str) -> tuple[str, str]:
        """Extract CoT and final answer from few-shot prompted output.

        Looks for a final "Answer: X" or "The answer is X" pattern.
        Everything before is treated as CoT.

        Returns:
            (cot_text, final_answer)
        """
        import re

        # Try common answer patterns (case-insensitive).
        patterns = [
            r"(?:The answer is|Answer:)\s*\(?([A-Da-d])\)?",
            r"\b([A-D])\s*$",  # Single letter at end of output.
        ]

        for pattern in patterns:
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                cot = raw_output[: match.start()].strip()
                return cot, answer

        return raw_output.strip(), ""

    def get_prefix_at_step(self, step_idx: int) -> str:
        """Return the CoT prefix up to and including step_idx."""
        return "\n\n".join(self.cot_steps[: step_idx + 1])

    def num_steps(self) -> int:
        """Number of reasoning steps in the trace."""
        return len(self.cot_steps)

    def is_correct(self) -> bool:
        """Check if the model's answer matches ground truth."""
        if not self.correct_answer or not self.final_answer:
            return False
        return self.final_answer.upper() == self.correct_answer.upper()

    def to_dict(self) -> dict:
        """Serialize to dict (without activations)."""
        d = asdict(self)
        d.pop("activations", None)
        return d

    def save(self, path: Path, save_activations: bool = True):
        """Save trace to disk.

        JSON for text fields, .pt for activations.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        trace_file = path / f"{self.instance_id}.json"
        with open(trace_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        if save_activations and self.activations is not None:
            act_file = path / f"{self.instance_id}_activations.pt"
            torch.save(self.activations, act_file)

    @classmethod
    def load(cls, path: Path, instance_id: str,
             load_activations: bool = True) -> Trace:
        """Load a trace from disk."""
        path = Path(path)

        trace_file = path / f"{instance_id}.json"
        with open(trace_file) as f:
            d = json.load(f)

        activations = None
        if load_activations:
            act_file = path / f"{instance_id}_activations.pt"
            if act_file.exists():
                activations = torch.load(act_file, weights_only=True)

        return cls(activations=activations, **d)
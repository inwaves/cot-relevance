"""MCQ benchmark loader for MMLU-Redux and GPQA-Diamond.

Loads datasets from HuggingFace, formats them into prompts,
and provides iteration over instances for the RL-pair experiment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)

ANSWER_LETTERS = ["A", "B", "C", "D"]


@dataclass
class MCQInstance:
    """A single multiple-choice question."""

    instance_id: str
    question: str
    choices: list[str]
    correct_answer: str  # "A", "B", "C", or "D"
    subject: str = ""
    source: str = ""  # "mmlu" or "gpqa"


# Few-shot examples for eliciting step-by-step reasoning from base models.
FEWSHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": "C",
        "reasoning": (
            "Let me think through each option. Berlin is the capital of Germany, "
            "not France. Madrid is the capital of Spain. Paris is well-known as "
            "the capital of France. Rome is the capital of Italy. "
            "The answer is Paris."
        ),
    },
    {
        "question": "Which of these is a prime number?",
        "choices": ["4", "6", "9", "11"],
        "answer": "D",
        "reasoning": (
            "I need to check which numbers are prime. 4 = 2 × 2, not prime. "
            "6 = 2 × 3, not prime. 9 = 3 × 3, not prime. "
            "11 has no divisors other than 1 and itself, so it is prime."
        ),
    },
]


def format_mcq_prompt(
    instance: MCQInstance,
    fewshot: bool = True,
    think_tag: bool = False,
) -> str:
    """Format an MCQ instance into a prompt.

    Args:
        instance: The question to format.
        fewshot: Whether to prepend few-shot examples.
        think_tag: If True, use <think> tag format (for reasoning models).
            If False, use few-shot step-by-step format (for base models).

    Returns:
        Formatted prompt string.
    """
    parts = []

    if fewshot and not think_tag:
        parts.append(
            "Answer each question by reasoning step-by-step, "
            "then give your final answer.\n"
        )
        for ex in FEWSHOT_EXAMPLES:
            parts.append(f"Question: {ex['question']}")
            for i, choice in enumerate(ex["choices"]):
                parts.append(f"  {ANSWER_LETTERS[i]}. {choice}")
            parts.append(f"\nReasoning: {ex['reasoning']}")
            parts.append(f"Answer: {ex['answer']}\n")

    # The actual question.
    parts.append(f"Question: {instance.question}")
    for i, choice in enumerate(instance.choices):
        parts.append(f"  {ANSWER_LETTERS[i]}. {choice}")

    if think_tag:
        parts.append("\n<think>")
    else:
        parts.append("\nReasoning:")

    return "\n".join(parts)


def load_mmlu(
    split: str = "test",
    max_items: Optional[int] = None,
    subjects: Optional[list[str]] = None,
) -> list[MCQInstance]:
    """Load MMLU-Redux from HuggingFace.

    Uses the 'TIGER-Lab/MMLU-Pro' dataset as a readily available
    MMLU variant. Falls back to 'cais/mmlu' if unavailable.

    Args:
        split: Dataset split ("test", "validation", etc.).
        max_items: Maximum number of items to load.
        subjects: Optional list of subjects to filter by.

    Returns:
        List of MCQInstance objects.
    """
    logger.info("Loading MMLU (split=%s, max=%s)", split, max_items)

    try:
        ds = load_dataset("cais/mmlu", "all", split=split)
    except Exception:
        logger.warning("Failed to load cais/mmlu, trying TIGER-Lab/MMLU-Pro")
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)

    instances = []
    for i, row in enumerate(ds):
        if max_items and i >= max_items:
            break

        # Handle different column naming conventions.
        question = row.get("question", row.get("input", ""))
        subject = row.get("subject", row.get("category", ""))

        if subjects and subject not in subjects:
            continue

        # Get choices — different datasets format these differently.
        if "choices" in row:
            choices = row["choices"]
        else:
            choices = [row.get(f"choice_{l}", row.get(l, ""))
                       for l in ["A", "B", "C", "D"]]

        # Get answer.
        answer = row.get("answer", row.get("target", ""))
        if isinstance(answer, int):
            answer = ANSWER_LETTERS[answer]

        instances.append(
            MCQInstance(
                instance_id=f"mmlu_{i:05d}",
                question=question,
                choices=choices[:4],  # Ensure exactly 4.
                correct_answer=str(answer).upper(),
                subject=subject,
                source="mmlu",
            )
        )

    logger.info("Loaded %d MMLU instances", len(instances))
    return instances


def load_gpqa(
    split: str = "train",
    max_items: Optional[int] = None,
) -> list[MCQInstance]:
    """Load GPQA-Diamond from HuggingFace.

    Args:
        split: Dataset split.
        max_items: Maximum number of items to load.

    Returns:
        List of MCQInstance objects.
    """
    logger.info("Loading GPQA-Diamond (split=%s, max=%s)", split, max_items)

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split)

    instances = []
    for i, row in enumerate(ds):
        if max_items and i >= max_items:
            break

        question = row.get("Question", row.get("question", ""))
        choices = []
        correct_idx = -1

        # GPQA has Correct Answer and Incorrect Answer 1/2/3.
        if "Correct Answer" in row:
            all_answers = [
                row["Correct Answer"],
                row.get("Incorrect Answer 1", ""),
                row.get("Incorrect Answer 2", ""),
                row.get("Incorrect Answer 3", ""),
            ]
            # Shuffle but track correct index.
            import random

            rng = random.Random(i)  # Deterministic shuffle per question.
            indexed = list(enumerate(all_answers))
            rng.shuffle(indexed)
            choices = [a for _, a in indexed]
            correct_idx = next(
                j for j, (orig_idx, _) in enumerate(indexed) if orig_idx == 0
            )
        else:
            for key in ["choice_a", "choice_b", "choice_c", "choice_d",
                         "A", "B", "C", "D"]:
                if key in row:
                    choices.append(row[key])
            answer_raw = row.get("answer", row.get("Answer", 0))
            if isinstance(answer_raw, int):
                correct_idx = answer_raw
            elif isinstance(answer_raw, str) and answer_raw.upper() in ANSWER_LETTERS:
                correct_idx = ANSWER_LETTERS.index(answer_raw.upper())

        correct_answer = ANSWER_LETTERS[correct_idx] if 0 <= correct_idx < 4 else ""

        instances.append(
            MCQInstance(
                instance_id=f"gpqa_{i:04d}",
                question=question,
                choices=choices[:4],
                correct_answer=correct_answer,
                subject=row.get("Subdomain", row.get("subdomain", "")),
                source="gpqa",
            )
        )

    logger.info("Loaded %d GPQA instances", len(instances))
    return instances
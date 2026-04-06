"""Local model inference with activation capture.

Loads HuggingFace models, generates text, and captures residual stream
activations at all layers for probe training. Designed for Apple Silicon
(MPS backend) but falls back to CPU.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .trace import Trace

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Select the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _default_dtype(device: torch.device) -> torch.dtype:
    """Select a safe default dtype for the device.

    float16 is efficient on GPU/MPS but can cause issues on CPU.
    """
    if device.type == "cpu":
        return torch.float32
    return torch.float16


class LocalModel:
    """Local model inference with optional activation capture.

    Usage:
        model = LocalModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        trace = model.generate_trace(
            prompt="...",
            instance_id="mmlu_001",
            correct_answer="B",
            max_new_tokens=2048,
            capture_activations=True,
        )
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device or _get_device()
        self.dtype = dtype or _default_dtype(self.device)

        logger.info("Loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            "Loading model: %s → %s (%s)", model_name, self.device, self.dtype
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        logger.info(
            "Model loaded: %d layers, hidden_dim=%d",
            self.num_layers,
            self.hidden_dim,
        )

    def generate_trace(
        self,
        prompt: str,
        instance_id: str,
        correct_answer: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        capture_activations: bool = True,
    ) -> Trace:
        """Generate a completion and optionally capture activations.

        Steps:
            1. Generate the full response autoregressively.
            2. If capture_activations, run a single forward pass over the
               complete sequence (prompt + response) to extract hidden states
               at every layer and position.
            3. Parse CoT and final answer from the response.
            4. Return a Trace object.

        Args:
            prompt: The formatted prompt string.
            instance_id: Unique ID for this task instance.
            correct_answer: Ground truth (e.g. "A", "B", "C", "D").
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            capture_activations: Whether to run the activation-capture pass.

        Returns:
            A Trace with all fields populated.
        """
        t0 = time.time()

        # Tokenize prompt.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate.
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode.
        generated_ids = output_ids[0, prompt_len:]
        raw_output = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        gen_time = time.time() - t0
        num_tokens = len(generated_ids)

        # Parse CoT and answer.
        if "<think>" in raw_output:
            cot_text, final_answer = Trace.parse_think_tags(raw_output)
        else:
            cot_text, final_answer = Trace.parse_fewshot_output(raw_output)

        # Capture activations via a forward pass over the full sequence.
        activations = None
        if capture_activations:
            activations = self._capture_activations(output_ids)

        metadata = {
            "generation_time_s": round(gen_time, 2),
            "num_generated_tokens": num_tokens,
            "prompt_tokens": prompt_len,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

        return Trace(
            model_id=self.model_name,
            instance_id=instance_id,
            prompt=prompt,
            raw_output=raw_output,
            cot_text=cot_text,
            final_answer=final_answer,
            correct_answer=correct_answer,
            activations=activations,
            metadata=metadata,
        )

    def _capture_activations(
        self, full_ids: torch.Tensor
    ) -> torch.Tensor:
        """Run a forward pass and extract all hidden states.

        Args:
            full_ids: Token IDs for the complete sequence (prompt + response),
                shape (1, total_seq_len).

        Returns:
            Tensor of shape (num_layers+1, seq_len, hidden_dim) containing
            residual stream activations at each layer. Layer 0 is the
            embedding output; layers 1..N are transformer block outputs.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                output_hidden_states=True,
            )

        # outputs.hidden_states is a tuple of (num_layers+1,) tensors,
        # each of shape (batch=1, seq_len, hidden_dim).
        # Stack into (num_layers+1, seq_len, hidden_dim).
        hidden_states = torch.stack(
            [h.squeeze(0) for h in outputs.hidden_states]
        )

        # Move to CPU to free device memory.
        return hidden_states.cpu().float()

    def forced_answer(
        self,
        prompt: str,
        cot_prefix: str,
        forced_prompt: str = (
            "\n\nBased on my reasoning above, the answer is ("
        ),
        max_new_tokens: int = 5,
    ) -> str:
        """Forced answering: append a prefix of CoT, then force an answer.

        Reconstructs the input as:
            {prompt}\\n{cot_prefix}{forced_prompt}

        The newline between prompt and cot_prefix ensures proper separation
        regardless of whether the prompt ends with a reasoning cue
        (e.g. "\\nReasoning:" or "\\n<think>").

        Args:
            prompt: The original question prompt (as produced by
                format_mcq_prompt, which may end with "\\nReasoning:"
                or "\\n<think>").
            cot_prefix: The CoT text up to the truncation point.
            forced_prompt: Text appended to elicit the answer.
            max_new_tokens: Short, since we just want a letter.

        Returns:
            The predicted answer letter (A/B/C/D) or empty string.
        """
        # Ensure clean boundaries between prompt, CoT prefix, and
        # forced-answer cue. The prompt already ends with a reasoning
        # marker; cot_prefix is the reasoning text; forced_prompt
        # transitions to the answer.
        full_input = prompt.rstrip() + "\n" + cot_prefix.strip() + forced_prompt

        inputs = self.tokenizer(full_input, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        # Extract the first A-D letter.
        for char in text:
            if char.upper() in "ABCD":
                return char.upper()
        return ""
"""Attention-pooling probe for answer prediction from hidden states.

Replicates the architecture from Boppana et al. (2026), "Reasoning Theater":
    z = W_v @ H @ softmax(W_q @ H)

where H is (seq_len, hidden_dim) hidden states at a chosen layer.
W_q produces attention weights over tokens; W_v projects to class logits.
The probe is trained on random prefixes of reasoning traces to be robust
to varying sequence positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionProbe(nn.Module):
    """Attention-pooling probe for multi-class prediction.

    Args:
        hidden_dim: Dimensionality of the model's hidden states.
        num_classes: Number of output classes (4 for MCQ A/B/C/D).
    """

    def __init__(self, hidden_dim: int, num_classes: int = 4):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, 1, bias=False)
        self.W_v = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
                or (seq_len, hidden_dim) for a single example.

        Returns:
            Logits of shape (batch, num_classes) or (num_classes,).
        """
        squeeze = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze = True

        # Attention weights: (batch, seq_len, 1).
        attn_logits = self.W_q(hidden_states)
        attn_weights = torch.softmax(attn_logits, dim=1)

        # Project and pool: (batch, seq_len, num_classes) * (batch, seq_len, 1)
        # -> sum over seq_len -> (batch, num_classes).
        projected = self.W_v(hidden_states)
        pooled = (projected * attn_weights).sum(dim=1)

        if squeeze:
            pooled = pooled.squeeze(0)
        return pooled

    def predict(self, hidden_states: torch.Tensor) -> tuple[int, float]:
        """Predict class and confidence.

        Args:
            hidden_states: (seq_len, hidden_dim) for a single example.

        Returns:
            (predicted_class, confidence) where confidence is the
            softmax probability of the predicted class.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(hidden_states)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1).item()
            conf = probs[pred].item()
        return pred, conf

    def predict_at_prefixes(
        self,
        hidden_states: torch.Tensor,
        prefix_lengths: list[int],
    ) -> list[tuple[int, float]]:
        """Predict at multiple prefix lengths.

        Args:
            hidden_states: (seq_len, hidden_dim) full sequence hidden states.
            prefix_lengths: List of token positions to evaluate at.

        Returns:
            List of (predicted_class, confidence) at each prefix length.
        """
        results = []
        for length in prefix_lengths:
            prefix = hidden_states[:length]
            results.append(self.predict(prefix))
        return results
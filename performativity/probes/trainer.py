"""Probe training with random prefix sampling.

The key training detail from Reasoning Theater (Boppana et al., 2026):
during training, for each trace we sample a random prefix length and
train the probe to predict the final answer from that prefix. This makes
the probe robust to varying sequence positions and prevents it from
relying on position-specific features.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .attention_probe import AttentionProbe

logger = logging.getLogger(__name__)


# Maps answer letters to class indices.
ANSWER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


class PrefixDataset(Dataset):
    """Dataset that yields random prefixes of activation sequences.

    Each item is (hidden_states_prefix, label) where the prefix length
    is sampled uniformly at random from [min_prefix, seq_len].
    """

    def __init__(
        self,
        activations: list[torch.Tensor],
        labels: list[int],
        prompt_lengths: list[int],
        min_prefix_tokens: int = 10,
        samples_per_trace: int = 5,
    ):
        """
        Args:
            activations: List of tensors, each (seq_len_i, hidden_dim).
            labels: List of integer class labels (0-3 for A-D).
            prompt_lengths: List of prompt token counts (prefix starts after prompt).
            min_prefix_tokens: Minimum number of response tokens in a prefix.
            samples_per_trace: Number of random prefixes to sample per trace per epoch.
        """
        self.activations = activations
        self.labels = labels
        self.prompt_lengths = prompt_lengths
        self.min_prefix_tokens = min_prefix_tokens
        self.samples_per_trace = samples_per_trace

    def __len__(self) -> int:
        return len(self.activations) * self.samples_per_trace

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        trace_idx = idx % len(self.activations)
        acts = self.activations[trace_idx]
        label = self.labels[trace_idx]
        prompt_len = self.prompt_lengths[trace_idx]
        seq_len = acts.shape[0]

        # Sample a random prefix length (in total tokens, including prompt).
        min_len = prompt_len + self.min_prefix_tokens
        if min_len >= seq_len:
            prefix_len = seq_len
        else:
            prefix_len = random.randint(min_len, seq_len)

        return acts[:prefix_len], label


def collate_variable_length(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences by padding.

    Returns:
        (padded_activations, labels, attention_mask)
    """
    sequences, labels = zip(*batch)
    max_len = max(s.shape[0] for s in sequences)
    hidden_dim = sequences[0].shape[1]

    padded = torch.zeros(len(sequences), max_len, hidden_dim)
    mask = torch.zeros(len(sequences), max_len, 1)

    for i, seq in enumerate(sequences):
        padded[i, : seq.shape[0]] = seq
        mask[i, : seq.shape[0]] = 1.0

    return padded, torch.tensor(labels, dtype=torch.long), mask


class MaskedAttentionProbe(nn.Module):
    """Wrapper that applies attention masking during training."""

    def __init__(self, probe: AttentionProbe):
        super().__init__()
        self.probe = probe

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward with padding mask.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len, 1) — 1 for real tokens, 0 for padding.
        """
        # Mask attention logits: set padding positions to -inf.
        attn_logits = self.probe.W_q(hidden_states)  # (batch, seq_len, 1)
        attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=1)

        projected = self.probe.W_v(hidden_states)
        pooled = (projected * attn_weights).sum(dim=1)
        return pooled


def train_probe(
    activations: list[torch.Tensor],
    labels: list[str],
    prompt_lengths: list[int],
    hidden_dim: int,
    num_classes: int = 4,
    layer: Optional[int] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 16,
    min_prefix_tokens: int = 10,
    samples_per_trace: int = 5,
    device: Optional[torch.device] = None,
) -> AttentionProbe:
    """Train an attention probe on activation prefixes.

    Args:
        activations: List of tensors, each (seq_len_i, hidden_dim) or
            (num_layers, seq_len_i, hidden_dim). If 3D and layer is specified,
            selects that layer.
        labels: List of answer strings ("A", "B", "C", "D").
        prompt_lengths: Number of prompt tokens per trace (so we only sample
            prefixes that include at least some response tokens).
        hidden_dim: Model hidden dimension.
        num_classes: Number of answer choices.
        layer: If activations are 3D (num_layers, seq_len, hidden_dim),
            select this layer index.
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        min_prefix_tokens: Minimum response tokens in a sampled prefix.
        samples_per_trace: Random prefixes per trace per epoch.
        device: Training device.

    Returns:
        Trained AttentionProbe.
    """
    if device is None:
        device = torch.device("cpu")

    # Convert labels to indices.
    label_indices = []
    for lbl in labels:
        idx = ANSWER_TO_IDX.get(lbl.upper())
        if idx is None:
            raise ValueError(f"Unknown label: {lbl!r}. Expected A/B/C/D.")
        label_indices.append(idx)

    # Select layer if needed.
    proc_acts = []
    for act in activations:
        if act.dim() == 3 and layer is not None:
            act = act[layer]  # (seq_len, hidden_dim)
        elif act.dim() == 3:
            raise ValueError(
                "Activations are 3D but no layer specified. "
                "Pass layer= to select a layer."
            )
        proc_acts.append(act.float())

    dataset = PrefixDataset(
        activations=proc_acts,
        labels=label_indices,
        prompt_lengths=prompt_lengths,
        min_prefix_tokens=min_prefix_tokens,
        samples_per_trace=samples_per_trace,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
    )

    probe = AttentionProbe(hidden_dim, num_classes)
    masked_probe = MaskedAttentionProbe(probe).to(device)
    optimizer = torch.optim.Adam(masked_probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        masked_probe.train()
        for batch_acts, batch_labels, batch_mask in loader:
            batch_acts = batch_acts.to(device)
            batch_labels = batch_labels.to(device)
            batch_mask = batch_mask.to(device)

            logits = masked_probe(batch_acts, batch_mask)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        logger.info(
            "Epoch %d/%d — loss: %.4f, accuracy: %.3f",
            epoch + 1,
            epochs,
            avg_loss,
            accuracy,
        )

    # Return the inner probe (without the masking wrapper).
    return probe.cpu()
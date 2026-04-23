"""
Task 2: State Space Model (SSM) for Next Word Prediction.
Implemented from scratch using only nn.Linear, nn.Embedding, nn.Dropout, nn.LayerNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SSMLayer(nn.Module):
    """
    Diagonal State Space Model layer.

    Recurrence:
        h_t = exp(-exp(A)) * h_{t-1} + B @ x_t
        y_t = C @ h_t + D * x_t

    where:
        A: (d_state,)               — diagonal SSM A matrix (log-log parameterization)
        B: (d_state, d_model)       — input projection
        C: (d_model, d_state)       — output projection
        D: (d_model,)               — skip connection (D * x_t)
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal A: parameterised as log of negative log (so stable)
        self.A = nn.Parameter(torch.randn(d_state))

        # B: (d_state, d_model)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)

        # C: (d_model, d_state)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        # D (skip): (d_model,)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        device = x.device

        # Stable transition: a_bar = exp(-exp(A))
        # This ensures 0 < a_bar < 1
        a_bar = torch.exp(-torch.exp(self.A))  # (d_state,)

        # Initialise hidden state
        h = torch.zeros(batch, self.d_state, device=device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]  # (batch, d_model)

            # h_t = a_bar * h_{t-1} + B @ x_t
            # B: (d_state, d_model), xt: (batch, d_model)
            # xt @ B.T -> (batch, d_state)
            h = a_bar * h + xt @ self.B.T  # (batch, d_state)

            # y_t = C @ h_t + D * x_t
            # h: (batch, d_state), C: (d_model, d_state)
            y = h @ self.C.T + self.D * xt  # (batch, d_model)
            outputs.append(y.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (batch, seq_len, d_model)


class SSMModel(nn.Module):
    """
    SSM-based language model for next word prediction.

    Architecture: Embedding → [SSMLayer + LayerNorm + Dropout] × num_layers → Linear(vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_state: int,
        num_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.ssm_layers = nn.ModuleList([SSMLayer(d_model, d_state) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) token ids
        returns: (batch, seq_len, vocab_size)
        """
        emb = self.dropout(self.embedding(x))  # (batch, seq_len, d_model)

        h = emb
        for ssm, norm in zip(self.ssm_layers, self.norms):
            h = norm(ssm(h) + h)  # residual connection + norm
            h = self.dropout(h)

        logits = self.output_proj(h)  # (batch, seq_len, vocab_size)
        return logits

    def next_word_logits(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (batch, context_len) — returns logits for next word (batch, vocab_size)
        """
        logits = self.forward(context)
        return logits[:, -1, :]  # last position

"""
Task 2: Bidirectional LSTM from scratch for Masked Language Modeling.
Uses only nn.Linear, nn.Embedding, nn.Dropout, nn.LayerNorm.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    """Single LSTM cell."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_io = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        i = torch.sigmoid(self.W_ii(x) + self.W_hi(h))
        f = torch.sigmoid(self.W_if(x) + self.W_hf(h))
        g = torch.tanh(self.W_ig(x) + self.W_hg(h))
        o = torch.sigmoid(self.W_io(x) + self.W_ho(h))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class BiLSTMLayer(nn.Module):
    """
    Bidirectional multi-layer LSTM from scratch.
    Forward and backward LSTM stacks, concatenated outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Forward cells: first layer takes input_size, rest take hidden_size
        fwd_cells = []
        bwd_cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            fwd_cells.append(LSTMCell(in_size, hidden_size))
            bwd_cells.append(LSTMCell(in_size, hidden_size))

        self.fwd_cells = nn.ModuleList(fwd_cells)
        self.bwd_cells = nn.ModuleList(bwd_cells)

    def _run_direction(
        self,
        x: torch.Tensor,
        cells: nn.ModuleList,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Run one direction of the LSTM.
        x: (batch, seq_len, input_size)
        Returns: (batch, seq_len, hidden_size) — outputs at top layer
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        h_all = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        c_all = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]

        seq_range = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        top_outputs = []

        for t in seq_range:
            inp = x[:, t, :]
            for layer_idx, cell in enumerate(cells):
                h_all[layer_idx], c_all[layer_idx] = cell(inp, h_all[layer_idx], c_all[layer_idx])
                inp = self.dropout(h_all[layer_idx]) if layer_idx < self.num_layers - 1 else h_all[layer_idx]
            top_outputs.append(inp)

        if reverse:
            top_outputs = top_outputs[::-1]

        return torch.stack(top_outputs, dim=1)  # (batch, seq_len, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, seq_len, 2 * hidden_size)
        """
        fwd_out = self._run_direction(x, self.fwd_cells, reverse=False)
        bwd_out = self._run_direction(x, self.bwd_cells, reverse=True)
        return torch.cat([fwd_out, bwd_out], dim=-1)  # (batch, seq_len, 2*hidden)


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM model for Masked Language Modeling.

    Architecture: Embedding → BiLSTMLayer → LayerNorm → Linear(vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = BiLSTMLayer(embed_dim, hidden_dim, num_layers, dropout)
        self.norm = nn.LayerNorm(2 * hidden_dim)
        self.output_proj = nn.Linear(2 * hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) token ids
        returns: (batch, seq_len, vocab_size)
        """
        emb = self.dropout(self.embedding(x))          # (batch, seq_len, embed_dim)
        h = self.bilstm(emb)                           # (batch, seq_len, 2*hidden_dim)
        h = self.norm(h)
        h = self.dropout(h)
        logits = self.output_proj(h)                   # (batch, seq_len, vocab_size)
        return logits

"""
Task 1: Seq2Seq Models with Bahdanau Attention, from scratch.
No nn.RNN, nn.LSTM, nn.GRU — only nn.Linear, nn.Embedding, nn.Dropout, nn.LayerNorm.
"""

import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cell implementations
# ---------------------------------------------------------------------------


class RNNCell(nn.Module):
    """Vanilla RNN cell: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.W_ih(x) + self.W_hh(h))


class LSTMCell(nn.Module):
    """LSTM cell with separate gate matrices."""

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


# ---------------------------------------------------------------------------
# Bahdanau Attention
# ---------------------------------------------------------------------------


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention:
        score(s_t, h_j) = v^T tanh(W_a s_t + U_a h_j)
        a_t = softmax(score)
        context = sum(a_t * encoder_outputs)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        decoder_hidden:  (batch, hidden_dim)
        encoder_outputs: (batch, src_len, hidden_dim)

        Returns:
            context:  (batch, hidden_dim)
            weights:  (batch, src_len)
        """
        # (batch, 1, hidden_dim) + (batch, src_len, hidden_dim)
        score = self.v(
            torch.tanh(
                self.W_a(decoder_hidden).unsqueeze(1) + self.U_a(encoder_outputs)
            )
        ).squeeze(
            -1
        )  # (batch, src_len)

        weights = F.softmax(score, dim=-1)  # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(
            1
        )  # (batch, hidden_dim)
        return context, weights


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        cells = []
        for i in range(num_layers):
            cells.append(RNNCell(embed_dim if i == 0 else hidden_dim, hidden_dim))
        self.cells = nn.ModuleList(cells)

    def forward(self, x, lengths):
        batch_size, seq_len = x.size()
        device = x.device
        emb = self.dropout(self.embedding(x))
        h = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        all_outputs = []
        for t in range(seq_len):
            inp = emb[:, t, :]
            for li, cell in enumerate(self.cells):
                h[li] = cell(inp, h[li])
                inp = self.dropout(h[li]) if li < self.num_layers - 1 else h[li]
            all_outputs.append(inp.unsqueeze(1))
        outputs = torch.cat(all_outputs, dim=1)
        hidden = torch.stack(h, dim=0)
        return outputs, hidden


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        cells = []
        for i in range(num_layers):
            cells.append(LSTMCell(embed_dim if i == 0 else hidden_dim, hidden_dim))
        self.cells = nn.ModuleList(cells)

    def forward(self, x, lengths):
        batch_size, seq_len = x.size()
        device = x.device
        emb = self.dropout(self.embedding(x))
        h = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        all_outputs = []
        for t in range(seq_len):
            inp = emb[:, t, :]
            for li, cell in enumerate(self.cells):
                h[li], c[li] = cell(inp, h[li], c[li])
                inp = self.dropout(h[li]) if li < self.num_layers - 1 else h[li]
            all_outputs.append(inp.unsqueeze(1))
        outputs = torch.cat(all_outputs, dim=1)
        return outputs, (torch.stack(h, dim=0), torch.stack(c, dim=0))


# ---------------------------------------------------------------------------
# Attention Decoders
# ---------------------------------------------------------------------------


class RNNAttnDecoder(nn.Module):
    """RNN decoder with Bahdanau attention."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim)
        # First cell takes concat(embed, context)
        cells = []
        for i in range(num_layers):
            in_size = embed_dim + hidden_dim if i == 0 else hidden_dim
            cells.append(RNNCell(in_size, hidden_dim))
        self.cells = nn.ModuleList(cells)
        # Output from concat(hidden, context)
        self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        """
        input_token:     (batch,)
        hidden:          (num_layers, batch, hidden_dim)
        encoder_outputs: (batch, src_len, hidden_dim)
        Returns: logits (batch, vocab_size), new_hidden (num_layers, batch, hidden_dim)
        """
        if input_token.dim() == 2:
            input_token = input_token.squeeze(1)
        emb = self.dropout(self.embedding(input_token))  # (batch, embed_dim)
        # Attention using top-layer hidden
        context, _ = self.attention(hidden[-1], encoder_outputs)  # (batch, hidden_dim)
        # First cell input = concat(embed, context)
        h_list = [hidden[i] for i in range(self.num_layers)]
        inp = torch.cat([emb, context], dim=-1)  # (batch, embed_dim + hidden_dim)
        for li, cell in enumerate(self.cells):
            h_list[li] = cell(inp, h_list[li])
            inp = self.dropout(h_list[li]) if li < self.num_layers - 1 else h_list[li]
        # Output projection from concat(top_hidden, context)
        logits = self.output_proj(torch.cat([inp, context], dim=-1))
        return logits, torch.stack(h_list, dim=0)


class LSTMAttnDecoder(nn.Module):
    """LSTM decoder with Bahdanau attention."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim)
        cells = []
        for i in range(num_layers):
            in_size = embed_dim + hidden_dim if i == 0 else hidden_dim
            cells.append(LSTMCell(in_size, hidden_dim))
        self.cells = nn.ModuleList(cells)
        self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        if input_token.dim() == 2:
            input_token = input_token.squeeze(1)
        h_all, c_all = hidden
        emb = self.dropout(self.embedding(input_token))
        context, _ = self.attention(h_all[-1], encoder_outputs)
        h_list = [h_all[i] for i in range(self.num_layers)]
        c_list = [c_all[i] for i in range(self.num_layers)]
        inp = torch.cat([emb, context], dim=-1)
        for li, cell in enumerate(self.cells):
            h_list[li], c_list[li] = cell(inp, h_list[li], c_list[li])
            inp = self.dropout(h_list[li]) if li < self.num_layers - 1 else h_list[li]
        logits = self.output_proj(torch.cat([inp, context], dim=-1))
        return logits, (torch.stack(h_list, dim=0), torch.stack(c_list, dim=0))


# ---------------------------------------------------------------------------
# Seq2Seq Models (with attention)
# ---------------------------------------------------------------------------


class Seq2SeqRNN(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.encoder = RNNEncoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = RNNAttnDecoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        device = src.device
        if src_lengths is None:
            src_lengths = torch.full((batch_size,), src.size(1), dtype=torch.long, device=device)

        encoder_outputs, hidden = self.encoder(src, src_lengths)
        decoder_input = tgt[:, 0]
        outputs = []

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs.append(logits.unsqueeze(1))
            if random.random() < teacher_forcing_ratio:
                decoder_input = tgt[:, t]
            else:
                decoder_input = logits.argmax(dim=-1)

        return torch.cat(outputs, dim=1)

    def decode(self, src, max_len, sos_idx, eos_idx, device):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            src_lengths = torch.full((batch_size,), src.size(1), dtype=torch.long, device=device)
            encoder_outputs, hidden = self.encoder(src, src_lengths)
            decoder_input = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
            decoded = [[] for _ in range(batch_size)]
            finished = [False] * batch_size

            for _ in range(max_len):
                logits, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
                top_tokens = logits.argmax(dim=-1)
                for b in range(batch_size):
                    if not finished[b]:
                        tok = int(top_tokens[b])
                        if tok == eos_idx:
                            finished[b] = True
                        else:
                            decoded[b].append(tok)
                if all(finished):
                    break
                decoder_input = top_tokens

        return decoded


class Seq2SeqLSTM(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.encoder = LSTMEncoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMAttnDecoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        device = src.device
        if src_lengths is None:
            src_lengths = torch.full((batch_size,), src.size(1), dtype=torch.long, device=device)

        encoder_outputs, (h_n, c_n) = self.encoder(src, src_lengths)
        decoder_input = tgt[:, 0]
        hidden = (h_n, c_n)
        outputs = []

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs.append(logits.unsqueeze(1))
            if random.random() < teacher_forcing_ratio:
                decoder_input = tgt[:, t]
            else:
                decoder_input = logits.argmax(dim=-1)

        return torch.cat(outputs, dim=1)

    def decode(self, src, max_len, sos_idx, eos_idx, device):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            src_lengths = torch.full((batch_size,), src.size(1), dtype=torch.long, device=device)
            encoder_outputs, (h_n, c_n) = self.encoder(src, src_lengths)
            decoder_input = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
            hidden = (h_n, c_n)
            decoded = [[] for _ in range(batch_size)]
            finished = [False] * batch_size

            for _ in range(max_len):
                logits, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
                top_tokens = logits.argmax(dim=-1)
                for b in range(batch_size):
                    if not finished[b]:
                        tok = int(top_tokens[b])
                        if tok == eos_idx:
                            finished[b] = True
                        else:
                            decoded[b].append(tok)
                if all(finished):
                    break
                decoder_input = top_tokens

        return decoded

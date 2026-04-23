"""
Task 1: Shared training/evaluation utilities for seq2seq decryption.
"""

import logging
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.task1.dataset import CipherDataset, collate_fn, PAD_IDX, SOS_IDX, EOS_IDX


def _fmt_time(seconds: float) -> str:
    """Format seconds into Xm Ys string."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float = 0.5,
    clip_grad: float = 1.0,
    logger: Optional[logging.Logger] = None,
    log_every: int = 50,
) -> float:
    """
    Run one training epoch.
    Returns average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    num_batches = len(loader)
    t0 = time.time()

    for i, batch in enumerate(tqdm(loader, desc="Train", leave=False), 1):
        cipher_padded, plain_padded, cipher_lengths, plain_lengths = batch
        cipher_padded = cipher_padded.to(device)
        plain_padded = plain_padded.to(device)
        cipher_lengths = cipher_lengths.to(device)

        optimizer.zero_grad()

        # Forward: tgt includes SOS; model predicts positions 1..T
        outputs = model(
            cipher_padded,
            plain_padded,
            src_lengths=cipher_lengths,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        # outputs: (batch, tgt_len-1, vocab_size)
        # targets: plain_padded[:, 1:]  (shifted by 1, EOS at end)

        tgt_out = plain_padded[:, 1:]  # (batch, tgt_len-1)
        batch_size, seq_len, vocab_size = outputs.size()

        loss = criterion(
            outputs.reshape(-1, vocab_size),
            tgt_out.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        # Count non-pad tokens for normalisation
        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if logger and (i % log_every == 0 or i == num_batches):
            elapsed = time.time() - t0
            avg_loss = total_loss / max(total_tokens, 1)
            eta = elapsed / i * (num_batches - i)
            logger.info(
                f"  [Batch {i:>4d}/{num_batches}]  loss={loss.item():.4f}  "
                f"avg_loss={avg_loss:.4f}  elapsed={_fmt_time(elapsed)}  eta={_fmt_time(eta)}"
            )

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_tokens, 1)
    if logger:
        logger.info(f"  Train complete: {num_batches} batches, avg_loss={avg_loss:.4f}, time={_fmt_time(elapsed)}")

    return avg_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Evaluate model on a DataLoader.
    Returns average loss (no teacher forcing).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = len(loader)

    if logger:
        logger.info(f"  Evaluating ({num_batches} batches)...")

    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            cipher_padded, plain_padded, cipher_lengths, plain_lengths = batch
            cipher_padded = cipher_padded.to(device)
            plain_padded = plain_padded.to(device)
            cipher_lengths = cipher_lengths.to(device)

            outputs = model(
                cipher_padded,
                plain_padded,
                src_lengths=cipher_lengths,
                teacher_forcing_ratio=0.0,  # no teacher forcing during eval
            )

            tgt_out = plain_padded[:, 1:]
            batch_size, seq_len, vocab_size = outputs.size()

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                tgt_out.reshape(-1),
            )

            n_tokens = (tgt_out != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    elapsed = time.time() - t0
    if logger:
        logger.info(f"  Eval complete: avg_loss={avg_loss:.4f}, time={_fmt_time(elapsed)}")

    return avg_loss


def decode_outputs(
    model: nn.Module,
    dataset: CipherDataset,
    cipher_lines: List[str],
    device: torch.device,
    batch_size: int = 64,
) -> List[str]:
    """
    Run the model on a list of cipher text lines.
    Returns list of decoded plain text strings.
    """
    model.eval()
    results = []

    # Process in batches
    for start in tqdm(range(0, len(cipher_lines), batch_size), desc="Decoding"):
        batch_cipher = cipher_lines[start : start + batch_size]

        # Encode each cipher line
        tensors = [dataset.encode_cipher(c) for c in batch_cipher]

        # Pad to max len in this mini-batch
        max_len = max(t.size(0) for t in tensors)
        padded = torch.zeros(len(tensors), max_len, dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, : t.size(0)] = t
        padded = padded.to(device)

        # Greedy decode
        max_decode_len = dataset.max_plain_len
        decoded_indices = model.decode(padded, max_decode_len, SOS_IDX, EOS_IDX, device)

        for indices in decoded_indices:
            text = dataset.decode_plain(indices)
            results.append(text)

    return results


def levenshtein_distance(s1: str, s2: str) -> int:
    """Dynamic programming Levenshtein distance."""
    m, n = len(s1), len(s2)
    # Use two rows only for memory efficiency
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute character accuracy, word accuracy, and average Levenshtein distance.

    Returns dict with keys: char_accuracy, word_accuracy, avg_levenshtein
    """
    assert len(predictions) == len(references), "Lengths must match"

    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0
    total_lev = 0.0

    for pred, ref in zip(predictions, references):
        # Character accuracy (align by min length)
        min_len = min(len(pred), len(ref))
        total_chars += len(ref)
        correct_chars += sum(1 for i in range(min_len) if pred[i] == ref[i])

        # Word accuracy
        pred_words = pred.split()
        ref_words = ref.split()
        total_words += len(ref_words)
        correct_words += sum(1 for pw, rw in zip(pred_words, ref_words) if pw == rw)

        # Levenshtein
        total_lev += levenshtein_distance(pred, ref)

    n = len(predictions)
    return {
        "char_accuracy": correct_chars / max(total_chars, 1),
        "word_accuracy": correct_words / max(total_words, 1),
        "avg_levenshtein": total_lev / max(n, 1),
    }


def save_results(
    predictions: List[str],
    references: List[str],
    filepath: str,
) -> None:
    """Save prediction/reference pairs to a text file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for pred, ref in zip(predictions, references):
            f.write(f"PRED: {pred} | REF: {ref}\n")

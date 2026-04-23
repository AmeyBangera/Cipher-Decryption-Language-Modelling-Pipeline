"""
Task 2: Language Modeling Dataset
Word-level datasets for Next Word Prediction (SSM) and Masked Language Modeling (Bi-LSTM).
"""

import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# Special token indices
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def preprocess_line(line: str) -> List[str]:
    """Lowercase, keep only a-z and space, return word tokens."""
    line = line.lower()
    line = re.sub(r"[^a-z ]", "", line)
    line = re.sub(r" +", " ", line).strip()
    return line.split()


class Vocabulary:
    """Word-level vocabulary with frequency threshold."""

    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._setup_specials()

    def _setup_specials(self):
        specials = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        for i, tok in enumerate(specials):
            self.word2idx[tok] = i
            self.idx2word[i] = tok

    def build(self, word_counts: Counter, min_freq: int = 2):
        for word, count in sorted(word_counts.items()):
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, word: str) -> int:
        return self.word2idx.get(word, UNK_IDX)

    def decode(self, idx: int) -> str:
        return self.idx2word.get(idx, "<UNK>")


def build_vocabulary(lines: List[str], min_freq: int = 2) -> Vocabulary:
    """Build vocabulary from list of text lines."""
    counts: Counter = Counter()
    for line in lines:
        words = preprocess_line(line)
        counts.update(words)
    vocab = Vocabulary()
    vocab.build(counts, min_freq=min_freq)
    return vocab


class NWPDataset(Dataset):
    """
    Next Word Prediction dataset for SSM.
    Each sample: (context_ids, next_word_id)
    Context window → next word.
    """

    def __init__(self, lines: List[str], vocab: Vocabulary, context_len: int = 20):
        self.vocab = vocab
        self.context_len = context_len
        self.samples: List[Tuple[List[int], int]] = []

        for line in lines:
            words = preprocess_line(line)
            if len(words) < 2:
                continue
            ids = [vocab.encode(w) for w in words]
            # Slide over the sequence
            for i in range(len(ids)):
                # context: up to context_len words before position i+1
                start = max(0, i - context_len + 1)
                context = ids[start : i + 1]
                if i + 1 < len(ids):
                    target = ids[i + 1]
                    self.samples.append((context, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target = self.samples[idx]
        # Pad context to context_len
        padded = [PAD_IDX] * (self.context_len - len(context)) + context
        padded = padded[-self.context_len :]  # truncate if somehow longer
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


class MLMDataset(Dataset):
    """
    Masked Language Modeling dataset for Bi-LSTM.
    Each sample: (masked_ids, labels) where non-masked positions have label -100.
    """

    def __init__(
        self,
        lines: List[str],
        vocab: Vocabulary,
        seq_len: int = 30,
        mask_prob: float = 0.15,
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.mask_prob = mask_prob

        # Build a flat token list from all lines
        all_ids: List[int] = []
        for line in lines:
            words = preprocess_line(line)
            if not words:
                continue
            ids = [vocab.encode(w) for w in words]
            all_ids.extend(ids)

        # Chunk into segments of seq_len
        self.segments: List[List[int]] = []
        for i in range(0, len(all_ids) - seq_len, seq_len):
            self.segments.append(all_ids[i : i + seq_len])

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        segment = self.segments[idx]
        input_ids = list(segment)
        labels = [-100] * len(segment)

        for i in range(len(input_ids)):
            if random.random() < self.mask_prob:
                labels[i] = input_ids[i]  # True label
                # 80% replace with UNK (mask token), 10% random, 10% unchanged
                r = random.random()
                if r < 0.8:
                    input_ids[i] = UNK_IDX  # use UNK as mask
                elif r < 0.9:
                    input_ids[i] = random.randint(4, len(self.vocab) - 1)
                # else: keep original

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def load_lm_data(
    plain_path: str,
    context_len: int = 20,
    seq_len: int = 30,
    mask_prob: float = 0.15,
    train_split: float = 0.8,
    val_split: float = 0.1,
    min_freq: int = 2,
) -> Tuple:
    """
    Load plain text, build vocabulary, create NWP and MLM datasets.

    Returns:
        train_nwp, val_nwp, test_nwp,
        train_mlm, val_mlm, test_mlm,
        vocab
    """
    with open(plain_path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    n = len(lines)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_lines = lines[:n_train]
    val_lines = lines[n_train : n_train + n_val]
    test_lines = lines[n_train + n_val :]

    # Build vocabulary on training data only
    vocab = build_vocabulary(train_lines, min_freq=min_freq)

    train_nwp = NWPDataset(train_lines, vocab, context_len)
    val_nwp = NWPDataset(val_lines, vocab, context_len)
    test_nwp = NWPDataset(test_lines, vocab, context_len)

    train_mlm = MLMDataset(train_lines, vocab, seq_len, mask_prob)
    val_mlm = MLMDataset(val_lines, vocab, seq_len, mask_prob)
    test_mlm = MLMDataset(test_lines, vocab, seq_len, mask_prob)

    return (
        train_nwp, val_nwp, test_nwp,
        train_mlm, val_mlm, test_mlm,
        vocab,
    )

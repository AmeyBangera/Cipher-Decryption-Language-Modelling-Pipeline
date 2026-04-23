"""
Task 1: Cipher Dataset
Handles preprocessing and batching for seq2seq cipher decryption.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# Special token indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def preprocess_plain(text: str) -> str:
    """Lowercase and keep only a-z and space."""
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r" +", " ", text).strip()
    return text


def preprocess_cipher(text: str) -> str:
    """Keep only digit characters."""
    return re.sub(r"[^0-9]", "", text)


def build_vocabs(
    plain_lines: List[str], cipher_lines: List[str]
) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """
    Build vocabularies for plain and cipher text.

    Plain vocab: a-z + space + PAD/SOS/EOS
    Cipher vocab: digits 0-9 + PAD/SOS/EOS

    Returns:
        plain_vocab (char->idx), plain_idx2char (idx->char),
        cipher_vocab (char->idx), cipher_idx2char (idx->char)
    """
    # Plain vocab
    plain_chars = list("abcdefghijklmnopqrstuvwxyz ")
    plain_vocab: Dict[str, int] = {"<PAD>": PAD_IDX, "<SOS>": SOS_IDX, "<EOS>": EOS_IDX}
    for ch in plain_chars:
        plain_vocab[ch] = len(plain_vocab)
    plain_idx2char: Dict[int, str] = {v: k for k, v in plain_vocab.items()}

    # Cipher vocab
    cipher_chars = list("0123456789")
    cipher_vocab: Dict[str, int] = {"<PAD>": PAD_IDX, "<SOS>": SOS_IDX, "<EOS>": EOS_IDX}
    for ch in cipher_chars:
        cipher_vocab[ch] = len(cipher_vocab)
    cipher_idx2char: Dict[int, str] = {v: k for k, v in cipher_vocab.items()}

    return plain_vocab, plain_idx2char, cipher_vocab, cipher_idx2char


class CipherDataset(Dataset):
    """
    Dataset for seq2seq cipher decryption.

    Each sample: (cipher_tensor, plain_tensor)
    Both tensors include SOS and EOS tokens.
    """

    def __init__(
        self,
        plain_lines: List[str],
        cipher_lines: List[str],
        plain_vocab: Dict[str, int],
        cipher_vocab: Dict[str, int],
        max_plain_len: int = 200,
        max_cipher_len: int = 400,
    ):
        assert len(plain_lines) == len(cipher_lines), "Mismatched line counts"

        self.plain_vocab = plain_vocab
        self.cipher_vocab = cipher_vocab
        self.max_plain_len = max_plain_len
        self.max_cipher_len = max_cipher_len

        # Preprocess
        self.plain_lines = [preprocess_plain(l) for l in plain_lines]
        self.cipher_lines = [preprocess_cipher(l) for l in cipher_lines]

        # Filter out empty lines or lines exceeding max length
        valid = []
        for i, (p, c) in enumerate(zip(self.plain_lines, self.cipher_lines)):
            if len(p) == 0 or len(c) == 0:
                continue
            # +2 for SOS/EOS
            if len(p) + 2 > max_plain_len or len(c) + 2 > max_cipher_len:
                continue
            valid.append(i)

        self.plain_lines = [self.plain_lines[i] for i in valid]
        self.cipher_lines = [self.cipher_lines[i] for i in valid]

    def encode_cipher(self, text: str) -> torch.Tensor:
        """Encode cipher text string to tensor with SOS and EOS."""
        text = preprocess_cipher(text)
        indices = (
            [SOS_IDX]
            + [self.cipher_vocab.get(ch, PAD_IDX) for ch in text]
            + [EOS_IDX]
        )
        return torch.tensor(indices, dtype=torch.long)

    def encode_plain(self, text: str) -> torch.Tensor:
        """Encode plain text string to tensor with SOS and EOS."""
        text = preprocess_plain(text)
        indices = (
            [SOS_IDX]
            + [self.plain_vocab.get(ch, PAD_IDX) for ch in text]
            + [EOS_IDX]
        )
        return torch.tensor(indices, dtype=torch.long)

    def decode_plain(self, indices) -> str:
        """Decode a list/tensor of plain vocab indices back to string."""
        idx2char: Dict[int, str] = {v: k for k, v in self.plain_vocab.items()}
        chars = []
        for idx in indices:
            idx = int(idx)
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            ch = idx2char.get(idx, "")
            if ch not in ("<PAD>", "<SOS>", "<EOS>"):
                chars.append(ch)
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.plain_lines)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cipher_tensor = self.encode_cipher(self.cipher_lines[idx])
        plain_tensor = self.encode_plain(self.plain_lines[idx])
        return cipher_tensor, plain_tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Pad batch to max lengths within the batch.

    Returns:
        cipher_padded: (batch, max_cipher_len)
        plain_padded:  (batch, max_plain_len)
        cipher_lengths: (batch,) actual lengths
        plain_lengths:  (batch,) actual lengths
    """
    cipher_seqs, plain_seqs = zip(*batch)

    cipher_lengths = torch.tensor([s.size(0) for s in cipher_seqs], dtype=torch.long)
    plain_lengths = torch.tensor([s.size(0) for s in plain_seqs], dtype=torch.long)

    cipher_padded = torch.nn.utils.rnn.pad_sequence(
        cipher_seqs, batch_first=True, padding_value=PAD_IDX
    )
    plain_padded = torch.nn.utils.rnn.pad_sequence(
        plain_seqs, batch_first=True, padding_value=PAD_IDX
    )

    return cipher_padded, plain_padded, cipher_lengths, plain_lengths


def load_data(
    plain_path: str,
    cipher_path: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    max_plain_len: int = 200,
    max_cipher_len: int = 400,
) -> Tuple[CipherDataset, CipherDataset, CipherDataset, Dict, Dict, Dict, Dict]:
    """
    Load plain and cipher data, build vocabularies, and split into train/val/test.

    Returns:
        train_dataset, val_dataset, test_dataset,
        plain_vocab, plain_idx2char, cipher_vocab, cipher_idx2char
    """
    with open(plain_path, "r", encoding="utf-8") as f:
        plain_lines = [line.rstrip("\n") for line in f]
    with open(cipher_path, "r", encoding="utf-8") as f:
        cipher_lines = [line.rstrip("\n") for line in f]

    # Align lengths
    n = min(len(plain_lines), len(cipher_lines))
    plain_lines = plain_lines[:n]
    cipher_lines = cipher_lines[:n]

    plain_vocab, plain_idx2char, cipher_vocab, cipher_idx2char = build_vocabs(
        plain_lines, cipher_lines
    )

    # Split indices
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_plain = plain_lines[:n_train]
    train_cipher = cipher_lines[:n_train]

    val_plain = plain_lines[n_train : n_train + n_val]
    val_cipher = cipher_lines[n_train : n_train + n_val]

    test_plain = plain_lines[n_train + n_val :]
    test_cipher = cipher_lines[n_train + n_val :]

    train_dataset = CipherDataset(
        train_plain, train_cipher, plain_vocab, cipher_vocab, max_plain_len, max_cipher_len
    )
    val_dataset = CipherDataset(
        val_plain, val_cipher, plain_vocab, cipher_vocab, max_plain_len, max_cipher_len
    )
    test_dataset = CipherDataset(
        test_plain, test_cipher, plain_vocab, cipher_vocab, max_plain_len, max_cipher_len
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        plain_vocab,
        plain_idx2char,
        cipher_vocab,
        cipher_idx2char,
    )

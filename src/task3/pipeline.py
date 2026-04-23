"""
Task 3: Decryption + Language Model correction pipeline.

Loads a trained Task 1 decryption model and a Task 2 language model,
then for each cipher file:
  1. Decrypts using seq2seq model alone -> metrics
  2. Applies LM correction -> metrics

LM correction strategy (word_rescore):
  For each word in the decoded sequence, compute LM confidence;
  if confidence < threshold, replace with LM top-1 prediction.

Also supports automated evaluation via config keys:
  test_file: path to cipher text
  output_file: path to write decoded output
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.task1.dataset import (
    CipherDataset,
    load_data as load_task1_data,
    collate_fn,
    PAD_IDX as T1_PAD,
    SOS_IDX as T1_SOS,
    EOS_IDX as T1_EOS,
    preprocess_plain,
    preprocess_cipher,
)
from src.task1.model import Seq2SeqRNN, Seq2SeqLSTM
from src.task1.train_utils import decode_outputs
from src.task2.dataset import build_vocabulary, preprocess_line, PAD_IDX as T2_PAD
from src.utils.checkpoints import load_checkpoint
from src.utils.metrics import compute_all_metrics


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logger = logging.getLogger("task3_pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def _load_decryption_model(
    model_type: str,
    model_cfg: Dict,
    src_vocab_size: int,
    tgt_vocab_size: int,
    ckpt_path: str,
    hf_cfg: Dict,
    device: torch.device,
    logger: logging.Logger,
) -> nn.Module:
    """Load Task 1 decryption model from checkpoint or HuggingFace."""
    if model_type == "rnn":
        model = Seq2SeqRNN(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=model_cfg.get("embed_dim", 64),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
        ).to(device)
    else:  # lstm
        model = Seq2SeqLSTM(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=model_cfg.get("embed_dim", 64),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
        ).to(device)

    if os.path.isfile(ckpt_path):
        load_checkpoint(ckpt_path, model, device=str(device))
        logger.info(f"Loaded decryption model from {ckpt_path}")
    elif hf_cfg.get("enabled", False):
        try:
            from src.utils.hf_wandb import load_from_hub
            model = load_from_hub(
                model,
                hf_cfg["repo_id"],
                hf_cfg.get("filename", f"task1_{model_type}.pt"),
                device=str(device),
            )
            logger.info("Loaded decryption model from HuggingFace")
        except Exception as e:
            raise RuntimeError(f"Cannot load decryption model: {e}")
    else:
        raise FileNotFoundError(f"Decryption checkpoint not found: {ckpt_path}")

    model.eval()
    return model


def _load_lm_model(
    lm_type: str,
    lm_cfg: Dict,
    vocab_size: int,
    ckpt_path: str,
    hf_cfg: Dict,
    device: torch.device,
    logger: logging.Logger,
) -> nn.Module:
    """Load Task 2 language model from checkpoint or HuggingFace."""
    model_cfg = lm_cfg.get("model", {})

    if lm_type == "ssm":
        from src.task2.ssm_model import SSMModel
        lm = SSMModel(
            vocab_size=vocab_size,
            d_model=model_cfg.get("d_model", 256),
            d_state=model_cfg.get("d_state", 64),
            num_layers=model_cfg.get("num_layers", 4),
            dropout=model_cfg.get("dropout", 0.2),
        ).to(device)
    else:  # bilstm
        from src.task2.bilstm_model import BiLSTMModel
        lm = BiLSTMModel(
            vocab_size=vocab_size,
            embed_dim=model_cfg.get("embed_dim", 128),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
        ).to(device)

    if os.path.isfile(ckpt_path):
        load_checkpoint(ckpt_path, lm, device=str(device))
        logger.info(f"Loaded LM from {ckpt_path}")
    elif hf_cfg.get("enabled", False):
        try:
            from src.utils.hf_wandb import load_from_hub
            lm = load_from_hub(
                lm,
                hf_cfg["repo_id"],
                hf_cfg.get("filename", f"task2_{lm_type}.pt"),
                device=str(device),
            )
            logger.info("Loaded LM from HuggingFace")
        except Exception as e:
            raise RuntimeError(f"Cannot load LM: {e}")
    else:
        raise FileNotFoundError(f"LM checkpoint not found: {ckpt_path}")

    lm.eval()
    return lm


def _lm_correct_words(
    decoded_sentences: List[str],
    lm: nn.Module,
    vocab,
    device: torch.device,
    lm_type: str,
    context_len: int = 20,
    confidence_threshold: float = 0.5,
    lm_weight: float = 0.3,
) -> List[str]:
    """
    Word-level LM correction.

    For each word in the decoded sentence:
      - Compute LM probability for the word given its context.
      - If probability < confidence_threshold, replace with LM's top-1 prediction.

    This uses a sliding window approach with the LM.
    """
    corrected = []

    with torch.no_grad():
        for sentence in decoded_sentences:
            words = sentence.split()
            if not words:
                corrected.append(sentence)
                continue

            word_ids = [vocab.encode(w) for w in words]
            result_words = list(words)

            for i, (word, wid) in enumerate(zip(words, word_ids)):
                # Build context: up to context_len words before position i
                start = max(0, i - context_len)
                context = word_ids[start:i]
                if not context:
                    # No context, can't correct
                    continue

                # Pad context to context_len
                padded = [T2_PAD] * (context_len - len(context)) + context
                padded = padded[-context_len:]
                x = torch.tensor([padded], dtype=torch.long, device=device)

                # Get LM logits for next word
                logits = lm(x)  # (1, seq_len, vocab) for SSM; same for BiLSTM
                if lm_type == "ssm":
                    next_logits = logits[0, -1, :]  # (vocab,)
                else:
                    # BiLSTM: use the logit at the target position
                    # For correction, use middle/last position
                    next_logits = logits[0, -1, :]  # (vocab,)

                probs = F.softmax(next_logits, dim=-1)
                word_prob = probs[wid].item()
                top_id = probs.argmax().item()

                if word_prob < confidence_threshold and top_id not in (T2_PAD, 1, 2, 3):
                    # Replace with LM prediction
                    top_word = vocab.decode(top_id)
                    if top_word not in ("<PAD>", "<UNK>", "<SOS>", "<EOS>"):
                        result_words[i] = top_word

            corrected.append(" ".join(result_words))

    return corrected


def _decrypt_cipher_file(
    cipher_path: str,
    plain_path: str,
    decrypt_model: nn.Module,
    task1_dataset_ref: CipherDataset,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[List[str], List[str]]:
    """
    Load a cipher file, decrypt it, and return (predictions, references).
    Loads the corresponding plain text from plain_path for references.
    """
    with open(cipher_path, "r", encoding="utf-8") as f:
        cipher_lines = [l.rstrip("\n") for l in f]
    with open(plain_path, "r", encoding="utf-8") as f:
        plain_lines = [l.rstrip("\n") for l in f]

    n = min(len(cipher_lines), len(plain_lines))
    cipher_lines = cipher_lines[:n]
    plain_lines = plain_lines[:n]

    # Filter empties and overlength
    valid_cipher, valid_plain = [], []
    for c, p in zip(cipher_lines, plain_lines):
        c_proc = preprocess_cipher(c)
        p_proc = preprocess_plain(p)
        if c_proc and p_proc and len(c_proc) + 2 <= task1_dataset_ref.max_cipher_len:
            valid_cipher.append(c_proc)
            valid_plain.append(p_proc)

    predictions = decode_outputs(
        decrypt_model, task1_dataset_ref, valid_cipher, device, batch_size=batch_size
    )
    return predictions, valid_plain


def main(config_path: str, mode: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = _get_device()
    lm_type = cfg["language_model"]["model_type"]

    log_path = cfg.get("output", {}).get(
        "log_file", f"outputs/logs/task3_{lm_type}.log"
    )
    logger = _setup_logging(log_path)
    logger.info(f"Device: {device}, LM type: {lm_type}, Mode: {mode}")

    # Create output dirs
    for key in ("result_file", "plot_file"):
        p = cfg.get("output", {}).get(key, "")
        if p:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # -----------------------------------------------------------------------
    # Load Task 1 data + build vocabs
    # -----------------------------------------------------------------------
    dec_cfg = cfg["decryption"]
    with open(dec_cfg["config_path"], "r") as f:
        task1_cfg = yaml.safe_load(f)

    t1_data_cfg = task1_cfg["data"]
    (
        train_ds, val_ds, test_ds,
        plain_vocab, plain_idx2char,
        cipher_vocab, cipher_idx2char,
    ) = load_task1_data(
        plain_path=t1_data_cfg["plain_path"],
        cipher_path=t1_data_cfg["cipher_path"],
        train_split=t1_data_cfg.get("train_split", 0.8),
        val_split=t1_data_cfg.get("val_split", 0.1),
        max_plain_len=t1_data_cfg.get("max_plain_len", 200),
        max_cipher_len=t1_data_cfg.get("max_cipher_len", 400),
    )

    # Use train_ds as the reference dataset for encode/decode utilities
    ref_dataset = train_ds

    # -----------------------------------------------------------------------
    # Load decryption model (Task 1)
    # -----------------------------------------------------------------------
    hf_cfg_dec = task1_cfg.get("huggingface", {})
    decrypt_model = _load_decryption_model(
        model_type=dec_cfg["model_type"],
        model_cfg=task1_cfg["model"],
        src_vocab_size=len(cipher_vocab),
        tgt_vocab_size=len(plain_vocab),
        ckpt_path=dec_cfg["checkpoint_path"],
        hf_cfg=hf_cfg_dec,
        device=device,
        logger=logger,
    )

    # -----------------------------------------------------------------------
    # Load Task 2 language model
    # -----------------------------------------------------------------------
    lm_info = cfg["language_model"]
    with open(lm_info["config_path"], "r") as f:
        task2_cfg = yaml.safe_load(f)

    # Build LM vocabulary from plain text (same file used in task 2)
    t2_data_cfg = task2_cfg["data"]
    with open(t2_data_cfg["plain_path"], "r", encoding="utf-8") as f:
        lm_plain_lines = [l.rstrip("\n") for l in f]
    n_train = int(len(lm_plain_lines) * t2_data_cfg.get("train_split", 0.8))
    lm_vocab = build_vocabulary(
        lm_plain_lines[:n_train],
        min_freq=t2_data_cfg.get("min_freq", 2),
    )
    logger.info(f"LM vocab size: {len(lm_vocab)}")

    hf_cfg_lm = task2_cfg.get("huggingface", {})
    lm_model = _load_lm_model(
        lm_type=lm_type,
        lm_cfg=task2_cfg,
        vocab_size=len(lm_vocab),
        ckpt_path=lm_info["checkpoint_path"],
        hf_cfg=hf_cfg_lm,
        device=device,
        logger=logger,
    )

    correction_cfg = cfg.get("correction", {})
    confidence_threshold = correction_cfg.get("confidence_threshold", 0.5)
    lm_weight = correction_cfg.get("lm_weight", 0.3)
    context_len = t2_data_cfg.get("context_len", 20)
    batch_size = 64

    # -----------------------------------------------------------------------
    # Automated evaluation: test_file -> output_file
    # -----------------------------------------------------------------------
    if "test_file" in cfg and "output_file" in cfg:
        test_file = cfg["test_file"]
        output_file = cfg["output_file"]
        logger.info(f"Automated evaluation: {test_file} -> {output_file}")

        with open(test_file, "r", encoding="utf-8") as f:
            test_cipher_lines = [l.rstrip("\n") for l in f]

        predictions = decode_outputs(
            decrypt_model, ref_dataset, test_cipher_lines, device, batch_size=batch_size
        )

        # Apply LM correction
        corrected = _lm_correct_words(
            predictions,
            lm_model,
            lm_vocab,
            device,
            lm_type=lm_type,
            context_len=context_len,
            confidence_threshold=confidence_threshold,
            lm_weight=lm_weight,
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for line in corrected:
                f.write(line + "\n")
        logger.info(f"Written {len(corrected)} lines to {output_file}")
        return

    # -----------------------------------------------------------------------
    # Per-noise-level evaluation
    # -----------------------------------------------------------------------
    data_cfg = cfg["data"]
    cipher_files = data_cfg.get("cipher_files", [])
    noise_labels = data_cfg.get("noise_labels", [f"noise_{i:02d}" for i in range(len(cipher_files))])
    plain_path_ref = data_cfg["plain_path"]

    all_results: List[Dict] = []

    result_file = cfg.get("output", {}).get("result_file", f"outputs/results/task3_{lm_type}.txt")
    os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)

    with open(result_file, "w", encoding="utf-8") as out_f:
        out_f.write(f"Task 3 Pipeline Results — LM: {lm_type}\n")
        out_f.write("=" * 70 + "\n\n")

        for cipher_path, label in zip(cipher_files, noise_labels):
            logger.info(f"Processing {label}: {cipher_path}")

            # --- Step 1: Decrypt alone ---
            try:
                preds_raw, refs = _decrypt_cipher_file(
                    cipher_path, plain_path_ref, decrypt_model, ref_dataset, device, batch_size
                )
            except Exception as e:
                logger.error(f"Failed to decrypt {cipher_path}: {e}")
                continue

            metrics_raw = compute_all_metrics(preds_raw, refs)
            logger.info(f"[{label}] Raw decryption: {metrics_raw}")

            # --- Step 2: LM correction ---
            preds_corrected = _lm_correct_words(
                preds_raw,
                lm_model,
                lm_vocab,
                device,
                lm_type=lm_type,
                context_len=context_len,
                confidence_threshold=confidence_threshold,
                lm_weight=lm_weight,
            )
            metrics_corrected = compute_all_metrics(preds_corrected, refs)
            logger.info(f"[{label}] After LM correction: {metrics_corrected}")

            # Write to result file
            out_f.write(f"=== {label} ===\n")
            out_f.write("-- Decryption only --\n")
            for k, v in metrics_raw.items():
                out_f.write(f"  {k}: {v:.4f}\n")
            out_f.write("-- Decryption + LM correction --\n")
            for k, v in metrics_corrected.items():
                out_f.write(f"  {k}: {v:.4f}\n")
            out_f.write("\n-- Sample Predictions (first 5) --\n")
            for i in range(min(5, len(preds_corrected))):
                out_f.write(f"  PRED: {preds_corrected[i]}\n")
                out_f.write(f"  REF:  {refs[i]}\n")
            out_f.write("\n")

            all_results.append({
                "label": label,
                "raw": metrics_raw,
                "corrected": metrics_corrected,
            })

    logger.info(f"Results saved to {result_file}")

    # -----------------------------------------------------------------------
    # Plot: char_acc vs noise level
    # -----------------------------------------------------------------------
    if all_results:
        plot_file = cfg.get("output", {}).get("plot_file", f"outputs/plots/task3_{lm_type}.png")
        os.makedirs(os.path.dirname(os.path.abspath(plot_file)), exist_ok=True)

        labels = [r["label"] for r in all_results]
        raw_char = [r["raw"]["char_acc"] for r in all_results]
        cor_char = [r["corrected"]["char_acc"] for r in all_results]
        x = list(range(len(labels)))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(x, raw_char, marker="o", label="Raw decrypt")
        axes[0].plot(x, cor_char, marker="s", label="+ LM correction")
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45)
        axes[0].set_ylabel("Char Accuracy"); axes[0].set_title("Character Accuracy")
        axes[0].legend()

        raw_word = [r["raw"]["word_acc"] for r in all_results]
        cor_word = [r["corrected"]["word_acc"] for r in all_results]
        axes[1].plot(x, raw_word, marker="o", label="Raw decrypt")
        axes[1].plot(x, cor_word, marker="s", label="+ LM correction")
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45)
        axes[1].set_ylabel("Word Accuracy"); axes[1].set_title("Word Accuracy")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(plot_file)
        plt.close(fig)
        logger.info(f"Saved plot to {plot_file}")

"""
Task 2: SSM training and evaluation for Next Word Prediction.
Metric: perplexity = exp(cross_entropy_loss).
"""

import logging
import math
import os
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.task2.dataset import load_lm_data, PAD_IDX
from src.task2.ssm_model import SSMModel
from src.utils.checkpoints import save_checkpoint, load_checkpoint


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logger = logging.getLogger("task2_ssm")
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


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def train_epoch_nwp(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip: float = 1.0,
    logger: Optional[logging.Logger] = None,
    log_every: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    num_batches = len(loader)
    t0 = time.time()

    for i, (context, target) in enumerate(tqdm(loader, desc="Train NWP", leave=False), 1):
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(context)
        last_logits = logits[:, -1, :]
        loss = criterion(last_logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item() * context.size(0)
        total_samples += context.size(0)

        if logger and (i % log_every == 0 or i == num_batches):
            elapsed = time.time() - t0
            avg_loss = total_loss / max(total_samples, 1)
            ppl = math.exp(min(avg_loss, 20))
            eta = elapsed / i * (num_batches - i)
            logger.info(
                f"  [Batch {i:>4d}/{num_batches}]  loss={loss.item():.4f}  "
                f"avg_loss={avg_loss:.4f}  ppl={ppl:.2f}  elapsed={_fmt_time(elapsed)}  eta={_fmt_time(eta)}"
            )

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_samples, 1)
    if logger:
        logger.info(f"  Train complete: {num_batches} batches, avg_loss={avg_loss:.4f}, time={_fmt_time(elapsed)}")

    return avg_loss


def evaluate_nwp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    num_batches = len(loader)

    if logger:
        logger.info(f"  Evaluating ({num_batches} batches)...")

    t0 = time.time()

    with torch.no_grad():
        for context, target in tqdm(loader, desc="Eval NWP", leave=False):
            context = context.to(device)
            target = target.to(device)

            logits = model(context)
            last_logits = logits[:, -1, :]
            loss = criterion(last_logits, target)

            total_loss += loss.item() * context.size(0)
            total_samples += context.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    elapsed = time.time() - t0
    if logger:
        logger.info(f"  Eval complete: avg_loss={avg_loss:.4f}, ppl={math.exp(min(avg_loss, 20)):.2f}, time={_fmt_time(elapsed)}")

    return avg_loss


def generate_samples(
    model: nn.Module,
    vocab,
    device: torch.device,
    seed_text: str = "the",
    num_words: int = 20,
    context_len: int = 20,
) -> str:
    """Generate a text sample from the SSM model."""
    model.eval()
    words = seed_text.lower().split()
    with torch.no_grad():
        for _ in range(num_words):
            context = [vocab.encode(w) for w in words[-context_len:]]
            # Pad to context_len
            padded = [PAD_IDX] * (context_len - len(context)) + context
            x = torch.tensor([padded], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :]  # (1, vocab)
            next_id = logits.argmax(dim=-1).item()
            next_word = vocab.decode(next_id)
            if next_word in ("<EOS>", "<PAD>", "<UNK>"):
                break
            words.append(next_word)
    return " ".join(words)


def main(config_path: str, mode: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = _get_device()

    for key in ("checkpoint_path", "result_file", "plot_file"):
        p = cfg["output"].get(key, "")
        if p:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    log_path = cfg["output"].get("log_file", "outputs/logs/task2_ssm.log")
    logger = _setup_logging(log_path)
    logger.info(f"Device: {device}, Mode: {mode}")

    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        try:
            from src.utils.hf_wandb import init_wandb, log_wandb, finish_wandb
            init_wandb(project=cfg["wandb"]["project"], config=cfg, name=cfg["wandb"].get("name"))
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
            use_wandb = False

    data_cfg = cfg["data"]
    train_nwp, val_nwp, test_nwp, _, _, _, vocab = load_lm_data(
        plain_path=data_cfg["plain_path"],
        context_len=data_cfg.get("context_len", 20),
        train_split=data_cfg.get("train_split", 0.8),
        val_split=data_cfg.get("val_split", 0.1),
        min_freq=data_cfg.get("min_freq", 2),
    )
    logger.info(f"Vocab size: {len(vocab)}, Train samples: {len(train_nwp)}")

    train_cfg = cfg["training"]
    batch_size = train_cfg.get("batch_size", 256)

    train_loader = DataLoader(train_nwp, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_nwp,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_nwp,  batch_size=batch_size, shuffle=False, num_workers=0)

    model_cfg = cfg["model"]
    model = SSMModel(
        vocab_size=len(vocab),
        d_model=model_cfg.get("d_model", 256),
        d_state=model_cfg.get("d_state", 64),
        num_layers=model_cfg.get("num_layers", 4),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)
    logger.info(f"SSM params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    ckpt_path = cfg["output"]["checkpoint_path"]
    train_losses, val_losses = [], []

    if mode in ("train", "both"):
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 0.001))
        best_val = float("inf")
        epochs = train_cfg.get("epochs", 20)
        clip = train_cfg.get("clip_grad", 1.0)

        for epoch in range(1, epochs + 1):
            logger.info(f"--- Epoch {epoch}/{epochs} starting (lr={optimizer.param_groups[0]['lr']:.6f}) ---")
            tr_loss = train_epoch_nwp(model, train_loader, optimizer, criterion, device, clip, logger=logger)
            val_loss = evaluate_nwp(model, val_loader, criterion, device, logger=logger)
            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            tr_ppl = math.exp(min(tr_loss, 20))
            val_ppl = math.exp(min(val_loss, 20))
            logger.info(
                f"Epoch {epoch}/{epochs} | Train Loss: {tr_loss:.4f} PPL: {tr_ppl:.2f} "
                f"| Val Loss: {val_loss:.4f} PPL: {val_ppl:.2f}"
            )
            if use_wandb:
                try:
                    log_wandb({"train_loss": tr_loss, "val_loss": val_loss,
                               "train_ppl": tr_ppl, "val_ppl": val_ppl}, step=epoch)
                except Exception:
                    pass
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
                logger.info(f"  -> Checkpoint saved (val_ppl={val_ppl:.2f})")

        # Plot
        plot_file = cfg["output"].get("plot_file", "outputs/plots/task2_ssm_loss.png")
        os.makedirs(os.path.dirname(os.path.abspath(plot_file)), exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
        ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Task 2 SSM - NWP Loss")
        ax.legend()
        fig.savefig(plot_file); plt.close(fig)
        logger.info(f"Saved plot to {plot_file}")

        use_hf = cfg.get("huggingface", {}).get("enabled", False)
        if use_hf:
            try:
                from src.utils.hf_wandb import push_to_hub
                push_to_hub(ckpt_path, cfg["huggingface"]["repo_id"],
                            cfg["huggingface"].get("filename", "task2_ssm.pt"))
            except Exception as e:
                logger.warning(f"HF push failed: {e}")

    if mode in ("evaluate", "both"):
        loaded_from_hub = False
        if not os.path.isfile(ckpt_path):
            use_hf = cfg.get("huggingface", {}).get("enabled", False)
            if use_hf:
                try:
                    from src.utils.hf_wandb import load_from_hub
                    model = load_from_hub(model, cfg["huggingface"]["repo_id"],
                                         cfg["huggingface"].get("filename", "task2_ssm.pt"),
                                         device=str(device))
                    loaded_from_hub = True
                    logger.info("Loaded model from HuggingFace")
                except Exception as e:
                    logger.error(f"Cannot load checkpoint: {e}"); return
            else:
                logger.error(f"Checkpoint not found: {ckpt_path}"); return
        if not loaded_from_hub:
            load_checkpoint(ckpt_path, model, device=str(device))

        test_loss = evaluate_nwp(model, test_loader, criterion, device)
        test_ppl = math.exp(min(test_loss, 20))
        logger.info(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")

        # Generate sample predictions
        result_file = cfg["output"].get("result_file", "outputs/results/task2_ssm.txt")
        os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
        seed_texts = ["the president", "a new law", "united states", "economic growth", "city of"]
        context_len = data_cfg.get("context_len", 20)

        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Perplexity: {test_ppl:.2f}\n\n")
            f.write("=== Sample Generations ===\n")
            for seed in seed_texts:
                sample = generate_samples(model, vocab, device, seed, context_len=context_len)
                f.write(f"SEED: {seed}\nOUTPUT: {sample}\n\n")
        logger.info(f"Saved results to {result_file}")

        if use_wandb:
            try:
                log_wandb({"test_loss": test_loss, "test_ppl": test_ppl})
            except Exception:
                pass

    if use_wandb:
        try:
            finish_wandb()
        except Exception:
            pass

"""
Task 1: RNN Seq2Seq training and evaluation entry point.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.task1.dataset import load_data, collate_fn, PAD_IDX
from src.task1.model import Seq2SeqRNN
from src.task1.train_utils import (
    train_epoch,
    evaluate,
    decode_outputs,
    compute_metrics,
    save_results,
)
from src.utils.checkpoints import save_checkpoint, load_checkpoint


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logger = logging.getLogger("task1_rnn")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def main(config_path: str, mode: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = _get_device()

    # Setup output dirs
    for key in ("checkpoint_path", "result_file", "plot_file", "log_file"):
        p = cfg["output"].get(key, "")
        if p:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    logger = _setup_logging(cfg["output"].get("log_file", "outputs/logs/task1_rnn.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Mode: {mode}")

    # Optional wandb
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    wandb_run = None
    if use_wandb:
        try:
            from src.utils.hf_wandb import init_wandb, log_wandb, finish_wandb
            wandb_run = init_wandb(
                project=cfg["wandb"]["project"],
                config=cfg,
                name=cfg["wandb"].get("name"),
            )
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
            use_wandb = False

    train_losses = []
    val_losses = []

    if mode in ("train", "both"):
        logger.info("Loading data...")
        data_cfg = cfg["data"]
        (
            train_ds, val_ds, test_ds,
            plain_vocab, plain_idx2char,
            cipher_vocab, cipher_idx2char,
        ) = load_data(
            plain_path=data_cfg["plain_path"],
            cipher_path=data_cfg["cipher_path"],
            train_split=data_cfg.get("train_split", 0.8),
            val_split=data_cfg.get("val_split", 0.1),
            max_plain_len=data_cfg.get("max_plain_len", 200),
            max_cipher_len=data_cfg.get("max_cipher_len", 400),
        )

        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

        train_cfg = cfg["training"]
        batch_size = train_cfg.get("batch_size", 128)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        model_cfg = cfg["model"]
        model = Seq2SeqRNN(
            src_vocab_size=len(cipher_vocab),
            tgt_vocab_size=len(plain_vocab),
            embed_dim=model_cfg.get("embed_dim", 64),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
        ).to(device)

        logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 0.001))
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        best_val_loss = float("inf")
        epochs = train_cfg.get("epochs", 20)
        clip = train_cfg.get("clip_grad", 1.0)
        tf_ratio = train_cfg.get("teacher_forcing_ratio", 0.5)

        for epoch in range(1, epochs + 1):
            logger.info(f"--- Epoch {epoch}/{epochs} starting (lr={optimizer.param_groups[0]['lr']:.6f}) ---")
            tr_loss = train_epoch(
                model, train_loader, optimizer, criterion, device,
                teacher_forcing_ratio=tf_ratio,
                clip_grad=clip,
                logger=logger,
            )

            val_loss = evaluate(model, val_loader, criterion, device, logger=logger)
            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

            if use_wandb:
                try:
                    log_wandb({"train_loss": tr_loss, "val_loss": val_loss}, step=epoch)
                except Exception:
                    pass

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    cfg["output"]["checkpoint_path"],
                )
                logger.info(f"  -> Saved checkpoint (val_loss={val_loss:.4f})")

        # Save training plot
        plot_file = cfg["output"].get("plot_file", "outputs/plots/task1_rnn_loss.png")
        os.makedirs(os.path.dirname(os.path.abspath(plot_file)), exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
        ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Task 1 RNN - Training Loss")
        ax.legend()
        fig.savefig(plot_file)
        plt.close(fig)
        logger.info(f"Saved plot to {plot_file}")

        # Push to HuggingFace if configured
        use_hf = cfg.get("huggingface", {}).get("enabled", False)
        if use_hf:
            try:
                from src.utils.hf_wandb import push_to_hub
                push_to_hub(
                    cfg["output"]["checkpoint_path"],
                    cfg["huggingface"]["repo_id"],
                    cfg["huggingface"].get("filename", "task1_rnn.pt"),
                )
                logger.info("Pushed checkpoint to HuggingFace")
            except Exception as e:
                logger.warning(f"HuggingFace push failed: {e}")

    if mode in ("evaluate", "both"):
        logger.info("Loading data for evaluation...")
        data_cfg = cfg["data"]
        (
            train_ds, val_ds, test_ds,
            plain_vocab, plain_idx2char,
            cipher_vocab, cipher_idx2char,
        ) = load_data(
            plain_path=data_cfg["plain_path"],
            cipher_path=data_cfg["cipher_path"],
            train_split=data_cfg.get("train_split", 0.8),
            val_split=data_cfg.get("val_split", 0.1),
            max_plain_len=data_cfg.get("max_plain_len", 200),
            max_cipher_len=data_cfg.get("max_cipher_len", 400),
        )

        model_cfg = cfg["model"]
        model = Seq2SeqRNN(
            src_vocab_size=len(cipher_vocab),
            tgt_vocab_size=len(plain_vocab),
            embed_dim=model_cfg.get("embed_dim", 64),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
        ).to(device)

        ckpt_path = cfg["output"]["checkpoint_path"]
        loaded_from_hub = False

        if not os.path.isfile(ckpt_path):
            use_hf = cfg.get("huggingface", {}).get("enabled", False)
            if use_hf:
                try:
                    from src.utils.hf_wandb import load_from_hub
                    model = load_from_hub(
                        model,
                        cfg["huggingface"]["repo_id"],
                        cfg["huggingface"].get("filename", "task1_rnn.pt"),
                        device=str(device),
                    )
                    loaded_from_hub = True
                    logger.info("Loaded model from HuggingFace")
                except Exception as e:
                    logger.error(f"Could not load from HuggingFace: {e}")
                    return
            else:
                logger.error(f"Checkpoint not found: {ckpt_path}")
                return

        if not loaded_from_hub:
            load_checkpoint(ckpt_path, model, device=str(device))
            logger.info(f"Loaded checkpoint from {ckpt_path}")

        model.eval()

        # Evaluate on test set
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        train_cfg = cfg["training"]
        batch_size = train_cfg.get("batch_size", 128)

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        test_loss = evaluate(model, test_loader, criterion, device)
        logger.info(f"Test Loss: {test_loss:.4f}")

        # Decode test set
        test_cipher_lines = [test_ds.cipher_lines[i] for i in range(len(test_ds))]
        predictions = decode_outputs(model, test_ds, test_cipher_lines, device, batch_size=64)
        references = [test_ds.plain_lines[i] for i in range(len(test_ds))]

        metrics = compute_metrics(predictions, references)
        logger.info(f"Metrics: {metrics}")

        if use_wandb:
            try:
                log_wandb(metrics)
            except Exception:
                pass

        result_file = cfg["output"].get("result_file", "outputs/results/task1_rnn.txt")
        save_results(predictions, references, result_file)
        logger.info(f"Saved results to {result_file}")

    if use_wandb:
        try:
            finish_wandb()
        except Exception:
            pass

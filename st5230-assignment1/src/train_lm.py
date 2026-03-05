"""
Usage
-----
Run directly::

    python train_lm.py                           # default config (lstm)
    python train_lm.py --model_type rnn
    python train_lm.py --model_type transformer --embed_dim 128 --epochs 15

Or import and call from a notebook / script::

    from train_lm import run_experiment
    result = run_experiment(model_type="lstm", embed_dim=128, epochs=10)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# ---- project imports ----
from config import ExperimentConfig, make_config
from utils import (
    set_seed,
    get_device,
    TrainingLogger,
    save_checkpoint,
    Timer,
    epoch_progress_bar,
    compute_perplexity,
)
from data import (
    load_imdb,
    tokenize,
    build_vocab,
    texts_to_ids,
    get_lm_dataloaders,
)
from eval import evaluate_lm, summarize_history, benchmark_generation

# Model factory imports (deferred to avoid circular import on model side)
from models.rnn_lm import build_rnn_lm
from models.lstm_lm import build_lstm_lm
from models.transformer_lm import build_transformer_lm


# ==============================================================
# 1. Model factory – select model by config.model.model_type
# ==============================================================
MODEL_BUILDERS = {
    "rnn":         build_rnn_lm,
    "lstm":        build_lstm_lm,
    "transformer": build_transformer_lm,
}


def build_model(
    cfg: ExperimentConfig,
    word2idx: dict,
    tokenized_texts=None,
) -> nn.Module:
    """
    Dispatch to the correct model builder based on config.

    Parameters
    ----------
    cfg             : full experiment config
    word2idx        : vocabulary mapping
    tokenized_texts : needed when embedding mode is "word2vec"
    """

    model_type = cfg.model.model_type
    if model_type not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from {list(MODEL_BUILDERS.keys())}"
        )
    builder = MODEL_BUILDERS[model_type]
    return builder(cfg, word2idx, tokenized_texts=tokenized_texts)


# ==============================================================
# 2. Optimizer & scheduler construction
# ==============================================================
def build_optimizer(cfg: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from TrainConfig settings."""
    opt_name = cfg.train.optimizer.lower()
    lr = cfg.train.learning_rate
    wd = cfg.train.weight_decay

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'. Use adam/adamw/sgd.")


def build_scheduler(
    cfg: ExperimentConfig, optimizer: torch.optim.Optimizer
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning-rate scheduler (or None)."""
    name = cfg.train.scheduler.lower()
    if name == "none":
        return None
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.scheduler_step_size,
            gamma=cfg.train.scheduler_gamma,
        )
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs
        )
    else:
        raise ValueError(f"Unknown scheduler '{name}'. Use none/step/cosine.")


# ==============================================================
# 3. Single-epoch training step
# ==============================================================
def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: ExperimentConfig,
    epoch: int,
) -> Dict[str, float]:
    """
    Run one full training epoch and return metrics.
    """

    model.train()
    total_loss = 0.0
    total_tokens = 0
    pad_id = cfg.data.pad_id
    is_rnn = cfg.model.model_type in ("rnn", "lstm")
    log_every = cfg.train.log_every_n_steps

    start = time.perf_counter()

    pbar = epoch_progress_bar(
        dataloader, epoch, cfg.train.epochs, desc="Train"
    )

    for step, (x, y) in enumerate(pbar, 1):
        x = x.to(device)  # (B, T)
        y = y.to(device)  # (B, T)

        # Forward
        output = model(x)

        # Unpack: RNN/LSTM return (logits, hidden); Transformer returns logits
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output

        B, T, V = logits.shape
        loss = criterion(
            logits.reshape(B * T, V),
            y.reshape(B * T),
        )

        # Count non-pad tokens for accurate per-token loss
        non_pad = (y != pad_id).sum().item()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if cfg.train.grad_clip_max_norm > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip_max_norm
            )

        optimizer.step()

        total_loss += loss.item()
        total_tokens += non_pad

        # Step-level logging
        if log_every > 0 and step % log_every == 0:
            step_loss = loss.item() / max(non_pad, 1)
            pbar.set_postfix(loss=f"{step_loss:.4f}")

    elapsed = time.perf_counter() - start
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = compute_perplexity(avg_loss)

    return {
        "train_loss": round(avg_loss, 6),
        "train_ppl":  round(ppl, 2),
        "elapsed_sec": round(elapsed, 2),
    }


# ==============================================================
# 4. Main experiment runner
# ==============================================================
def run_experiment(
    cfg: Optional[ExperimentConfig] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    End-to-end: config → data → model → train → eval → save.

    Parameters
    ----------
    cfg       : pre-built ExperimentConfig (overrides ignored if given)
    overrides : keyword args forwarded to make_config() when cfg is None
    """

    # -------------------------------------------------------
    # Step 1  Config
    # -------------------------------------------------------
    if cfg is None:
        cfg = make_config(**overrides)
    else:
        # Ensure paths and ID are populated
        if not cfg.experiment_id:
            cfg.generate_experiment_id()
        cfg.build_paths()
        cfg.validate()

    print("\n" + "=" * 60)
    print("  Experiment: " + cfg.summary())
    print("=" * 60 + "\n")

    # -------------------------------------------------------
    # Step 2  Seed & device
    # -------------------------------------------------------
    set_seed(cfg.seed)
    device = get_device()

    # -------------------------------------------------------
    # Step 3  Data pipeline
    # -------------------------------------------------------
    with Timer("Data loading & tokenisation"):
        train_texts, train_labels, test_texts, test_labels = load_imdb()

        # Optional: limit training data size for faster iteration
        max_samples = cfg.data.max_samples
        if max_samples and max_samples < len(train_texts):
            train_texts  = train_texts[:max_samples]
            train_labels = train_labels[:max_samples]
            print(f"[data] Using {max_samples}/{25000} training samples")

        train_tokenized = [tokenize(t) for t in train_texts]
        test_tokenized  = [tokenize(t) for t in test_texts]

        word2idx = build_vocab(
            train_tokenized,
            max_vocab_size=cfg.data.max_vocab_size,
            min_freq=cfg.data.min_freq,
        )
        idx2word = {i: w for w, i in word2idx.items()}

        train_ids = texts_to_ids(train_tokenized, word2idx)
        test_ids  = texts_to_ids(test_tokenized,  word2idx)

        train_loader, test_loader = get_lm_dataloaders(
            train_ids, test_ids, word2idx,
            seq_len=cfg.data.lm_seq_len,
            batch_size=cfg.data.batch_size,
            eval_batch_size=cfg.data.eval_batch_size,
            num_workers=cfg.data.num_workers,
        )

    print(f"[data] Vocab size : {len(word2idx)}")
    print(f"[data] Train batches: {len(train_loader)}  |  "
          f"Test batches: {len(test_loader)}")

    # -------------------------------------------------------
    # Step 4  Model  (embedding is built inside the model constructor)
    # -------------------------------------------------------
    with Timer("Model construction"):
        model = build_model(
            cfg, word2idx,
            tokenized_texts=train_tokenized,  # needed for word2vec mode
        )
        model = model.to(device)

    # -------------------------------------------------------
    # Step 5  Optimizer, scheduler, criterion
    # -------------------------------------------------------
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # Use reduction="sum" so that train_one_epoch can compute
    # accurate per-token loss across variable non-pad counts
    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.data.pad_id, reduction="sum"
    )

    # -------------------------------------------------------
    # Step 6  Training loop
    # -------------------------------------------------------
    logger = TrainingLogger(log_dir=cfg.paths.experiment_dir)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = -1

    # Save config before training starts
    cfg.save()

    print(f"\n{'─' * 60}")
    print(f"  Training for {cfg.train.epochs} epochs")
    print(f"{'─' * 60}\n")

    total_timer_start = time.perf_counter()

    for epoch in range(1, cfg.train.epochs + 1):

        # ---- Train ----
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg, epoch
        )

        # ---- Evaluate on test set ----
        val_metrics = evaluate_lm(
            model, test_loader, pad_id=cfg.data.pad_id, device=device,
        )

        # ---- Scheduler step ----
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # ---- Log ----
        logger.log(
            epoch=epoch,
            train_loss=train_metrics["train_loss"],
            train_ppl=train_metrics["train_ppl"],
            val_loss=round(val_metrics["loss"], 6),
            val_ppl=round(val_metrics["ppl"], 2),
            lr=current_lr,
            train_sec=train_metrics["elapsed_sec"],
            val_sec=val_metrics["elapsed_sec"],
        )

        # ---- Checkpoint best model ----
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_metrics["loss"],
                path=cfg.paths.best_model_path,
                val_ppl=val_metrics["ppl"],
            )
        else:
            patience_counter += 1

        # ---- Early stopping ----
        if cfg.train.patience > 0 and patience_counter >= cfg.train.patience:
            print(f"\n[early stop] No improvement for {cfg.train.patience} "
                  f"epochs. Stopping at epoch {epoch}.")
            break

    total_elapsed = time.perf_counter() - total_timer_start

    # -------------------------------------------------------
    # Step 7  Final evaluation (load best checkpoint)
    # -------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"  Final evaluation (best model from epoch {best_epoch})")
    print(f"{'─' * 60}\n")

    # Reload best model weights
    if os.path.exists(cfg.paths.best_model_path):
        ckpt = torch.load(cfg.paths.best_model_path,
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[eval] Loaded best checkpoint (epoch {best_epoch})")

    final_test = evaluate_lm(
        model, test_loader, pad_id=cfg.data.pad_id, device=device,
    )
    print(f"[eval] Test loss = {final_test['loss']:.4f}  |  "
          f"Test ppl = {final_test['ppl']:.2f}")

    # -------------------------------------------------------
    # Step 7b  Inference-time benchmark (fixed-length generation)
    # -------------------------------------------------------
    sos_id = cfg.data.sos_id
    gen_len = cfg.eval.generate_max_len        # default 100 tokens
    prompt = torch.tensor([[sos_id]], dtype=torch.long, device=device)

    gen_bench = benchmark_generation(
        model=model,
        prompt=prompt,
        gen_len=gen_len,
        num_runs=5,
        temperature=cfg.eval.temperature,
        top_k=cfg.eval.top_k,
        eos_id=None,                           # force full gen_len
        device=device,
        model_type=cfg.model.model_type,
    )

    # -------------------------------------------------------
    # Step 8  Summary & save
    # -------------------------------------------------------
    summary = summarize_history(
        logger.history,
        primary_metric="val_loss",
        higher_is_better=False,
        convergence_threshold=cfg.eval.convergence_ppl_threshold,
    )

    # Save training log
    log_path = logger.save(filename="training_log.json")

    # Save a final results file
    results = {
        "experiment_id": cfg.experiment_id,
        "model_type": cfg.model.model_type,
        "embedding_mode": cfg.embedding.mode,
        "embed_dim": cfg.embedding.embed_dim,
        "embedding_freeze": cfg.embedding.freeze,
        "seed": cfg.seed,
        "total_epochs_run": len(logger.history),
        "total_train_sec": round(total_elapsed, 2),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_ppl": round(compute_perplexity(best_val_loss), 2),
        "final_test_loss": round(final_test["loss"], 6),
        "final_test_ppl": round(final_test["ppl"], 2),
        "convergence_epoch": summary.get("convergence_epoch"),
        "vocab_size": len(word2idx),
        "model_params": sum(p.numel() for p in model.parameters()),
        # Inference-time benchmark (fixed-length generation)
        "inference_gen_len": gen_bench["gen_len"],
        "inference_num_runs": gen_bench["num_runs"],
        "inference_avg_sec": gen_bench["avg_sec"],
        "inference_min_sec": gen_bench["min_sec"],
        "inference_max_sec": gen_bench["max_sec"],
        "inference_tokens_per_sec": gen_bench["tokens_per_sec"],
    }
    results_path = os.path.join(cfg.paths.experiment_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] Saved → {results_path}")

    return {
        "config": cfg,
        "best_metrics": logger.get_best("val_loss", mode="min"),
        "summary": summary,
        "model_path": cfg.paths.best_model_path,
        "log_path": str(log_path),
        "results": results,
    }


# ==============================================================
# 5. CLI entry point
# ==============================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for quick experiments."""
    p = argparse.ArgumentParser(
        description="Train a neural language model on IMDB."
    )
    p.add_argument("--model_type", type=str, default="lstm",
                   choices=["rnn", "lstm", "transformer"],
                   help="Model architecture")
    p.add_argument("--embed_dim", type=int, default=128,
                   help="Embedding dimension")
    p.add_argument("--embedding_mode", type=str, default="scratch",
                   choices=["scratch", "word2vec", "pretrained"],
                   help="Embedding strategy")
    p.add_argument("--freeze", action="store_true", default=False,
                   help="Freeze embedding weights (for word2vec / pretrained)")
    p.add_argument("--pretrained_path", type=str, default=None,
                   help="Path to pretrained embedding file (for mode=pretrained)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size")
    p.add_argument("--eval_batch_size", type=int, default=0,
                   help="Eval batch size (0 = 2x batch_size)")
    p.add_argument("--seq_len", type=int, default=128,
                   help="LM sequence chunk length (smaller = fewer batches)")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Max training reviews to use (0 = all 25k)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--patience", type=int, default=3,
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--scheduler", type=str, default="none",
                   choices=["none", "step", "cosine"],
                   help="LR scheduler")
    p.add_argument("--grad_clip", type=float, default=5.0,
                   help="Max gradient norm for clipping")
    p.add_argument("--output_dir", type=str, default="outputs",
                   help="Base output directory")
    return p.parse_args()


def main() -> None:
    """CLI entry point: parse args → build config → run experiment."""
    args = parse_args()

    cfg = make_config(
        model_type=args.model_type,
        task="lm",
        embed_dim=args.embed_dim,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        # Pass through to sub-configs via **overrides
        eval_batch_size=args.eval_batch_size,
        lm_seq_len=args.seq_len,
        max_samples=args.max_samples,
        mode=args.embedding_mode,
        freeze=args.freeze,
        pretrained_path=args.pretrained_path,
        learning_rate=args.lr,
        patience=args.patience,
        scheduler=args.scheduler,
        grad_clip_max_norm=args.grad_clip,
        output_dir=args.output_dir,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()

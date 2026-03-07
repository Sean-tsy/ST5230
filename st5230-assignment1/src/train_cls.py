"""
Usage
-----
::

    python train_cls.py --lm_checkpoint outputs/<exp_id>/best_model.pt \\
                        --model_type lstm --epochs 5

Or from Python::

    from train_cls import run_cls_experiment
    result = run_cls_experiment(
        lm_checkpoint="outputs/.../best_model.pt",
        model_type="lstm",
        epochs=5,
    )


"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ---- project imports ----
from config import ExperimentConfig, make_config
from utils import (
    set_seed,
    get_device,
    TrainingLogger,
    save_checkpoint,
    load_checkpoint,
    Timer,
    epoch_progress_bar,
)
from data import (
    load_imdb,
    tokenize,
    build_vocab,
    texts_to_ids,
    get_sentiment_dataloaders,
)
from eval import evaluate_sentiment, summarize_history

from models.rnn_lm import build_rnn_lm
from models.lstm_lm import build_lstm_lm
from models.transformer_lm import build_transformer_lm


# ==============================================================
# 1. Classifier head on top of a pretrained LM encoder
# ==============================================================
class SentimentClassifier(nn.Module):
    """
    Wraps a pretrained LM encoder with a classification head.

    The LM encoder includes the embedding layer and the recurrent /
    transformer layers, but NOT the final vocabulary projection (fc).

    Parameters
    ----------
    lm_model     : a pretrained RNNLM, LSTMLM, or TransformerLM
    model_type   : "rnn", "lstm", or "transformer"
    num_classes  : number of output classes (2 for binary sentiment)
    classifier_dropout : dropout before the linear head
    """

    def __init__(
        self,
        lm_model: nn.Module,
        model_type: str,
        num_classes: int = 2,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lm_model = lm_model
        self.model_type = model_type

        # Determine hidden dimension from the LM model
        if model_type in ("rnn", "lstm"):
            self.hidden_dim = lm_model.hidden_size
        elif model_type == "transformer":
            self.hidden_dim = lm_model.embed_dim
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Always freeze the LM encoder — only train the classifier head
        for param in self.lm_model.parameters():
            param.requires_grad = False
        print(f"[cls] Froze all LM encoder parameters")

        # Classification head
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (B, T)  padded token-id tensor
        lengths   : (B,)    actual sequence lengths (before padding)
        """

        # ---- Extract representations from the LM encoder ----
        if self.model_type in ("rnn", "lstm"):
            return self._forward_rnn(input_ids, lengths)
        elif self.model_type == "transformer":
            return self._forward_transformer(input_ids, lengths)

    def _forward_rnn(
        self, input_ids: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        RNN / LSTM: use last-hidden-state pooling.

        We run the full sequence through the encoder and pick the
        hidden vector at the position of the last non-pad token.
        """
        lm = self.lm_model

        # Embedding
        emb = lm.embedding(input_ids)        # (B, T, embed_dim)
        emb = lm.embed_drop(emb)

        # RNN / LSTM forward
        if self.model_type == "rnn":
            rnn_out, _ = lm.rnn(emb)         # (B, T, hidden_size)
        else:
            rnn_out, _ = lm.lstm(emb)         # (B, T, hidden_size)

        # Gather the last valid hidden state for each sample
        # lengths: (B,), we need index = lengths - 1
        B = input_ids.size(0)
        last_idx = (lengths - 1).clamp(min=0).long()  # (B,)
        last_idx = last_idx.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        last_idx = last_idx.expand(-1, -1, rnn_out.size(2))  # (B, 1, H)
        pooled = rnn_out.gather(1, last_idx).squeeze(1)  # (B, H)

        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits

    def _forward_transformer(
        self, input_ids: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Transformer: use mean-pooling over non-pad positions.

        We run through embedding + positional encoding + transformer
        blocks, then average the hidden states of non-pad tokens.
        """
        lm = self.lm_model
        B, T = input_ids.shape

        # Embedding + positional encoding
        emb = lm.embedding(input_ids)       # (B, T, embed_dim)
        emb = lm.pos_encoder(emb)           # (B, T, embed_dim)

        # Build padding mask: True where input is pad (id=0)
        pad_mask = (input_ids == 0)          # (B, T)

        # Causal mask
        causal_mask = lm._generate_causal_mask(T, input_ids.device)

        # Transformer forward
        hidden = lm.transformer(
            emb,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )  # (B, T, embed_dim)

        # Mean pooling over non-pad positions
        # Create a float mask: 1.0 for valid, 0.0 for pad
        valid_mask = (~pad_mask).float().unsqueeze(2)  # (B, T, 1)
        summed = (hidden * valid_mask).sum(dim=1)       # (B, embed_dim)
        counts = valid_mask.sum(dim=1).clamp(min=1.0)   # (B, 1)
        pooled = summed / counts                         # (B, embed_dim)

        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits


# ==============================================================
# 2. Build LM + load checkpoint
# ==============================================================
LM_BUILDERS = {
    "rnn":         build_rnn_lm,
    "lstm":        build_lstm_lm,
    "transformer": build_transformer_lm,
}


def load_pretrained_lm(
    cfg: ExperimentConfig,
    word2idx: dict,
    checkpoint_path: str,
    device: torch.device,
    tokenized_texts=None,
) -> nn.Module:
    """
    Build a fresh LM from config, then load pretrained weights.

    Parameters
    ----------
    cfg              : config used to build the LM architecture
    word2idx         : vocabulary mapping
    checkpoint_path  : path to the best_model.pt saved by train_lm.py
    device           : target device
    tokenized_texts  : needed only when embedding mode is "word2vec"
    """

    model_type = cfg.model.model_type
    if model_type not in LM_BUILDERS:
        raise ValueError(
            f"model_type '{model_type}' not supported for classification. "
            f"Choose from {list(LM_BUILDERS.keys())}"
        )

    # Build fresh model (same architecture as the checkpoint)
    builder = LM_BUILDERS[model_type]
    lm_model = builder(cfg, word2idx, tokenized_texts=tokenized_texts)
    lm_model = lm_model.to(device)

    # Load pretrained weights
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"LM checkpoint not found: {checkpoint_path}"
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    lm_model.load_state_dict(ckpt["model_state_dict"])
    print(f"[cls] Loaded pretrained LM checkpoint ← {checkpoint_path}")

    return lm_model


# ==============================================================
# 3. Optimizer for classifier (only non-frozen params)
# ==============================================================
def build_cls_optimizer(
    cfg: ExperimentConfig, model: nn.Module
) -> torch.optim.Optimizer:
    """Create optimizer over trainable parameters only."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt_name = cfg.train.optimizer.lower()
    lr = cfg.train.learning_rate
    wd = cfg.train.weight_decay

    print(f"[cls] Optimizer over {sum(p.numel() for p in trainable):,} "
          f"trainable params (out of "
          f"{sum(p.numel() for p in model.parameters()):,} total)")

    if opt_name == "adam":
        return torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(trainable, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'.")


# ==============================================================
# 4. Single-epoch training step
# ==============================================================
def train_cls_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: ExperimentConfig,
    epoch: int,
) -> Dict[str, float]:
    """
    Train the classifier for one epoch.

    Returns
    -------
    dict with keys: train_loss, train_acc, elapsed_sec
    """
    model.train()

    # Keep frozen encoder in eval mode (disables dropout / batchnorm)
    model.lm_model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start = time.perf_counter()

    pbar = epoch_progress_bar(
        dataloader, epoch, cfg.train.epochs, desc="Train"
    )

    for step, (padded_ids, lengths, labels) in enumerate(pbar, 1):
        padded_ids = padded_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        logits = model(padded_ids, lengths)  # (B, C)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        if cfg.train.grad_clip_max_norm > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.train.grad_clip_max_norm,
            )

        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    elapsed = time.perf_counter() - start
    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)

    return {
        "train_loss": round(avg_loss, 6),
        "train_acc":  round(acc, 4),
        "elapsed_sec": round(elapsed, 2),
    }


# ==============================================================
# 5. Main experiment runner
# ==============================================================
def run_cls_experiment(
    cfg: Optional[ExperimentConfig] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    End-to-end: config → data → pretrained LM (frozen) → classifier → train → eval → save.

    Parameters
    ----------
    cfg       : pre-built ExperimentConfig (overrides ignored if given)
    overrides : keyword args forwarded to make_config() when cfg is None

    The LM checkpoint path and classifier dropout are read from
    cfg.paths.lm_checkpoint_path and cfg.model.classifier_dropout.
    """

    # -------------------------------------------------------
    # Step 1  Config
    # -------------------------------------------------------
    if cfg is None:
        overrides.setdefault("task", "sentiment")
        cfg = make_config(**overrides)
    else:
        if not cfg.experiment_id:
            cfg.generate_experiment_id()
        cfg.build_paths()
        cfg.validate()

    print("\n" + "=" * 60)
    print("  Experiment: " + cfg.summary())
    print("  LM checkpoint: " + cfg.paths.lm_checkpoint_path)
    print("  Classifier dropout: " + str(cfg.model.classifier_dropout))
    print("  Encoder: frozen (only train classifier head)")
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

        train_tokenized = [tokenize(t) for t in train_texts]
        test_tokenized  = [tokenize(t) for t in test_texts]

        # Load vocab from LM experiment dir to guarantee consistency
        lm_dir = os.path.dirname(cfg.paths.lm_checkpoint_path)
        vocab_path = os.path.join(lm_dir, "vocab.json")
        if os.path.isfile(vocab_path):
            with open(vocab_path) as f:
                word2idx = json.load(f)
            print(f"[data] Loaded LM vocab ({len(word2idx)} words) ← {vocab_path}")
        else:
            print(f"[data] WARNING: vocab.json not found in {lm_dir}, rebuilding from scratch")
            word2idx = build_vocab(
                train_tokenized,
                max_vocab_size=cfg.data.max_vocab_size,
                min_freq=cfg.data.min_freq,
            )

        train_ids = texts_to_ids(train_tokenized, word2idx)
        test_ids  = texts_to_ids(test_tokenized, word2idx)

        train_loader, test_loader = get_sentiment_dataloaders(
            train_ids, train_labels, test_ids, test_labels,
            max_len=cfg.data.sentiment_max_len,
            batch_size=cfg.data.batch_size,
            eval_batch_size=cfg.data.eval_batch_size,
            num_workers=cfg.data.num_workers,
        )

    print(f"[data] Vocab size: {len(word2idx)}")
    print(f"[data] Train batches: {len(train_loader)}  |  "
          f"Test batches: {len(test_loader)}")

    # -------------------------------------------------------
    # Step 4  Load pretrained LM encoder
    # -------------------------------------------------------
    with Timer("Loading pretrained LM"):
        lm_model = load_pretrained_lm(
            cfg, word2idx,
            checkpoint_path=cfg.paths.lm_checkpoint_path,
            device=device,
            tokenized_texts=train_tokenized,
        )

    # -------------------------------------------------------
    # Step 5  Build classifier on top of LM
    # -------------------------------------------------------
    model = SentimentClassifier(
        lm_model=lm_model,
        model_type=cfg.model.model_type,
        num_classes=cfg.model.num_classes,
        classifier_dropout=cfg.model.classifier_dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cls] Total params: {total_params:,}  |  "
          f"Trainable: {trainable_params:,}")

    # -------------------------------------------------------
    # Step 6  Optimizer & criterion
    # -------------------------------------------------------
    optimizer = build_cls_optimizer(cfg, model)
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------
    # Step 7  Training loop
    # -------------------------------------------------------
    logger = TrainingLogger(log_dir=cfg.paths.experiment_dir)

    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = -1

    cfg.save()

    print(f"\n{'─' * 60}")
    print(f"  Training classifier for {cfg.train.epochs} epochs")
    print(f"{'─' * 60}\n")

    total_timer_start = time.perf_counter()

    for epoch in range(1, cfg.train.epochs + 1):

        # ---- Train ----
        train_metrics = train_cls_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg, epoch,
        )

        # ---- Evaluate on test set ----
        val_metrics = evaluate_sentiment(
            model, test_loader, device=device,
        )

        # ---- Log ----
        logger.log(
            epoch=epoch,
            train_loss=train_metrics["train_loss"],
            train_acc=train_metrics["train_acc"],
            val_loss=round(val_metrics["loss"], 6),
            val_acc=round(val_metrics["acc"], 4),
            lr=optimizer.param_groups[0]["lr"],
            train_sec=train_metrics["elapsed_sec"],
            val_sec=val_metrics["elapsed_sec"],
        )

        # ---- Checkpoint best model ----
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_metrics["loss"],
                path=cfg.paths.best_model_path,
                val_acc=val_metrics["acc"],
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
    # Step 8  Final evaluation (load best checkpoint)
    # -------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"  Final evaluation (best model from epoch {best_epoch})")
    print(f"{'─' * 60}\n")

    if os.path.exists(cfg.paths.best_model_path):
        ckpt = torch.load(cfg.paths.best_model_path,
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[eval] Loaded best checkpoint (epoch {best_epoch})")

    final_test = evaluate_sentiment(
        model, test_loader, device=device,
    )
    print(f"[eval] Test loss = {final_test['loss']:.4f}  |  "
          f"Test acc = {final_test['acc']:.4f}")

    # -------------------------------------------------------
    # Step 9  Summary & save
    # -------------------------------------------------------
    summary = summarize_history(
        logger.history,
        primary_metric="val_acc",
        higher_is_better=True,
        convergence_threshold=cfg.eval.convergence_acc_threshold,
    )

    log_path = logger.save(filename="training_log.json")

    results = {
        "experiment_id": cfg.experiment_id,
        "task": "sentiment",
        "model_type": cfg.model.model_type,
        "embedding_mode": cfg.embedding.mode,
        "embed_dim": cfg.embedding.embed_dim,
        "freeze_encoder": True,
        "classifier_dropout": cfg.model.classifier_dropout,
        "seed": cfg.seed,
        "total_epochs_run": len(logger.history),
        "total_train_sec": round(total_elapsed, 2),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "final_test_loss": round(final_test["loss"], 6),
        "final_test_acc": round(final_test["acc"], 4),
        "vocab_size": len(word2idx),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lm_checkpoint": cfg.paths.lm_checkpoint_path,
    }

    results_path = os.path.join(cfg.paths.experiment_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] Saved → {results_path}")

    return {
        "config": cfg,
        "best_metrics": logger.get_best("val_acc", mode="max"),
        "summary": summary,
        "model_path": cfg.paths.best_model_path,
        "log_path": str(log_path),
        "results": results,
    }


# ==============================================================
# 6. CLI entry point
# ==============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a sentiment classifier using a pretrained LM."
    )
    p.add_argument("--lm_checkpoint", type=str, required=True,
                   help="Path to pretrained LM checkpoint (best_model.pt)")
    p.add_argument("--model_type", type=str, default="lstm",
                   choices=["rnn", "lstm", "transformer"],
                   help="LM architecture type (must match the checkpoint)")
    p.add_argument("--embed_dim", type=int, default=128,
                   help="Embedding dimension (must match the LM)")
    p.add_argument("--embedding_mode", type=str, default="scratch",
                   choices=["scratch", "word2vec", "pretrained"],
                   help="Embedding strategy (must match the LM)")
    p.add_argument("--pretrained_path", type=str, default=None,
                   help="Path to pretrained embedding file")
    p.add_argument("--classifier_dropout", type=float, default=0.3,
                   help="Dropout for classification head")
    p.add_argument("--epochs", type=int, default=5,
                   help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size")
    p.add_argument("--eval_batch_size", type=int, default=0,
                   help="Eval batch size (0 = 2x batch_size)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--patience", type=int, default=3,
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--output_dir", type=str, default="outputs",
                   help="Base output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = make_config(
        model_type=args.model_type,
        task="sentiment",
        embed_dim=args.embed_dim,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        mode=args.embedding_mode,
        pretrained_path=args.pretrained_path,
        learning_rate=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        classifier_dropout=args.classifier_dropout,
        lm_checkpoint_path=args.lm_checkpoint,
    )

    run_cls_experiment(cfg=cfg)


if __name__ == "__main__":
    main()

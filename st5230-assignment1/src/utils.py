"""
utility toolbox.

Provides helpers shared across n-gram, RNN, LSTM, and Transformer LM
experiments on IMDB.

"""

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm


# ============================================================
# 1. Seed fixing – reproducibility across models & embeddings
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds for Python, NumPy, and PyTorch (CPU + GPU).
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU

    # Deterministic algorithms (may hurt speed slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. CPU / GPU auto-detection
# ============================================================

def get_device() -> torch.device:
    """
    Return the best available device.
    Priority: CUDA GPU  >  Apple MPS  >  CPU
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[utils] Using device: {device}")
    return device


# ============================================================
# 3. Unified training logger (loss / ppl / acc)
# ============================================================
class TrainingLogger:
    """
    Accumulate and record per-epoch metrics.
    """

    def __init__(self, log_dir: str = "outputs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[Dict[str, Any]] = []

    def log(self, **kwargs: Any) -> None:
        """
        Append one record (typically per epoch).
        """

        self.history.append(kwargs)
        # Pretty-print to stdout
        parts = []
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print("[log] " + "  |  ".join(parts))

    def save(self, filename: str = "training_log.json") -> Path:
        """
        Write full history to a JSON file and return its path.
        """

        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[log] Training log saved to {path}")
        return path

    def get_best(self, metric: str = "val_loss", mode: str = "min") -> Dict[str, Any]:
        """
        Return the record with the best value of metric.
        """

        records = [r for r in self.history if metric in r]
        if not records:
            return {}
        if mode == "min":
            return min(records, key=lambda r: r[metric])
        return max(records, key=lambda r: r[metric])


# ============================================================
# 4. Checkpoint – save / load model & training state
# ============================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **extra: Any,
) -> None:
    """
    Save model weights, optimizer state, and metadata to path.

    Parameters
    ----------
    model     : the nn.Module to save
    optimizer : the optimizer whose state dict should be saved
    epoch     : current epoch number
    loss      : latest loss value for reference
    path      : file path 
    extra     : any additional key-value pairs to store
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    state.update(extra)
    torch.save(state, path)
    print(f"[ckpt] Checkpoint saved → {path}  (epoch {epoch}, loss {loss:.4f})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model and optionally optimizer state.
    """

    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    meta = {k: v for k, v in ckpt.items()
            if k not in ("model_state_dict", "optimizer_state_dict")}
    print(f"[ckpt] Loaded checkpoint ← {path}  (epoch {meta.get('epoch', '?')})")
    return meta


# ============================================================
# 5. Time tracking & progress bar helpers
# ============================================================

class Timer:
    """
    Simple wall-clock timer, usable as a context manager.
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start
        if self.label:
            print(f"[timer] {self.label} — {self.elapsed:.2f} s")


def epoch_progress_bar(
    dataloader,
    epoch: int,
    total_epochs: int,
    desc: str = "Train",
) -> tqdm:
    """
    Wrap a DataLoader with a tqdm progress bar for one epoch.
    """

    return tqdm(
        dataloader,
        desc=f"{desc} [{epoch}/{total_epochs}]",
        leave=True,
        dynamic_ncols=True,
    )


# ============================================================
# 6. Perplexity helper
# ============================================================

def compute_perplexity(avg_cross_entropy_loss: float) -> float:
    """
    Convert average cross-entropy loss to perplexity.
    math::
        PPL = e^{\\text{CE loss}}

    If the loss is unreasonably large (> 100), cap the result at 1e30
    to avoid overflow.
    """
    
    if avg_cross_entropy_loss > 100:
        return float("inf")
    return math.exp(avg_cross_entropy_loss)

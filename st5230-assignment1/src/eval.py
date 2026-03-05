
import math
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import compute_perplexity


# ============================================================
# 1. Neural Language-Model evaluation
# ============================================================

@torch.no_grad() 
def evaluate_lm(
    model: nn.Module,
    dataloader: DataLoader,
    pad_id: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate a neural language model (RNN / LSTM / Transformer).

    The model is expected to return logits of shape (B, T, V) where
        B = batch size, T = sequence length, V = vocab size.
    Targets are (B, T).  Padding positions (target == pad_id) are
    ignored in the loss computation.

    Parameters
    ----------
    model      : nn.Module that maps input_ids (B, T) -> logits (B, T, V)
    dataloader : yields (x, y) pairs from LanguageModelDataset
    pad_id     : token index used for padding (default 0)
    device     : device to run on; inferred from model if None
    """

    if device is None:
        device = next(model.parameters()).device 

    model.eval()

    # CrossEntropyLoss with reduction='sum' so we can divide by exact number of non-pad tokens across all batches.
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    total_loss = 0.0 
    total_tokens = 0  
    start = time.perf_counter() # Start the timer for evaluation

    for x, y in dataloader:
        x = x.to(device)           # (B, T)
        y = y.to(device)           # (B, T)

        logits = model(x)          # (B, T, V), the model's raw predictions for the next token at each position

        # If the model returns a tuple (e.g. (logits, hidden)), take first
        if isinstance(logits, tuple):
            logits = logits[0]

        B, T, V = logits.shape
        loss = criterion(
            logits.reshape(B * T, V),   # (B*T, V)
            y.reshape(B * T),           # (B*T,)
        ) 

        # Count non-pad tokens in the target
        non_pad = (y != pad_id).sum().item()

        total_loss += loss.item()
        total_tokens += non_pad

    elapsed = time.perf_counter() - start # End the timer and compute elapsed time

    avg_loss = total_loss / max(total_tokens, 1) # Average cross-entropy loss per token
    ppl = compute_perplexity(avg_loss) # Compute perplexity from the average loss

    return {
        "loss": avg_loss,
        "ppl": ppl,
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 3),
    }


# ============================================================
# 2. N-gram Language-Model evaluation
# ============================================================

def evaluate_ngram_lm(
    ngram_model,
    token_ids: Sequence[int],
    n: int,
    vocab_size: int,
    pad_id: int = 0,
) -> Dict[str, float]:
    """
    Evaluate an n-gram LM on a flat token-id stream.

    Parameters
    ----------
    ngram_model : object with get_log_prob(history, next_id)
    token_ids   : flat sequence of token ids (the evaluation stream)
    n           : n-gram order 
    vocab_size  : vocabulary size (used only for the fallback uniform prob)
    pad_id      : padding index to skip 
    """

    start = time.perf_counter()

    total_nll = 0.0   # negative log-likelihood (natural log)
    total_tokens = 0

    # Minimum log-prob for unseen contexts (uniform fallback)
    floor_logp = math.log(1.0 / vocab_size) if vocab_size > 0 else -30.0

    for i in range(n - 1, len(token_ids)):
        target = token_ids[i]
        if target == pad_id:
            continue

        history = tuple(token_ids[i - n + 1 : i])
        logp = ngram_model.get_log_prob(history, target)

        # Guard against -inf / nan
        if math.isinf(logp) or math.isnan(logp):
            logp = floor_logp

        total_nll -= logp           # nll = -sum(log P)
        total_tokens += 1

    elapsed = time.perf_counter() - start

    avg_loss = total_nll / max(total_tokens, 1)
    ppl = compute_perplexity(avg_loss)

    return {
        "loss": avg_loss,
        "ppl": ppl,
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 3),
    }


# ============================================================
# 3. Sentiment classification evaluation
# ============================================================

@torch.no_grad()
def evaluate_sentiment(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate a sentiment classifier.

    The DataLoader is expected to yield (padded_ids, lengths, labels)
    as produced by collate_sentiment in data.py.  The model should
    accept at least (padded_ids, lengths) and return logits of shape
    (B, num_classes).

    Parameters
    ----------
    model      : nn.Module mapping (ids, lengths) -> logits (B, C)
    dataloader : yields (padded_ids, lengths, labels) triplets
    device     : device to run on; inferred from model if None
    """

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start = time.perf_counter()

    for padded_ids, lengths, labels in dataloader:
        padded_ids = padded_ids.to(device)   # (B, max_len)
        lengths    = lengths.to(device)       # (B,)
        labels     = labels.to(device)        # (B,)

        logits = model(padded_ids, lengths)   # (B, C)

        # If the model returns a tuple, take the first element
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)         # (B,)
        total_loss += loss.item()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    elapsed = time.perf_counter() - start

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)

    return {
        "loss": avg_loss,
        "acc": acc,
        "total": total_samples,
        "elapsed_sec": round(elapsed, 3),
    }


# ============================================================
# 4. Generation inference-time benchmark
# ============================================================

@torch.no_grad()
def benchmark_generation(
    model,
    prompt,
    gen_len: int = 100,
    num_runs: int = 5,
    temperature: float = 1.0,
    top_k: int = 0,
    eos_id: Optional[int] = None,
    device: Optional[torch.device] = None,
    model_type: str = "neural",
) -> Dict[str, Any]:
    """Time autoregressive generation for a fixed output length.

    Works with all four model types:
    * Neural models (RNN / LSTM / Transformer) — call model.generate()
    * N-gram model — call model.generate()

    Parameters
    ----------
    model      : any LM with a generate() method
    prompt     : for neural models: (1, P) token-id tensor
                 for n-gram: list[int] of prompt token ids
    gen_len    : number of tokens to generate (fixed length)
    num_runs   : repeat generation this many times and average
    temperature: sampling temperature
    top_k      : top-k sampling (0 = unrestricted)
    eos_id     : early stopping token (None = always generate gen_len)
    device     : device for neural models; ignored for n-gram
    model_type : "rnn", "lstm", "transformer", or "ngram"

    Returns
    -------
    dict with keys:
        total_sec      : total wall-clock across all runs
        avg_sec        : average time per run
        min_sec        : fastest run
        max_sec        : slowest run
        tokens_per_sec : average generation speed
        gen_len        : requested generation length
        num_runs       : number of runs
    """
    is_ngram = model_type == "ngram"
    timings: List[float] = []

    if not is_ngram:
        # Ensure model is on the right device and in eval mode
        if device is None:
            device = next(model.parameters()).device
        model.eval()
        if isinstance(prompt, list):
            prompt = torch.tensor([prompt], dtype=torch.long, device=device)
        prompt = prompt.to(device)

    for _ in range(num_runs):
        start = time.perf_counter()

        if is_ngram:
            # N-gram generate() takes a list, not a tensor
            prompt_list = prompt if isinstance(prompt, list) else prompt[0].tolist()
            _ = model.generate(
                prompt=prompt_list,
                max_len=gen_len,
                temperature=temperature,
                eos_id=eos_id,
            )
        else:
            # Neural model generate()
            _ = model.generate(
                prompt=prompt,
                max_len=gen_len,
                temperature=temperature,
                top_k=top_k,
                eos_id=eos_id,
            )

        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    total = sum(timings)
    avg = total / num_runs
    tokens_per_sec = gen_len / avg if avg > 0 else float("inf")

    result = {
        "total_sec":      round(total, 4),
        "avg_sec":        round(avg, 4),
        "min_sec":        round(min(timings), 4),
        "max_sec":        round(max(timings), 4),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "gen_len":        gen_len,
        "num_runs":       num_runs,
    }

    print(f"[benchmark] {model_type} generation: "
          f"{gen_len} tokens × {num_runs} runs  |  "
          f"avg={avg:.4f}s  min={min(timings):.4f}s  max={max(timings):.4f}s  |  "
          f"{tokens_per_sec:.1f} tok/s")

    return result


# ============================================================
# 5. Training-history summary / aggregation
# ============================================================

def summarize_history(
    history: List[Dict[str, Any]],
    primary_metric: str = "val_loss",
    higher_is_better: bool = False,
    convergence_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Aggregate a list of per-epoch records into a concise summary.

    Parameters
    ----------
    history              : list of dicts, one per epoch, as stored by
                           TrainingLogger.history 
    primary_metric       : the key used to pick the best epoch
    higher_is_better     : False → lower is better (loss, ppl);
                           True  → higher is better (acc)
    convergence_threshold: if provided, find the first epoch where
                           primary_metric crosses this value
                           (below threshold when lower-is-better,
                            above threshold when higher-is-better)
    """

    if not history:
        return {
            "best_epoch": None,
            "best_metrics": {},
            "convergence_epoch": None,
            "stats": {},
            "num_epochs": 0,
        }

    # --- Filter to records that contain the primary metric ---
    valid = [r for r in history if primary_metric in r]
    if not valid:
        return {
            "best_epoch": None,
            "best_metrics": {},
            "convergence_epoch": None,
            "stats": {},
            "num_epochs": len(history),
        }

    values = [r[primary_metric] for r in valid]

    # --- Best epoch ---
    if higher_is_better:
        best_record = max(valid, key=lambda r: r[primary_metric])
    else:
        best_record = min(valid, key=lambda r: r[primary_metric])

    best_epoch = best_record.get("epoch", None)

    # --- Convergence: first epoch that crosses the threshold ---
    convergence_epoch = None
    if convergence_threshold is not None:
        for r in valid:
            v = r[primary_metric]
            if (not higher_is_better and v <= convergence_threshold) or \
               (higher_is_better and v >= convergence_threshold):
                convergence_epoch = r.get("epoch", None)
                break

    # --- Descriptive statistics for the metric across epochs ---
    import numpy as np
    arr = np.array(values, dtype=float)
    stats = {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
    }

    summary = {
        "best_epoch": best_epoch,
        "best_metrics": best_record,
        "convergence_epoch": convergence_epoch,
        "stats": stats,
        "num_epochs": len(history),
    }

    # --- Pretty-print ---
    print("=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Total epochs       : {summary['num_epochs']}")
    print(f"  Primary metric     : {primary_metric}  "
          f"({'↑' if higher_is_better else '↓'} is better)")
    print(f"  Best epoch         : {best_epoch}")
    for k, v in best_record.items():
        if isinstance(v, float):
            print(f"    {k:20s} = {v:.4f}")
        else:
            print(f"    {k:20s} = {v}")
    print(f"  Convergence epoch  : {convergence_epoch}  "
          f"(threshold={convergence_threshold})")
    print(f"  {primary_metric} stats:")
    print(f"    mean = {stats['mean']:.4f}  |  std = {stats['std']:.4f}  "
          f"|  min = {stats['min']:.4f}  |  max = {stats['max']:.4f}")
    print("=" * 60)

    return summary

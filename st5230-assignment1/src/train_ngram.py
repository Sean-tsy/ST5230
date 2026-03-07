

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

# ---- project imports ----
from config import ExperimentConfig, make_config
from utils import set_seed, Timer
from data import (
    load_imdb,
    tokenize,
    build_vocab,
    texts_to_ids,
    SOS_TOKEN,
    EOS_TOKEN,
)
from eval import evaluate_ngram_lm, benchmark_generation
from models.ngram import build_ngram_lm


# ==============================================================
# 1. Prepare token-id streams for eval
# ==============================================================
def _build_token_stream(
    id_sequences, word2idx: dict
) -> list:
    """
    Concatenate id sequences into a single flat token stream.

    Inserts <sos> and <eos> around each review, matching what
    LanguageModelDataset does in data.py so that perplexity
    numbers are comparable between neural and n-gram models.
    """
    
    sos_id = word2idx[SOS_TOKEN]
    eos_id = word2idx[EOS_TOKEN]
    stream = []
    for ids in id_sequences:
        stream.append(sos_id)
        stream.extend(ids)
        stream.append(eos_id)
    return stream


# ==============================================================
# 2. Main experiment runner
# ==============================================================
def run_ngram_experiment(
    cfg: Optional[ExperimentConfig] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    End-to-end: config → data → model → fit → eval → save.

    Parameters
    ----------
    cfg       : pre-built ExperimentConfig (overrides ignored if given)
    overrides : keyword args forwarded to make_config() when cfg is None
    """

    # -------------------------------------------------------
    # Step 1  Config
    # -------------------------------------------------------
    if cfg is None:
        overrides.setdefault("model_type", "ngram")
        overrides.setdefault("task", "lm")
        cfg = make_config(**overrides)
    else:
        if not cfg.experiment_id:
            cfg.generate_experiment_id()
        cfg.build_paths()
        cfg.validate()

    print("\n" + "=" * 60)
    print("  Experiment: " + cfg.summary())
    print("=" * 60 + "\n")

    # -------------------------------------------------------
    # Step 2  Seed
    # -------------------------------------------------------
    set_seed(cfg.seed)

    # -------------------------------------------------------
    # Step 3  Data pipeline
    # -------------------------------------------------------
    with Timer("Data loading & tokenisation"):
        train_texts, train_labels, test_texts, test_labels = load_imdb()

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

    vocab_size = len(word2idx)
    print(f"[data] Vocab size: {vocab_size}")

    # Build flat token streams for n-gram counting and evaluation
    train_stream = _build_token_stream(train_ids, word2idx)
    test_stream  = _build_token_stream(test_ids,  word2idx)
    print(f"[data] Train stream: {len(train_stream):,} tokens  |  "
          f"Test stream: {len(test_stream):,} tokens")

    # -------------------------------------------------------
    # Step 4  Model construction
    # -------------------------------------------------------
    model = build_ngram_lm(cfg, vocab_size=vocab_size)

    # -------------------------------------------------------
    # Step 5  Training (= counting n-grams)
    # -------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"  Fitting {cfg.model.ngram_order}-gram model")
    print(f"{'─' * 60}\n")

    with Timer("N-gram fitting (counting)") as fit_timer:
        # fit() expects list-of-lists; we pass the per-review id lists
        # (with <sos>/<eos> added) so that boundary n-grams are correct
        sos_id = word2idx[SOS_TOKEN]
        eos_id = word2idx[EOS_TOKEN]
        train_sequences_with_boundaries = [
            [sos_id] + ids + [eos_id] for ids in train_ids
        ]
        model.fit(train_sequences_with_boundaries)

    train_elapsed = fit_timer.elapsed

    # -------------------------------------------------------
    # Step 6  Evaluation
    # -------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"  Evaluation")
    print(f"{'─' * 60}\n")

    n = cfg.model.ngram_order

    # Train-set perplexity (sanity check)
    train_results = evaluate_ngram_lm(
        model, train_stream, n=n, vocab_size=vocab_size,
        pad_id=cfg.data.pad_id,
    )
    print(f"[eval] Train  loss={train_results['loss']:.4f}  "
          f"ppl={train_results['ppl']:.2f}  "
          f"({train_results['elapsed_sec']:.2f}s)")

    # Test-set perplexity
    test_results = evaluate_ngram_lm(
        model, test_stream, n=n, vocab_size=vocab_size,
        pad_id=cfg.data.pad_id,
    )
    print(f"[eval] Test   loss={test_results['loss']:.4f}  "
          f"ppl={test_results['ppl']:.2f}  "
          f"({test_results['elapsed_sec']:.2f}s)")

    # -------------------------------------------------------
    # Step 7  Inference-time benchmark (fixed-length generation)
    # -------------------------------------------------------
    gen_len = cfg.eval.generate_max_len
    prompt = [sos_id]

    gen_bench = benchmark_generation(
        model=model,
        prompt=prompt,
        gen_len=gen_len,
        num_runs=5,
        temperature=cfg.eval.temperature,
        eos_id=None,              # force full gen_len
        model_type="ngram",
    )

    # -------------------------------------------------------
    # Step 8  Save config, vocab & results
    # -------------------------------------------------------
    cfg.save()
    vocab_path = os.path.join(cfg.paths.experiment_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(word2idx, f)
    print(f"[data] Vocab saved → {vocab_path}")

    results = {
        "experiment_id": cfg.experiment_id,
        "model_type": "ngram",
        "ngram_order": cfg.model.ngram_order,
        "smoothing": cfg.model.smoothing,
        "smoothing_alpha": cfg.model.smoothing_alpha,
        "seed": cfg.seed,
        "vocab_size": vocab_size,
        "num_contexts": len(model.ngram_counts),
        "total_ngrams": sum(model.context_totals.values()),
        # Timing
        "fit_sec": round(train_elapsed, 2),
        # Train-set metrics
        "train_loss": round(train_results["loss"], 6),
        "train_ppl": round(train_results["ppl"], 2),
        "train_eval_sec": train_results["elapsed_sec"],
        # Test-set metrics
        "test_loss": round(test_results["loss"], 6),
        "test_ppl": round(test_results["ppl"], 2),
        "test_eval_sec": test_results["elapsed_sec"],
        # Inference benchmark
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
    print(f"\n[results] Saved → {results_path}")

    # -------------------------------------------------------
    # Step 9  Sample generation (qualitative check)
    # -------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("  Sample generation")
    print(f"{'─' * 60}")
    sample = model.generate(
        prompt=[sos_id], max_len=30,
        temperature=cfg.eval.temperature, eos_id=eos_id,
    )
    sample_text = " ".join(idx2word.get(t, "<?>") for t in sample)
    print(f"  {sample_text}")
    print(f"{'─' * 60}\n")

    return {
        "config": cfg,
        "train_results": train_results,
        "test_results": test_results,
        "inference_bench": gen_bench,
        "results_path": results_path,
        "results": results,
    }


# ==============================================================
# 3. CLI entry point
# ==============================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for n-gram experiments."""
    p = argparse.ArgumentParser(
        description="Fit and evaluate an n-gram language model on IMDB."
    )
    p.add_argument("--ngram_order", type=int, default=3,
                   help="N-gram order (2=bigram, 3=trigram, ...)")
    p.add_argument("--smoothing", type=str, default="laplace",
                   choices=["none", "laplace", "kneser_ney"],
                   help="Smoothing method")
    p.add_argument("--smoothing_alpha", type=float, default=1.0,
                   help="Smoothing parameter (Laplace alpha / KN discount)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--max_vocab_size", type=int, default=30_000,
                   help="Maximum vocabulary size")
    p.add_argument("--min_freq", type=int, default=2,
                   help="Minimum word frequency to include in vocab")
    p.add_argument("--output_dir", type=str, default="outputs",
                   help="Base output directory")
    return p.parse_args()


def main() -> None:
    """CLI entry point: parse args → build config → run experiment."""
    args = parse_args()

    cfg = make_config(
        model_type="ngram",
        task="lm",
        seed=args.seed,
        epochs=1,            # not used, but satisfies validation
        batch_size=64,       # not used for n-gram
        # N-gram specific overrides
        ngram_order=args.ngram_order,
        smoothing=args.smoothing,
        smoothing_alpha=args.smoothing_alpha,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        output_dir=args.output_dir,
    )

    run_ngram_experiment(cfg)


if __name__ == "__main__":
    main()

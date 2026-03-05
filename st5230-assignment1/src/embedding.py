"""
Supports three embedding strategies (all return a plain ``nn.Embedding``):

    1. **scratch**      Trainable embeddings learned from scratch
                         (random initialisation, updated during training).
    2. **word2vec**     Fixed embeddings trained by yourself on the IMDB
                         corpus via gensim Word2Vec.
    3. **pretrained**   Fixed embeddings loaded from a public source
                         (e.g. GloVe, fastText).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import EmbeddingConfig, DataConfig


# ============================================================
# 1. Unified public interface  (config-driven)
# ============================================================
def build_embedding_layer(
    emb_cfg: EmbeddingConfig,
    data_cfg: DataConfig,
    word2idx: Dict[str, int],
    tokenized_texts: Optional[List[List[str]]] = None,
    seed: int = 42,
) -> nn.Embedding:
    """Build and return a ``nn.Embedding`` layer driven by config.

    All hyper-parameters (mode, embed_dim, freeze, pretrained_path,
    Word2Vec settings …) are read from *emb_cfg* and *data_cfg*
    so that the caller only passes config objects.

    Parameters
    ----------
    emb_cfg          : EmbeddingConfig  (mode, embed_dim, freeze, …)
    data_cfg         : DataConfig       (pad_id, vocab size info)
    word2idx         : word → index mapping (from data.build_vocab)
    tokenized_texts  : list of token lists, needed when mode="word2vec"
    seed             : random seed (passed to Word2Vec for reproducibility)
    """
    vocab_size = len(word2idx)
    embed_dim  = emb_cfg.embed_dim
    freeze     = emb_cfg.freeze
    pad_idx    = data_cfg.pad_id
    mode       = emb_cfg.mode

    if mode == "scratch":
        emb = _build_scratch(vocab_size, embed_dim, pad_idx)
        print(f"[embedding] mode=scratch  dim={embed_dim}  "
              f"freeze={freeze}")

    elif mode == "word2vec":
        if tokenized_texts is None:
            raise ValueError(
                "mode='word2vec' requires tokenized_texts "
                "(training corpus for Word2Vec)."
            )
        weight = train_word2vec(
            tokenized_texts,
            word2idx,
            embed_dim=embed_dim,
            window=emb_cfg.w2v_window,
            min_count=emb_cfg.w2v_min_count,
            sg=emb_cfg.w2v_sg,
            epochs=emb_cfg.w2v_epochs,
            workers=emb_cfg.w2v_workers,
            seed=seed,
            save_path=emb_cfg.w2v_save_path,
        )
        emb = _build_from_weight(weight, pad_idx)
        print(f"[embedding] mode=word2vec  dim={embed_dim}  "
              f"freeze={freeze}")

    elif mode == "pretrained":
        if emb_cfg.pretrained_path is None:
            raise ValueError(
                "mode='pretrained' requires EmbeddingConfig.pretrained_path "
                "(e.g. 'glove.6B.100d.txt')."
            )
        weight = load_pretrained_vectors(
            emb_cfg.pretrained_path, word2idx, embed_dim
        )
        emb = _build_from_weight(weight, pad_idx)
        print(f"[embedding] mode=pretrained  dim={embed_dim}  "
              f"freeze={freeze}  file={emb_cfg.pretrained_path}")

    else:
        raise ValueError(
            f"Unknown embedding mode '{mode}'.  "
            f"Choose from: scratch, word2vec, pretrained."
        )

    # Apply freeze setting
    emb.weight.requires_grad_(not freeze)

    return emb


# ------------------------------------------------------------------
# Internal builders
# ------------------------------------------------------------------
def _build_scratch(
    vocab_size: int, embed_dim: int, pad_idx: int
) -> nn.Embedding:
    """Random-initialised embedding; pad vector stays zero."""
    emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
    # Xavier-uniform init for non-pad rows
    nn.init.xavier_uniform_(emb.weight.data)
    emb.weight.data[pad_idx].zero_()
    return emb


def _build_from_weight(
    weight: torch.Tensor, pad_idx: int
) -> nn.Embedding:
    """Construct nn.Embedding from an existing weight matrix."""
    vocab_size, embed_dim = weight.shape
    emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
    emb.weight = nn.Parameter(weight)
    emb.weight.data[pad_idx].zero_()
    return emb


# ============================================================
# 2. Train your own embeddings – Word2Vec (gensim)
# ============================================================
def train_word2vec(
    tokenized_texts: List[List[str]],
    word2idx: Dict[str, int],
    embed_dim: int = 128,
    window: int = 5,
    min_count: int = 1,
    sg: int = 1,
    epochs: int = 10,
    workers: int = 4,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Train a Word2Vec model on tokenized_texts and return a weight
    matrix aligned with *word2idx*.

    Parameters
    ----------
    tokenized_texts : list of token-lists (training corpus)
    word2idx        : project vocabulary mapping
    embed_dim       : vector dimensionality (= Word2Vec ``vector_size``)
    window          : Word2Vec context window
    min_count       : ignore words with frequency < min_count in W2V
    sg              : 1 = Skip-gram, 0 = CBOW
    epochs          : Word2Vec training epochs
    workers         : CPU threads for training
    seed            : random seed for reproducibility
    save_path       : (optional) save the gensim model to disk
    """

    from gensim.models import Word2Vec

    print("[embedding] Training Word2Vec …")
    w2v = Word2Vec(
        sentences=tokenized_texts,
        vector_size=embed_dim,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers,
        seed=seed,
    )
    print(f"[embedding] Word2Vec done – "
          f"{len(w2v.wv)} vectors of dim {w2v.wv.vector_size}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        w2v.save(save_path)
        print(f"[embedding] Word2Vec model saved → {save_path}")

    # Build weight matrix aligned to word2idx
    vocab_size = len(word2idx)
    weight = torch.randn(vocab_size, embed_dim) * 0.01  # small random init

    found = 0
    for word, idx in word2idx.items():
        if word in w2v.wv:
            weight[idx] = torch.from_numpy(w2v.wv[word].copy())
            found += 1

    print(f"[embedding] Word2Vec coverage: {found}/{vocab_size} "
          f"({found / vocab_size * 100:.1f}%)")

    return weight


# ============================================================
# 3. Load public pretrained embeddings (GloVe / fastText format)
# ============================================================
def load_pretrained_vectors(
    path: str,
    word2idx: Dict[str, int],
    embed_dim: int,
) -> torch.Tensor:
    """
    Read a text-format embedding file and return a weight matrix.
    The function checks that every loaded vector has exactly
    ``embed_dim`` floats.  A mismatch raises ``ValueError``.

    Parameters
    ----------
    path      : path to the embedding text file
    word2idx  : project vocabulary mapping
    embed_dim : expected dimensionality
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained file not found: {path}")

    vocab_size = len(word2idx)
    weight = torch.randn(vocab_size, embed_dim) * 0.01  # fallback init

    found = 0
    dim_checked = False

    print(f"[embedding] Loading pretrained vectors from {path} …")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip().split(" ")

            # Some files have a header line (e.g. fastText):  "400000 300"
            if line_no == 1 and len(parts) == 2:
                try:
                    int(parts[0])
                    int(parts[1])
                    continue      # skip header
                except ValueError:
                    pass          # not a header; treat as normal line

            word = parts[0]
            try:
                vec = [float(x) for x in parts[1:]]
            except ValueError:
                continue  # skip malformed lines

            # Dimension check (once)
            if not dim_checked:
                if len(vec) != embed_dim:
                    raise ValueError(
                        f"Pretrained dim mismatch: file has {len(vec)}-d "
                        f"vectors but embed_dim={embed_dim}.  "
                        f"Use a matching file or change embed_dim."
                    )
                dim_checked = True

            if word in word2idx:
                weight[word2idx[word]] = torch.tensor(vec, dtype=torch.float)
                found += 1

    print(f"[embedding] Pretrained coverage: {found}/{vocab_size} "
          f"({found / vocab_size * 100:.1f}%)")

    return weight


# ============================================================
# 4. Embedding quality evaluation
# ============================================================
def embedding_coverage(
    weight: torch.Tensor,
    word2idx: Dict[str, int],
    special_tokens: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute coverage statistics for an embedding weight matrix.

    Parameters
    ----------
    weight         : (vocab_size, embed_dim) tensor
    word2idx       : vocabulary mapping
    special_tokens : tokens to exclude from statistics
    """

    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]

    special_ids = {word2idx[t] for t in special_tokens if t in word2idx}

    norms = weight.norm(dim=1)                # (vocab_size,)
    threshold = 0.05                          # near-zero → not initialised

    total = 0
    covered = 0
    uncovered_examples: List[str] = []
    idx2word = {i: w for w, i in word2idx.items()}

    for idx in range(weight.size(0)):
        if idx in special_ids:
            continue
        total += 1
        if norms[idx].item() > threshold:
            covered += 1
        else:
            if len(uncovered_examples) < 20:
                uncovered_examples.append(idx2word.get(idx, f"idx={idx}"))

    coverage_pct = covered / max(total, 1) * 100

    return {
        "total": total,
        "covered": covered,
        "coverage_pct": round(coverage_pct, 2),
        "uncovered_examples": uncovered_examples,
    }


def similarity_sanity_check(
    weight: torch.Tensor,
    word2idx: Dict[str, int],
    query_pairs: Optional[List[Tuple[str, str]]] = None,
    top_k: int = 5,
    query_words: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Run cosine-similarity spot checks on the embedding matrix.

    Parameters
    ----------
    weight       : (vocab_size, embed_dim)
    word2idx     : vocabulary mapping
    query_pairs  : list of (word_a, word_b), compute pairwise similarity
    top_k        : for each query_word, find top-k nearest neighbours
    query_words  : words whose nearest neighbours will be listed
    """

    idx2word = {i: w for w, i in word2idx.items()}

    # Normalise once
    norms = weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = weight / norms                  # (V, D)

    # ---- Pairwise similarities ----
    if query_pairs is None:
        query_pairs = [
            ("good", "great"),
            ("good", "bad"),
            ("movie", "film"),
            ("terrible", "awful"),
            ("happy", "sad"),
        ]

    pair_sims: List[Dict[str, object]] = []
    for w1, w2 in query_pairs:
        if w1 not in word2idx or w2 not in word2idx:
            pair_sims.append({"pair": (w1, w2), "cosine": None,
                              "note": "OOV"})
            continue
        v1 = normed[word2idx[w1]]
        v2 = normed[word2idx[w2]]
        cos = (v1 @ v2).item()
        pair_sims.append({"pair": (w1, w2), "cosine": round(cos, 4)})

    # ---- Nearest neighbours ----
    if query_words is None:
        query_words = ["good", "movie", "terrible", "love", "boring"]

    nn_results: Dict[str, List[Tuple[str, float]]] = {}
    for w in query_words:
        if w not in word2idx:
            nn_results[w] = []
            continue
        qvec = normed[word2idx[w]].unsqueeze(0)     # (1, D)
        sims = (qvec @ normed.T).squeeze(0)          # (V,)
        # Exclude self
        sims[word2idx[w]] = -1.0
        topk_vals, topk_ids = sims.topk(top_k)
        neighbours = [
            (idx2word.get(idx.item(), "?"), round(val.item(), 4))
            for val, idx in zip(topk_vals, topk_ids)
        ]
        nn_results[w] = neighbours

    return {
        "pair_similarities": pair_sims,
        "nearest_neighbours": nn_results,
    }


def embedding_report(
    emb_layer: nn.Embedding,
    word2idx: Dict[str, int],
    label: str = "",
) -> Dict[str, object]:
    """
    Print a combined quality report and return all metrics.

    Parameters
    ----------
    emb_layer : nn.Embedding produced by build_embedding_layer
    word2idx  : vocabulary mapping
    label     : optional label for display 
    """

    weight = emb_layer.weight.data.clone().cpu()

    cov = embedding_coverage(weight, word2idx)
    sim = similarity_sanity_check(weight, word2idx)

    # ---- Pretty-print ----
    header = f"Embedding Report" + (f" [{label}]" if label else "")
    print("\n" + "=" * 60)
    print(f"  {header}")
    print("=" * 60)

    print(f"  Shape        : {tuple(weight.shape)}")
    print(f"  Coverage     : {cov['covered']}/{cov['total']} "
          f"({cov['coverage_pct']}%)")
    if cov["uncovered_examples"]:
        print(f"  Uncovered ex : {cov['uncovered_examples'][:10]}")

    print("\n  Pair cosine similarities:")
    for entry in sim["pair_similarities"]:
        w1, w2 = entry["pair"]
        cos = entry.get("cosine")
        note = entry.get("note", "")
        if cos is not None:
            print(f"    {w1:>12s} – {w2:<12s}  cos = {cos:+.4f}")
        else:
            print(f"    {w1:>12s} – {w2:<12s}  ({note})")

    print("\n  Nearest neighbours:")
    for word, neighbours in sim["nearest_neighbours"].items():
        if not neighbours:
            print(f"    {word}: (OOV)")
            continue
        nns = ", ".join(f"{w}({s:+.3f})" for w, s in neighbours)
        print(f"    {word}: {nns}")

    print("=" * 60 + "\n")

    return {"coverage": cov, "similarity": sim}

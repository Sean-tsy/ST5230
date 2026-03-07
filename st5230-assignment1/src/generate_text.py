"""
Load trained models from exp1 and generate sample text for the report.

Usage:
    cd st5230-assignment1
    python src/generate_text.py

No retraining needed — neural models are loaded from checkpoints.
The ngram model is re-fit (counting only, takes a few seconds).
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import ExperimentConfig
from data import load_imdb, tokenize, build_vocab, texts_to_ids, SOS_TOKEN, EOS_TOKEN
from models.rnn_lm import build_rnn_lm
from models.lstm_lm import build_lstm_lm
from models.transformer_lm import build_transformer_lm
from models.ngram import build_ngram_lm

# ── Experiment directories ──────────────────────────────────
EXP1_DIR = "outputs/exp1"
MODEL_DIRS = {
    "rnn":         "rnn_lm_rnn_emb128_s42_0307_0406_be7f97ee",
    "lstm":        "lstm_lm_lstm_emb128_s42_0307_0529_74a5a180",
    "transformer": "transformer_lm_transformer_emb128_s42_0307_0715_3e18b82a",
    "ngram":       "ngram_lm_ngram_emb128_s42_0307_0405_a046f7c4",
}

NEURAL_BUILDERS = {
    "rnn":         build_rnn_lm,
    "lstm":        build_lstm_lm,
    "transformer": build_transformer_lm,
}

# ── Generation parameters ───────────────────────────────────
GEN_LEN     = 50
TEMPERATURE = 0.8
TOP_K       = 10
SEED        = 42


def ids_to_text(ids, idx2word):
    """Convert a list of token ids back to a readable string."""
    tokens = [idx2word.get(i, "<unk>") for i in ids]
    # Remove special tokens for readability
    tokens = [t for t in tokens if t not in ("<pad>", "<sos>", "<eos>")]
    return " ".join(tokens)


def generate_neural(model_type, device):
    """Load a neural model checkpoint and generate text."""
    exp_dir = os.path.join(EXP1_DIR, MODEL_DIRS[model_type])
    cfg = ExperimentConfig.load(os.path.join(exp_dir, "config.json"))

    with open(os.path.join(exp_dir, "vocab.json")) as f:
        word2idx = json.load(f)
    idx2word = {int(i): w for w, i in word2idx.items()}

    # Build model architecture (embedding from scratch, weights will be overwritten)
    model = NEURAL_BUILDERS[model_type](cfg, word2idx)
    model = model.to(device)

    # Load trained weights
    ckpt_path = os.path.join(exp_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Generate
    sos_id = cfg.data.sos_id
    prompt = torch.tensor([[sos_id]], dtype=torch.long, device=device)

    torch.manual_seed(SEED)
    generated = model.generate(
        prompt=prompt,
        max_len=GEN_LEN,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        eos_id=cfg.data.eos_id,
    )

    token_ids = generated[0].tolist()
    return ids_to_text(token_ids, idx2word)


def generate_ngram():
    """Re-fit the ngram model (fast counting) and generate text."""
    import random
    exp_dir = os.path.join(EXP1_DIR, MODEL_DIRS["ngram"])
    cfg = ExperimentConfig.load(os.path.join(exp_dir, "config.json"))

    with open(os.path.join(exp_dir, "vocab.json")) as f:
        word2idx = json.load(f)
    idx2word = {int(i): w for w, i in word2idx.items()}

    # Need to re-fit: load data → count n-grams (a few seconds)
    print("[ngram] Re-fitting (counting only, no gradient training) ...")
    train_texts, _, _, _ = load_imdb()
    train_tokenized = [tokenize(t) for t in train_texts]
    train_ids = texts_to_ids(train_tokenized, word2idx)

    sos_id = word2idx[SOS_TOKEN]
    eos_id = word2idx[EOS_TOKEN]
    train_seqs = [[sos_id] + ids + [eos_id] for ids in train_ids]

    model = build_ngram_lm(cfg, vocab_size=len(word2idx))
    model.fit(train_seqs)

    # Generate
    random.seed(SEED)
    token_ids = model.generate(
        prompt=[sos_id],
        max_len=GEN_LEN,
        temperature=TEMPERATURE,
        eos_id=eos_id,
    )

    return ids_to_text(token_ids, idx2word)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = {}

    for model_type in ["ngram", "rnn", "lstm", "transformer"]:
        print(f"\n{'='*60}")
        print(f"  Generating with: {model_type.upper()}")
        print(f"{'='*60}")

        if model_type == "ngram":
            text = generate_ngram()
        else:
            text = generate_neural(model_type, device)

        results[model_type] = text
        print(f"\n>>> Generated text:\n{text}\n")

    # Print summary table for the report
    print("\n" + "=" * 60)
    print("  SUMMARY (for report)")
    print("=" * 60)
    for model_type, text in results.items():
        print(f"\n[{model_type.upper()}]")
        print(f"  {text}")


if __name__ == "__main__":
    main()

# ST5230 Assignment 1 — Language Model Comparison

Comparison of n-gram, RNN, LSTM, and Transformer language models on the IMDB dataset, with embedding ablation and downstream sentiment classification.

## Project Structure

```
st5230-assignment1/
├── src/
│   ├── config.py            # Dataclass-based experiment configuration
│   ├── data.py              # Data loading, tokenisation, vocabulary, dataloaders
│   ├── embedding.py         # Embedding layer construction (scratch / word2vec / pretrained)
│   ├── eval.py              # Evaluation: perplexity, accuracy, generation benchmarking
│   ├── utils.py             # Utilities: seeding, logging, checkpointing, progress bars
│   ├── train_lm.py          # Train neural language models (RNN / LSTM / Transformer)
│   ├── train_ngram.py       # Train n-gram language model
│   ├── train_cls.py         # Downstream sentiment classification with frozen LM encoder
│   ├── generate_text.py     # Load trained models and generate sample text
│   └── models/
│       ├── ngram.py          # Count-based n-gram LM with Laplace / Kneser-Ney smoothing
│       ├── rnn_lm.py         # Vanilla RNN language model
│       ├── lstm_lm.py        # LSTM language model
│       └── transformer_lm.py # Decoder-only Transformer language model
├── outputs/
│   ├── exp1/                 # Experiment 1: LM architecture comparison
│   ├── exp2/                 # Experiment 2: Embedding ablation
│   └── exp3/                 # Experiment 3: Downstream sentiment classification
├── src/glove.6B.100d.txt     # GloVe pretrained embeddings (not included, see Setup)
└── README.md
```

## Setup

```bash
pip install torch numpy datasets tqdm
```

Download [GloVe embeddings](https://nlp.stanford.edu/data/glove.6B.zip), extract `glove.6B.100d.txt`, and place it under `src/glove.6B.100d.txt`. This file is required for the Experiment 2 pretrained embedding ablation.

## Experiments

### Experiment 1 — Language Model Comparison

Train four LM architectures (n-gram, RNN, LSTM, Transformer) on IMDB and compare perplexity and inference speed.

```bash
# N-gram (trigram, Laplace smoothing)
python src/train_ngram.py --ngram_order 3 --smoothing laplace --seed 42 --output_dir outputs/exp1

# RNN
python src/train_lm.py --model_type rnn --embed_dim 128 --batch_size 512 --seq_len 64 \
    --max_samples 6000 --epochs 5 --seed 42 --output_dir outputs/exp1

# LSTM
python src/train_lm.py --model_type lstm --embed_dim 128 --batch_size 512 --seq_len 64 \
    --max_samples 6000 --epochs 5 --seed 42 --output_dir outputs/exp1

# Transformer
python src/train_lm.py --model_type transformer --embed_dim 128 --batch_size 512 --seq_len 64 \
    --max_samples 6000 --epochs 5 --seed 42 --output_dir outputs/exp1
```

### Experiment 2 — Embedding Ablation

Compare scratch, Word2Vec, and frozen GloVe embeddings on the Transformer LM, each with 3 seeds.
 
run individually:

```bash
# Scratch
python src/train_lm.py --model_type transformer --embedding_mode scratch \
    --embed_dim 100 --batch_size 512 --seq_len 64 --max_samples 6000 \
    --epochs 3 --seed 42 --output_dir outputs/exp2

# Word2Vec
python src/train_lm.py --model_type transformer --embedding_mode word2vec \
    --embed_dim 100 --batch_size 512 --seq_len 64 --max_samples 6000 \
    --epochs 3 --seed 42 --output_dir outputs/exp2

# GloVe (frozen)
python src/train_lm.py --model_type transformer --embedding_mode pretrained \
    --pretrained_path src/glove.6B.100d.txt --freeze --embed_dim 100 \
    --batch_size 512 --seq_len 64 --max_samples 6000 \
    --epochs 3 --seed 42 --output_dir outputs/exp2
```

### Experiment 3 — Downstream Sentiment Classification

Use frozen LM encoders from Exp1 as feature extractors for binary sentiment classification.

```bash
# RNN encoder
python src/train_cls.py --model_type rnn --embed_dim 128 \
    --lm_checkpoint outputs/exp1/rnn_lm_rnn_emb128_s42_*/best_model.pt \
    --batch_size 512 --epochs 10 --seed 42 --output_dir outputs/exp3

# LSTM encoder
python src/train_cls.py --model_type lstm --embed_dim 128 \
    --lm_checkpoint outputs/exp1/lstm_lm_lstm_emb128_s42_*/best_model.pt \
    --batch_size 512 --epochs 10 --seed 42 --output_dir outputs/exp3

# Transformer encoder
python src/train_cls.py --model_type transformer --embed_dim 128 \
    --lm_checkpoint outputs/exp1/transformer_lm_transformer_emb128_s42_*/best_model.pt \
    --batch_size 512 --epochs 10 --seed 42 --output_dir outputs/exp3
```

### Text Generation

Generate sample text from trained Exp1 models (no retraining needed):

```bash
python src/generate_text.py
```

## Output Format

Each experiment saves to its own directory under `outputs/`:

| File | Description |
|------|-------------|
| `config.json` | Full experiment configuration |
| `vocab.json` | Word-to-index vocabulary mapping |
| `best_model.pt` | Best model checkpoint (neural models only) |
| `training_log.json` | Per-epoch training metrics |
| `results.json` | Final evaluation summary |

## Key Results

| Model | Test PPL | Parameters | Inference (tok/s) |
|-------|----------|------------|-------------------|
| N-gram (trigram) | 8511.55 | -- | ~37.6M |
| RNN | 259.75 | 9.3M | 1198.7 |
| LSTM | 252.64 | 10.0M | 1159.7 |
| Transformer | 257.79 | 6.6M | 673.3 |

#!/bin/bash
set -e

echo "===== Step 1: Experiment 2 - Word2Vec first ====="

python src/train_lm.py \
--model_type transformer \
--embedding_mode word2vec \
--embed_dim 100 \
--batch_size 512 \
--seq_len 64 \
--max_samples 6000 \
--epochs 3 \
--seed 42 \
--output_dir outputs/exp2

python src/train_lm.py \
--model_type transformer \
--embedding_mode word2vec \
--embed_dim 100 \
--batch_size 512 \
--seq_len 64 \
--max_samples 6000 \
--epochs 3 \
--seed 123 \
--output_dir outputs/exp2

python src/train_lm.py \
--model_type transformer \
--embedding_mode word2vec \
--embed_dim 100 \
--batch_size 512 \
--seq_len 64 \
--max_samples 6000 \
--epochs 3 \
--seed 456 \
--output_dir outputs/exp2

echo "===== Step 2: Experiment 1 ====="

python src/train_ngram.py \
--ngram_order 3 \
--smoothing laplace \
--seed 42 \
--output_dir outputs/exp1

python src/train_lm.py \
--model_type rnn \
--embed_dim 128 \
--batch_size 512 \
--seq_len 64 \
--max_samples 6000 \
--epochs 5 \
--seed 42 \
--output_dir outputs/exp1

python src/train_lm.py \
--model_type lstm \
--embed_dim 128 \
--batch_size 512 \
--seq_len 64 \
--max_samples 6000 \
--epochs 5 \
--seed 42 \
--output_dir outputs/exp1

python src/train_lm.py \
--model_type transformer \
--embed_dim 128 \
--batch_size 512 \
--seq_len 64 \
--max_samples 6000 \
--epochs 5 \
--seed 42 \
--output_dir outputs/exp1

echo "===== ALL DONE ====="

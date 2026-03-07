#!/bin/bash

echo "===== Exp2: Scratch ====="

python src/train_lm.py --model_type transformer --embedding_mode scratch --embed_dim 100 --batch_size 512 --seq_len 64 --max_samples 6000 --epochs 3 --seed 42 --output_dir outputs/exp2

python src/train_lm.py --model_type transformer --embedding_mode scratch --embed_dim 100 --batch_size 512 --seq_len 64 --max_samples 6000 --epochs 3 --seed 123 --output_dir outputs/exp2

python src/train_lm.py --model_type transformer --embedding_mode scratch --embed_dim 100 --batch_size 512 --seq_len 64 --max_samples 6000 --epochs 3 --seed 456 --output_dir outputs/exp2


echo "===== Exp2: Word2Vec ====="

python src/train_lm.py --model_type transformer --embedding_mode word2vec --embed_dim 100 --batch_size 512 --seq_len 64 --max_samples 6000 --epochs 3 --seed 42 --output_dir outputs/exp2

python src/train_lm.py --model_type transformer --embedding_mode word2vec --embed_dim 100 --batch_size 512 --seq_len 64 --max_samples 6000 --epochs 3 --seed 123 --output_dir outputs/exp2

python src/train_lm.py --model_type transformer --embedding_mode word2vec --embed_dim 100 --batch_size 512 --seq_len 64 --max_samples 6000 --epochs 3 --seed 456 --output_dir outputs/exp2


echo "===== Exp2: Pretrained ====="

python src/train_lm.py --model_type transformer --embedding_mode pretrained --pretrained_path src/glove.6B.100d.txt --freeze --embed_dim 100 --batch_size 256 --seq_len 64 --max_samples 6000 --epochs 3 --seed 42 --output_dir outputs/exp2

python src/train_lm.py --model_type transformer --embedding_mode pretrained --pretrained_path src/glove.6B.100d.txt --freeze --embed_dim 100 --batch_size 256 --seq_len 64 --max_samples 6000 --epochs 3 --seed 123 --output_dir outputs/exp2

python src/train_lm.py --model_type transformer --embedding_mode pretrained --pretrained_path src/glove.6B.100d.txt --freeze --embed_dim 100 --batch_size 256 --seq_len 64 --max_samples 6000 --epochs 3 --seed 456 --output_dir outputs/exp2


echo "===== Exp2 DONE ====="

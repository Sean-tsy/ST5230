
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig, ModelConfig, EmbeddingConfig, DataConfig
from embedding import build_embedding_layer


# ==============================================================
# Learned Positional Embedding
# ==============================================================
class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embedding 
    Each position [0, max_len) gets a *trainable* embedding vector.  
    The position embeddings are simply added to the token embeddings.
    """

    def __init__(self, embed_dim: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, embed_dim)  token embeddings
        """

        B, T, _ = x.shape
        assert T <= self.max_len, (
            f"Sequence length {T} exceeds max_position_len {self.max_len}"
        )
        positions = torch.arange(T, device=x.device)  # (T,)
        x = x + self.pos_embedding(positions)          # broadcast over B
        return self.dropout(x)


# ==============================================================
# Transformer Language Model (decoder-only)
# ==============================================================
class TransformerLM(nn.Module):
    """
    Decoder-only Transformer LM, fully driven by config objects.

    Parameters
    ----------
    cfg             : ExperimentConfig (sub-configs used: embedding, model, data)
    word2idx        : vocabulary mapping  {token: index}
    tokenized_texts : training corpus tokens (needed only when
                      cfg.embedding.mode == "word2vec")
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        word2idx: dict[str, int],
        tokenized_texts=None,
    ) -> None:
        super().__init__()

        # ---- unpack configs ----
        emb_cfg: EmbeddingConfig = cfg.embedding
        mdl_cfg: ModelConfig     = cfg.model
        data_cfg: DataConfig     = cfg.data

        self.vocab_size   = len(word2idx)
        self.embed_dim    = emb_cfg.embed_dim
        self.num_heads    = mdl_cfg.num_heads
        self.num_layers   = mdl_cfg.num_transformer_layers
        self.ff_dim       = mdl_cfg.ff_dim
        self.dropout      = mdl_cfg.transformer_dropout
        self.max_pos_len  = mdl_cfg.max_position_len

        # Sanity check: embed_dim must be divisible by num_heads
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )

        # ---- layers ----

        # 1. Token embedding (built by embedding.py)
        self.embedding: nn.Embedding = build_embedding_layer(
            emb_cfg=emb_cfg,
            data_cfg=data_cfg,
            word2idx=word2idx,
            tokenized_texts=tokenized_texts,
            seed=cfg.seed,
        )

        # 2. Learned positional embedding (trainable)
        self.pos_encoder = LearnedPositionalEmbedding(
            embed_dim=self.embed_dim,
            max_len=self.max_pos_len,
            dropout=self.dropout,
        )

        # 4. Transformer encoder stack (used as decoder-only with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            batch_first=True,          # (B, T, D) convention
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.embed_dim),  # final layer norm
        )

        # 5. Projection to vocabulary
        self.fc = nn.Linear(self.embed_dim, self.vocab_size)

        # ---- weight initialisation ----
        self._init_weights()

    # ----------------------------------------------------------
    # Weight initialisation
    # ----------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier-uniform for linear layers, normal for embedding (if trainable)."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and "embedding" not in name:
                nn.init.xavier_uniform_(p)

    # ----------------------------------------------------------
    # Causal mask generation
    # ----------------------------------------------------------
    @staticmethod
    def _generate_causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """
        Create an upper-triangular causal mask for self-attention.

        Position i can attend to positions [0, ..., i] only.
        """

        # torch.triu with diagonal=1 gives True for future positions
        mask = torch.triu(
            torch.ones(T, T, device=device), diagonal=1
        ).bool()
        # Convert to float mask: 0.0 for attend, -inf for block
        return mask.float().masked_fill(mask, float("-inf"))

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x                    : (B, T)  token-id input
        src_key_padding_mask : (B, T)  bool tensor, True for PAD positions
                               (optional — LanguageModelDataset does not pad,
                               but useful if you feed variable-length batches)

        Returns
        -------
        logits : (B, T, vocab_size)  raw scores for next-token prediction

        Note: Unlike RNN/LSTM models this returns a *single tensor* (no hidden
        state), but eval.py handles both cases via ``isinstance(logits, tuple)``.
        """
        B, T = x.shape

        # 1. Token embedding + learned positional embedding
        emb = self.embedding(x)                         # (B, T, embed_dim)
        emb = self.pos_encoder(emb)                     # (B, T, embed_dim)

        # 2. Causal mask so position i cannot attend to j > i
        causal_mask = self._generate_causal_mask(T, x.device)  # (T, T)

        # 3. Transformer encoder stack (acts as decoder-only with causal mask)
        hidden = self.transformer(
            emb,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, T, embed_dim)

        # 4. Project to vocabulary
        logits = self.fc(hidden)  # (B, T, vocab_size)

        return logits


    # ----------------------------------------------------------
    # Autoregressive generation (for inference-time benchmarking)
    # ----------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_len: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate token ids autoregressively.

        Parameters
        ----------
        prompt      : (1, P) token-id tensor (batch size must be 1)
        max_len     : number of *new* tokens to generate
        temperature : sampling temperature (lower → more greedy)
        top_k       : if > 0, restrict sampling to top-k tokens
        eos_id      : stop early when this token is generated
        """

        self.eval()
        generated = prompt.clone()  # (1, P)

        for _ in range(max_len):
            # Truncate to max_position_len if needed
            inp = generated[:, -self.max_pos_len:]

            logits = self.forward(inp)  # (1, T, V)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (1, V)

            if top_k > 0:
                topk_vals, topk_idx = logits.topk(top_k, dim=-1)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(-1, topk_idx, topk_vals)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat([generated, next_id], dim=1)

            if eos_id is not None and next_id.item() == eos_id:
                break

        return generated


# ==============================================================
# Convenience: build from config in one call
# ==============================================================
def build_transformer_lm(
    cfg: ExperimentConfig,
    word2idx: dict[str, int],
    tokenized_texts=None,
) -> TransformerLM:
    """
    Instantiate a TransformerLM from an ExperimentConfig.
    """

    model = TransformerLM(cfg, word2idx, tokenized_texts=tokenized_texts)
    print(f"[transformer_lm] Built TransformerLM: "
          f"embed={model.embed_dim}, heads={model.num_heads}, "
          f"layers={model.num_layers}, ff={model.ff_dim}, "
          f"vocab={model.vocab_size}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model

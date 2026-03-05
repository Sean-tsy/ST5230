
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig, ModelConfig, EmbeddingConfig, DataConfig
from embedding import build_embedding_layer


# ==============================================================
# LSTM Language Model
# ==============================================================
class LSTMLM(nn.Module):
    """LSTM language model, fully driven by config objects.

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
        self.hidden_size  = mdl_cfg.hidden_size
        self.num_layers   = mdl_cfg.num_layers
        # nn.LSTM ignores the dropout arg when num_layers == 1
        self.rnn_dropout  = mdl_cfg.rnn_dropout if mdl_cfg.num_layers > 1 else 0.0

        # ---- layers ----

        # 1. Embedding (built by embedding.py, respects mode / freeze / pretrained)
        self.embedding: nn.Embedding = build_embedding_layer(
            emb_cfg=emb_cfg,
            data_cfg=data_cfg,
            word2idx=word2idx,
            tokenized_texts=tokenized_texts,
            seed=cfg.seed,
        )

        # 2. Dropout after embedding
        self.embed_drop = nn.Dropout(emb_cfg.embed_dropout)

        # 3. LSTM core  (the key difference from RNNLM)
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.rnn_dropout,
        )

        # 4. Dropout before projection
        self.out_drop = nn.Dropout(mdl_cfg.rnn_dropout)

        # 5. Projection to vocabulary
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x      : (B, T)  token-id input
        hidden : tuple (h_0, c_0), each of shape (num_layers, B, hidden_size),
                 or None (auto-initialised to zeros)

        """
        B, T = x.shape

        if hidden is None:
            hidden = self.init_hidden(B, x.device)

        # Embedding:  (B, T) → (B, T, embed_dim)
        emb = self.embedding(x)
        emb = self.embed_drop(emb)

        # LSTM:  (B, T, embed_dim) → (B, T, hidden_size), (h_n, c_n)
        lstm_out, hidden = self.lstm(emb, hidden)

        # Projection:  (B, T, hidden_size) → (B, T, vocab_size)
        lstm_out = self.out_drop(lstm_out)
        logits = self.fc(lstm_out)

        return logits, hidden

    # ----------------------------------------------------------
    # Hidden-state initialisation
    # ----------------------------------------------------------
    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return zero-initialised (h_0, c_0) for the LSTM.

        Each tensor has shape (num_layers, batch_size, hidden_size).
        """

        if device is None:
            device = next(self.parameters()).device

        h_0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
        c_0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
        return (h_0, c_0)


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
        """Generate token ids autoregressively.

        Parameters
        ----------
        prompt      : (1, P) token-id tensor (batch size must be 1)
        max_len     : number of *new* tokens to generate
        temperature : sampling temperature (lower → more greedy)
        top_k       : if > 0, restrict sampling to top-k tokens
        eos_id      : stop early when this token is generated

        Returns
        -------
        (1, P + generated) token-id tensor
        """
        self.eval()
        device = prompt.device
        generated = prompt.clone()  # (1, P)
        hidden = self.init_hidden(1, device)

        # Process prompt to warm up hidden state
        if prompt.size(1) > 1:
            _, hidden = self.forward(prompt[:, :-1], hidden)

        # Current input token
        inp = prompt[:, -1:]  # (1, 1)

        for _ in range(max_len):
            logits, hidden = self.forward(inp, hidden)  # (1,1,V)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (1, V)

            if top_k > 0:
                topk_vals, topk_idx = logits.topk(top_k, dim=-1)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(-1, topk_idx, topk_vals)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat([generated, next_id], dim=1)
            inp = next_id

            if eos_id is not None and next_id.item() == eos_id:
                break

        return generated


# ==============================================================
# Convenience: build from config in one call
# ==============================================================
def build_lstm_lm(
    cfg: ExperimentConfig,
    word2idx: dict[str, int],
    tokenized_texts=None,
) -> LSTMLM:
    """
    Instantiate an LSTMLM from an ExperimentConfig.
    """

    model = LSTMLM(cfg, word2idx, tokenized_texts=tokenized_texts)
    print(f"[lstm_lm] Built LSTMLM: "
          f"embed={model.embed_dim}, hidden={model.hidden_size}, "
          f"layers={model.num_layers}, vocab={model.vocab_size}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model

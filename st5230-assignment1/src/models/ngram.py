
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig, ModelConfig, DataConfig


# ==============================================================
# N-gram Language Model
# ==============================================================
class NgramLM:
    """
    Count-based n-gram language model with configurable smoothing.

    Parameters
    ----------
    n          : n-gram order (e.g. 2 = bigram, 3 = trigram)
    vocab_size : total vocabulary size (including special tokens)
    smoothing  : "none", "laplace", or "kneser_ney"
    alpha      : smoothing parameter (Laplace alpha or KN discount)
    pad_id     : token id for <pad>, excluded from training counts
    """

    def __init__(
        self,
        n: int = 3,
        vocab_size: int = 30_000,
        smoothing: str = "laplace",
        alpha: float = 1.0,
        pad_id: int = 0,
    ) -> None:
        assert n >= 1, f"n must be >= 1, got {n}"
        self.n = n
        self.vocab_size = vocab_size
        self.smoothing = smoothing.lower()
        self.alpha = alpha
        self.pad_id = pad_id

        # Count tables filled by fit()
        # ngram_counts[history_tuple][next_id] = count
        self.ngram_counts: Dict[tuple, Counter] = defaultdict(Counter)
        # context_totals[history_tuple] = sum of counts for that context
        self.context_totals: Dict[tuple, int] = defaultdict(int)

        # For Kneser-Ney: continuation counts
        # continuation_counts[next_id] = number of distinct histories
        #   that precede next_id (used for lower-order KN estimate)
        self.continuation_counts: Counter = Counter()
        self.total_bigram_types: int = 0  # total distinct (h, w) bigram types

        self._fitted = False

    # ----------------------------------------------------------
    # Training (counting)
    # ----------------------------------------------------------
    def fit(self, token_id_sequences: List[List[int]]) -> "NgramLM":
        """
        Build n-gram count tables from token-id sequences.

        Parameters
        ----------
        token_id_sequences : list of int lists (one per review / document).
                             These should already include <sos>/<eos> if desired.
        """

        print(f"[ngram] Fitting {self.n}-gram model "
              f"(smoothing={self.smoothing}, alpha={self.alpha}) ...")

        for seq in token_id_sequences:
            # Slide an n-sized window over the sequence
            for i in range(len(seq) - self.n + 1):
                ngram = seq[i : i + self.n]

                # Skip windows that contain pad tokens
                if self.pad_id in ngram:
                    continue

                history = tuple(ngram[:-1])   # (n-1) context tokens
                target  = ngram[-1]           # next token

                self.ngram_counts[history][target] += 1
                self.context_totals[history] += 1

        # Pre-compute Kneser-Ney continuation counts if needed
        if self.smoothing == "kneser_ney":
            self._build_kn_counts()

        total_ngrams = sum(self.context_totals.values())
        print(f"[ngram] Done — {len(self.ngram_counts):,} distinct contexts, "
              f"{total_ngrams:,} total n-grams counted.")

        self._fitted = True
        return self

    def _build_kn_counts(self) -> None:
        """Pre-compute continuation counts for Kneser-Ney smoothing.

        continuation_counts[w] = |{h : C(h, w) > 0}|
            i.e. how many distinct histories precede word w.
        """
        self.continuation_counts = Counter()
        bigram_types = set()

        for history, counter in self.ngram_counts.items():
            for target in counter:
                self.continuation_counts[target] += 1
                bigram_types.add((history, target))

        self.total_bigram_types = len(bigram_types)

    # ----------------------------------------------------------
    # Probability queries (required by eval.py)
    # ----------------------------------------------------------
    def get_log_prob(self, history: tuple, next_id: int) -> float:
        """
        Return ln P(next_id | history).

        Parameters
        ----------
        history : tuple of (n-1) token ids
        next_id : the token whose probability we want
        """

        if self.smoothing == "none":
            return self._logp_mle(history, next_id)
        elif self.smoothing == "laplace":
            return self._logp_laplace(history, next_id)
        elif self.smoothing == "kneser_ney":
            return self._logp_kneser_ney(history, next_id)
        else:
            raise ValueError(f"Unknown smoothing '{self.smoothing}'")

    # ---- MLE (no smoothing) ----
    def _logp_mle(self, history: tuple, next_id: int) -> float:
        """Maximum likelihood estimate: C(h,w) / C(h)."""
        total = self.context_totals.get(history, 0)
        if total == 0:
            # Unseen context → uniform fallback
            return math.log(1.0 / self.vocab_size)

        count = self.ngram_counts[history].get(next_id, 0)
        if count == 0:
            return math.log(1.0 / self.vocab_size)

        return math.log(count / total)

    # ---- Laplace (add-alpha) smoothing ----
    def _logp_laplace(self, history: tuple, next_id: int) -> float:
        """Add-alpha smoothing: (C(h,w) + α) / (C(h) + α·V)."""
        count = self.ngram_counts.get(history, {}).get(next_id, 0)
        total = self.context_totals.get(history, 0)

        numerator   = count + self.alpha
        denominator = total + self.alpha * self.vocab_size

        return math.log(numerator / denominator)

    # ---- Interpolated Kneser-Ney ----
    def _logp_kneser_ney(self, history: tuple, next_id: int) -> float:
        """Interpolated Kneser-Ney smoothing.

        P_KN(w | h) = max(C(h,w) - d, 0) / C(h)
                     + lambda(h) * P_continuation(w)

        where:
            d = self.alpha  (fixed discount, typically 0.75)
            lambda(h) = d * N_+(h, •) / C(h)
            N_+(h, •) = number of distinct words following h
            P_continuation(w) = |{h' : C(h', w) > 0}| / total_bigram_types
        """
        d = self.alpha  # discount (commonly 0.75)
        total = self.context_totals.get(history, 0)

        if total == 0:
            # Unseen context: fall back to continuation probability
            return self._logp_continuation(next_id)

        count = self.ngram_counts[history].get(next_id, 0)

        # First term: discounted probability
        first_term = max(count - d, 0.0) / total

        # Lambda: interpolation weight
        num_distinct = len(self.ngram_counts[history])  # N_+(h, •)
        lam = d * num_distinct / total

        # Continuation probability (lower-order estimate)
        p_cont = self._p_continuation(next_id)

        prob = first_term + lam * p_cont

        # Safety: prob should be > 0, but guard against floating issues
        if prob <= 0:
            return math.log(1.0 / self.vocab_size)

        return math.log(prob)

    def _p_continuation(self, word_id: int) -> float:
        """P_continuation(w) = |{h : C(h,w) > 0}| / total_bigram_types."""
        if self.total_bigram_types == 0:
            return 1.0 / self.vocab_size
        cont = self.continuation_counts.get(word_id, 0)
        if cont == 0:
            # Unseen word in any context — small floor
            return 1.0 / (self.total_bigram_types + self.vocab_size)
        return cont / self.total_bigram_types

    def _logp_continuation(self, word_id: int) -> float:
        """Log continuation probability (for unseen contexts)."""
        return math.log(max(self._p_continuation(word_id), 1e-30))

    # ----------------------------------------------------------
    # Text generation (for qualitative inspection)
    # ----------------------------------------------------------
    def generate(
        self,
        prompt: List[int],
        max_len: int = 50,
        temperature: float = 1.0,
        eos_id: Optional[int] = None,
    ) -> List[int]:
        """
        Generate token ids autoregressively from a prompt.
        Uses simple temperature-scaled sampling from the n-gram
        distribution.

        Parameters
        ----------
        prompt      : initial token ids (at least n-1 tokens)
        max_len     : maximum number of tokens to generate
        temperature : sampling temperature (lower → more greedy)
        eos_id      : stop when this token is generated (optional)
        """

        import random

        result = list(prompt)
        n = self.n

        for _ in range(max_len):
            # Take the last (n-1) tokens as history
            if len(result) < n - 1:
                break
            history = tuple(result[-(n - 1):])

            # Get distribution over vocab
            if history not in self.ngram_counts or not self.ngram_counts[history]:
                # Unseen context — stop or sample uniformly
                break

            counter = self.ngram_counts[history]
            words = list(counter.keys())
            counts = [counter[w] for w in words]

            # Apply temperature
            if temperature != 1.0:
                log_counts = [math.log(c) / temperature for c in counts]
                max_lc = max(log_counts)
                exp_counts = [math.exp(lc - max_lc) for lc in log_counts]
            else:
                exp_counts = counts

            total = sum(exp_counts)
            probs = [c / total for c in exp_counts]

            # Sample
            chosen = random.choices(words, weights=probs, k=1)[0]
            result.append(chosen)

            if eos_id is not None and chosen == eos_id:
                break

        return result

    # ----------------------------------------------------------
    # Top-k next tokens (for debugging / inspection)
    # ----------------------------------------------------------
    
    def topk_next(
        self, history: tuple, k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Return top-k (token_id, probability) pairs for a context.

        Uses the same smoothing as get_log_prob.
        """
        scored = []
        candidates = set(self.ngram_counts.get(history, {}).keys())

        # For Laplace, every vocab token has non-zero prob
        if self.smoothing == "laplace":
            candidates = set(range(self.vocab_size))

        for w in candidates:
            if w == self.pad_id:
                continue
            logp = self.get_log_prob(history, w)
            scored.append((w, math.exp(logp)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ==============================================================
# Convenience: build from config
# ==============================================================
def build_ngram_lm(
    cfg: ExperimentConfig,
    vocab_size: int,
) -> NgramLM:
    """
    Instantiate an NgramLM from an ExperimentConfig.
    """
    model = NgramLM(
        n=cfg.model.ngram_order,
        vocab_size=vocab_size,
        smoothing=cfg.model.smoothing,
        alpha=cfg.model.smoothing_alpha,
        pad_id=cfg.data.pad_id,
    )
    print(f"[ngram] Created {model.n}-gram LM: "
          f"vocab={vocab_size}, smoothing={model.smoothing}, "
          f"alpha={model.alpha}")
    return model

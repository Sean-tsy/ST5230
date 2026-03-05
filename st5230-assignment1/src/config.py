
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# 1. Data configuration
# ============================================================
@dataclass
class DataConfig:
    """Parameters that govern data loading and tokenisation."""

    dataset_name: str = "imdb"

    # Vocabulary
    max_vocab_size: int = 30_000
    min_freq: int = 2

    # Special-token indices (must match data.py SPECIAL_TOKENS order)
    pad_id: int = 0
    unk_id: int = 1
    sos_id: int = 2
    eos_id: int = 3

    # Language-model chunking
    lm_seq_len: int = 128

    # Sentiment truncation
    sentiment_max_len: int = 512

    # DataLoader
    batch_size: int = 64
    eval_batch_size: int = 0       # 0 = use 2× batch_size
    num_workers: int = 0

    # Subset sampling (0 = use all data)
    max_samples: int = 0


# ============================================================
# 2. Embedding configuration
# ============================================================
@dataclass
class EmbeddingConfig:
    """Controls the word-embedding layer shared by neural models."""

    # Embedding strategy: "scratch" | "word2vec" | "pretrained"
    mode: str = "scratch"

    embed_dim: int = 128

    # Pre-trained embeddings (used when mode="pretrained")
    pretrained_path: Optional[str] = None     # e.g. "glove.6B.100d.txt"
    freeze: bool = False                       # freeze embedding weights?

    # Dropout applied right after the embedding layer
    embed_dropout: float = 0.1

    # Word2Vec hyper-parameters (used when mode="word2vec")
    w2v_window: int = 5
    w2v_min_count: int = 1
    w2v_sg: int = 1               # 1 = Skip-gram, 0 = CBOW
    w2v_epochs: int = 10
    w2v_workers: int = 4
    w2v_save_path: Optional[str] = None  # save trained w2v model to disk


# ============================================================
# 3. Model configuration
# ============================================================
@dataclass
class ModelConfig:
    """Architecture hyper-parameters for every supported model type."""

    # ---- model selector ----
    model_type: str = "lstm"   # one of: "ngram", "rnn", "lstm", "transformer"

    # ---- n-gram specific ----
    ngram_order: int = 3       # n in n-gram (e.g. 2=bigram, 3=trigram)
    smoothing: str = "laplace" # smoothing method: "none", "laplace", "kneser_ney"
    smoothing_alpha: float = 1.0  # Laplace alpha or discount parameter

    # ---- RNN / LSTM shared ----
    hidden_size: int = 256
    num_layers: int = 2
    rnn_dropout: float = 0.3   # dropout between RNN layers (only if num_layers > 1)
    bidirectional: bool = False # only meaningful for sentiment classifier

    # ---- Transformer specific ----
    num_heads: int = 4
    num_transformer_layers: int = 2
    ff_dim: int = 512          # feed-forward inner dimension
    transformer_dropout: float = 0.1
    max_position_len: int = 1024  # max positional-encoding length

    # ---- classifier head (sentiment) ----
    num_classes: int = 2              # for sentiment classification (pos / neg)
    classifier_dropout: float = 0.3   # dropout before the linear classifier head


# ============================================================
# 4. Training configuration
# ============================================================
@dataclass
class TrainConfig:
    """Optimiser, scheduler, and training-loop parameters."""

    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"          # "adam", "adamw", "sgd"

    # Learning-rate scheduler
    scheduler: str = "none"         # "none", "step", "cosine"
    scheduler_step_size: int = 5    # for StepLR
    scheduler_gamma: float = 0.5    # for StepLR
    warmup_steps: int = 0           # linear warmup (transformer)

    # Gradient clipping
    grad_clip_max_norm: float = 5.0

    # Early stopping (0 = disabled)
    patience: int = 3

    # Logging frequency (print every N batches; 0 = epoch-level only)
    log_every_n_steps: int = 0


# ============================================================
# 5. Evaluation configuration
# ============================================================
@dataclass
class EvalConfig:
    """Parameters for the evaluation / summary phase."""

    # Convergence threshold (for summarize_history)
    convergence_ppl_threshold: Optional[float] = None   # e.g. 100.0
    convergence_acc_threshold: Optional[float] = None    # e.g. 0.80

    # Text generation (LM sampling after training)
    generate_max_len: int = 100
    temperature: float = 1.0
    top_k: int = 10


# ============================================================
# 6. Path configuration
# ============================================================
@dataclass
class PathConfig:
    """All filesystem paths used by the project."""

    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"

    # Pretrained LM checkpoint for classification transfer learning
    lm_checkpoint_path: str = ""    # path to best_model.pt from train_lm.py

    # These are filled automatically by ExperimentConfig.build_paths()
    experiment_dir: str = ""
    config_path: str = ""
    best_model_path: str = ""
    training_log_path: str = ""


# ============================================================
# 7. Top-level experiment configuration
# ============================================================
@dataclass
class ExperimentConfig:
    """
    Root configuration object that bundles every sub-config.

    Attributes
    ----------
    experiment_name : human-readable label 
    experiment_id   : auto-generated unique ID (timestamp + hash)
    seed            : global random seed for reproducibility
    """

    experiment_name: str = "default"
    experiment_id: str = ""          # auto-filled by generate_experiment_id()
    seed: int = 42
    task: str = "lm"                 # "lm" or "sentiment"

    data: DataConfig = field(default_factory=DataConfig) 
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # ----------------------------------------------------------
    # Experiment ID generation
    # ----------------------------------------------------------
    def generate_experiment_id(self) -> str:
        """
        Create a unique, deterministic experiment ID.

        Format: ``<experiment_name>_<model_type>_emb<dim>_s<seed>_<short_hash>``

        The short hash is derived from all serialisable parameters so that
        any config change produces a different ID.
        """
        cfg_bytes = json.dumps(asdict(self), sort_keys=True).encode()
        short_hash = hashlib.md5(cfg_bytes).hexdigest()[:8]
        ts = time.strftime("%m%d_%H%M")

        self.experiment_id = (
            f"{self.experiment_name}"
            f"_{self.model.model_type}"
            f"_emb{self.embedding.embed_dim}"
            f"_s{self.seed}"
            f"_{ts}"
            f"_{short_hash}"
        )
        return self.experiment_id

    # ----------------------------------------------------------
    # Derived paths
    # ----------------------------------------------------------
    def build_paths(self) -> None:
        """
        Populate PathConfig fields based on experiment_id.

        Call this after generate_experiment_id().
        """
        if not self.experiment_id:
            self.generate_experiment_id()

        base = Path(self.paths.output_dir) / self.experiment_id
        self.paths.experiment_dir = str(base)
        self.paths.config_path = str(base / "config.json")
        self.paths.best_model_path = str(base / "best_model.pt")
        self.paths.training_log_path = str(base / "training_log.json")

        # Ensure directories exist
        base.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------
    def validate(self) -> None:
        """
        Run basic sanity checks on the configuration.

        Raises ValueError if any constraint is violated.
        """
        errors: List[str] = []

        # ---- model_type must be known ----
        valid_models = {"ngram", "rnn", "lstm", "transformer"}
        if self.model.model_type not in valid_models:
            errors.append(
                f"model_type='{self.model.model_type}' not in {valid_models}"
            )

        # ---- task must be known ----
        if self.task not in ("lm", "sentiment"):
            errors.append(f"task='{self.task}' must be 'lm' or 'sentiment'")

        # ---- pad_id consistency ----
        if self.data.pad_id != 0:
            errors.append(
                f"pad_id={self.data.pad_id} but data.py hard-codes "
                f"padding_value=0 in collate_sentiment"
            )

        # ---- Sentiment task requires LM checkpoint ----
        if self.task == "sentiment" and not self.paths.lm_checkpoint_path:
            errors.append(
                "task='sentiment' requires paths.lm_checkpoint_path "
                "pointing to a pretrained LM checkpoint (best_model.pt)"
            )

        # ---- Embedding mode must be known ----
        valid_emb_modes = {"scratch", "word2vec", "pretrained"}
        if self.embedding.mode not in valid_emb_modes:
            errors.append(
                f"embedding.mode='{self.embedding.mode}' not in "
                f"{valid_emb_modes}"
            )

        # ---- pretrained mode requires a path ----
        if self.embedding.mode == "pretrained" and not self.embedding.pretrained_path:
            errors.append(
                "embedding.mode='pretrained' but pretrained_path is None"
            )

        # ---- Embedding dim vs pretrained ----
        # (If a pretrained path is given, the user must ensure dimensions
        #  match externally; we just warn if embed_dim looks wrong.)
        if self.embedding.pretrained_path and self.embedding.embed_dim not in (
            50, 100, 200, 300
        ):
            errors.append(
                f"embed_dim={self.embedding.embed_dim} with a pretrained file – "
                f"common GloVe dims are 50/100/200/300. Double-check."
            )

        # ---- Transformer dimension constraints ----
        if self.model.model_type == "transformer":
            # embed_dim must be divisible by num_heads
            if self.embedding.embed_dim % self.model.num_heads != 0:
                errors.append(
                    f"embed_dim ({self.embedding.embed_dim}) must be divisible "
                    f"by num_heads ({self.model.num_heads}) for Transformer"
                )
            # ff_dim should typically be >= embed_dim
            if self.model.ff_dim < self.embedding.embed_dim:
                errors.append(
                    f"ff_dim ({self.model.ff_dim}) < embed_dim "
                    f"({self.embedding.embed_dim}) – transformer FFN is "
                    f"usually wider than the model dimension"
                )
            # max_position_len should cover lm_seq_len / sentiment_max_len
            max_len = max(self.data.lm_seq_len, self.data.sentiment_max_len)
            if self.model.max_position_len < max_len:
                errors.append(
                    f"max_position_len ({self.model.max_position_len}) < "
                    f"max sequence length ({max_len})"
                )

        # ---- RNN / LSTM: dropout only makes sense with >1 layer ----
        if self.model.model_type in ("rnn", "lstm"):
            if self.model.num_layers == 1 and self.model.rnn_dropout > 0:
                errors.append(
                    f"rnn_dropout={self.model.rnn_dropout} has no effect "
                    f"when num_layers=1 (PyTorch ignores it)"
                )

        # ---- N-gram: no embedding / training params needed ----
        if self.model.model_type == "ngram":
            if self.model.ngram_order < 1:
                errors.append(
                    f"ngram_order={self.model.ngram_order} must be >= 1"
                )

        # ---- General ----
        if self.train.epochs < 1 and self.model.model_type != "ngram":
            errors.append(f"epochs={self.train.epochs} must be >= 1")

        if self.train.learning_rate <= 0 and self.model.model_type != "ngram":
            errors.append(
                f"learning_rate={self.train.learning_rate} must be > 0"
            )

        if self.data.max_vocab_size < len(["<pad>", "<unk>", "<sos>", "<eos>"]):
            errors.append(
                f"max_vocab_size={self.data.max_vocab_size} is too small "
                f"to hold special tokens"
            )

        if errors:
            msg = "Config validation failed:\n  • " + "\n  • ".join(errors)
            raise ValueError(msg)

        print("[config] Validation passed ✓")

    # ----------------------------------------------------------
    # Serialisation helpers
    # ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return the full config as a plain dict (JSON-safe)."""
        return asdict(self)

    def save(self, path: Optional[str] = None) -> str:
        """Save config to JSON.  Uses paths.config_path by default."""
        if path is None:
            if not self.paths.config_path:
                self.build_paths()
            path = self.paths.config_path

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[config] Saved → {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from a JSON file."""
        with open(path) as f:
            d = json.load(f)

        cfg = cls(
            experiment_name=d.get("experiment_name", "default"),
            experiment_id=d.get("experiment_id", ""),
            seed=d.get("seed", 42),
            task=d.get("task", "lm"),
            data=DataConfig(**d.get("data", {})),
            embedding=EmbeddingConfig(**d.get("embedding", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            eval=EvalConfig(**d.get("eval", {})),
            paths=PathConfig(**d.get("paths", {})),
        )
        print(f"[config] Loaded ← {path}")
        return cfg

    def summary(self) -> str:
        """One-line summary string for logging / filenames."""
        return (
            f"{self.experiment_name} | {self.model.model_type} | "
            f"emb={self.embedding.embed_dim} "
            f"{'(frozen)' if self.embedding.freeze else '(trainable)'} | "
            f"seed={self.seed} | task={self.task}"
        )


# ============================================================
# Convenience: quick preset builders
# ============================================================
def make_config(
    model_type: str = "lstm",
    task: str = "lm",
    embed_dim: int = 128,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 64,
    **overrides: Any,
) -> ExperimentConfig:
    """
    Create, validate, and return an ExperimentConfig with sensible defaults.
    Any additional keyword arguments are applied as overrides to the
    matching sub-config. 
    """

    cfg = ExperimentConfig(
        experiment_name=f"{model_type}_{task}",
        seed=seed,
        task=task,
        data=DataConfig(batch_size=batch_size),
        embedding=EmbeddingConfig(embed_dim=embed_dim),
        model=ModelConfig(model_type=model_type),
        train=TrainConfig(epochs=epochs),
    )

    # Apply overrides to the appropriate sub-config
    for key, val in overrides.items():
        applied = False
        for sub in (cfg.data, cfg.embedding, cfg.model, cfg.train, cfg.eval, cfg.paths):
            if hasattr(sub, key):
                setattr(sub, key, val)
                applied = True
                break
        if not applied:
            if hasattr(cfg, key):
                setattr(cfg, key, val)
            else:
                raise ValueError(f"Unknown config key: '{key}'")

    cfg.generate_experiment_id()
    cfg.build_paths()
    cfg.validate()

    return cfg




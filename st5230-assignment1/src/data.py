
import re
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# ============================================================
# Define Special tokens before we build the vocab
# ============================================================
PAD_TOKEN = "<pad>" # padding token (used by SentimentDataset)
UNK_TOKEN = "<unk>" # unknown token (used by both LM and SentimentDataset, for Out-of-Vocabulary words in test set)
SOS_TOKEN = "<sos>"   # start-of-sequence (used by LM)
EOS_TOKEN = "<eos>"   # end-of-sequence   (used by LM)

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


# ============================================================
# Part 1 – Raw data download & loading
# ============================================================
def load_imdb() -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    The IMDB dataset contains 50 000 movie reviews, split evenly into 25 000 for training and 25 000 for testing. 
    train_texts : list[str]   25 000 training reviews 
    train_labels: list[int]   0 = negative, 1 = positive
    test_texts  : list[str]   25 000 test reviews
    test_labels : list[int]
    """
    dataset = load_dataset("imdb")

    train_texts  = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts   = dataset["test"]["text"]
    test_labels  = dataset["test"]["label"]

    return train_texts, train_labels, test_texts, test_labels


# ============================================================
# Part 2 – Text processing: tokenize / build_vocab / texts_to_ids
# ============================================================
def tokenize(text: str) -> List[str]:
    
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)        # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)        # keep only letters + spaces
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text.split()


def build_vocab(
    tokenized_texts: List[List[str]],
    max_vocab_size: int = 30_000,
    min_freq: int = 2,
) -> Dict[str, int]:
    """
    Build a word-to-index dictionary from pre-tokenized training texts.

    The vocabulary includes SPECIAL_TOKENS at the front (indices 0-3),
    then the most frequent words that appear >= min_freq times,
    up to max_vocab_size total entries.

    Parameters
    ----------
    tokenized_texts : list of token lists  (training set only)
    max_vocab_size  : cap on vocabulary size including special tokens
    min_freq        : minimum corpus frequency to keep a word
    """

    # Count token frequencies across all texts
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    # Start with special tokens
    word2idx: Dict[str, int] = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        word2idx[tok] = i

    # Add most-common words (respecting min_freq and max_vocab_size)
    remaining_slots = max_vocab_size - len(SPECIAL_TOKENS) 
    for word, freq in counter.most_common():
        if len(word2idx) >= max_vocab_size:
            break
        if freq < min_freq:
            break
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    return word2idx


def texts_to_ids(
    tokenized_texts: List[List[str]],
    word2idx: Dict[str, int],
) -> List[List[int]]:
    """
    Convert pre-tokenized texts into lists of token indices.
    Words not in the vocabulary are mapped to UNK_TOKEN.
    """

    unk_id = word2idx[UNK_TOKEN]
    return [
        [word2idx.get(tok, unk_id) for tok in tokens]
        for tokens in tokenized_texts
    ]


# ============================================================
# Part 3 – Language-Model dataloader
# ============================================================
class LanguageModelDataset(Dataset):
    """
    All review texts are concatenated into a single long token stream
    (separated by <sos>...<eos> boundaries).  The stream is then split
    into fixed-length chunks of `seq_len` tokens.
    """

    def __init__(self, id_sequences: List[List[int]], word2idx: Dict[str, int],
                 seq_len: int = 128):
        sos_id = word2idx[SOS_TOKEN]
        eos_id = word2idx[EOS_TOKEN]

        # Build one long token stream: <sos> review_1 <eos> <sos> review_2 <eos> ...
        stream: List[int] = [] 
        for ids in id_sequences:
            stream.append(sos_id)
            stream.extend(ids)
            stream.append(eos_id)

        self.data = torch.tensor(stream, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Number of valid (x, y) pairs we can extract
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx     : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def get_lm_dataloaders(
    train_ids: List[List[int]],
    test_ids: List[List[int]],
    word2idx: Dict[str, int],
    seq_len: int = 128,
    batch_size: int = 64,
    eval_batch_size: int = 0,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for the language-modelling task.

    Parameters
    ----------
    train_ids        : tokenised & id-converted training texts
    test_ids         : tokenised & id-converted test texts
    word2idx         : shared vocabulary
    seq_len          : chunk length for each (x, y) sample
    batch_size       : mini-batch size for training
    eval_batch_size  : mini-batch size for evaluation (0 = 2× batch_size)
    num_workers      : parallel data-loading workers
    """
    if eval_batch_size <= 0:
        eval_batch_size = batch_size * 2

    train_dataset = LanguageModelDataset(train_ids, word2idx, seq_len=seq_len)
    test_dataset  = LanguageModelDataset(test_ids,  word2idx, seq_len=seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,   # drop incomplete last batch for stable training
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, test_loader


# ============================================================
# Part 4 – Sentiment-Classification dataloader
# ============================================================
class SentimentDataset(Dataset):
    """
    Dataset for binary sentiment classification task.

    Each sample is a variable-length sequence of token ids paired with
    a label (0 = negative, 1 = positive).  Padding is handled at
    collation time (see collate_sentiment).
    """

    def __init__(self, id_sequences: List[List[int]], labels: List[int],
                 max_len: int = 512):
        """
        Parameters
        ----------
        id_sequences : list of int lists (token ids for each review)
        labels       : list of int (0 or 1)
        max_len      : truncate reviews longer than this many tokens
        """
        self.ids = id_sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        token_ids = self.ids[idx][: self.max_len]   # truncate if needed
        return torch.tensor(token_ids, dtype=torch.long), self.labels[idx]


def collate_sentiment(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function that pads variable-length sequences
    to the longest sequence in the current mini-batch.
    """

    sequences, labels = zip(*batch) # unzip the batch into separate lists of sequences and labels

    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long) 
    labels  = torch.tensor(labels, dtype=torch.long) 

    # Pad all sequences to the max length in this batch
    padded_ids = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0  # 0 = PAD_TOKEN index
    )

    return padded_ids, lengths, labels


def get_sentiment_dataloaders(
    train_ids: List[List[int]],
    train_labels: List[int],
    test_ids: List[List[int]],
    test_labels: List[int],
    max_len: int = 512,
    batch_size: int = 64,
    eval_batch_size: int = 0,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for sentiment classification.

    Parameters
    ----------
    train_ids       : tokenised & id-converted training texts
    train_labels    : training sentiment labels (0/1)
    test_ids        : tokenised & id-converted test texts
    test_labels     : test sentiment labels (0/1)
    max_len         : max tokens per review (longer reviews are truncated)
    batch_size      : mini-batch size for training
    eval_batch_size : mini-batch size for evaluation (0 = 2× batch_size)
    num_workers     : parallel data-loading workers
    """
    if eval_batch_size <= 0:
        eval_batch_size = batch_size * 2

    train_dataset = SentimentDataset(train_ids, train_labels, max_len=max_len)
    test_dataset  = SentimentDataset(test_ids,  test_labels,  max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collate_sentiment,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_sentiment,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, test_loader


# ============================================================
# Convenience: end-to-end pipeline
# ============================================================

def prepare_data(
    max_vocab_size: int = 30_000,
    min_freq: int = 2,
    lm_seq_len: int = 128,
    sentiment_max_len: int = 512,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict:
    """
    Run the full pipeline and return everything needed for training.
    """

    # ---- Part 1: download raw data ----
    train_texts, train_labels, test_texts, test_labels = load_imdb()

    # ---- Part 2: tokenize & build shared vocab ----
    train_tokenized = [tokenize(t) for t in train_texts]
    test_tokenized  = [tokenize(t) for t in test_texts]

    word2idx = build_vocab(train_tokenized, max_vocab_size, min_freq)
    idx2word = {i: w for w, i in word2idx.items()}

    train_ids = texts_to_ids(train_tokenized, word2idx)
    test_ids  = texts_to_ids(test_tokenized,  word2idx)

    # ---- Part 3: LM dataloaders (shared vocab) ----
    lm_train_loader, lm_test_loader = get_lm_dataloaders(
        train_ids, test_ids, word2idx,
        seq_len=lm_seq_len, batch_size=batch_size, num_workers=num_workers,
    )

    # ---- Part 4: Sentiment dataloaders (shared vocab) ----
    sentiment_train_loader, sentiment_test_loader = get_sentiment_dataloaders(
        train_ids, train_labels, test_ids, test_labels,
        max_len=sentiment_max_len, batch_size=batch_size, num_workers=num_workers,
    )

    return {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "vocab_size": len(word2idx),
        "lm_train_loader": lm_train_loader,
        "lm_test_loader": lm_test_loader,
        "sentiment_train_loader": sentiment_train_loader,
        "sentiment_test_loader": sentiment_test_loader,
    }


# ============================================================
# Quick sanity check (run: python data.py)
# ============================================================
if __name__ == "__main__":
    data = prepare_data(batch_size=4)
    print(f"Vocab size : {data['vocab_size']}")

    # -- LM sample --
    x, y = next(iter(data["lm_train_loader"]))
    print(f"\n[LM] x.shape={x.shape}, y.shape={y.shape}")
    print(f"[LM] first 10 input tokens : {x[0, :10].tolist()}")
    print(f"[LM] first 10 target tokens: {y[0, :10].tolist()}")

    # -- Sentiment sample --
    ids, lengths, labels = next(iter(data["sentiment_train_loader"]))
    print(f"\n[Sentiment] ids.shape={ids.shape}, labels={labels.tolist()}")
    print(f"[Sentiment] lengths={lengths.tolist()}")

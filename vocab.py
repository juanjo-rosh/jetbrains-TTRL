"""
vocab.py — Vocabulary Building, Preprocessing & Negative Sampling
=================================================================

This module handles all data-pipeline concerns for Skip-gram with Negative
Sampling (SGNS):

1.  Corpus loading and tokenisation
2.  Vocabulary construction with minimum-count filtering
3.  Mikolov-style subsampling of frequent words
4.  Unigram^(3/4) negative-sampling distribution
5.  Training-pair generation with dynamic context windows

References
----------
* Mikolov et al. (2013) — "Distributed Representations of Words and Phrases and their Compositionality"
* Rong (2014) — "word2vec Parameter Learning Explained"
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Corpus Loading
# ---------------------------------------------------------------------------

# A small fallback corpus used when no external file is provided.
# It is large enough to demonstrate the algorithm but small enough to train
# in seconds on a laptop.
_FALLBACK_CORPUS = (
    "the king is a man the queen is a woman "
    "the prince is a young man the princess is a young woman "
    "a man is strong a woman is beautiful "
    "the king and the queen rule the kingdom "
    "the prince and the princess are the children of the king and queen "
    "a boy is a young man a girl is a young woman "
    "the king lives in a castle the queen lives in a castle "
    "the prince will become a king the princess will become a queen "
    "a man and a woman can have a boy or a girl "
    "the kingdom is a large country ruled by a king or a queen "
    "the dog chased the cat the cat climbed the tree "
    "a dog is an animal a cat is an animal "
    "the bird flew over the tree the fish swam in the river "
    "the city is large the village is small "
    "a car drives on the road a boat sails on the river "
    "the teacher teaches the student the student learns from the teacher "
    "a doctor heals the patient a lawyer defends the client "
    "the sun rises in the east the moon shines at night "
    "a computer processes data a phone connects people "
    "the book has many pages the page has many words "
) * 200  # Repeat to give enough co-occurrence signal


def download_text8(dest_dir: str = ".") -> str:
    """Download and extract the *text8* dataset (~31 MB compressed).

    The text8 corpus is a cleaned excerpt of a 2006 Wikipedia dump that has
    been widely used to benchmark word-embedding algorithms.

    Parameters
    ----------
    dest_dir : str
        Directory where the extracted ``text8`` file will be stored.

    Returns
    -------
    str
        Absolute path to the extracted plain-text file.
    """
    import requests  # deferred import — only needed for download

    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = os.path.join(dest_dir, "text8.zip")
    txt_path = os.path.join(dest_dir, "text8")

    if os.path.isfile(txt_path):
        print(f"[vocab] text8 already exists at {txt_path}")
        return txt_path

    print(f"[vocab] Downloading text8 from {url} …")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    print("[vocab] Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)
    print(f"[vocab] Saved to {txt_path}")
    return txt_path


def load_corpus(
    path: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> List[str]:
    """Load and tokenise a plain-text corpus.

    Tokenisation is deliberately simple: lower-case, then extract sequences
    of alphabetic characters.  This mirrors the standard text8 preparation.

    Parameters
    ----------
    path : str or None
        Path to a plain-text file.  If *None*, the built-in fallback corpus
        is used (useful for quick smoke-tests).
    max_tokens : int or None
        If given, truncate the token list to this many words.  Useful for
        running on a small slice of text8 during development.

    Returns
    -------
    list[str]
        A flat list of lowercase word tokens.
    """
    if path is None:
        text = _FALLBACK_CORPUS
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    # Simple whitespace + alpha tokenisation (matches text8 conventions)
    tokens: List[str] = re.findall(r"[a-z]+", text.lower())

    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    print(f"[vocab] Loaded {len(tokens):,} tokens")
    return tokens


# ---------------------------------------------------------------------------
# 2. Vocabulary Construction
# ---------------------------------------------------------------------------

class Vocabulary:
    """Holds word ↔ index mappings, counts, and a negative-sampling table.

    Attributes
    ----------
    word2idx : dict[str, int]
        Mapping from word string to integer index.
    idx2word : dict[int, str]
        Reverse mapping from index to word string.
    counts : np.ndarray
        counts[i] = raw frequency of word with index i.
    size : int
        Number of words in the vocabulary.
    """

    def __init__(
        self,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
        counts: np.ndarray,
    ) -> None:
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.counts = counts
        self.size = len(word2idx)

    # Convenience -----------------------------------------------------------
    def __len__(self) -> int:
        return self.size

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx


def build_vocab(tokens: List[str], min_count: int = 5) -> Vocabulary:
    """Build a vocabulary from a list of tokens.

    Words appearing fewer than *min_count* times are discarded.  The
    remaining words are sorted by descending frequency so that the most
    common word has index 0.

    Parameters
    ----------
    tokens : list[str]
        The full (pre-subsampled) token list coming from ``load_corpus``.
    min_count : int
        Minimum occurrence count to keep a word.

    Returns
    -------
    Vocabulary
        The constructed vocabulary object.
    """
    freq: Counter = Counter(tokens)

    # Filter by minimum count and sort by frequency (descending)
    filtered = sorted(
        ((w, c) for w, c in freq.items() if c >= min_count),
        key=lambda wc: -wc[1],
    )

    word2idx: Dict[str, int] = {}
    idx2word: Dict[int, str] = {}
    counts_list: List[int] = []

    for idx, (word, count) in enumerate(filtered):
        word2idx[word] = idx
        idx2word[idx] = word
        counts_list.append(count)

    counts = np.array(counts_list, dtype=np.float64)

    vocab = Vocabulary(word2idx, idx2word, counts)
    print(f"[vocab] Vocabulary size: {vocab.size:,}  (min_count={min_count})")
    return vocab


# ---------------------------------------------------------------------------
# 3. Frequent-Word Subsampling
# ---------------------------------------------------------------------------

def subsample_corpus(
    corpus_indices: np.ndarray,
    counts: np.ndarray,
    threshold: float = 1e-5,
) -> np.ndarray:
    """Discard frequent words with Mikolov's subsampling formula.

    Each token w_i in the corpus is *kept* with probability:

        P(keep w_i) = min(1, sqrt(t / f(w_i)))

    where f(w_i) = count(w_i) / total_count  is the word's relative
    frequency, and t is the threshold (typically 1e-5).

    This aggressively removes stop-words like "the", "a", "is", freeing
    up co-occurrence signal for more informative words.

    Parameters
    ----------
    corpus_indices : np.ndarray, shape (N,)
        Array of word indices representing the corpus.
    counts : np.ndarray, shape (V,)
        counts[i] = raw frequency of vocabulary word i.
    threshold : float
        Subsampling threshold *t*.

    Returns
    -------
    np.ndarray
        Filtered corpus indices (shorter than or equal to the input).
    """
    total = counts.sum()
    freqs = counts / total  # f(w_i) for every vocab word

    # Per-word keep-probability (vectorised over the vocabulary)
    keep_prob = np.minimum(1.0, np.sqrt(threshold / freqs))

    # Map each token in the corpus to its keep probability
    token_keep_prob = keep_prob[corpus_indices]  # shape (N,)

    # Draw uniform random numbers and keep tokens whose random < keep_prob
    rand = np.random.rand(len(corpus_indices))
    mask = rand < token_keep_prob

    filtered = corpus_indices[mask]
    n_removed = len(corpus_indices) - len(filtered)
    print(
        f"[vocab] Subsampling: kept {len(filtered):,} / {len(corpus_indices):,} "
        f"tokens  (removed {n_removed:,})"
    )
    return filtered


# ---------------------------------------------------------------------------
# 4. Negative Sampling Distribution
# ---------------------------------------------------------------------------

class NegativeSampler:
    """Draws negative samples from the smoothed unigram distribution.

    The sampling probability for word w_i is:

        P_n(w_i)  =  count(w_i)^{3/4}  /  Σ_j count(w_j)^{3/4}

    The 3/4 exponent (empirically chosen by Mikolov et al.) *flattens* the
    unigram distribution, boosting the probability of rare words relative
    to a pure unigram baseline.  This makes negative samples more
    informative — extremely common words (stop words) would otherwise
    dominate the negatives, providing little contrastive signal.
    """

    def __init__(self, counts: np.ndarray, power: float = 0.75) -> None:
        """
        Parameters
        ----------
        counts : np.ndarray, shape (V,)
            Raw frequency counts for each word in the vocabulary.
        power : float
            Smoothing exponent (default 3/4 = 0.75).
        """
        # Smooth and normalise  →  P_n(w_i)
        weighted = np.power(counts, power)               # count(w_i)^{3/4}
        self._probs = weighted / weighted.sum()           # normalise to a distribution
        self._vocab_size = len(counts)

    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Draw negative-sample indices.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output array, e.g. (batch_size, K).

        Returns
        -------
        np.ndarray
            Array of word indices drawn from P_n, with the given shape.
        """
        n_total = int(np.prod(shape))
        indices = np.random.choice(
            self._vocab_size,
            size=n_total,
            replace=True,
            p=self._probs,
        )
        return indices.reshape(shape)


# ---------------------------------------------------------------------------
# 5. Training-Pair Generation
# ---------------------------------------------------------------------------

def corpus_to_indices(tokens: List[str], vocab: Vocabulary) -> np.ndarray:
    """Convert a token list to a NumPy array of vocabulary indices.

    Tokens not present in the vocabulary (filtered by min_count) are
    silently dropped.

    Parameters
    ----------
    tokens : list[str]
        Raw token list from ``load_corpus``.
    vocab : Vocabulary
        The vocabulary built by ``build_vocab``.

    Returns
    -------
    np.ndarray, shape (M,)
        Corpus as an array of integer word indices.
    """
    indices = [vocab.word2idx[t] for t in tokens if t in vocab.word2idx]
    return np.array(indices, dtype=np.int64)


def generate_training_pairs(
    corpus: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Generate (center, context) training pairs with a *dynamic* window.

    For every position *i* in the corpus, a window size *w* is sampled
    uniformly from [1, window_size].  For every position *j* in
    [i − w, i + w] (j ≠ i and within bounds), the pair (corpus[i],
    corpus[j]) is emitted.

    Using a *dynamic* (randomly shortened) window effectively gives closer
    context words a higher sampling weight, which matches the intuition
    that nearer words provide stronger co-occurrence signal.

    Parameters
    ----------
    corpus : np.ndarray, shape (N,)
        Integer-encoded corpus (after subsampling).
    window_size : int
        Maximum one-sided context window size *W*.

    Returns
    -------
    np.ndarray, shape (P, 2)
        Each row is ``[center_word_idx, context_word_idx]``.
    """
    n = len(corpus)
    pairs: List[Tuple[int, int]] = []

    # Pre-sample dynamic window sizes for every position (vectorised)
    dyn_windows = np.random.randint(1, window_size + 1, size=n)

    for i in range(n):
        w = dyn_windows[i]
        center = corpus[i]
        start = max(0, i - w)
        end = min(n, i + w + 1)
        for j in range(start, end):
            if j != i:
                pairs.append((center, corpus[j]))

    pairs_array = np.array(pairs, dtype=np.int64)
    print(f"[vocab] Generated {len(pairs_array):,} training pairs  (window={window_size})")
    return pairs_array


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke-test with the fallback corpus
    tokens = load_corpus(path=None, max_tokens=50_000)
    vocab = build_vocab(tokens, min_count=3)
    corpus_idx = corpus_to_indices(tokens, vocab)
    corpus_idx = subsample_corpus(corpus_idx, vocab.counts, threshold=1e-5)
    sampler = NegativeSampler(vocab.counts)
    pairs = generate_training_pairs(corpus_idx, window_size=5)
    neg_samples = sampler.sample((len(pairs), 5))
    print(f"[vocab] Pairs shape: {pairs.shape}  Negatives shape: {neg_samples.shape}")
    print("[vocab] ✓ Self-test passed")

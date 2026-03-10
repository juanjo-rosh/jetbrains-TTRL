"""
vocab.py — Corpus loading, vocabulary construction, and sampling utilities.

Responsibilities:
    1. Fetch the text8 corpus via gensim.downloader (download only; no Gensim
       training models are used anywhere in this project).
    2. Build a word-to-index mapping, filtering by minimum frequency.
    3. Compute word-subsampling probabilities (Mikolov et al., 2013 §2.3).
    4. Build the negative-sampling lookup table using the 3/4-power noise
       distribution also described in the original word2vec paper.
    5. Generate all (center, context) positive training pairs with a sliding
       window and dynamic window-size sampling.
    6. Sample negative indices efficiently from the lookup table.
"""

from __future__ import annotations

import collections
import random
from typing import Dict, List, Optional, Tuple

import gensim.downloader as api
import numpy as np


# ---------------------------------------------------------------------------
# 1.  Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(dataset_name: str = "text8") -> List[List[str]]:
    """
    Download (if necessary) and return a corpus via gensim's model zoo.

    The gensim downloader is *only* used to obtain raw text; we do not call
    any gensim Word2Vec training routines.

    Parameters
    ----------
    dataset_name:
        Any name accepted by ``gensim.downloader.load``.
        Common choices: ``"text8"``, ``"text9"``, ``"20-newsgroups"``.

    Returns
    -------
    sentences:
        A list of sentences, where each sentence is itself a list of
        lower-cased string tokens.
    """
    print(f"[vocab] Loading corpus '{dataset_name}' via gensim.downloader …")
    corpus = api.load(dataset_name)
    sentences: List[List[str]] = list(corpus)
    n_tokens = sum(len(s) for s in sentences)
    print(f"[vocab] Loaded {len(sentences):,} sentences / {n_tokens:,} tokens.")
    return sentences


# ---------------------------------------------------------------------------
# 2.  Vocabulary construction
# ---------------------------------------------------------------------------

def build_vocabulary(
    sentences: List[List[str]],
    min_count: int = 5,
    max_vocab_size: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int]]:
    """
    Build vocabulary from a tokenised corpus.

    Words that appear fewer than ``min_count`` times are discarded.
    The remaining words are sorted by descending frequency so that
    the most common words receive the lowest indices (handy for
    debugging and for building a compact negative-sampling table).

    Parameters
    ----------
    sentences:
        Tokenised corpus as returned by ``load_corpus``.
    min_count:
        Minimum number of occurrences required to keep a word.
    max_vocab_size:
        If provided, only the top-N most frequent words are kept
        after the ``min_count`` filter.

    Returns
    -------
    word2idx:
        ``{word: integer_index}`` mapping.
    idx2word:
        ``{integer_index: word}`` reverse mapping.
    word_freq:
        ``{word: raw_count}`` dictionary (filtered vocabulary only).
    """
    raw_counts: collections.Counter = collections.Counter()
    for sentence in sentences:
        raw_counts.update(sentence)

    # --- min_count filter ------------------------------------------------
    filtered = {w: c for w, c in raw_counts.items() if c >= min_count}

    # --- sort descending by frequency -----------------------------------
    sorted_vocab: List[Tuple[str, int]] = sorted(
        filtered.items(), key=lambda wc: wc[1], reverse=True
    )

    # --- optional hard vocabulary cap -----------------------------------
    if max_vocab_size is not None:
        sorted_vocab = sorted_vocab[:max_vocab_size]

    # --- build index mappings -------------------------------------------
    word2idx: Dict[str, int] = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
    idx2word: Dict[int, str] = {idx: word for word, idx in word2idx.items()}
    word_freq: Dict[str, int] = {word: cnt for word, cnt in sorted_vocab}

    print(f"[vocab] Vocabulary size: {len(word2idx):,}  (min_count={min_count})")
    return word2idx, idx2word, word_freq


# ---------------------------------------------------------------------------
# 3.  Subsampling probabilities
# ---------------------------------------------------------------------------

def compute_subsampling_probs(
    word_freq: Dict[str, int],
    word2idx: Dict[str, int],
    t: float = 1e-4,
) -> np.ndarray:
    """
    Compute the probability of *keeping* each word during corpus iteration.

    Mikolov et al. (2013) propose discarding the word ``w`` with probability:

        P(discard | w) = 1 − √(t / f(w))

    Equivalently, the probability of *keeping* ``w`` is:

        P(keep | w) = min(1, √(t / f(w))  +  t / f(w))

    The additive ``t / f(w)`` term is present in the original C code and
    provides a small non-zero keep probability even for very frequent words.

    Parameters
    ----------
    word_freq:
        Raw frequency counts for every word in the vocabulary.
    word2idx:
        Word-to-index mapping (same vocabulary as ``word_freq``).
    t:
        Subsampling threshold.  Smaller values discard more frequent words.
        The original paper recommends values in the range [1e-5, 1e-3].

    Returns
    -------
    keep_probs:
        Float32 array of shape ``(vocab_size,)``; ``keep_probs[i]`` is the
        probability of retaining a token with index ``i``.
    """
    total: int = sum(word_freq.values())
    keep_probs = np.ones(len(word2idx), dtype=np.float32)

    for word, idx in word2idx.items():
        relative_freq: float = word_freq[word] / total          # f(w)
        # Mikolov et al. keep-probability formula
        keep_probs[idx] = float(
            min(1.0, np.sqrt(t / relative_freq) + (t / relative_freq))
        )

    return keep_probs


# ---------------------------------------------------------------------------
# 4.  Negative-sampling lookup table
# ---------------------------------------------------------------------------

def build_negative_sampling_table(
    word_freq: Dict[str, int],
    word2idx: Dict[str, int],
    table_size: int = 1_000_000,
    power: float = 0.75,
) -> np.ndarray:
    """
    Pre-compute a large integer lookup table for fast negative sampling.

    Each word ``w`` occupies a fraction of the table proportional to:

        P_n(w) ∝ f(w)^power

    The 3/4-power rule (power=0.75) dampens very common words while giving
    rare words a better chance of being selected as negatives.  Random
    access into this flat array is O(1) per negative sample.

    Parameters
    ----------
    word_freq:
        Raw frequency counts for the vocabulary.
    word2idx:
        Word-to-index mapping.
    table_size:
        Number of entries in the lookup table.  Larger values give a more
        accurate approximation to the true noise distribution.
    power:
        Exponent applied to word frequencies.  The original paper uses 0.75.

    Returns
    -------
    table:
        Int32 array of shape ``(table_size,)``; ``table[i]`` is a word index.
    """
    vocab_size = len(word2idx)

    # ---- compute f(w)^power for every word in vocabulary ---------------
    freq_powered = np.zeros(vocab_size, dtype=np.float64)
    for word, idx in word2idx.items():
        freq_powered[idx] = word_freq[word] ** power

    # ---- normalise to a probability distribution -----------------------
    noise_probs: np.ndarray = freq_powered / freq_powered.sum()

    # ---- fill table using cumulative probabilities ---------------------
    # np.searchsorted gives O(log V) per entry; total O(M log V) once.
    cumulative_probs: np.ndarray = np.cumsum(noise_probs)
    # Positions in [0, 1) for each table slot
    table_positions: np.ndarray = np.arange(table_size, dtype=np.float64) / table_size
    # Map each position to a word index
    table: np.ndarray = np.searchsorted(cumulative_probs, table_positions).astype(np.int32)

    print(f"[vocab] Negative-sampling table built  (size={table_size:,}, power={power}).")
    return table


# ---------------------------------------------------------------------------
# 5.  Training pair generation
# ---------------------------------------------------------------------------

def generate_training_pairs(
    sentences: List[List[str]],
    word2idx: Dict[str, int],
    keep_probs: np.ndarray,
    window_size: int = 5,
) -> List[Tuple[int, int]]:
    """
    Slide a window over every sentence and emit (center_idx, context_idx) pairs.

    Two standard word2vec tricks are applied:
        • **Vocabulary filtering** — tokens not in ``word2idx`` are skipped.
        • **Subsampling** — each token is retained with probability
          ``keep_probs[idx]``, thinning frequent words.
        • **Dynamic window** — the actual context radius is sampled uniformly
          from ``[1, window_size]``, giving closer context words slightly
          more weight.

    Parameters
    ----------
    sentences:
        Tokenised corpus.
    word2idx:
        Vocabulary mapping.
    keep_probs:
        Per-word keep probabilities from ``compute_subsampling_probs``.
    window_size:
        Maximum half-window size (the original paper uses 5).

    Returns
    -------
    pairs:
        List of ``(center_word_index, context_word_index)`` integer tuples.
    """
    pairs: List[Tuple[int, int]] = []

    for sentence in sentences:
        # Map to indices, discard OOV tokens, apply subsampling
        indexed: List[int] = []
        for token in sentence:
            if token not in word2idx:
                continue
            idx = word2idx[token]
            # Stochastic keep — Bernoulli trial with keep probability
            if random.random() < keep_probs[idx]:
                indexed.append(idx)

        # Slide window over the (subsampled) sentence
        n = len(indexed)
        for center_pos in range(n):
            # Dynamic window size: sample actual radius ∈ {1, …, window_size}
            actual_radius: int = random.randint(1, window_size)
            left  = max(0, center_pos - actual_radius)
            right = min(n, center_pos + actual_radius + 1)

            center_idx = indexed[center_pos]
            for ctx_pos in range(left, right):
                if ctx_pos == center_pos:
                    continue                         # skip center itself
                pairs.append((center_idx, indexed[ctx_pos]))

    print(f"[vocab] Generated {len(pairs):,} (center, context) training pairs.")
    return pairs


# ---------------------------------------------------------------------------
# 6.  Negative sample drawing
# ---------------------------------------------------------------------------

def sample_negatives(
    negative_table: np.ndarray,
    n_negatives: int,
    exclude: Tuple[int, ...] = (),
) -> List[int]:
    """
    Draw ``n_negatives`` word indices from the noise distribution.

    Any index in ``exclude`` (typically the current center and context word)
    is rejected and resampled.  We over-sample by a factor of 3× and fall
    back to an explicit loop only when the batch does not contain enough
    valid candidates — this is rarely needed in practice.

    Parameters
    ----------
    negative_table:
        Pre-built noise table from ``build_negative_sampling_table``.
    n_negatives:
        How many negative samples to return.
    exclude:
        Tuple of word indices that must not appear in the negatives.

    Returns
    -------
    negatives:
        List of ``n_negatives`` integer word indices.
    """
    exclude_set = set(exclude)
    table_len   = len(negative_table)

    # ---- Fast path: over-sample and filter in one NumPy call -----------
    candidates = negative_table[
        np.random.randint(0, table_len, size=n_negatives * 3)
    ].tolist()

    negatives: List[int] = [c for c in candidates if c not in exclude_set]

    # ---- Fallback: keep sampling until we have enough ------------------
    while len(negatives) < n_negatives:
        c = int(negative_table[random.randint(0, table_len - 1)])
        if c not in exclude_set:
            negatives.append(c)

    return negatives[:n_negatives]

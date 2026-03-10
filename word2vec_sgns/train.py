"""
train.py — Main training loop for Skip-gram with Negative Sampling (SGNS).

Usage
-----
    python train.py                              # use all defaults
    python train.py --embed_dim 200 --epochs 3  # custom settings

The script:
    1. Loads and preprocesses the text8 corpus (via vocab.py).
    2. Generates all positive skip-gram training pairs.
    3. Initialises the SGNS model (via word2vec.py).
    4. Runs the training loop with:
           • per-step negative sampling
           • forward + backward passes
           • linearly decayed learning rate (SGD)
           • configurable logging
    5. Saves the learned embeddings as a .npy file.
    6. Prints a cosine-similarity sanity check for a handful of test words.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List, Optional, Tuple

import numpy as np

from vocab import (
    build_negative_sampling_table,
    build_vocabulary,
    compute_subsampling_probs,
    generate_training_pairs,
    load_corpus,
    sample_negatives,
)
from word2vec import SkipGramNegativeSampling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_loss(
    step: int,
    total_steps: int,
    window_loss: float,
    window_size: int,
    lr: float,
    t0: float,
) -> None:
    """Print a one-line training progress update."""
    avg_loss      = window_loss / window_size
    pairs_per_sec = window_size / (time.time() - t0)
    pct           = 100.0 * step / total_steps
    print(
        f"  step {step:>10,}/{total_steps:,} ({pct:4.1f}%) "
        f"| lr={lr:.6f} "
        f"| avg_loss={avg_loss:.4f} "
        f"| {pairs_per_sec:>8,.0f} pairs/s"
    )


def _similarity_check(
    model: SkipGramNegativeSampling,
    idx2word: dict,
    word2idx: dict,
    test_words: Optional[List[str]] = None,
    top_n: int = 5,
) -> None:
    """Print nearest neighbours for a few probe words."""
    if test_words is None:
        test_words = ["king", "man", "woman", "computer", "france"]

    print("\n[eval] Nearest-neighbour sanity check:")
    for word in test_words:
        if word not in word2idx:
            print(f"  '{word}' not in vocabulary — skipped.")
            continue
        neighbours = model.most_similar(word2idx[word], idx2word, top_n=top_n)
        neighbours_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbours)
        print(f"  {word:12s} → {neighbours_str}")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    # ---- corpus / vocabulary ------------------------------------------
    dataset:       str   = "text8",
    min_count:     int   = 5,
    max_vocab_size: Optional[int] = None,
    subsample_t:   float = 1e-4,
    # ---- skip-gram window -------------------------------------------
    window_size:   int   = 5,
    # ---- model -------------------------------------------------------
    embed_dim:     int   = 100,
    # ---- negative sampling ------------------------------------------
    n_negatives:   int   = 5,
    ns_table_size: int   = 1_000_000,
    ns_power:      float = 0.75,
    # ---- optimisation -----------------------------------------------
    n_epochs:      int   = 1,
    learning_rate: float = 0.025,
    lr_min:        float = 0.0001,
    # ---- logging / IO -----------------------------------------------
    log_every:     int   = 100_000,
    save_path:     str   = "embeddings.npy",
    seed:          int   = 42,
) -> SkipGramNegativeSampling:
    """
    End-to-end SGNS training pipeline.

    Learning rate schedule
    ----------------------
    The original word2vec C implementation uses a linear decay:

        η(t) = max(η_min,  η_0 · (1 − t / T))

    where t is the current global step and T is the total number of steps
    across all epochs.  This ensures the learning rate reaches η_min exactly
    at the end of training.

    Parameters
    ----------
    dataset:
        gensim corpus name.  ``"text8"`` is the standard benchmark choice.
    min_count:
        Words appearing fewer times are removed from the vocabulary.
    max_vocab_size:
        Hard cap on vocabulary size (None = no cap).
    subsample_t:
        Subsampling threshold t in Mikolov et al. 2013.
    window_size:
        Maximum context half-window size.
    embed_dim:
        Embedding dimensionality d.
    n_negatives:
        K — number of negative samples per training pair.
    ns_table_size:
        Size of the negative-sampling lookup table.
    ns_power:
        Exponent in the noise distribution P_n(w) ∝ f(w)^power.
    n_epochs:
        Number of full passes over the training data.
    learning_rate:
        Initial SGD learning rate η_0.  The original paper uses 0.025.
    lr_min:
        Minimum learning rate η_min (floor for the decay schedule).
    log_every:
        Print a progress line every this many training steps.
    save_path:
        Path to save the final W_in embedding matrix as a NumPy .npy file.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    model:
        Trained ``SkipGramNegativeSampling`` instance.
    """
    random.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------ #
    # Step 1 — Load and preprocess the corpus                             #
    # ------------------------------------------------------------------ #
    sentences = load_corpus(dataset)

    word2idx, idx2word, word_freq = build_vocabulary(
        sentences,
        min_count=min_count,
        max_vocab_size=max_vocab_size,
    )
    vocab_size = len(word2idx)

    # Subsampling probabilities: P(keep | w) for each word index
    keep_probs = compute_subsampling_probs(word_freq, word2idx, t=subsample_t)

    # Negative-sampling lookup table (built once, reused every step)
    neg_table = build_negative_sampling_table(
        word_freq, word2idx,
        table_size=ns_table_size,
        power=ns_power,
    )

    # ------------------------------------------------------------------ #
    # Step 2 — Pre-generate all positive training pairs                   #
    # ------------------------------------------------------------------ #
    # NOTE: For very large corpora or many epochs, consider regenerating
    # pairs each epoch so that subsampling and dynamic windows provide
    # different thinning each pass (adds stochastic regularisation).
    pairs: List[Tuple[int, int]] = generate_training_pairs(
        sentences, word2idx, keep_probs, window_size
    )
    n_pairs      = len(pairs)
    total_steps  = n_epochs * n_pairs

    print(
        f"\n[train] Vocab={vocab_size:,}  |  pairs={n_pairs:,}  "
        f"|  total_steps={total_steps:,}  |  embed_dim={embed_dim}"
    )

    # ------------------------------------------------------------------ #
    # Step 3 — Initialise the model                                       #
    # ------------------------------------------------------------------ #
    model = SkipGramNegativeSampling(vocab_size=vocab_size, embed_dim=embed_dim)

    # ------------------------------------------------------------------ #
    # Step 4 — Training loop                                              #
    # ------------------------------------------------------------------ #
    global_step = 0     # counts total training steps across all epochs

    for epoch in range(1, n_epochs + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch} / {n_epochs}")
        print(f"{'='*60}")

        # Shuffle pairs at the start of each epoch for stochastic
        # gradient descent (reduces correlation across consecutive steps).
        random.shuffle(pairs)

        epoch_loss  = 0.0   # accumulated over the full epoch
        window_loss = 0.0   # accumulated over the last `log_every` steps
        t_window    = time.time()

        for center_idx, context_idx in pairs:

            # --- Linear learning rate decay ----------------------------
            # η(t) = max(η_min,  η_0 · (1 − t / T))
            lr: float = max(
                lr_min,
                learning_rate * (1.0 - global_step / total_steps),
            )

            # --- Draw K negative samples -------------------------------
            # Exclude the center and context word from the negatives so
            # the model is not asked to suppress a true context relationship.
            neg_indices = sample_negatives(
                neg_table, n_negatives,
                exclude=(center_idx, context_idx),
            )

            # --- Forward pass ------------------------------------------
            # Computes loss and caches intermediate values needed for
            # the backward pass without redundant look-ups.
            loss, v_center, sig_pos, sig_neg, v_neg_matrix = model.forward(
                center_idx, context_idx, neg_indices
            )

            # --- Backward pass + SGD update ----------------------------
            # All gradients are computed before any parameter is mutated
            # (correct SGD semantics for this per-sample update scheme).
            model.backward(
                center_idx, context_idx, neg_indices,
                v_center, sig_pos, sig_neg, v_neg_matrix,
                lr,
            )

            # --- Bookkeeping ------------------------------------------
            epoch_loss  += loss
            window_loss += loss
            global_step += 1

            # --- Logging -----------------------------------------------
            if global_step % log_every == 0:
                _log_loss(
                    global_step, total_steps,
                    window_loss, log_every,
                    lr, t_window,
                )
                window_loss = 0.0
                t_window    = time.time()

        avg_epoch_loss = epoch_loss / n_pairs
        print(f"\n[train] Epoch {epoch} done.  Avg loss = {avg_epoch_loss:.4f}")

    # ------------------------------------------------------------------ #
    # Step 5 — Save embeddings                                            #
    # ------------------------------------------------------------------ #
    embeddings = model.get_embeddings()   # W_in, shape (|V|, d)
    np.save(save_path, embeddings)

    # Optionally save the vocabulary mapping alongside the embeddings
    # so downstream code can map indices back to words.
    vocab_save = save_path.replace(".npy", "_vocab.npy")
    np.save(vocab_save, np.array(list(word2idx.keys())))   # word strings

    print(f"\n[train] Embeddings saved → '{save_path}'  shape={embeddings.shape}")
    print(f"[train] Vocabulary saved  → '{vocab_save}'")

    # ------------------------------------------------------------------ #
    # Step 6 — Quick similarity sanity check                              #
    # ------------------------------------------------------------------ #
    _similarity_check(model, idx2word, word2idx)

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Word2Vec Skip-gram with Negative Sampling (pure NumPy)."
    )
    # Corpus
    p.add_argument("--dataset",       default="text8",      help="gensim corpus name")
    p.add_argument("--min_count",     type=int,   default=5)
    p.add_argument("--max_vocab",     type=int,   default=None)
    p.add_argument("--subsample_t",   type=float, default=1e-4)
    # Model
    p.add_argument("--embed_dim",     type=int,   default=100)
    p.add_argument("--window_size",   type=int,   default=5)
    # Negative sampling
    p.add_argument("--n_negatives",   type=int,   default=5)
    p.add_argument("--ns_power",      type=float, default=0.75)
    # Training
    p.add_argument("--epochs",        type=int,   default=1)
    p.add_argument("--lr",            type=float, default=0.025)
    p.add_argument("--lr_min",        type=float, default=0.0001)
    p.add_argument("--log_every",     type=int,   default=100_000)
    p.add_argument("--save_path",     default="embeddings.npy")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        dataset        = args.dataset,
        min_count      = args.min_count,
        max_vocab_size = args.max_vocab,
        subsample_t    = args.subsample_t,
        embed_dim      = args.embed_dim,
        window_size    = args.window_size,
        n_negatives    = args.n_negatives,
        ns_power       = args.ns_power,
        n_epochs       = args.epochs,
        learning_rate  = args.lr,
        lr_min         = args.lr_min,
        log_every      = args.log_every,
        save_path      = args.save_path,
        seed           = args.seed,
    )

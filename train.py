"""
train.py — Training Loop for Skip-gram with Negative Sampling
==============================================================

Entry-point script that orchestrates the full SGNS training pipeline:

1.  Load and tokenise the corpus        (vocab.py)
2.  Build vocabulary and preprocess      (vocab.py)
3.  Initialise the SGNS model            (word2vec.py)
4.  Run the training loop with:
      • Mini-batch SGD
      • Linear learning-rate decay
      • Periodic loss logging
5.  Evaluate embedding quality (most-similar queries)
6.  Save the trained embeddings to disk

Usage
-----
    # Quick smoke-test with the built-in fallback corpus
    python train.py

    # Train on text8 (first 1 000 000 tokens)
    python train.py --corpus text8 --max_tokens 1000000

    # Full text8 training (≈ 17 M tokens, ~20 min on a modern CPU)
    python train.py --corpus text8
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional

import numpy as np

from vocab import (
    NegativeSampler,
    Vocabulary,
    build_vocab,
    corpus_to_indices,
    generate_training_pairs,
    load_corpus,
    subsample_corpus,
)
from word2vec import Word2VecSGNS


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train word2vec (Skip-gram + Negative Sampling) in pure NumPy."
    )
    # Data
    p.add_argument("--corpus", type=str, default=None,
                    help="Path to a plain-text corpus file.  If omitted, uses "
                        "the built-in fallback corpus for a quick smoke-test.")
    p.add_argument("--max_tokens", type=int, default=None,
                    help="Truncate the corpus to this many tokens (useful for "
                        "rapid iteration).")
    # Vocabulary
    p.add_argument("--min_count", type=int, default=5,
                    help="Discard words that appear fewer than min_count times.")
    p.add_argument("--subsample_t", type=float, default=1e-5,
                    help="Subsampling threshold for frequent words.")
    # Model
    p.add_argument("--embedding_dim", type=int, default=100,
                    help="Dimensionality of word embeddings.")
    p.add_argument("--window_size", type=int, default=5,
                    help="Max one-sided context window size.")
    p.add_argument("--num_negatives", type=int, default=10,
                    help="Number of negative samples per positive pair.")
    # Training
    p.add_argument("--epochs", type=int, default=5,
                    help="Number of full passes over the training pairs.")
    p.add_argument("--batch_size", type=int, default=256,
                    help="Mini-batch size.")
    p.add_argument("--lr", type=float, default=0.025,
                    help="Initial learning rate (linearly decayed to ~0).")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    # Output
    p.add_argument("--save_path", type=str, default="embeddings.npz",
                    help="File path for saving trained embeddings.")
    p.add_argument("--log_every", type=int, default=5000,
                    help="Print running loss every N batches.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: Word2VecSGNS,
    pairs: np.ndarray,
    sampler: NegativeSampler,
    *,
    epochs: int,
    batch_size: int,
    num_negatives: int,
    lr_init: float,
    log_every: int,
) -> List[float]:
    """Run the full mini-batch SGD training loop.

    Learning-rate schedule: linear decay from *lr_init* to *lr_min* over
    the total number of weight updates, where lr_min = lr_init × 1e-4.
    This matches the original word2vec C implementation.

    Parameters
    ----------
    model : Word2VecSGNS
        The model instance to train (modified in-place).
    pairs : np.ndarray, shape (P, 2)
        Training pairs ``[center_idx, context_idx]``.
    sampler : NegativeSampler
        Negative-sampling distribution.
    epochs : int
        Number of full passes over ``pairs``.
    batch_size : int
        Mini-batch size.
    num_negatives : int
        Number of negative samples drawn per positive pair.
    lr_init : float
        Initial learning rate.
    log_every : int
        Print loss every this many batches.

    Returns
    -------
    list[float]
        Per-log-step average losses (for later plotting / inspection).
    """
    num_pairs = len(pairs)
    num_batches = (num_pairs + batch_size - 1) // batch_size
    total_steps = epochs * num_batches
    lr_min = lr_init * 1e-4       # floor for the learning rate

    loss_history: List[float] = []
    global_step = 0

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # Shuffle training pairs at the start of each epoch
        perm = np.random.permutation(num_pairs)
        pairs_shuffled = pairs[perm]

        running_loss = 0.0
        running_count = 0

        for b in range(num_batches):
            # --- Slice the mini-batch ------------------------------------
            start = b * batch_size
            end = min(start + batch_size, num_pairs)
            batch = pairs_shuffled[start:end]
            B = len(batch)

            center_batch = batch[:, 0]              # (B,)
            context_batch = batch[:, 1]             # (B,)

            # --- Draw negative samples for the whole batch (one call) ----
            neg_batch = sampler.sample((B, num_negatives))  # (B, K)

            # --- Linear learning-rate decay ------------------------------
            #   η_t  =  η_0 · (1 − progress)   clamped to [η_min, η_0]
            progress = global_step / max(total_steps, 1)
            lr = max(lr_init * (1.0 - progress), lr_min)

            # --- Forward → backward → update (all vectorised) -----------
            batch_loss = model.train_step_batch(
                center_batch, context_batch, neg_batch, lr
            )

            running_loss += batch_loss * B
            running_count += B
            global_step += 1

            # --- Logging -------------------------------------------------
            if global_step % log_every == 0 or b == num_batches - 1:
                avg_loss = running_loss / max(running_count, 1)
                elapsed = time.time() - t_start
                pairs_per_sec = (global_step * batch_size) / max(elapsed, 1e-6)
                print(
                    f"  epoch {epoch}/{epochs}  "
                    f"batch {b + 1}/{num_batches}  "
                    f"loss={avg_loss:.4f}  "
                    f"lr={lr:.6f}  "
                    f"speed={pairs_per_sec:,.0f} pairs/s"
                )
                loss_history.append(avg_loss)
                running_loss = 0.0
                running_count = 0

    total_time = time.time() - t_start
    print(f"\n[train] Training complete in {total_time:.1f}s  "
          f"({global_step:,} steps, {global_step * batch_size:,} pair updates)")
    return loss_history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_similarity(
    model: Word2VecSGNS,
    vocab: Vocabulary,
    query_words: List[str],
    topn: int = 8,
) -> None:
    """Print the most-similar words for a list of query words."""
    print("\n" + "=" * 60)
    print("  Nearest-Neighbour Evaluation (cosine similarity)")
    print("=" * 60)
    for word in query_words:
        if word not in vocab.word2idx:
            print(f"  '{word}' not in vocabulary — skipping")
            continue
        idx = vocab.word2idx[word]
        neighbours = model.most_similar(idx, vocab.idx2word, topn=topn)
        neighbour_str = ", ".join(
            f"{w} ({s:.3f})" for w, s in neighbours
        )
        print(f"  {word:>12s} → {neighbour_str}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # 1. Load and tokenise ---------------------------------------------------
    print("\n[1/5] Loading corpus …")
    tokens = load_corpus(path=args.corpus, max_tokens=args.max_tokens)

    # 2. Build vocabulary ----------------------------------------------------
    print("[2/5] Building vocabulary …")
    vocab = build_vocab(tokens, min_count=args.min_count)
    corpus_idx = corpus_to_indices(tokens, vocab)
    corpus_idx = subsample_corpus(corpus_idx, vocab.counts, args.subsample_t)

    # 3. Generate training pairs and sampler ---------------------------------
    print("[3/5] Generating training pairs …")
    pairs = generate_training_pairs(corpus_idx, window_size=args.window_size)
    sampler = NegativeSampler(vocab.counts)

    # 4. Initialise model and train ------------------------------------------
    print("[4/5] Training …\n")
    print(f"  Vocabulary size : {vocab.size:,}")
    print(f"  Embedding dim   : {args.embedding_dim}")
    print(f"  Window size     : {args.window_size}")
    print(f"  Num negatives   : {args.num_negatives}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Initial LR      : {args.lr}")
    print(f"  Training pairs  : {len(pairs):,}")
    print()

    model = Word2VecSGNS(
        vocab_size=vocab.size,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )

    loss_history = train(
        model,
        pairs,
        sampler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        lr_init=args.lr,
        log_every=args.log_every,
    )

    # 5. Evaluate and save ---------------------------------------------------
    print("[5/5] Evaluation & saving …")

    # Pick some probe words that are likely in the vocabulary
    probe_words = [
        "king", "queen", "man", "woman", "prince", "princess",
        "dog", "cat", "city", "castle", "young", "strong",
    ]
    evaluate_similarity(model, vocab, probe_words)

    model.save(args.save_path)
    print(f"\nDone.  Embeddings saved to {args.save_path}")


if __name__ == "__main__":
    main()

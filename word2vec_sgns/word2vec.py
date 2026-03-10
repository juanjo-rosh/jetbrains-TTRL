"""
word2vec.py — Pure-NumPy Skip-gram with Negative Sampling (SGNS) model.

Mathematical notation used throughout this file
------------------------------------------------
    d         : embedding dimension
    |V|       : vocabulary size
    v_w       : W_in[center]    — center word's INPUT  vector  ∈ ℝ^d
    v'_c      : W_out[context]  — context word's OUTPUT vector ∈ ℝ^d
    v'_{n_k}  : W_out[neg_k]   — k-th negative word's OUTPUT vector ∈ ℝ^d
    s_+       : v'_c  · v_w      — positive dot-product score
    s_k       : v'_{n_k} · v_w  — k-th negative dot-product score
    σ(x)      : 1 / (1 + e^{-x}) — sigmoid / logistic function
    K         : number of negative samples per training pair

SGNS Loss (per training pair, see math_derivation.md for full derivation)
--------------------------------------------------------------------------
    L = −log σ(s_+) − Σ_{k=1}^{K} log(1 − σ(s_k))
      = −log σ(s_+) − Σ_{k=1}^{K} log σ(−s_k)

Gradient summary
----------------
    ∂L/∂v_w      = (σ(s_+) − 1)·v'_c  +  Σ_k σ(s_k)·v'_{n_k}
    ∂L/∂v'_c     = (σ(s_+) − 1)·v_w
    ∂L/∂v'_{n_k} =  σ(s_k)·v_w      for each k ∈ {1,…,K}
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class SkipGramNegativeSampling:
    """
    Word2Vec Skip-gram model trained with Negative Sampling (pure NumPy).

    Two embedding matrices are maintained:

        W_in  (|V| × d)  — *input*  (center-word)  embedding matrix.
        W_out (|V| × d)  — *output* (context-word) embedding matrix.

    After training, ``W_in`` is typically used as the final word embeddings,
    though some applications average ``W_in`` and ``W_out``.

    Initialisation
    --------------
    Following the original word2vec C code:
        • W_in  is initialised with *uniform noise* in (−0.5/d, 0.5/d).
        • W_out is initialised to *zeros*.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """
        Parameters
        ----------
        vocab_size:
            Total number of unique tokens in the filtered vocabulary (|V|).
        embed_dim:
            Dimensionality of the word embedding vectors (d).
        """
        self.vocab_size: int = vocab_size
        self.embed_dim:  int = embed_dim

        # W_in: shape (|V|, d) — centre-word (input) embeddings
        # Uniform initialisation scaled by 1/d keeps initial dot products small
        self.W_in: np.ndarray = np.random.uniform(
            low  = -0.5 / embed_dim,
            high =  0.5 / embed_dim,
            size = (vocab_size, embed_dim),
        ).astype(np.float32)

        # W_out: shape (|V|, d) — context/output word embeddings
        # Initialised to zero, matching the original C implementation
        self.W_out: np.ndarray = np.zeros(
            (vocab_size, embed_dim), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Numerically stable element-wise sigmoid.

        Avoids overflow for very negative inputs and underflow for very
        positive inputs by branching on the sign of each element:

            σ(x) = 1 / (1 + e^{−x})    for x ≥ 0
            σ(x) = e^x / (1 + e^x)     for x < 0  (equivalent, but stable)
        """
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        center_idx:  int,
        context_idx: int,
        neg_indices: List[int],
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute all quantities needed for the loss and backward pass.

        Parameters
        ----------
        center_idx:
            Index of the center word in W_in.
        context_idx:
            Index of the positive context word in W_out.
        neg_indices:
            List of K negative sample indices, all indexing W_out.

        Returns
        -------
        loss:
            Scalar SGNS loss L = −log σ(s_+) − Σ_k log(1 − σ(s_k)).
        v_center:
            Center word vector v_w = W_in[center_idx], shape (d,).
        sig_pos:
            σ(s_+) — sigmoid of the positive score; used in the backward pass.
        sig_neg:
            σ(s_k) for each negative; shape (K,).
        v_neg_matrix:
            Stacked negative output vectors V'_neg = W_out[neg_indices],
            shape (K, d); cached to avoid a second index lookup in backward.
        """
        # ---- Retrieve embedding vectors -----------------------------------
        # v_w = W_in[center_idx]           shape: (d,)
        v_center: np.ndarray = self.W_in[center_idx]

        # v'_c = W_out[context_idx]        shape: (d,)
        v_context: np.ndarray = self.W_out[context_idx]

        # V'_neg = W_out[neg_indices]      shape: (K, d)
        v_neg_matrix: np.ndarray = self.W_out[neg_indices]   # fancy indexing

        # ---- Dot-product scores -------------------------------------------
        # s_+ = v'_c · v_w                 scalar
        s_pos: float = float(np.dot(v_context, v_center))

        # s_k  = v'_{n_k} · v_w for each k  shape: (K,)
        # Matrix–vector product: (K, d) × (d,) → (K,)
        s_neg: np.ndarray = v_neg_matrix @ v_center

        # ---- Sigmoid activations -----------------------------------------
        # σ(s_+)    scalar — probability assigned to the positive pair
        sig_pos: np.ndarray = self.sigmoid(np.array([s_pos]))[0]

        # σ(s_k)    shape (K,) — probability assigned to each negative pair
        sig_neg: np.ndarray = self.sigmoid(s_neg)            # (K,)

        # ---- SGNS Loss ---------------------------------------------------
        # L = −log σ(s_+) − Σ_k log(1 − σ(s_k))
        # Clip arguments to log to avoid log(0) = −∞
        loss: float = (
            -np.log(sig_pos + 1e-10)
            - float(np.sum(np.log(1.0 - sig_neg + 1e-10)))
        )

        return loss, v_center, sig_pos, sig_neg, v_neg_matrix

    # ------------------------------------------------------------------
    # Backward pass + SGD update
    # ------------------------------------------------------------------

    def backward(
        self,
        center_idx:   int,
        context_idx:  int,
        neg_indices:  List[int],
        v_center:     np.ndarray,   # cached from forward — shape (d,)
        sig_pos:      float,        # σ(s_+) — scalar
        sig_neg:      np.ndarray,   # σ(s_k) for each k — shape (K,)
        v_neg_matrix: np.ndarray,   # W_out[neg_indices] at forward time — (K, d)
        lr:           float,        # current learning rate η
    ) -> None:
        """
        Compute the SGNS gradients and apply SGD updates in-place.

        The gradient derivations are listed in full in the module docstring;
        here we annotate each line with the corresponding formula.

        Crucially, we compute ALL gradients before modifying any parameter
        so that concurrent updates do not contaminate one another.

        Parameters
        ----------
        center_idx, context_idx, neg_indices:
            Same as in ``forward``; used to index into W_in / W_out.
        v_center:
            Center word vector returned by ``forward`` (read-only snapshot).
        sig_pos:
            σ(s_+) returned by ``forward``.
        sig_neg:
            σ(s_k) for each negative, shape (K,), returned by ``forward``.
        v_neg_matrix:
            Stacked negative output vectors (K, d), returned by ``forward``.
        lr:
            Current scalar learning rate η.
        """
        # ---- Error signals -----------------------------------------------
        # For the positive sample the "error" is:  σ(s_+) − 1  ∈ (−1, 0)
        # (negative because the model should push s_+ toward +∞)
        err_pos: float = sig_pos - 1.0              # scalar

        # For each negative sample the error is:  σ(s_k)  ∈ (0, 1)
        # (positive because the model should push s_k toward −∞)
        # sig_neg already has shape (K,) — no extra work needed

        # ---- Gradient w.r.t. W_out[context_idx] (positive) --------------
        # ∂L/∂v'_c = (σ(s_+) − 1) · v_w       shape: (d,)
        grad_v_ctx: np.ndarray = err_pos * v_center                 # (d,)

        # ---- Gradient w.r.t. W_out[neg_k] for each negative k -----------
        # ∂L/∂v'_{n_k} = σ(s_k) · v_w          shape: (K, d)
        #
        # Outer product via broadcasting:
        #   sig_neg[:, np.newaxis] → (K, 1)
        #   v_center[np.newaxis, :] → (1, d)
        #   product → (K, d)
        grad_v_neg: np.ndarray = sig_neg[:, np.newaxis] * v_center[np.newaxis, :]

        # ---- Gradient w.r.t. W_in[center_idx] (center) ------------------
        # ∂L/∂v_w = (σ(s_+)−1)·v'_c  +  Σ_k σ(s_k)·v'_{n_k}
        #
        # Term 1: err_pos · W_out[context_idx]   shape (d,)
        #   We re-read W_out[context_idx] here because it has NOT yet
        #   been updated — all gradient computations precede all updates.
        #
        # Term 2: sig_neg @ v_neg_matrix          shape (d,)
        #   Matrix–vector:  (K,) × (K, d)  →  (d,)
        #   This is the vectorised form of  Σ_k σ(s_k) · v'_{n_k}
        grad_v_center: np.ndarray = (
            err_pos * self.W_out[context_idx]       # contribution from positive
            + sig_neg @ v_neg_matrix                # contribution from negatives
        )                                           # shape: (d,)

        # ---- SGD in-place updates ----------------------------------------
        # Rule: param ← param − η · ∂L/∂param

        # Update center word input vector
        self.W_in[center_idx] -= lr * grad_v_center                 # (d,)

        # Update positive context output vector
        self.W_out[context_idx] -= lr * grad_v_ctx                  # (d,)

        # Update all negative output vectors
        # np.add.at is used instead of direct indexing because the same
        # negative index *could* appear multiple times in neg_indices;
        # np.add.at accumulates gradients correctly in that case.
        np.add.at(self.W_out, neg_indices, -lr * grad_v_neg)        # (K, d)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def get_embeddings(self) -> np.ndarray:
        """
        Return the input (center-word) embedding matrix W_in.

        Shape: ``(vocab_size, embed_dim)``.  These are the standard "word
        vectors" used for downstream tasks such as analogy evaluation or
        initialising neural network embedding layers.
        """
        return self.W_in

    def most_similar(
        self,
        word_idx: int,
        idx2word: Dict[int, str],
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Return the ``top_n`` most similar words by cosine similarity.

        Cosine similarity:  sim(u, v) = (u · v) / (||u|| · ||v||)

        We normalise all rows of W_in once and then use a single
        matrix–vector product for all pairwise similarities — O(|V|·d).

        Parameters
        ----------
        word_idx:
            Index of the query word.
        idx2word:
            Reverse vocabulary mapping.
        top_n:
            Number of neighbours to return.

        Returns
        -------
        List of (word_string, cosine_similarity) sorted by descending
        similarity.
        """
        # L2-normalise every embedding row to unit length
        norms: np.ndarray = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)                    # guard /0
        W_normed: np.ndarray = self.W_in / norms            # (|V|, d)

        # Query vector (already normalised)
        query: np.ndarray = W_normed[word_idx]              # (d,)

        # All pairwise cosine similarities in one matrix–vector multiply
        similarities: np.ndarray = W_normed @ query         # (|V|,)

        # Exclude the query word from results
        similarities[word_idx] = -np.inf

        # Partial sort: O(|V|) rather than a full O(|V| log |V|) sort
        top_idx = np.argpartition(similarities, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(similarities[top_idx])[::-1]]

        return [(idx2word[int(i)], float(similarities[i])) for i in top_idx]

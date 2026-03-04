"""
word2vec.py — Skip-gram with Negative Sampling (SGNS) Model
============================================================

Pure-NumPy implementation of the SGNS forward pass, loss computation,
analytical gradient calculation, and SGD parameter updates.

Mathematical Background
-----------------------
For a (center, context) pair (w_c, w_o) and K negative samples {w_1, …, w_K},
the SGNS loss is:

    L  =  −log σ(u_o · v_c)  −  Σ_{k=1}^{K} log σ(−u_k · v_c)

where:
    v_c  ∈  ℝ^d   is the *center* (input) embedding of w_c
    u_o  ∈  ℝ^d   is the *context* (output) embedding of w_o
    u_k  ∈  ℝ^d   is the *context* embedding of the k-th negative sample
    σ(x) = 1 / (1 + exp(−x))   is the sigmoid function

Gradients (derived in math_derivation.md):

    ∂L/∂v_c  =  −(1 − σ(u_o · v_c)) u_o  +  Σ_k σ(u_k · v_c) u_k
    ∂L/∂u_o  =  −(1 − σ(u_o · v_c)) v_c
    ∂L/∂u_k  =   σ(u_k · v_c) v_c           for each k

SGD update rule (learning rate η):

    θ  ←  θ  −  η · ∂L/∂θ

References
----------
* Mikolov et al. (2013) — "Distributed Representations of Words and Phrases
  and their Compositionality"
* Levy & Goldberg (2014) — "Neural Word Embedding as Implicit Matrix
  Factorization"
* Rong (2014) — "word2vec Parameter Learning Explained"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid.

    Avoids overflow in exp() by clipping the input to [−20, 20].
    Outside this range σ(x) is effectively 0 or 1.
    """
    x_clip = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class Word2VecSGNS:
    """Skip-gram with Negative Sampling — pure NumPy.

    Parameters
    ----------
    vocab_size : int
        Number of unique words in the vocabulary (V).
    embedding_dim : int
        Dimensionality of word vectors (d).
    seed : int or None
        Random seed for reproducible weight initialisation.

    Attributes
    ----------
    W_center : np.ndarray, shape (V, d)
        Center (input) embedding matrix.  Row i is v_i.
    W_context : np.ndarray, shape (V, d)
        Context (output) embedding matrix.  Row i is u_i.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        rng = np.random.RandomState(seed)

        # -------------------------------------------------------------------
        # Weight initialisation
        #
        # Center embeddings: small uniform random values in (−0.5/d, 0.5/d).
        # Context embeddings: zeros (standard practice — makes initial
        #   sigmoid outputs 0.5, giving balanced gradients at the start).
        # -------------------------------------------------------------------
        scale = 0.5 / embedding_dim
        self.W_center: np.ndarray = rng.uniform(
            -scale, scale, size=(vocab_size, embedding_dim)
        )
        self.W_context: np.ndarray = np.zeros(
            (vocab_size, embedding_dim), dtype=np.float64
        )

    # -----------------------------------------------------------------------
    # Single-pair forward + backward  (educational / reference version)
    # -----------------------------------------------------------------------

    def train_step_single(
        self,
        center_idx: int,
        context_idx: int,
        neg_indices: np.ndarray,
        lr: float,
    ) -> float:
        """Process one (center, context) pair and update parameters.

        This method is the *un-batched* reference implementation.  It maps
        one-to-one onto the mathematical derivation and is easy to verify
        against finite-difference gradient checks.  For production training,
        use :meth:`train_step_batch` instead.

        Parameters
        ----------
        center_idx : int
            Vocabulary index of the center word.
        context_idx : int
            Vocabulary index of the (positive) context word.
        neg_indices : np.ndarray, shape (K,)
            Vocabulary indices of the K negative samples.
        lr : float
            Current learning rate η.

        Returns
        -------
        float
            The SGNS loss for this pair.
        """
        K = len(neg_indices)

        # === FORWARD PASS ===================================================

        # Look up embeddings  ─────────────────────────────────────────────
        v_c = self.W_center[center_idx]            # (d,)  center vector
        u_o = self.W_context[context_idx]           # (d,)  positive context
        u_neg = self.W_context[neg_indices]          # (K, d) negative contexts

        # Dot products  ────────────────────────────────────────────────────
        # Positive:  s_pos = u_o · v_c   →  scalar
        s_pos = np.dot(u_o, v_c)

        # Negatives: s_neg[k] = u_k · v_c  →  shape (K,)
        s_neg = u_neg @ v_c                          # matrix-vector product

        # Sigmoid activations  ─────────────────────────────────────────────
        sig_pos = _sigmoid(s_pos)                    # σ(u_o · v_c)
        sig_neg = _sigmoid(s_neg)                    # σ(u_k · v_c) for each k

        # === LOSS ============================================================
        #   L  =  −log σ(s_pos)  −  Σ_k log σ(−s_neg_k)
        #      =  −log(sig_pos)  −  Σ_k log(1 − sig_neg_k)
        loss = -np.log(sig_pos + 1e-10) - np.sum(np.log(1.0 - sig_neg + 1e-10))

        # === BACKWARD PASS (analytical gradients) ============================
        #
        # See math_derivation.md for the full chain-rule derivation.
        #
        # ∂L/∂v_c  =  −(1 − σ_pos) u_o  +  Σ_k σ_neg_k u_k
        grad_v_c = -(1.0 - sig_pos) * u_o + (sig_neg[:, None] * u_neg).sum(axis=0)

        # ∂L/∂u_o  =  −(1 − σ_pos) v_c
        grad_u_o = -(1.0 - sig_pos) * v_c

        # ∂L/∂u_k  =  σ_neg_k v_c   for each k
        grad_u_neg = sig_neg[:, None] * v_c[None, :]  # (K, d)

        # === SGD UPDATES =====================================================
        #   θ  ←  θ − η · ∂L/∂θ
        self.W_center[center_idx]   -= lr * grad_v_c
        self.W_context[context_idx] -= lr * grad_u_o

        # Use np.subtract.at for scattered updates — handles duplicate
        # indices in neg_indices correctly (plain indexing would silently
        # skip repeated index updates).
        np.subtract.at(self.W_context, neg_indices, lr * grad_u_neg)

        return float(loss)

    # -----------------------------------------------------------------------
    # Batched forward + backward  (vectorised for NumPy performance)
    # -----------------------------------------------------------------------

    def train_step_batch(
        self,
        center_batch: np.ndarray,
        context_batch: np.ndarray,
        neg_batch: np.ndarray,
        lr: float,
    ) -> float:
        """Process a mini-batch of (center, context) pairs.

        All dot products and gradient computations are fully vectorised as
        matrix operations, which is the key optimisation for a pure-NumPy
        implementation (avoids Python-level per-pair loops).

        Parameters
        ----------
        center_batch : np.ndarray, shape (B,)
            Center-word indices for B training pairs.
        context_batch : np.ndarray, shape (B,)
            Positive context-word indices for B pairs.
        neg_batch : np.ndarray, shape (B, K)
            Negative-sample indices for each pair.
        lr : float
            Current learning rate η.

        Returns
        -------
        float
            Mean SGNS loss over the batch.
        """
        B, K = neg_batch.shape
        d = self.embedding_dim

        # === FORWARD PASS (batched) =========================================

        # Look up embeddings  ─────────────────────────────────────────────
        v_c = self.W_center[center_batch]           # (B, d)
        u_o = self.W_context[context_batch]          # (B, d)
        u_neg = self.W_context[neg_batch]             # (B, K, d)

        # Dot products  ────────────────────────────────────────────────────
        # Positive: element-wise multiply then sum along d  →  (B,)
        s_pos = np.sum(v_c * u_o, axis=1)           # (B,)

        # Negatives: for each sample b, dot product of each of K negatives
        #   u_neg[b, k, :] · v_c[b, :]  →  (B, K)
        # Achieved via Einstein summation: "bd,bkd->bk"
        s_neg = np.einsum("bd,bkd->bk", v_c, u_neg)  # (B, K)

        # Sigmoid activations  ─────────────────────────────────────────────
        sig_pos = _sigmoid(s_pos)                    # (B,)
        sig_neg = _sigmoid(s_neg)                    # (B, K)

        # === LOSS (batched) ==================================================
        #   L_b  =  −log σ(s_pos_b)  −  Σ_k log σ(−s_neg_{b,k})
        #        =  −log(sig_pos_b)  −  Σ_k log(1 − sig_neg_{b,k})
        loss_per_pair = (
            -np.log(sig_pos + 1e-10)
            - np.sum(np.log(1.0 - sig_neg + 1e-10), axis=1)
        )  # (B,)
        mean_loss = loss_per_pair.mean()

        # === BACKWARD PASS (batched) =========================================
        #
        # Gradient notation uses · for element-wise and @ for matmul where
        # the shape comments make the broadcasting explicit.
        #
        # ∂L/∂v_c  (B, d)
        #   = −(1 − sig_pos)[:,None] * u_o                  positive term
        #   + Σ_k sig_neg[:,:,None] * u_neg   summed over k  negative term
        pos_coeff = -(1.0 - sig_pos)[:, None]                # (B, 1)
        neg_coeff = sig_neg[:, :, None]                       # (B, K, 1)

        grad_v_c = pos_coeff * u_o + np.sum(neg_coeff * u_neg, axis=1)  # (B, d)

        # ∂L/∂u_o  (B, d)
        #   = −(1 − sig_pos)[:,None] * v_c
        grad_u_o = pos_coeff * v_c                            # (B, d)

        # ∂L/∂u_k  (B, K, d)
        #   = sig_neg[:,:,None] * v_c[:,None,:]
        grad_u_neg = neg_coeff * v_c[:, None, :]              # (B, K, d)

        # === SGD UPDATES (scattered) =========================================
        #
        # np.subtract.at performs *unbuffered* in-place subtraction, which
        # correctly accumulates when the same index appears more than once
        # (as may happen in center_batch, context_batch, or neg_batch).
        # Plain fancy-indexing (W[indices] -= ...) would only apply the
        # *last* update for duplicate indices.

        np.subtract.at(self.W_center,  center_batch, lr * grad_v_c)
        np.subtract.at(self.W_context, context_batch, lr * grad_u_o)

        # For the (B, K, d) negative gradients, flatten the first two axes
        # so np.subtract.at sees a 1-D index array of length B*K.
        neg_flat = neg_batch.reshape(-1)                      # (B*K,)
        grad_neg_flat = (lr * grad_u_neg).reshape(-1, d)      # (B*K, d)
        np.subtract.at(self.W_context, neg_flat, grad_neg_flat)

        return float(mean_loss)

    # -----------------------------------------------------------------------
    # Embedding access & similarity
    # -----------------------------------------------------------------------

    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Return the center embedding for a single word.

        Parameters
        ----------
        word_idx : int
            Vocabulary index.

        Returns
        -------
        np.ndarray, shape (d,)
        """
        return self.W_center[word_idx].copy()

    def get_all_embeddings(self) -> np.ndarray:
        """Return the full center embedding matrix.

        Returns
        -------
        np.ndarray, shape (V, d)
        """
        return self.W_center.copy()

    def most_similar(
        self,
        word_idx: int,
        idx2word: Dict[int, str],
        topn: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find the *topn* most similar words by cosine similarity.

        Cosine similarity between vectors a and b:

            cos(a, b)  =  (a · b) / (‖a‖ ‖b‖)

        Parameters
        ----------
        word_idx : int
            Index of the query word.
        idx2word : dict[int, str]
            Index-to-word mapping.
        topn : int
            Number of results to return.

        Returns
        -------
        list of (word, similarity) tuples, sorted descending by similarity.
        """
        query = self.W_center[word_idx]               # (d,)

        # Normalise all center embeddings
        norms = np.linalg.norm(self.W_center, axis=1, keepdims=True) + 1e-10
        normed = self.W_center / norms                 # (V, d)

        # Cosine similarity as a single matrix-vector product
        q_norm = query / (np.linalg.norm(query) + 1e-10)
        similarities = normed @ q_norm                 # (V,)

        # Exclude the word itself and find top-n
        similarities[word_idx] = -np.inf
        top_indices = np.argsort(similarities)[::-1][:topn]

        return [
            (idx2word[int(i)], float(similarities[i]))
            for i in top_indices
        ]

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save embeddings to a compressed .npz file."""
        np.savez_compressed(
            path,
            W_center=self.W_center,
            W_context=self.W_context,
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
        )
        print(f"[model] Saved embeddings to {path}")

    @classmethod
    def load(cls, path: str) -> "Word2VecSGNS":
        """Load embeddings from a .npz file."""
        data = np.load(path)
        model = cls(
            vocab_size=int(data["vocab_size"]),
            embedding_dim=int(data["embedding_dim"]),
        )
        model.W_center = data["W_center"]
        model.W_context = data["W_context"]
        print(f"[model] Loaded embeddings from {path}")
        return model


# ---------------------------------------------------------------------------
# Gradient check  (finite-difference verification)
# ---------------------------------------------------------------------------

def gradient_check(
    model: Word2VecSGNS,
    center_idx: int = 0,
    context_idx: int = 1,
    neg_indices: Optional[np.ndarray] = None,
    epsilon: float = 1e-5,
    tolerance: float = 1e-5,
) -> bool:
    """Verify analytical gradients against numerical (finite-difference) ones.

    For each parameter θ_i, the numerical gradient is:

        ∂L/∂θ_i  ≈  (L(θ_i + ε) − L(θ_i − ε)) / (2ε)

    The relative error between analytical and numerical gradients should
    be well below 1e-4 for a correct implementation.

    Parameters
    ----------
    model : Word2VecSGNS
        A model instance (will be mutated — call on a throwaway copy).
    center_idx, context_idx : int
        Word indices for the test pair.
    neg_indices : np.ndarray or None
        Negative-sample indices.  Defaults to [2, 3, 4].
    epsilon : float
        Finite-difference step size.
    tolerance : float
        Maximum acceptable relative error.

    Returns
    -------
    bool
        True if all checks pass.
    """
    if neg_indices is None:
        neg_indices = np.array([2, 3, 4])
    K = len(neg_indices)

    def _compute_loss() -> float:
        """Compute the SGNS loss *without* modifying parameters."""
        v_c = model.W_center[center_idx]
        u_o = model.W_context[context_idx]
        u_neg = model.W_context[neg_indices]
        s_pos = np.dot(u_o, v_c)
        s_neg = u_neg @ v_c
        sig_pos = _sigmoid(s_pos)
        sig_neg = _sigmoid(s_neg)
        return float(
            -np.log(sig_pos + 1e-10)
            - np.sum(np.log(1.0 - sig_neg + 1e-10))
        )

    def _compute_analytical_grads():
        """Compute analytical gradients for the test pair."""
        v_c = model.W_center[center_idx]
        u_o = model.W_context[context_idx]
        u_neg = model.W_context[neg_indices]
        s_pos = np.dot(u_o, v_c)
        s_neg = u_neg @ v_c
        sig_pos = _sigmoid(s_pos)
        sig_neg = _sigmoid(s_neg)

        grad_v_c = -(1.0 - sig_pos) * u_o + (sig_neg[:, None] * u_neg).sum(axis=0)
        grad_u_o = -(1.0 - sig_pos) * v_c
        grad_u_neg = sig_neg[:, None] * v_c[None, :]
        return grad_v_c, grad_u_o, grad_u_neg

    grad_v_c, grad_u_o, grad_u_neg = _compute_analytical_grads()

    all_ok = True

    # --- Check grad_v_c ---
    for i in range(model.embedding_dim):
        orig = model.W_center[center_idx, i]
        model.W_center[center_idx, i] = orig + epsilon
        loss_plus = _compute_loss()
        model.W_center[center_idx, i] = orig - epsilon
        loss_minus = _compute_loss()
        model.W_center[center_idx, i] = orig  # restore

        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
        analytical = grad_v_c[i]
        rel_err = abs(analytical - numerical) / (abs(numerical) + 1e-10)
        if rel_err > tolerance:
            print(f"  ✗ grad_v_c[{i}]: analytical={analytical:.8f}  "
                    f"numerical={numerical:.8f}  rel_err={rel_err:.2e}")
            all_ok = False

    # --- Check grad_u_o ---
    for i in range(model.embedding_dim):
        orig = model.W_context[context_idx, i]
        model.W_context[context_idx, i] = orig + epsilon
        loss_plus = _compute_loss()
        model.W_context[context_idx, i] = orig - epsilon
        loss_minus = _compute_loss()
        model.W_context[context_idx, i] = orig

        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
        analytical = grad_u_o[i]
        rel_err = abs(analytical - numerical) / (abs(numerical) + 1e-10)
        if rel_err > tolerance:
            print(f"  ✗ grad_u_o[{i}]: analytical={analytical:.8f}  "
                    f"numerical={numerical:.8f}  rel_err={rel_err:.2e}")
            all_ok = False

    # --- Check grad_u_neg ---
    for k in range(K):
        for i in range(model.embedding_dim):
            orig = model.W_context[neg_indices[k], i]
            model.W_context[neg_indices[k], i] = orig + epsilon
            loss_plus = _compute_loss()
            model.W_context[neg_indices[k], i] = orig - epsilon
            loss_minus = _compute_loss()
            model.W_context[neg_indices[k], i] = orig

            numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
            analytical = grad_u_neg[k, i]
            rel_err = abs(analytical - numerical) / (abs(numerical) + 1e-10)
            if rel_err > tolerance:
                print(f"  ✗ grad_u_neg[{k},{i}]: analytical={analytical:.8f}  "
                        f"numerical={numerical:.8f}  rel_err={rel_err:.2e}")
                all_ok = False

    if all_ok:
        print("[grad_check] ✓ All analytical gradients match numerical gradients")
    else:
        print("[grad_check] ✗ Some gradients failed — see above")
    return all_ok


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Gradient check ===")
    model = Word2VecSGNS(vocab_size=20, embedding_dim=8, seed=42)
    gradient_check(model, center_idx=0, context_idx=1,
                    neg_indices=np.array([2, 3, 4]))

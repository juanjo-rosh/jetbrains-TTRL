# Mathematical Derivation — Skip-gram with Negative Sampling (SGNS)

This document provides a **complete, self-contained** mathematical derivation
of the loss function and gradients used in the SGNS implementation.  Every
equation maps directly to a line of code in `word2vec.py`.

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $V$ | Vocabulary size |
| $d$ | Embedding dimension |
| $w_c$ | The **center** (input) word |
| $w_o$ | The **context** (output) word — the positive sample |
| $w_k$ | The $k$-th **negative** sample ($k = 1, \dots, K$) |
| $\mathbf{v}_c \in \mathbb{R}^d$ | Center embedding vector for $w_c$ (row of $\mathbf{W}_{\text{center}}$) |
| $\mathbf{u}_o \in \mathbb{R}^d$ | Context embedding vector for $w_o$ (row of $\mathbf{W}_{\text{context}}$) |
| $\mathbf{u}_k \in \mathbb{R}^d$ | Context embedding vector for negative sample $w_k$ |
| $\sigma(x)$ | Sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$ |
| $K$ | Number of negative samples per positive pair |
| $\eta$ | Learning rate |

---

## 2. Objective Function

The SGNS objective **maximises** the log-probability that the model can
distinguish a true (center, context) pair from $K$ randomly drawn "noise"
pairs.  For a single training pair $(w_c, w_o)$:

$$
J = \log \sigma\!\left(\mathbf{u}_o^\top \mathbf{v}_c\right) + \sum_{k=1}^{K} \log \sigma\!\left(-\mathbf{u}_k^\top \mathbf{v}_c\right)
$$

**Interpretation as binary classification:**

* The first term says *"the positive pair should have a large positive dot
  product"* — $\sigma(\mathbf{u}_o^\top \mathbf{v}_c) \to 1$.
* Each term in the sum says *"a noise pair should have a large negative dot
  product"* — $\sigma(-\mathbf{u}_k^\top \mathbf{v}_c) \to 1$, equivalently
  $\sigma(\mathbf{u}_k^\top \mathbf{v}_c) \to 0$.

Since we implement gradient **descent**, we minimise the **negated** objective:

$$
\boxed{
\mathcal{L} = -\log \sigma\!\left(\mathbf{u}_o^\top \mathbf{v}_c\right) - \sum_{k=1}^{K} \log \sigma\!\left(-\mathbf{u}_k^\top \mathbf{v}_c\right)
}
$$

---

## 3. Prerequisite: Derivatives of the Sigmoid

Before deriving the loss gradients we need two identities.

### Identity 1 — Sigmoid derivative

$$
\frac{d\sigma}{dx}(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)
$$

### Identity 2 — Derivative of $\log\sigma$

$$
\frac{d}{dx}\log\sigma(x) = \frac{\sigma'(x)}{\sigma(x)}
= \frac{\sigma(x)(1-\sigma(x))}{\sigma(x)} = 1 - \sigma(x)
$$

### Identity 3 — Derivative of $\log\sigma(-x)$

$$
\frac{d}{dx}\log\sigma(-x)
= \frac{d}{dx}\log\sigma(-x)
= -\bigl(1 - \sigma(-x)\bigr)
= -\sigma(x)
$$

The last step uses the reflection property $1 - \sigma(-x) = \sigma(x)$.

---

## 4. Gradient Derivations

### 4.1 Gradient with respect to the center embedding $\mathbf{v}_c$

We differentiate $\mathcal{L}$ with respect to $\mathbf{v}_c$.  The key
intermediate quantity in each term is the dot product $s = \mathbf{u}^\top
\mathbf{v}_c$, for which $\frac{\partial s}{\partial \mathbf{v}_c} = \mathbf{u}$.

**Positive term:**

$$
\frac{\partial}{\partial \mathbf{v}_c}
\bigl[-\log\sigma(\mathbf{u}_o^\top \mathbf{v}_c)\bigr]
= -\bigl(1 - \sigma(\mathbf{u}_o^\top \mathbf{v}_c)\bigr)\;\mathbf{u}_o
$$

Step-by-step:
1. Let $s_+ = \mathbf{u}_o^\top \mathbf{v}_c$.
2. $\frac{\partial}{\partial \mathbf{v}_c}(-\log\sigma(s_+))
     = -\frac{d}{ds_+}\log\sigma(s_+) \cdot \frac{\partial s_+}{\partial \mathbf{v}_c}$
3. By Identity 2: $\frac{d}{ds_+}\log\sigma(s_+) = 1 - \sigma(s_+)$
4. $\frac{\partial s_+}{\partial \mathbf{v}_c} = \mathbf{u}_o$
5. Combining:  $= -(1 - \sigma(s_+))\;\mathbf{u}_o$

**Each negative term ($k$-th):**

$$
\frac{\partial}{\partial \mathbf{v}_c}
\bigl[-\log\sigma(-\mathbf{u}_k^\top \mathbf{v}_c)\bigr]
= \sigma(\mathbf{u}_k^\top \mathbf{v}_c)\;\mathbf{u}_k
$$

Step-by-step:
1. Let $s_k = \mathbf{u}_k^\top \mathbf{v}_c$.
2. $\frac{\partial}{\partial \mathbf{v}_c}(-\log\sigma(-s_k))
     = -\frac{d}{ds_k}\log\sigma(-s_k) \cdot \frac{\partial s_k}{\partial \mathbf{v}_c}$
3. By Identity 3: $\frac{d}{ds_k}\log\sigma(-s_k) = -\sigma(s_k)$
4. $\frac{\partial s_k}{\partial \mathbf{v}_c} = \mathbf{u}_k$
5. Combining: $= -(-\sigma(s_k))\;\mathbf{u}_k = \sigma(s_k)\;\mathbf{u}_k$

**Full gradient:**

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c}
= -\bigl(1 - \sigma(\mathbf{u}_o^\top \mathbf{v}_c)\bigr)\;\mathbf{u}_o
  \;+\; \sum_{k=1}^{K} \sigma(\mathbf{u}_k^\top \mathbf{v}_c)\;\mathbf{u}_k
}
$$

In code (`word2vec.py`, `train_step_batch`):

```python
grad_v_c = pos_coeff * u_o + np.sum(neg_coeff * u_neg, axis=1)  # (B, d)
```

---

### 4.2 Gradient with respect to the positive context embedding $\mathbf{u}_o$

Only the first term of $\mathcal{L}$ involves $\mathbf{u}_o$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_o}
= \frac{\partial}{\partial \mathbf{u}_o}
  \bigl[-\log\sigma(\mathbf{u}_o^\top \mathbf{v}_c)\bigr]
$$

1. Let $s_+ = \mathbf{u}_o^\top \mathbf{v}_c$.
2. $\frac{\partial s_+}{\partial \mathbf{u}_o} = \mathbf{v}_c$
3. Apply Identity 2:

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_o}
= -\bigl(1 - \sigma(\mathbf{u}_o^\top \mathbf{v}_c)\bigr)\;\mathbf{v}_c
}
$$

In code:

```python
grad_u_o = pos_coeff * v_c  # (B, d)
```

---

### 4.3 Gradient with respect to negative sample embedding $\mathbf{u}_k$

Only the $k$-th term of the negative sum involves $\mathbf{u}_k$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_k}
= \frac{\partial}{\partial \mathbf{u}_k}
  \bigl[-\log\sigma(-\mathbf{u}_k^\top \mathbf{v}_c)\bigr]
$$

1. Let $s_k = \mathbf{u}_k^\top \mathbf{v}_c$.
2. $\frac{\partial s_k}{\partial \mathbf{u}_k} = \mathbf{v}_c$
3. Apply Identity 3 and the chain rule:

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_k}
= \sigma(\mathbf{u}_k^\top \mathbf{v}_c)\;\mathbf{v}_c
}
$$

In code:

```python
grad_u_neg = neg_coeff * v_c[:, None, :]  # (B, K, d)
```

---

## 5. SGD Parameter Updates

With learning rate $\eta$, the stochastic gradient descent update for each
parameter is:

| Parameter | Update rule |
|-----------|-------------|
| $\mathbf{v}_c$ | $\mathbf{v}_c \;\leftarrow\; \mathbf{v}_c - \eta \,\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c}$ |
| $\mathbf{u}_o$ | $\mathbf{u}_o \;\leftarrow\; \mathbf{u}_o - \eta \,\frac{\partial \mathcal{L}}{\partial \mathbf{u}_o}$ |
| $\mathbf{u}_k$ (each $k$) | $\mathbf{u}_k \;\leftarrow\; \mathbf{u}_k - \eta \,\frac{\partial \mathcal{L}}{\partial \mathbf{u}_k}$ |

**Important implementation detail:** When updating via scattered indices
(many training pairs in a batch may reference the same vocabulary word), we
use `np.subtract.at` instead of plain fancy-indexing (`W[indices] -= grad`).
Plain fancy-indexing only applies the *last* write for duplicate indices,
whereas `np.subtract.at` correctly **accumulates** all updates.

---

## 6. Learning-Rate Schedule

Following the original word2vec C implementation, we linearly decay the
learning rate over the course of training:

$$
\eta_t = \eta_0 \cdot \max\!\left(1 - \frac{t}{T},\; 10^{-4}\right)
$$

where $t$ is the current step and $T$ is the total number of steps.
The floor $10^{-4} \cdot \eta_0$ prevents the learning rate from reaching
exactly zero.

---

## 7. Negative Sampling Distribution

Negative samples are drawn from a smoothed unigram distribution:

$$
P_n(w_i) = \frac{\text{count}(w_i)^{3/4}}
                 {\sum_{j=1}^{V} \text{count}(w_j)^{3/4}}
$$

The $\frac{3}{4}$ exponent was empirically chosen by Mikolov et al.
It **flattens** the unigram distribution:

* Very common words ("the", "a") are sampled *less often* than pure unigram.
* Rare words are sampled *more often*, providing more informative negatives.

---

## 8. Subsampling of Frequent Words

Before generating training pairs, each word $w_i$ in the corpus is
discarded with probability:

$$
P(\text{discard}\; w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}
$$

where $f(w_i)$ is the word's relative frequency and $t \approx 10^{-5}$.
This removes most occurrences of stop-words, improving both training speed
and embedding quality.

---

## 9. Connection to Implicit Matrix Factorisation

Levy & Goldberg (2014) proved that when SGNS is trained to convergence, the
embedding dot products implicitly factorise a **shifted PMI** matrix:

$$
\mathbf{u}_o^\top \mathbf{v}_c \;\approx\; \text{PMI}(w_c, w_o) - \log K
$$

where $\text{PMI}(w_c, w_o) = \log \frac{P(w_c, w_o)}{P(w_c)\,P(w_o)}$.

This provides a clean theoretical justification for SGNS and explains why
increasing $K$ effectively raises the "significance threshold" — only word
pairs with $\text{PMI} > \log K$ will have positive dot products.

---

## References

1. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).
   *Distributed Representations of Words and Phrases and their
   Compositionality.* NeurIPS.

2. Levy, O. & Goldberg, Y. (2014). *Neural Word Embedding as Implicit
   Matrix Factorization.* NeurIPS.

3. Rong, X. (2014). *word2vec Parameter Learning Explained.* arXiv:1411.2738.

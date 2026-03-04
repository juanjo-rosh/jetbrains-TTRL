# Interview Preparation — SGNS Word2Vec Q&A

This document prepares detailed, confident answers for the four most likely
follow-up questions a technical interviewer will ask about this implementation.

---

## Q1: "Walk me through your gradient derivation for the negative samples."

### Answer

Sure. Let me walk you through it step by step.

We start from the SGNS loss for a single (center, context) pair with $K$
negative samples:

$$
\mathcal{L}
= -\log \sigma(\mathbf{u}_o^\top \mathbf{v}_c)
  - \sum_{k=1}^{K} \log \sigma(-\mathbf{u}_k^\top \mathbf{v}_c)
$$

We want $\frac{\partial \mathcal{L}}{\partial \mathbf{u}_k}$ — the gradient
with respect to the embedding of the $k$-th negative sample.

**Step 1 — Isolate the relevant term.**
The only term in $\mathcal{L}$ that depends on $\mathbf{u}_k$ is the $k$-th
summand.  So:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_k}
= -\frac{\partial}{\partial \mathbf{u}_k}
  \log \sigma(-\mathbf{u}_k^\top \mathbf{v}_c)
$$

**Step 2 — Apply the chain rule.**
Define the intermediate scalar $s_k = \mathbf{u}_k^\top \mathbf{v}_c$.
The chain rule gives us:

$$
\frac{\partial}{\partial \mathbf{u}_k}
\log \sigma(-s_k)
= \underbrace{\frac{d}{ds_k}\log \sigma(-s_k)}_{\text{outer derivative}}
  \;\cdot\;
  \underbrace{\frac{\partial s_k}{\partial \mathbf{u}_k}}_{\text{inner derivative}}
$$

**Step 3 — Compute the outer derivative ("log-sigmoid of negative").**
Using the standard identity:

$$
\frac{d}{dx}\log\sigma(-x) = -\sigma(x)
$$

*Derivation of this identity:* $\log\sigma(-x) = \log\frac{1}{1+e^{x}} = -\log(1+e^{x})$.
Differentiating: $\frac{d}{dx}[-\log(1+e^{x})] = -\frac{e^x}{1+e^x} = -\sigma(x)$.

So the outer derivative is $-\sigma(s_k)$.

**Step 4 — Compute the inner derivative.**

$$
\frac{\partial s_k}{\partial \mathbf{u}_k}
= \frac{\partial}{\partial \mathbf{u}_k}(\mathbf{u}_k^\top \mathbf{v}_c)
= \mathbf{v}_c
$$

**Step 5 — Combine and negate.**

$$
\frac{\partial}{\partial \mathbf{u}_k}\log\sigma(-s_k)
= -\sigma(s_k) \cdot \mathbf{v}_c
$$

Negating (because of the minus sign in $\mathcal{L}$):

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_k}
= \sigma(\mathbf{u}_k^\top \mathbf{v}_c)\;\mathbf{v}_c
}
$$

**Intuition:** The gradient is proportional to $\mathbf{v}_c$ scaled by
$\sigma(\mathbf{u}_k^\top \mathbf{v}_c)$.  When the model *incorrectly*
thinks a negative pair is positive (high $\sigma$), the gradient is large,
pushing $\mathbf{u}_k$ away from $\mathbf{v}_c$.  When the model already
correctly scores the negative pair low ($\sigma \approx 0$), the gradient
vanishes and the parameters are left alone.  This is exactly the self-
correcting property we want from a well-designed loss function.

---

## Q2: "Why did you choose Negative Sampling over Hierarchical Softmax or standard Softmax?"

### Answer

The core problem is **computational cost**.  The standard softmax computes a
normalisation constant over the entire vocabulary:

$$
P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}
                       {\sum_{j=1}^{V} \exp(\mathbf{u}_j^\top \mathbf{v}_c)}
$$

This denominator requires $O(V \cdot d)$ work **per training pair**, which
is intractable when $V$ is in the hundreds of thousands or millions.

**Comparison of the three approaches:**

| Method | Cost per update | Key idea |
|--------|----------------|----------|
| Full softmax | $O(V \cdot d)$ | Exact normalisation over all words |
| Hierarchical softmax | $O(\log V \cdot d)$ | Binary tree (Huffman) decomposition of the output space |
| Negative sampling | $O(K \cdot d)$, $K \ll V$ | Replace the V-class problem with $K+1$ binary classifications |

**Why Negative Sampling wins for this assessment:**

1. **Simplicity.** NS is conceptually clean: for each positive pair, draw
   $K$ noise words and solve $K+1$ binary logistic regressions.  This is
   easy to implement, explain, and debug — which matters for a code
   assessment.

2. **Empirical quality.** Mikolov et al. showed NS produces embeddings as
   good as or better than hierarchical softmax for frequent words, and it
   has become the de facto standard in production systems (Gensim,
   fastText, etc.).

3. **Theoretical grounding.** Levy & Goldberg (2014) proved that SGNS
   implicitly factorises a shifted PMI matrix:
   $\mathbf{u}_o^\top \mathbf{v}_c \approx \text{PMI}(w_c, w_o) - \log K$.
   This gives a clear information-theoretic interpretation.

4. **Flexibility.** The noise distribution $P_n(w) \propto \text{count}(w)^{3/4}$
   is easy to control and tune.  With hierarchical softmax, the tree
   structure is fixed after construction and less amenable to tuning.

**When hierarchical softmax might be preferred:**

- **Rare words.** HS gives every word a path in the tree, so even very rare
  words get direct gradient signal.  In NS, rare words may rarely appear as
  either positives or negatives.
- **Exact gradients.** HS provides an exact (not approximate) decomposition
  of the full softmax, which can matter in some theoretical analyses.

---

## Q3: "What are the primary computational bottlenecks in your pure NumPy code? How would you optimise further?"

### Answer

There are **three main bottlenecks**, in decreasing order of severity:

### Bottleneck 1: Python-level training loop

Even though the *inner* computation (dot products, gradients) is vectorised
with NumPy, the **outer loop** — iterating over batches, generating pairs,
sampling negatives — runs in the Python interpreter ~100× slower than compiled
code.  The original word2vec by Mikolov is written in C for exactly this reason.

**Mitigation:**
- **Mini-batching** (already implemented): Process 256+ pairs per NumPy call,
  amortising the Python overhead.
- **Cython/Numba:** JIT-compile the inner training loop.  Gensim's word2vec
  uses Cython for its SGNS inner loop and achieves near-C speeds while
  remaining callable from Python.
- **C extension:** Write the entire `train_step_batch` as a C extension via
  `ctypes` or `cffi`.

### Bottleneck 2: Scattered memory access (`np.subtract.at`)

SGD updates only a few rows of the embedding matrices per step.  These
*scattered* reads and writes (`W[indices]`) don't benefit from cache-line
prefetching or SIMD vectorisation the way contiguous matrix multiplies do.
Moreover, `np.subtract.at` is even slower than plain fancy-indexing because
it must handle duplicate indices atomically (unbuffered).

**Mitigation:**
- **Larger batch sizes** concentrate updates on fewer unique indices.
- **Hogwild-style parallelism**: Run multiple threads doing SGD on shared
  embedding matrices without locks (Recht et al., 2011).  Works because
  word-embedding SGD is sparse and low-conflict.  The original word2vec
  uses this strategy.
- **Pre-sort batch** by center/context index to maximize cache locality.

### Bottleneck 3: Training-pair generation

`generate_training_pairs` iterates over every position in the corpus with a
Python `for` loop.  For a 17M-token corpus with window=5, this produces
~150M pairs and takes significant time.

**Mitigation:**
- **On-the-fly generation:** Don't materialise all pairs upfront.  Instead,
  iterate through the corpus during training and generate pairs for each
  window on the fly (as the original word2vec does).  This also saves memory.
- **Vectorise pair generation** with NumPy strides or Cython.

### Summary of optimisation hierarchy

| Level | Strategy | Expected speedup |
|-------|----------|-----------------|
| 1 (done) | Mini-batch vectorisation | 10–50× over single-pair loop |
| 2 | Pre-compute all negative samples per epoch | 2–3× |
| 3 | Cython inner loop | 20–100× over pure Python |
| 4 | Multi-threaded Hogwild | 4–8× (on 8 cores) |
| 5 | Full C rewrite | ≈ parity with original word2vec |

---

## Q4: "How does the choice of learning rate and negative sampling size affect the embeddings?"

### Answer

### Learning Rate ($\eta$)

The learning rate controls the *step size* of each SGD update.  Its impact is
both practical and theoretical:

**Too high ($\eta > 0.05$):**
- Embeddings oscillate wildly; loss may diverge.
- Dot products grow large in magnitude, saturating the sigmoid and producing
  near-zero gradients (the "saturation trap").

**Too low ($\eta < 0.001$):**
- Training converges very slowly; the model underfits given a fixed time/epoch
  budget.
- May get stuck in a poor local region if the initial learning rate never
  reaches high enough to escape.

**Best practice — linear decay:**
Starting at $\eta_0 = 0.025$ and linearly annealing to $\approx 0$ over
training is the standard schedule (used by the original word2vec and in this
implementation).  The early high LR explores broadly, while the late low LR
fine-tunes.  The floor $\eta_{\min} = \eta_0 \times 10^{-4}$ prevents the LR
from hitting exactly zero.

**Why not Adam / adaptive optimisers?**
Adaptive methods (Adam, AdaGrad) adjust per-parameter learning rates using
gradient history.  AdaGrad is actually excellent for word embeddings because
rare words get larger effective learning rates.  However, the original
word2vec uses plain SGD with linear decay and works very well — and for this
assessment, plain SGD is simpler to implement and explain correctly.

---

### Number of Negative Samples ($K$)

$K$ controls the **quality vs. speed trade-off** and has a deep connection to
the implicit PMI factorisation.

**Small $K$ (2–5):**
- Each gradient estimate is *noisier* (based on fewer contrastive examples).
- Training is *faster* (fewer dot products per pair).
- Mikolov recommends $K \in [2, 5]$ for **large** corpora, where the sheer
  number of training pairs provides enough gradient signal despite per-step
  noise.

**Large $K$ (10–20):**
- Gradient estimates are more *stable* (lower variance across mini-batches).
- Training is *slower* (more computation per pair).
- Mikolov recommends $K \in [10, 20]$ for **small** corpora.  With fewer
  training pairs, each one needs more contrastive signal to learn a good
  embedding.

**Theoretical perspective (Levy & Goldberg, 2014):**
At convergence, SGNS approximately factorises:

$$
\mathbf{u}_o^\top \mathbf{v}_c \approx \text{PMI}(w_c, w_o) - \log K
$$

Increasing $K$ shifts the factorised matrix downward by $\log K$.  This means
only word pairs with $\text{PMI} > \log K$ will have positive dot products.
In effect, **larger $K$ raises the "significance bar"** — only the strongest
co-occurrence patterns survive.  This can be beneficial (filtering noise) or
harmful (filtering genuine but weak associations).

**Practical recommendation for this codebase:**
I used $K=10$ as a default because the demonstration uses the small built-in
corpus or a truncated text8 slice.  For the full text8 (~17M tokens), $K=5$
would be sufficient and faster.

---

## Bonus Q: Why two separate embedding matrices?

SGNS maintains separate center ($\mathbf{W}_{\text{center}}$) and context
($\mathbf{W}_{\text{context}}$) matrices.  This is because words play
**asymmetric roles** as center vs. context:

- Using a single shared matrix would force $\mathbf{v}_w^\top \mathbf{v}_w$
  to be large for every word (a self-similarity bias that distorts the
  embedding space).
- Two matrices allow the dot product $\mathbf{u}_o^\top \mathbf{v}_c$ to
  encode *asymmetric* co-occurrence strength, which is the right inductive
  bias.

In practice, the final embedding is usually just $\mathbf{W}_{\text{center}}$.
Averaging $(\mathbf{W}_{\text{center}} + \mathbf{W}_{\text{context}}) / 2$
sometimes gives marginal improvement (as noted in the GloVe paper).

---

## Bonus Q: Why the 3/4 power in the noise distribution?

The Negative Sampling distribution is:

$$
P_n(w_i) = \frac{\text{count}(w_i)^{3/4}}{\sum_j \text{count}(w_j)^{3/4}}
$$

The $3/4$ exponent was determined **empirically** by Mikolov et al.
Without it (exponent = 1, pure unigram), extremely common stop-words
dominate the negative samples, providing almost no useful contrastive signal.
The $3/4$ power **flattens** the distribution:

- Common words are sampled *less* than their frequency would suggest.
- Rare words are sampled *more*, making the negatives more diverse and
  informative.

This is analogous to label smoothing: it prevents the model from only
learning to distinguish content words from stop-words, and forces it to
make finer-grained distinctions.

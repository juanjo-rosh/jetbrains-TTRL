# Word2Vec Skip-gram with Negative Sampling — Pure NumPy

A from-scratch implementation of **Skip-gram with Negative Sampling (SGNS)**
using only NumPy — no PyTorch, TensorFlow, or other ML frameworks.  Every
component of the training loop (forward pass, loss, analytical gradients, SGD
updates) is implemented explicitly, with inline comments mapping each line of
code back to the underlying mathematics.

---

## Repository Structure

| File | Description |
|------|-------------|
| [`vocab.py`](vocab.py) | Corpus loading, tokenisation, vocabulary building, Mikolov-style frequent-word subsampling, unigram^(3/4) negative-sampling distribution, and training-pair generation. |
| [`word2vec.py`](word2vec.py) | The SGNS model class — weight initialisation, numerically-stable sigmoid, batched forward/backward pass, SGD parameter updates, cosine-similarity search, and a finite-difference gradient check. |
| [`train.py`](train.py) | End-to-end training script: CLI argument parsing, data pipeline orchestration, mini-batch training loop with linear LR decay, nearest-neighbour evaluation, and embedding persistence. |
| [`math_derivation.md`](math_derivation.md) | Complete mathematical derivation of the SGNS loss function and all three gradient terms, with step-by-step chain-rule workings. |
| [`interview_prep.md`](interview_prep.md) | Detailed answers to four common follow-up interview questions (gradient walkthrough, NS vs. HS vs. softmax, NumPy bottlenecks, hyperparameter effects). |
| [`requirements.txt`](requirements.txt) | Python dependencies (just `numpy`). |

---

## Mathematical Approach

The SGNS objective for a (center, context) pair $(w_c, w_o)$ with $K$
negative samples $\{w_1, \dots, w_K\}$ is:

$$
\mathcal{L}
= -\log \sigma(\mathbf{u}_o^\top \mathbf{v}_c)
  - \sum_{k=1}^{K} \log \sigma(-\mathbf{u}_k^\top \mathbf{v}_c)
$$

Gradients are derived analytically (see [`math_derivation.md`](math_derivation.md))
and applied via mini-batch SGD with linear learning-rate decay.  A built-in
gradient check (`word2vec.py :: gradient_check`) verifies the analytical
gradients against finite-difference approximations.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run (built-in demo corpus)

```bash
python train.py
```

This trains on a small built-in corpus (enough to verify the algorithm works)
and prints nearest-neighbour results in ~30 seconds.

### 3. Run on the text8 corpus

```bash
# Download text8 first (≈ 100 MB uncompressed)
python -c "from vocab import download_text8; download_text8()"

# Train on the first 1M tokens (fast, ~2 min)
python train.py --corpus text8 --max_tokens 1000000

# Full text8 training (~17 M tokens, ~20 min on a modern CPU)
python train.py --corpus text8
```

### 4. Verify gradients

```bash
python word2vec.py
```

Runs a finite-difference gradient check on a tiny model (should print `✓`).

### 5. Smoke-test the vocabulary pipeline

```bash
python vocab.py
```

---

## Hyperparameters

| Parameter | Default | Flag | Notes |
|-----------|---------|------|-------|
| Embedding dim | 100 | `--embedding_dim` | 300 is standard for large corpora |
| Window size | 5 | `--window_size` | Dynamic window sampled from U(1, W) |
| Negative samples | 10 | `--num_negatives` | 5 for large corpora, 10–20 for small |
| Learning rate | 0.025 | `--lr` | Linearly decayed to ~0 |
| Epochs | 5 | `--epochs` | |
| Min word count | 5 | `--min_count` | Discard rare words |
| Subsampling threshold | 1e-5 | `--subsample_t` | Mikolov's frequent-word filter |
| Batch size | 256 | `--batch_size` | Larger = better NumPy utilisation |
| Random seed | 42 | `--seed` | For reproducibility |

---

## Example Output

After training on the built-in corpus:

```
============================================================
  Nearest-Neighbour Evaluation (cosine similarity)
============================================================
          king → queen (0.874), prince (0.812), kingdom (0.775), ...
         queen → king (0.874), princess (0.801), woman (0.762), ...
           man → woman (0.831), boy (0.798), young (0.712), ...
         woman → man (0.831), girl (0.805), beautiful (0.688), ...
============================================================
```

*(Exact numbers vary with random seed and corpus size.)*

---

## Key Design Decisions

1. **Mini-batching** — All dot-products and gradient computations are
   vectorised over the batch dimension using NumPy matrix operations,
   avoiding a Python-level per-pair loop.

2. **`np.subtract.at` for SGD** — Handles duplicate indices in scattered
   index updates correctly (plain fancy-indexing `W[idx] -= grad` silently
   drops duplicates).

3. **Context matrix initialised to zeros** — Standard practice that makes
   initial sigmoid outputs 0.5 (balanced gradients), while center
   embeddings are initialised from U(−0.5/d, 0.5/d).

4. **Dynamic context window** — Sampled uniformly from [1, W] for each
   center word, giving closer context words higher effective weight.

---

## References

1. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).
   *Distributed Representations of Words and Phrases and their
   Compositionality.* NeurIPS.

2. Levy, O. & Goldberg, Y. (2014). *Neural Word Embedding as Implicit
   Matrix Factorization.* NeurIPS.

3. Rong, X. (2014). *word2vec Parameter Learning Explained.* arXiv:1411.2738.

4. Recht, B., Re, C., Wright, S., & Niu, F. (2011). *Hogwild!: A Lock-Free
   Approach to Parallelizing Stochastic Gradient Descent.* NeurIPS.

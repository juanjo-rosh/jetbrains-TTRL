# Word2Vec — Skip-gram with Negative Sampling (Pure NumPy)

A clean, educational, and production-like implementation of the **word2vec
Skip-gram with Negative Sampling (SGNS)** algorithm written entirely in
**pure NumPy** — no PyTorch, TensorFlow, or other ML frameworks are used in
the training loop.

The [text8](http://mattmahoney.net/dc/textdata) corpus is fetched via
`gensim.downloader` as a convenient data source; Gensim's own `Word2Vec`
training code is **not** used anywhere.

---

## Repository Structure

```
word2vec_sgns/
├── vocab.py        Corpus loading, vocabulary, subsampling, neg-sampling table,
│                   training-pair generation
├── word2vec.py     SkipGramNegativeSampling model: forward pass, backward pass
│                   (gradients), SGD parameter updates
├── train.py        End-to-end training loop with CLI, logging, and embedding export
└── README.md       This file
```

---

## Mathematical Approach

### Objective Function

For a center word `w` with input vector **v**_w and a context word `c` with
output vector **v'**_c, the SGNS loss for K negative samples is:

```
L = −log σ(v'_c · v_w)  −  Σ_{k=1}^{K} log(1 − σ(v'_{n_k} · v_w))
```

where σ is the sigmoid function and n_1, …, n_K are drawn from the noise
distribution P_n(w) ∝ f(w)^0.75.

### Gradients (derived from scratch)

| Parameter | Gradient |
|---|---|
| Center vector **v**_w | (σ(s₊) − 1)·**v'**_c + Σ_k σ(s_k)·**v'**_{n_k} |
| Context vector **v'**_c | (σ(s₊) − 1)·**v**_w |
| Negative vector **v'**_{n_k} | σ(s_k)·**v**_w |

Only K+2 vectors are updated per step — the efficiency advantage of SGNS
over full-softmax.

### Learning Rate Schedule

Linear decay following the original C word2vec code:

```
η(t) = max(η_min,  η_0 · (1 − t / T))
```

---

## Dataset Loading

The text8 corpus is downloaded automatically by gensim on the first run:

```python
import gensim.downloader as api
sentences = list(api.load("text8"))   # ~17 M tokens
```

All subsequent preprocessing (vocabulary filtering, subsampling, pair
generation, negative sampling) is handled by `vocab.py` in pure Python/NumPy.

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy gensim
```

### 2. Run with defaults (text8, 1 epoch, dim=100)

```bash
python train.py
```

### 3. Custom configuration

```bash
python train.py \
    --embed_dim 200 \
    --window_size 5 \
    --n_negatives 10 \
    --epochs 3 \
    --lr 0.025 \
    --min_count 5 \
    --save_path embeddings_200d.npy
```

### 4. Load saved embeddings

```python
import numpy as np

embeddings = np.load("embeddings.npy")         # shape: (vocab_size, embed_dim)
vocab      = np.load("embeddings_vocab.npy")   # shape: (vocab_size,), dtype str
word2idx   = {w: i for i, w in enumerate(vocab)}
```

---

## Key Design Decisions

| Choice | Rationale |
|---|---|
| **Negative Sampling** over full Softmax | Reduces per-step cost from O(\|V\|·d) to O(K·d) |
| **3/4-power noise distribution** | Dampens very frequent words; upweights rare words as negatives |
| **Dynamic window size** | Closer context words implicitly receive more weight |
| **Subsampling frequent words** | Speeds training; improves quality of rare-word embeddings |
| **Two embedding matrices** | W_in (center) and W_out (context) — standard SGNS; W_in is used as final embeddings |
| **Linear LR decay** | Matches original C implementation; provides implicit curriculum |

---

## CLI Reference

```
usage: train.py [-h] [--dataset DATASET] [--min_count MIN_COUNT]
                [--max_vocab MAX_VOCAB] [--subsample_t SUBSAMPLE_T]
                [--embed_dim EMBED_DIM] [--window_size WINDOW_SIZE]
                [--n_negatives N_NEGATIVES] [--ns_power NS_POWER]
                [--epochs EPOCHS] [--lr LR] [--lr_min LR_MIN]
                [--log_every LOG_EVERY] [--save_path SAVE_PATH] [--seed SEED]
```

---

## References

1. Mikolov et al. (2013). *Efficient Estimation of Word Representations in
   Vector Space.* [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
2. Mikolov et al. (2013). *Distributed Representations of Words and Phrases
   and their Compositionality.* [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
3. Goldberg & Levy (2014). *word2vec Explained: Deriving Mikolov et al.'s
   Negative-Sampling Word-Embedding Method.*
   [arXiv:1402.3722](https://arxiv.org/abs/1402.3722)

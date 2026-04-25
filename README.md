# Torformer: Toroidal Embeddings for Transformers

**Satvik Prasad, Emile Anand**

Torformer places token embeddings and hidden states on a parameterized toroidal manifold, aligning the model's representation geometry with both the rotational structure of RoPE and the hierarchical structure of natural language.

Standard learned embeddings live in unconstrained $\mathbb{R}^D$, creating a geometric mismatch with RoPE, which implicitly acts on the flat torus $\mathbb{T}^{D/2}$. Torformer addresses this by natively parameterizing representations on a toroidal manifold. See [Method](#method) for the math.

---

## Pretraining Runs

### 1. Environment setup

```bash
cd nanochat/
uv sync --extra gpu        # CUDA (A100/H100)
# uv sync --extra cpu      # CPU-only / MPS
source .venv/bin/activate
```

The training scripts must be run from `nanochat/` with `torformer/` on `PYTHONPATH` so that `torus/` is importable. The run scripts handle this automatically; if invoking `base_train` directly, set:

```bash
export PYTHONPATH="/path/to/torformer:${PYTHONPATH}"
```

**Optional env vars:**

| Variable | Default | Description |
|---|---|---|
| `NANOCHAT_BASE_DIR` | `~/.cache/nanochat` | Checkpoints, data, tokenizer |
| `NANOCHAT_DTYPE` | `bfloat16` | Override compute dtype (`bfloat16`, `float16`, `float32`) |
| `WANDB_RUN` | `dummy` | Run name for wandb (set to `dummy` to disable) |

---

### 2. Data preparation

The pretraining dataset is **ClimbMix-400B** (NVIDIA), downloaded as Parquet shards from HuggingFace. The last shard (`shard_06542.parquet`) is pinned as the validation split.

```bash
# Download training shards (~100 MB each compressed)
# 8 shards is enough for ablations; 170 for a full GPT-2 capability run
python -m nanochat.dataset -n 8

# Train the BPE tokenizer (vocab_size=32768) on downloaded data
python -m scripts.tok_train

# Optional: evaluate tokenizer compression
python -m scripts.tok_eval
```

Both commands are idempotent — they skip work if outputs already exist.

---

### 3. Smoke test (CPU / MPS)

Verify the full stack without a GPU:

```bash
python -m scripts.base_train \
    --depth=4 --max-seq-len=512 --device-batch-size=1 \
    --eval-tokens=512 --core-metric-every=-1 \
    --total-batch-size=512 --num-iterations=20
```

---

### 4. The core ablation: `torus_probe.sh`

This is the primary experiment. It runs three back-to-back training runs at matched compute:

1. **Baseline** — standard `nn.Embedding`
2. **Torus B=2** — `ToroidalEmbedding` (flat torus, RoPE-compatible)
3. **Torus B=2 + ReToroidalization** — toroidal embed + re-projection at every-other layer

```bash
# From nanochat/
bash runs/torus_probe.sh

# With wandb logging
WANDB_RUN=torus-probe bash runs/torus_probe.sh
```

The script auto-detects GPU count and scales model size accordingly:

| GPUs | Depth | model_dim | ~Params | ~Time |
|---|---|---|---|---|
| 4× A100 | 12 | 1056 | 125M | 1.5 h |
| 2× A100 | 10 | 770 | 60M | 1 h |
| 1× A100 | 8 | 512 | 25M | 1 h |
| CPU | 4 | 128 | — | smoke test only |

Checkpoints are saved to `$NANOCHAT_BASE_DIR/base_checkpoints/`.

---

### 5. Manual pretraining

```bash
# Single GPU
python -m scripts.base_train [args]

# Multi-GPU (e.g. 4 GPUs)
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- [args]
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--depth` | 20 | Number of transformer layers |
| `--aspect-ratio` | 64 | `model_dim = depth × aspect_ratio` |
| `--head-dim` | 128 | Attention head dimension |
| `--max-seq-len` | 2048 | Context length |
| `--window-pattern` | `SSSL` | Sliding window pattern (`L`=full, `S`=quarter) |
| `--use-toroidal-embed` | off | Enable `ToroidalEmbedding` |
| `--torus-block-size` | 2 | Block size B (2=flat torus; >2=hierarchical) |
| `--retorus-layers` | `""` | Comma-separated layer indices, e.g. `"0,2,4,6"` |
| `--retorus-block-size` | 4 | Block size for `ReToroidalization` |
| `--num-iterations` | -1 | Steps (-1 = Chinchilla-optimal via `--target-param-data-ratio`) |
| `--target-param-data-ratio` | 12 | Tokens:params ratio for compute-optimal run length |
| `--device-batch-size` | 32 | Per-device micro-batch size |
| `--fp8` | off | FP8 training (H100+, requires torchao) |
| `--run` | `dummy` | wandb run name (`dummy` disables wandb) |
| `--resume-from-step` | -1 | Resume from checkpoint |
| `--eval-every` | 250 | Evaluate val BPB every N steps |
| `--save-every` | -1 | Save checkpoint every N steps |
| `--model-tag` | — | Label for checkpoint directory |

**Example: 125M baseline run**

```bash
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=12 --aspect-ratio=88 --head-dim=128 \
    --window-pattern=L --max-seq-len=2048 \
    --device-batch-size=32 --num-iterations=3000 \
    --eval-every=200 --model-tag=baseline \
    --run=my-run-baseline
```

**Example: same run with toroidal embedding + retorus**

```bash
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=12 --aspect-ratio=88 --head-dim=128 \
    --window-pattern=L --max-seq-len=2048 \
    --device-batch-size=32 --num-iterations=3000 \
    --eval-every=200 --model-tag=torus-b2-retorus \
    --use-toroidal-embed --torus-block-size=2 \
    --retorus-layers="0,2,4,6,8,10,11" --retorus-block-size=4 \
    --run=my-run-torus-b2-retorus
```

---

### 6. Evaluation

```bash
# Bits-per-byte on val split + DCLM CORE score
python -m scripts.base_eval --model-tag=baseline

# Interactive chat (after SFT)
python -m scripts.chat_cli --model-tag=baseline
```

---

### 7. Tests

```bash
# Run from nanochat/
PYTHONPATH=.. python -m pytest tests/test_retorus.py -v   # toroidal embedding + retorus
python -m pytest tests/test_engine.py -v                   # KV-cache engine
python -m pytest tests/ -v -m "not slow"                   # all fast tests
```

---

## Repository Structure

```
torformer/
├── torus/
│   └── embedding.py          # ToroidalEmbedding, ReToroidalization
├── nanochat/
│   ├── nanochat/             # Training library (GPT, dataloader, optimizer, ...)
│   ├── scripts/
│   │   ├── base_train.py     # Pretraining entry point
│   │   ├── base_eval.py      # BPB + DCLM CORE evaluation
│   │   └── ...
│   ├── runs/
│   │   ├── torus_probe.sh    # Core ablation: baseline vs torus vs torus+retorus
│   │   ├── speedrun.sh       # Full GPT-2 speedrun
│   │   └── ...
│   └── tests/
│       ├── test_retorus.py   # Torformer-specific tests
│       └── ...
├── plots/
│   └── plots.py              # Theory norm-bound plots
└── pyproject.toml
```

---

## Method

### Core parameterization

A rank-$D$ toroidal embedding maps radii $\vec{\rho}$ and angles $\vec{\theta}$ to a point in $\mathbb{R}^{D+1}$:

$$\vec{\tau}_D(\vec{\rho}, \vec{\theta}) = \begin{bmatrix} \sigma_1 \cos\theta_1 \\ \vdots \\ \sigma_D \cos\theta_D \\ \sigma_D \sin\theta_D \end{bmatrix}, \qquad \sigma_k = \sum_{i=1}^{k} R_i \prod_{j=i}^{k-1} \sin\theta_j$$

For large $D$, successive products of $\sin\theta$ terms cause vanishing gradients. We introduce **gain-corrected cumulative radii** with $D-1$ learnable gain factors $g_k$:

$$\tilde{\sigma}_k(\vec{\rho}, \vec{\theta}) = \sum_{i=1}^{k} \rho_i \prod_{j=i}^{k-1} g_j \sin\theta_j$$

### Blocked tori

To control hierarchical coupling depth, we use a **$B$-blocked** formulation: $D/B$ independent tori of dimension $B-1$ instead of one $(D-1)$-dimensional torus. This caps the sinusoid product chain at $\mathcal{O}(B)$.

| Block size $B$ | Behavior |
|---|---|
| $B = 2$ | Flat torus $\mathbb{T}^{D/2}$; decoupled circles; RoPE-compatible |
| $2 < B < D$ | Partial hierarchical coupling within each block |
| $B = D$ | Fully coupled geometric torus; maximum hierarchical expressivity |

### Re-toroidalization ($\mathcal{T}$)

After dense operations (attention, MLP), hidden states drift off the manifold. We define a smooth re-toroidalization layer $\mathcal{T}_D : \mathbb{R}^D \to \mathbb{R}^{D+1}$ via:

$$\vec{\rho} = W^{x\rho} \vec{x}, \qquad \theta_k = \pi - \arctan(x_k)$$

This is a smooth, injective map inserted at normalization points to periodically re-project drifted hidden states back onto the manifold.

---

## Citation

```bibtex
@article{prasad2026torformer,
  title   = {Toroidal Embeddings for Expressive, Hierarchical Self-Attention in Transformers},
  author  = {Prasad, Satvik and Anand, Emile},
  year    = {2026}
}
```

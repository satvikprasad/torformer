# Torformer: Toroidal Embeddings for Expressive, Hierarchical Self-Attention in Transformers

**Satvik Prasad, Emile Anand**

## Overview

Torformer places token embeddings and hidden states on a parameterized toroidal manifold, aligning the model's representation geometry with both the rotational structure of positional encodings (RoPE) and the hierarchical structure of natural language.

Standard learned embeddings live in unconstrained $\mathbb{R}^D$, creating a geometric mismatch with RoPE, which implicitly acts on the flat torus $\mathbb{T}^{D/2}$. This mismatch contributes to embedding norm divergence, destruction of rotational structure by dense Q/K projections, and attention sink artifacts. Torformer addresses this by natively parameterizing representations on a toroidal manifold.

## Method

### Core Parameterization

A rank-$D$ toroidal embedding maps radii $\vec{\rho}$ and angles $\vec{\theta}$ to a point in $\mathbb{R}^{D+1}$:

$$\vec{\tau}_D(\vec{\rho}, \vec{\theta}) = \begin{bmatrix} \sigma_1 \cos\theta_1 \\ \vdots \\ \sigma_D \cos\theta_D \\ \sigma_D \sin\theta_D \end{bmatrix}, \qquad \sigma_k = \sum_{i=1}^{k} R_i \prod_{j=i}^{k-1} \sin\theta_j$$

For large $D$, successive products of $\sin\theta$ terms cause vanishing gradients. We introduce **gain-corrected cumulative radii** with $D-1$ learnable gain factors $g_k$:

$$\tilde{\sigma}_k(\vec{\rho}, \vec{\theta}) = \sum_{i=1}^{k} \rho_i \prod_{j=i}^{k-1} g_j \sin\theta_j$$

Gain correction is equivalent to an affine rescaling of the original manifold and preserves homotopy type.

### Blocked Tori

To control the depth of hierarchical coupling, we use a **$B$-blocked** formulation: instead of one $(D-1)$-dimensional torus, we use $D/B$ independent tori of dimension $B-1$. This caps the sinusoid product chain at length $\mathcal{O}(B)$.

| Block size $B$ | Behavior |
|---|---|
| $B = 2$ | Flat torus $\mathbb{T}^{D/2}$; decoupled circles; standard RoPE-compatible |
| $2 < B < D$ | Partial hierarchical coupling within each block |
| $B = D$ | Fully coupled geometric torus; maximum hierarchical expressivity |

For $B = 2$, the parameterization exactly recovers a flat-torus projection, making it a drop-in replacement compatible with standard RoPE. For $B > 2$, a modified RoPE applies position-dependent rotations only to the outermost angle pair in each block, leaving inner angles free to encode content hierarchy.

### Re-Toroidalization Projection ($\mathcal{T}$)

After dense operations (attention, MLP, MoE routing), hidden states drift off the manifold. We define a smooth **re-toroidalization** layer $\mathcal{T}_D : \mathbb{R}^D \to \mathbb{R}^{D+1}$ via:

$$\vec{\rho} = W^{x\rho} \vec{x}, \qquad \theta_k = \pi - \arctan(x_k)$$

Under this reparameterization, $\mathcal{T}_D$ is a smooth, injective map that can be inserted at normalization points to periodically re-project drifted hidden states back onto the manifold.

## Experiments

1. **Block-size sweep** — Vary $B \in \{2, 4, 8, 16, 32, 64\}$ on a 125M-parameter model. Metrics: validation perplexity (OpenWebText), zero-shot accuracy (HellaSwag, PIQA, WinoGrande, ARC-Easy), wall-clock overhead, gradient norm statistics for $\theta_k$ and $g_k$.

2. **Length generalization** — Train at 512 tokens, evaluate at 1K–16K. Compare against standard RoPE, ALiBi, and YaRN across block sizes.

3. **Probing classifiers** — Freeze the best model, extract per-block angles, and train linear probes on Penn Treebank to test whether inner angles encode interpretable hierarchical features.

4. **Re-toroidalization ablation** — Test $\mathcal{T}$ insertions at various network depths for stability and generalization effects.

5. **Scaling** — If small-scale results are positive, scale to 350M–1B parameters.

## Repository Structure

```
torformer/
├── torus/
│   └── embedding.py     # ToroidalEmbedding and core _torus_map implementation
├── nanochat/            # Submodule: base decoder-only transformer (training harness)
├── main.py
└── pyproject.toml
```

## Installation

```bash
uv sync
```

Requires Python >= 3.10 and PyTorch >= 2.11.

## Usage

```python
from torus.embedding import ToroidalEmbedding

# B=2: flat torus, RoPE-compatible (default)
emb = ToroidalEmbedding(vocab_size=50257, embed_dim=768, block_size=2)

# B=8: partial hierarchical coupling
emb = ToroidalEmbedding(vocab_size=50257, embed_dim=768, block_size=8, gain=True)

x = emb(token_ids)  # (batch, seq_len, embed_dim)
```

## Citation

```bibtex
@article{prasad2026torformer,
  title   = {Toroidal Embeddings for Expressive, Hierarchical Self-Attention in Transformers},
  author  = {Prasad, Satvik and Anand, Emile},
  year    = {2026}
}
```

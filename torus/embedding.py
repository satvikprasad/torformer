"""
torformer: Toroidal Embeddings for Transformers

Implementations the parameterizations from:
    "Torformer: Toroidal Embeddings for Expressive, Hierarchical Self-Attention in Transformers"
    Prasad & Anand, 2026.
"""
import torch
import torch.nn as nn
import math

def _torus_map(rho: torch.Tensor, theta: torch.Tensor, gain: torch.Tensor | None = None) -> torch.Tensor:
    """
    Map radii and angles to a single rank D torus in R^{D+1}

    This is the core parameterization for one block.

    Args:
        rho:   (*, D)    radii per level of the torus
        theta: (*, D)    angles per level
        gain:  (D-1,)    learnable gain factors

    Returns:
        coords: (*, D+1) point on the torus in R^{D+1}
    """
    rank = rho.shape[-1] # Read the last dimension size D of the input tensors
    assert theta.shape[-1] == rank

    sin_theta = torch.sin(theta) # Broadcasted sin / cos
    cos_theta = torch.cos(theta)

    # B == 2 fast path for a simple circle
    if rank == 1:
        return torch.cat([ rho * cos_theta, rho * sin_theta ], dim = -1)

    if gain is not None:
        assert gain.shape == (rank - 1,), f"gain shape {gain.shape} != ({rank - 1},)"
        g_sin = gain * sin_theta[..., -1]
    else:
        g_sin = sin_theta[..., -1]

    sigma_list = [rho[..., 0:1]] # sigma_1 = rho_1, shape (*, 1)
    for k in range(1, rank):
        s_prev = sigma_list[k - 1]
        s_new = s_prev * g_sin[..., k - 1 : k] + rho[..., k : k + 1]
        sigma_list.append(s_new)
    sigma = torch.cat(sigma_list, dim=-1)

    coords_main = sigma * cos_theta
    coord_last = sigma[..., -1:] * sin_theta[..., -1:]

    return torch.cat([coords_main, coord_last], dim = -1)

class ToroidalEmbedding(nn.Module):
    """
    Token embedding on a B-blocked toroidal manifold.

    Each token is parameterized by (rho, theta) which are mapped
    throughout D/B independent rank-(B-1) tori, producing D
    output dimensions.

    Args:
        vocab_size: number of tokens
        embed_dim:  model dimension D (= n_embed in GPTConfig)
        block_size: B, the blocking factor. Must divide embed_dim
        gain:       whether to use learnable gain factors or not
    """
    def __init__(self, vocab_size: int, embed_dim: int, block_size: int = 2, gain: bool = True):
        super().__init__()
        assert embed_dim % block_size == 0, (
            f"embed_dim ({embed_dim}) must be divisible by block_size ({block_size})"
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.n_blocks = embed_dim // block_size
        self.rank = block_size - 1 # rank of each torus block

        # counts the total number of (rho, theta) parameters per token:
        self.params_per_token = self.n_blocks * self.rank

        self.rho = nn.Parameter(torch.empty(vocab_size, self.params_per_token))
        self.theta = nn.Parameter(torch.empty(vocab_size, self.params_per_token))

        self.gain_factors = None
        if gain and self.rank > 1:
            # We have n_blocks*(B-1) gain factors in total
            self.gain_factors = nn.Parameter(torch.ones(self.n_blocks, self.rank - 1))

        self._init_parameters()

    def _init_parameters():
        """Initialize radii and angles"""
        # TODO(satvik): Verify that negative rho make geometric sense here
        std = 0.8 / math.sqrt(self.rank) # Chosen to match typical std (~0.8)
                                         # Scaled down per-block to keep output
                                         # norm reasonable.
        nn.init.normal_(self.rho, mean=0.0, std=std)

        nn.init.uniform_(self.theta, 0.0, 2 * math.pi) # Initialize angles uniformly

        # TODO(satvik): Compute the optimal initialization pattern here
        if self.gain_factors is not None:
            nn.init.ones_(self.gain_factors)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Look up toroidal embeddings for each token index

        Args:
            idx: (batch_size, T) integer token indices

        Returns:
            x: (batch_size, T, embed_dim) toroidal coordinates
        """
        rho   = self.rho[idx]      # (batch_size, T, n_blocks * rank)
        theta = self.theta[idx]    # (batch_size, T, n_blocks * rank)

        batch_shape = idx.shape
        rho = rho.view(*batch_shape, self.n_blocks, self.rank)
        theta = theta.view(*batch_shape, self.n_blocks, self.rank)

        blocks = []
        for b in range(self.n_blocks):
            rho_b   = rho[..., b, :] # (B, T, rank)
            theta_b = theta[..., b, :] # (B, T, rank)
            gain_b  = self.gain_factors[b] if self.gain_factors is not None else None

            coords_b = _torus_map(rho_b, theta_b, gain_b)
            blocks.append(coords_b)

        x = torch.cat(blocks, dim=-1)
        return x
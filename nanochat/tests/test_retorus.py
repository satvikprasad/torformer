"""
Smoke tests for ReToroidalization and the supporting torus machinery.

Run from torformer/nanochat/:
    PYTHONPATH=.. python -m pytest tests/test_retorus.py -v
"""

import math
import pytest
import torch
import torch.nn as nn
import sys
import os

# Make the torus package importable when running from nanochat/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from torus.embedding import (
    _torus_map,
    _torus_map_weierstrass,
    ToroidalEmbedding,
    ReToroidalization,
)
from nanochat.gpt import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gpt(n_layer=4, n_embd=128, retorus_layers=(), retorus_block_size=4,
              use_toroidal_embed=False, torus_block_size=2):
    cfg = GPTConfig(
        n_layer=n_layer, n_head=4, n_kv_head=4, n_embd=n_embd,
        sequence_len=64, vocab_size=256,
        use_toroidal_embed=use_toroidal_embed,
        torus_block_size=torus_block_size,
        retorus_layers=retorus_layers,
        retorus_block_size=retorus_block_size,
    )
    with torch.device('meta'):
        model = GPT(cfg)
    model.to_empty(device='cpu')
    model.init_weights()
    return model


# ---------------------------------------------------------------------------
# _torus_map_weierstrass: shape contracts
# ---------------------------------------------------------------------------

class TestTorusMapWeierstrass:

    def test_rank1_shape(self):
        rho = torch.randn(3, 8, 1)
        phi = torch.randn(3, 8, 1)
        out = _torus_map_weierstrass(rho, phi)
        assert out.shape == (3, 8, 2)

    def test_rank3_shape(self):
        rho = torch.randn(2, 10, 3)
        phi = torch.randn(2, 10, 3)
        out = _torus_map_weierstrass(rho, phi)
        assert out.shape == (2, 10, 4)

    def test_flat_batch_shape(self):
        rho = torch.randn(7, 5)
        phi = torch.randn(7, 5)
        out = _torus_map_weierstrass(rho, phi)
        assert out.shape == (7, 6)

    def test_gain_shape(self):
        rank = 4
        rho  = torch.randn(2, rank)
        phi  = torch.randn(2, rank)
        gain = torch.ones(rank - 1)
        out  = _torus_map_weierstrass(rho, phi, gain)
        assert out.shape == (2, rank + 1)

    def test_agrees_with_torus_map(self):
        """Weierstrass form must match _torus_map when phi = tan(theta/2)."""
        torch.manual_seed(0)
        rho   = torch.randn(5, 3)
        theta = torch.rand(5, 3) * 2 * math.pi - math.pi   # in (-pi, pi)
        phi   = torch.tan(theta / 2)
        out_direct = _torus_map(rho, theta)
        out_weier  = _torus_map_weierstrass(rho, phi)
        assert torch.allclose(out_direct, out_weier, atol=1e-5), (
            f"max diff: {(out_direct - out_weier).abs().max():.2e}"
        )

    def test_gradient_flows(self):
        rho = torch.randn(4, 3, requires_grad=True)
        phi = torch.randn(4, 3, requires_grad=True)
        out = _torus_map_weierstrass(rho, phi)
        out.sum().backward()
        assert rho.grad is not None
        assert phi.grad is not None
        assert not rho.grad.isnan().any()
        assert not phi.grad.isnan().any()

    def test_no_nan_at_phi_zero(self):
        """phi=0 => theta=0 => cos=1, sin=0. Should not produce NaN."""
        rho = torch.randn(4, 3)
        phi = torch.zeros(4, 3)
        out = _torus_map_weierstrass(rho, phi)
        assert not out.isnan().any()

    def test_no_nan_at_large_phi(self):
        """Large phi values should not blow up due to the 1/(1+phi^2) denominator."""
        rho = torch.randn(4, 3)
        phi = torch.tensor([[1e6, -1e6, 0.5]] * 4)
        out = _torus_map_weierstrass(rho, phi)
        assert not out.isnan().any()
        assert not out.isinf().any()


# ---------------------------------------------------------------------------
# ReToroidalization: shape, parameters, gradient
# ---------------------------------------------------------------------------

class TestReToroidalization:

    @pytest.mark.parametrize("B", [2, 4, 8])
    def test_shape_preserved(self, B):
        D = 64
        layer = ReToroidalization(embed_dim=D, block_size=B)
        x = torch.randn(2, 16, D)
        y = layer(x)
        assert y.shape == x.shape

    def test_shape_unbatched(self):
        layer = ReToroidalization(embed_dim=32, block_size=4)
        x = torch.randn(32)
        y = layer(x)
        assert y.shape == (32,)

    def test_shape_3d_batch(self):
        layer = ReToroidalization(embed_dim=64, block_size=4)
        x = torch.randn(3, 5, 64)
        y = layer(x)
        assert y.shape == (3, 5, 64)

    def test_gradient_flows_through_layer(self):
        layer = ReToroidalization(embed_dim=64, block_size=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None
        assert not x.grad.isnan().any()

    def test_parameter_gradients(self):
        layer = ReToroidalization(embed_dim=64, block_size=4, gain=True)
        x = torch.randn(2, 8, 64)
        y = layer(x)
        y.sum().backward()
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert not p.grad.isnan().any(), f"{name} grad has NaN"

    def test_shared_projections_shape(self):
        layer = ReToroidalization(embed_dim=64, block_size=4, shared_projections=True)
        x = torch.randn(2, 8, 64)
        y = layer(x)
        assert y.shape == x.shape

    def test_shared_projections_fewer_params(self):
        D, B = 128, 4
        per_block = ReToroidalization(D, block_size=B, gain=False, shared_projections=False)
        shared    = ReToroidalization(D, block_size=B, gain=False, shared_projections=True)
        n_per    = sum(p.numel() for p in per_block.parameters())
        n_shared = sum(p.numel() for p in shared.parameters())
        assert n_shared < n_per, f"shared ({n_shared}) should be < per_block ({n_per})"

    def test_no_gain_b2(self):
        """B=2 rank=1: gain is skipped (only possible for rank>1), no error."""
        layer = ReToroidalization(embed_dim=32, block_size=2, gain=True)
        assert layer.gain_factors is None   # rank=1, gain irrelevant
        x = torch.randn(2, 8, 32)
        y = layer(x)
        assert y.shape == x.shape

    @pytest.mark.parametrize("D,B", [(32, 2), (64, 4), (128, 8), (256, 16)])
    def test_various_embed_block_combos(self, D, B):
        layer = ReToroidalization(D, B)
        x = torch.randn(1, 4, D)
        assert layer(x).shape == (1, 4, D)

    def test_embed_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            ReToroidalization(embed_dim=65, block_size=4)

    def test_output_finite(self):
        """Output should never contain NaN or Inf for normal inputs."""
        torch.manual_seed(42)
        layer = ReToroidalization(embed_dim=128, block_size=4, gain=True)
        x = torch.randn(4, 32, 128)
        y = layer(x)
        assert not y.isnan().any()
        assert not y.isinf().any()


# ---------------------------------------------------------------------------
# GPT integration: retorus_layers wiring
# ---------------------------------------------------------------------------

class TestGPTRetorus:

    def test_no_retorus_baseline(self):
        model = _make_gpt(n_layer=4, retorus_layers=())
        counts = model.num_scaling_params()
        assert counts['retorus'] == 0

    def test_retorus_params_nonzero(self):
        model = _make_gpt(n_layer=4, retorus_layers=(1, 3), retorus_block_size=4)
        counts = model.num_scaling_params()
        assert counts['retorus'] > 0

    def test_retorus_layers_attached(self):
        model = _make_gpt(n_layer=4, retorus_layers=(0, 2), retorus_block_size=4)
        for i, block in enumerate(model.transformer.h):
            if i in (0, 2):
                assert block.retorus is not None, f"layer {i} should have retorus"
            else:
                assert block.retorus is None, f"layer {i} should not have retorus"

    def test_forward_backward(self):
        model = _make_gpt(n_layer=4, retorus_layers=(1, 3), retorus_block_size=4)
        idx = torch.randint(0, 256, (2, 16))
        loss = model(idx, targets=idx)
        loss.backward()
        assert not loss.isnan()

    def test_forward_backward_toroidal_embed(self):
        model = _make_gpt(
            n_layer=4, retorus_layers=(0, 2), retorus_block_size=4,
            use_toroidal_embed=True, torus_block_size=2,
        )
        idx = torch.randint(0, 256, (2, 16))
        loss = model(idx, targets=idx)
        loss.backward()
        assert not loss.isnan()

    def test_forward_all_layers_retorus(self):
        model = _make_gpt(n_layer=4, retorus_layers=(0, 1, 2, 3), retorus_block_size=4)
        idx = torch.randint(0, 256, (2, 16))
        loss = model(idx, targets=idx)
        loss.backward()
        assert not loss.isnan()

    def test_optimizer_setup(self):
        model = _make_gpt(n_layer=4, retorus_layers=(1, 3), retorus_block_size=4)
        opt = model.setup_optimizer()
        # All parameters must appear in exactly one group
        all_ids = [id(p) for group in opt.param_groups for p in group['params']]
        assert len(all_ids) == len(set(all_ids)), "duplicate parameters in optimizer groups"
        assert len(set(all_ids)) == sum(1 for _ in model.parameters()), \
            "not all model parameters are in the optimizer"

    def test_optimizer_retorus_in_adamw(self):
        """Retorus params must land in an adamw group, not muon."""
        model = _make_gpt(n_layer=4, retorus_layers=(1,), retorus_block_size=4)
        retorus_param_ids = {
            id(p)
            for block in model.transformer.h if block.retorus is not None
            for p in block.retorus.parameters()
        }
        opt = model.setup_optimizer()
        for group in opt.param_groups:
            if group.get('kind') == 'muon':
                for p in group['params']:
                    assert id(p) not in retorus_param_ids, \
                        "retorus param found in muon group"

    def test_param_count_consistent(self):
        model = _make_gpt(n_layer=4, retorus_layers=(0, 2), retorus_block_size=4)
        counts = model.num_scaling_params()
        assert counts['total'] == sum(p.numel() for p in model.parameters())

    def test_retorus_grad_updates(self):
        """Retorus W_rho receives a nonzero gradient; a manual SGD step changes it."""
        model = _make_gpt(n_layer=4, retorus_layers=(1,), retorus_block_size=4)
        block = model.transformer.h[1]

        idx = torch.randint(0, 256, (2, 16))
        loss = model(idx, targets=idx)
        loss.backward()

        W_rho = block.retorus.W_rho
        assert W_rho.grad is not None, "W_rho has no gradient"
        assert W_rho.grad.abs().sum() > 0, "W_rho gradient is all zeros"

        # Manual SGD step to confirm the param actually changes
        W_before = W_rho.data.clone()
        with torch.no_grad():
            W_rho -= 0.01 * W_rho.grad
        assert not torch.equal(W_before, W_rho.data), "W_rho did not change after gradient step"


# ---------------------------------------------------------------------------
# CLI arg parsing: retorus_layers string -> tuple
# ---------------------------------------------------------------------------

class TestRetorusLayersParsing:
    """Verify the parsing logic used in base_train.py."""

    def _parse(self, s):
        return tuple(int(x) for x in s.split(',') if x.strip())

    def test_empty_string(self):
        assert self._parse("") == ()

    def test_single_layer(self):
        assert self._parse("3") == (3,)

    def test_multiple_layers(self):
        assert self._parse("0,2,4,6") == (0, 2, 4, 6)

    def test_spaces_ignored(self):
        assert self._parse("1, 3, 5") == (1, 3, 5)

    def test_last_layer(self):
        n_layer = 12
        s = ",".join(str(i) for i in range(0, n_layer, 2))
        layers = self._parse(s)
        assert all(l < n_layer for l in layers)

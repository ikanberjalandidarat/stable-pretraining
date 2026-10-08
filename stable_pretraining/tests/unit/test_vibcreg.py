"""Single-process unit tests for the VIbCReg building blocks.

Covers :class:`IterNorm` (whitening quality, running statistics, eval-mode
behavior, dtype handling, gradients) and :class:`VIbCRegLoss` (closed-form
values on collapsed and duplicated features, scale invariance of the
decorrelation term, coefficient weighting). Under a single process the
cross-GPU gather in the loss is a no-op.
"""

import math

import pytest
import torch

from stable_pretraining.backbone import IterNorm
from stable_pretraining.losses import VIbCRegLoss


def _max_group_whitening_error(y: torch.Tensor, num_groups: int) -> float:
    """Largest deviation of any per-group covariance of ``y`` from identity."""
    n, d = y.shape
    yg = y.T.reshape(num_groups, d // num_groups, n)
    yc = yg - yg.mean(dim=-1, keepdim=True)
    cov = yc @ yc.transpose(1, 2) / n
    eye = torch.eye(d // num_groups, dtype=y.dtype)
    return (cov - eye).abs().max().item()


def _correlated(n: int, d: int, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return (torch.randn(n, d, generator=g) @ torch.randn(d, d, generator=g) + 3).to(
        dtype
    )


@pytest.mark.unit
class TestIterNorm:
    """Unit tests for the IterNorm whitening layer."""

    def test_whitens_each_group(self):
        """With enough iterations every group covariance converges to identity."""
        x = _correlated(512, 64, torch.float64)
        layer = IterNorm(64, num_groups=4, T=10, affine=False).double().train()

        assert _max_group_whitening_error(x, 4) > 10
        assert _max_group_whitening_error(layer(x), 4) < 1e-4

    def test_default_iterations_reduce_correlation(self):
        """The default T=5 is an approximate whitening, far better than the input."""
        x = _correlated(512, 64)
        y = IterNorm(64, num_groups=4, affine=False).train()(x)

        assert _max_group_whitening_error(y, 4) < 0.5
        assert y.mean(dim=0).abs().max() < 1e-4

    def test_running_statistics_update_only_in_train(self):
        """Train mode updates running buffers; eval mode leaves them untouched."""
        layer = IterNorm(16, num_groups=2)
        x = _correlated(64, 16)

        layer.train()
        layer(x)
        assert not torch.equal(layer.running_wm, torch.eye(8).repeat(2, 1, 1)), (
            "running whitening matrix should move away from identity"
        )

        mean, wm = layer.running_mean.clone(), layer.running_wm.clone()
        layer.eval()
        layer(x)
        assert torch.equal(layer.running_mean, mean)
        assert torch.equal(layer.running_wm, wm)

    def test_eval_is_per_sample(self):
        """In eval mode each output depends only on its own input row."""
        layer = IterNorm(16, num_groups=2).train()
        x = _correlated(64, 16)
        layer(x)
        layer.eval()

        assert torch.allclose(layer(x)[:5], layer(x[:5]), atol=1e-6)

    def test_gradcheck(self):
        """Autograd gradients through the Newton-Schulz iterations are correct."""
        torch.manual_seed(0)
        layer = IterNorm(8, num_groups=2, T=3).double().train()
        x = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(layer, (x,))

    def test_affine_parameters_receive_gradients(self):
        """The learnable scale and shift are part of the computation graph."""
        layer = IterNorm(16, num_groups=2).train()
        layer(_correlated(32, 16)).pow(2).sum().backward()

        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
        assert layer.bias.grad is not None and layer.bias.grad.abs().sum() > 0

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_low_precision_input_is_whitened_in_float32(self, dtype):
        """Half-precision inputs are upcast so the iterations stay finite."""
        y = IterNorm(16, num_groups=2).train()(_correlated(64, 16, dtype))

        assert y.dtype == torch.float32
        assert torch.isfinite(y).all()

    def test_float64_module(self):
        """A float64 module keeps float64 buffers and outputs."""
        layer = IterNorm(16, num_groups=2).double().train()
        y = layer(_correlated(64, 16, torch.float64))

        assert y.dtype == torch.float64
        assert layer.running_wm.dtype == torch.float64

    def test_rejects_indivisible_groups(self):
        with pytest.raises(ValueError, match="divisible"):
            IterNorm(10, num_groups=3)

    def test_rejects_wrong_input_shape(self):
        layer = IterNorm(16, num_groups=2)
        with pytest.raises(ValueError, match=r"\[N, 16\]"):
            layer(torch.randn(4, 8))
        with pytest.raises(ValueError, match=r"\[N, 16\]"):
            layer(torch.randn(4, 16, 1))


@pytest.mark.unit
class TestVIbCRegLoss:
    """Unit tests for the VIbCRegLoss function (single process)."""

    def test_collapsed_embeddings(self):
        """Constant embeddings only pay the variance hinge, once per view."""
        z = torch.ones(8, 4)

        loss = VIbCRegLoss()(z, z)

        expected = 25 * 2 * (1 - math.sqrt(1e-4))
        assert torch.allclose(loss, torch.tensor(expected))

    def test_duplicated_feature_decorrelation(self):
        """Two copies of one feature have correlation 1 in both off-diagonals."""
        a = torch.randn(32, 1)
        z = torch.cat([a, a], dim=1)

        # 2 off-diagonal entries equal to 1, averaged over 2 * 2 entries
        assert torch.allclose(VIbCRegLoss._decorrelation(z), torch.tensor(0.5))

    def test_decorrelation_is_scale_invariant(self):
        """Unlike VICReg's covariance term, rescaling features has no effect."""
        z = _correlated(256, 32)

        base = VIbCRegLoss._decorrelation(z)
        assert torch.allclose(VIbCRegLoss._decorrelation(z * 1e3 - 7), base, atol=1e-6)
        assert torch.allclose(VIbCRegLoss._decorrelation(z * 1e-3), base, atol=1e-6)

    def test_decorrelation_lower_for_independent_features(self):
        torch.manual_seed(0)
        independent = VIbCRegLoss._decorrelation(torch.randn(1024, 32))
        correlated = VIbCRegLoss._decorrelation(_correlated(1024, 32))

        assert independent < 0.01 < correlated

    def test_coefficients_weight_each_term(self):
        """The loss is the weighted sum of invariance, variance and decorrelation."""
        torch.manual_seed(0)
        z_i, z_j = torch.randn(64, 16), torch.randn(64, 16) * 0.5

        sim = VIbCRegLoss(sim_coeff=1, std_coeff=0, cov_coeff=0)(z_i, z_j)
        std = VIbCRegLoss(sim_coeff=0, std_coeff=1, cov_coeff=0)(z_i, z_j)
        cov = VIbCRegLoss(sim_coeff=0, std_coeff=0, cov_coeff=1)(z_i, z_j)

        assert torch.allclose(sim, torch.nn.functional.mse_loss(z_i, z_j))
        assert std > 0 and cov > 0
        assert torch.allclose(
            VIbCRegLoss()(z_i, z_j), 25 * sim + 25 * std + 200 * cov, rtol=1e-5
        )

    def test_gradients_flow_to_both_views(self):
        torch.manual_seed(0)
        z_i = torch.randn(32, 16, requires_grad=True)
        z_j = torch.randn(32, 16, requires_grad=True)

        loss = VIbCRegLoss()(z_i, z_j)
        loss.backward()

        assert loss.ndim == 0 and torch.isfinite(loss)
        assert z_i.grad.abs().sum() > 0 and z_j.grad.abs().sum() > 0

    def test_projector_with_iternorm_trains(self):
        """End-to-end: a projector ending in IterNorm reduces the loss."""
        torch.manual_seed(0)
        projector = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            IterNorm(64, num_groups=8),
        )
        loss_fn = VIbCRegLoss()
        opt = torch.optim.Adam(projector.parameters(), lr=1e-2)
        x = torch.randn(128, 32)

        losses = []
        for _ in range(30):
            loss = loss_fn(projector(x + 0.1 * torch.randn_like(x)), projector(x))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert all(math.isfinite(v) for v in losses)
        assert losses[-1] < losses[0]

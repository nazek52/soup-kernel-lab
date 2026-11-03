"""CPU checks exercise fallbacks only. CUDA cases exercise the actual Triton kernels."""

import pytest
import torch
import torch.nn.functional as F

from solution.ops import cross_entropy, rmsnorm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("vocab", [31, 4099, 151936])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_ce_loss_and_gradient(device, vocab, reduction):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Triton requires CUDA; CPU results do not validate kernels")
    torch.manual_seed(42)
    # Strided input and irregular vocabulary exercise contiguous conversion and tail masks.
    x = (torch.randn(11, vocab * 2, device=device) * 4)[:, ::2].requires_grad_()
    labels = torch.randint(0, vocab, (11,), device=device)
    labels[::3] = -100
    ref = F.cross_entropy(x, labels, reduction=reduction)
    grad = torch.randn_like(ref)
    dx_ref = torch.autograd.grad(ref, x, grad)[0]
    actual = cross_entropy(x, labels, reduction=reduction)
    dx = torch.autograd.grad(actual, x, grad)[0]
    torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(dx, dx_ref, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_ce_all_ignored(device, reduction):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    x = torch.randn(4, 77, device=device, requires_grad=True)
    labels = torch.full((4,), -100, device=device)
    ref = F.cross_entropy(x, labels, reduction=reduction)
    actual = cross_entropy(x, labels, reduction=reduction)
    torch.testing.assert_close(actual, ref, equal_nan=True)
    dx = torch.autograd.grad(actual.sum(), x)[0]
    assert torch.count_nonzero(dx) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("width", [77, 1536])
def test_frozen_rmsnorm(device, dtype, width):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(71)
    x = torch.randn(2, 7, width, device=device, dtype=dtype).requires_grad_()
    weight = torch.randn(width, device=device, dtype=torch.float32)
    xf = x.float()
    ref = weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(dtype)
    grad = torch.randn_like(ref)
    dx_ref = torch.autograd.grad(ref, x, grad)[0]
    actual = rmsnorm(x, weight, 1e-6)
    dx = torch.autograd.grad(actual, x, grad)[0]
    assert actual.dtype == ref.dtype
    tol = 3e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, ref, atol=tol, rtol=tol)
    torch.testing.assert_close(dx, dx_ref, atol=tol, rtol=tol)


def test_trainable_norm_retains_weight_gradient():
    x = torch.randn(2, 17, requires_grad=True)
    w = torch.randn(17, requires_grad=True)
    rmsnorm(x, w, 1e-6).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_cpu_invalid_label_rejected():
    with pytest.raises((IndexError, RuntimeError)):
        cross_entropy(torch.randn(2, 4), torch.tensor([0, 5]))

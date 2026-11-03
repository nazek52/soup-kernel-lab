"""Execute our actual JIT bodies on CPU via the official Triton interpreter.

These tests validate arithmetic and indexing, not CUDA compilation, races or speed.
Run in Linux: TRITON_INTERPRET=1 python -m pytest dev_tests/test_interpreter.py -q
"""

import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1",
                                reason="Set TRITON_INTERPRET=1 on Linux with Triton")


@pytest.mark.parametrize("vocab", [31, 4099, 151936])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_actual_ce_kernel_bodies(vocab, all_ignored):
    from solution.ops import _CrossEntropyRows

    torch.manual_seed(7)
    x = torch.randn(5, vocab, requires_grad=True)
    labels = torch.randint(0, vocab, (5,))
    labels[::2] = -100
    if all_ignored:
        labels.fill_(-100)
    ref = F.cross_entropy(x, labels, reduction="none")
    actual = _CrossEntropyRows.apply(x, labels, -100)
    grad = torch.randn_like(ref)
    dx_ref = torch.autograd.grad(ref, x, grad)[0]
    dx = torch.autograd.grad(actual, x, grad)[0]
    torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(dx, dx_ref, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("width", [77, 1536])
def test_actual_rms_kernel_bodies(dtype, width):
    from solution.ops import _FrozenRMSNorm

    torch.manual_seed(8)
    x = torch.randn(2, 3, width, dtype=dtype).requires_grad_()
    weight = torch.randn(width, dtype=torch.float32)
    xf = x.float()
    ref = weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(dtype)
    actual = _FrozenRMSNorm.apply(x, weight, 1e-6)
    grad = torch.randn_like(ref)
    dx_ref = torch.autograd.grad(ref, x, grad)[0]
    dx = torch.autograd.grad(actual, x, grad)[0]
    tol = 3e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, ref, atol=tol, rtol=tol)
    torch.testing.assert_close(dx, dx_ref, atol=tol, rtol=tol)

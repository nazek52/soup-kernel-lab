"""Autograd bridges and guarded patches for the pinned transformers version."""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class _CrossEntropyRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index):
        import triton

        from .kernels import ce_row_loss, ce_tile_lse

        logits, labels = logits.contiguous(), labels.contiguous()
        rows, vocab = logits.shape
        # Invalid labels must fail, never silently produce a different training objective.
        valid = (labels == ignore_index) | ((labels >= 0) & (labels < vocab))
        torch._assert_async(valid.all(), "Class index out of bounds")
        tiles = triton.cdiv(vocab, 4096)
        part = torch.empty((rows, tiles), device=logits.device, dtype=torch.float32)
        lse = torch.empty(rows, device=logits.device, dtype=torch.float32)
        losses = torch.empty_like(lse)
        ce_tile_lse[(rows, tiles)](logits, part, vocab, tiles, 4096)
        ce_row_loss[(rows,)](logits, labels, part, lse, losses, vocab, tiles,
                             ignore_index, triton.next_power_of_2(tiles))
        ctx.save_for_backward(logits, labels, lse)
        ctx.ignore_index = ignore_index
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        import triton

        from .kernels import ce_tile_backward

        logits, labels, lse = ctx.saved_tensors
        rows, vocab = logits.shape
        dx = torch.empty_like(logits)
        ce_tile_backward[(rows, triton.cdiv(vocab, 4096))](
            logits, labels, lse, grad.contiguous(), dx, vocab, ctx.ignore_index, 4096
        )
        return dx, None, None


def cross_entropy(logits, labels, ignore_index=-100, reduction="mean"):
    """Index-label CE, no weights/smoothing; other shapes and CPU use PyTorch."""
    supported = (
        isinstance(logits, torch.Tensor) and isinstance(labels, torch.Tensor)
        and logits.is_cuda and labels.device == logits.device
        and logits.dtype == torch.float32 and labels.dtype == torch.long
        and logits.ndim == 2 and labels.ndim == 1
        and logits.shape[0] == labels.shape[0] and logits.numel() > 0
        and reduction in ("none", "mean", "sum")
    )
    if not supported:
        return F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    losses = _CrossEntropyRows.apply(logits, labels, ignore_index)
    if reduction == "none":
        return losses
    total = losses.sum()
    if reduction == "sum":
        return total
    count = (labels != ignore_index).sum()
    # Match PyTorch: all ignored => NaN loss but zero input gradients.
    mean = total / count.clamp_min(1)
    return torch.where(count > 0, mean, torch.full_like(mean, float("nan")))


class _FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        import triton

        from .kernels import rms_forward

        x = x.contiguous()
        width = x.shape[-1]
        rows = x.numel() // width
        dtype = torch.promote_types(x.dtype, weight.dtype)
        out = torch.empty(x.shape, dtype=dtype, device=x.device)
        inv = torch.empty(rows, dtype=torch.float32, device=x.device)
        rms_forward[(rows,)](x, weight, out, inv, width, eps,
                              triton.next_power_of_2(width), enable_fp_fusion=False)
        ctx.save_for_backward(x, weight, inv)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        import triton

        from .kernels import rms_backward

        x, weight, inv = ctx.saved_tensors
        width = x.shape[-1]
        dx = torch.empty_like(x)
        rms_backward[(x.numel() // width,)](
            x, weight, inv, grad.contiguous(), dx, width,
            triton.next_power_of_2(width), enable_fp_fusion=False
        )
        return dx, None, None


def rmsnorm(x, weight, eps):
    supported = (
        x.is_cuda and weight.device == x.device and not weight.requires_grad
        and weight.dtype == torch.float32 and weight.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.ndim >= 1 and 0 < x.shape[-1] <= 8192 and x.numel() > 0
        and weight.shape == (x.shape[-1],)
    )
    if supported:
        return _FrozenRMSNorm.apply(x, weight, eps)
    xf = x.float()
    return weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def install_patches():
    from transformers.loss import loss_utils
    from transformers.models.qwen2 import modeling_qwen2

    original_loss = loss_utils.fixed_cross_entropy
    if not getattr(original_loss, "_soup_candidate", False):
        def loss(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
            # Match fixed_cross_entropy's normalization, including TRL's item count.
            if not isinstance(source, torch.Tensor) or not source.is_cuda:
                return original_loss(source, target, num_items_in_batch, ignore_index, **kwargs)
            reduction = "sum" if num_items_in_batch is not None else "mean"
            value = cross_entropy(source, target, ignore_index, reduction)
            if num_items_in_batch is not None:
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(value.device)
                value = value / num_items_in_batch
            return value

        loss._soup_candidate = True
        loss_utils.fixed_cross_entropy = loss

    original_norm = modeling_qwen2.Qwen2RMSNorm.forward
    if not getattr(original_norm, "_soup_candidate", False):
        def norm(self, hidden_states):
            if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda:
                return original_norm(self, hidden_states)
            return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

        norm._soup_candidate = True
        modeling_qwen2.Qwen2RMSNorm.forward = norm

    # Keep the official example as an attributed additional optimization.
    # It never counts towards our required 5% kernel share.
    if torch.cuda.is_available() and not getattr(
        modeling_qwen2.Qwen2MLP.forward, "_soup_example_installed", False
    ):
        from .swiglu import patch_qwen2_mlp

        patch_qwen2_mlp()
        modeling_qwen2.Qwen2MLP.forward._soup_example_installed = True

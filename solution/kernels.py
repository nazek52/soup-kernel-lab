"""Original candidate kernels. GPU correctness and speed require a Kaggle T4 run.

Derivations: log-sum-exp/softmax derivative and the RMSNorm chain rule.
No starter SwiGLU code is used here. All reductions accumulate in float32.
"""

import triton
import triton.language as tl


@triton.jit
def ce_tile_lse(X, PART, V: tl.constexpr, TILES: tl.constexpr, B: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * B + tl.arange(0, B)
    x = tl.load(X + row * V + col, col < V, other=-float("inf")).to(tl.float32)
    m = tl.max(x, 0)
    lse = m + tl.log(tl.sum(tl.exp(x - m), 0))
    tl.store(PART + row * TILES + tile, lse)


@triton.jit
def ce_row_loss(X, Y, PART, LSE, LOSS, V: tl.constexpr, TILES: tl.constexpr,
                IGNORE: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    p = tl.load(PART + row * TILES + col, col < TILES, other=-float("inf"))
    m = tl.max(p, 0)
    lse = m + tl.log(tl.sum(tl.exp(p - m), 0))
    label = tl.load(Y + row)
    selected = tl.load(X + row * V + label, (label >= 0) & (label < V), other=0)
    tl.store(LSE + row, lse)
    tl.store(LOSS + row, tl.where(label == IGNORE, 0.0, lse - selected))


@triton.jit
def ce_tile_backward(X, Y, LSE, G, DX, V: tl.constexpr, IGNORE: tl.constexpr,
                     B: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * B + tl.arange(0, B)
    x = tl.load(X + row * V + col, col < V, other=-float("inf")).to(tl.float32)
    label = tl.load(Y + row)
    prob = tl.exp(x - tl.load(LSE + row))
    grad = (prob - (col == label).to(tl.float32)) * tl.load(G + row)
    grad = tl.where(label == IGNORE, 0.0, grad)
    tl.store(DX + row * V + col, grad, col < V)


@triton.jit
def rms_forward(X, W, OUT, INV, N: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    # Qwen casts normalized activations back before multiplying by its fp32 weight.
    norm = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    tl.store(OUT + row * N + col, norm * w, col < N)
    tl.store(INV + row, inv)


@triton.jit
def rms_backward(X, W, INV, G, DX, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    g = tl.load(G + row * N + col, col < N, other=0).to(tl.float32)
    # Autograd for the fp16/bf16 cast rounds this upstream gradient too.
    u = (g * w).to(X.dtype.element_ty).to(tl.float32)
    inv = tl.load(INV + row)
    dot = tl.sum(u * x, 0) / N
    dx = inv * u - x * (inv * inv * inv * dot)
    tl.store(DX + row * N + col, dx, col < N)

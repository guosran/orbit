"""Plan-sized BF16 GEMM recipe, using the installed Triton → AIR → AIE compiler.

The loop recipe and transform schedule originate in Triton-XDNA (MIT, AMD).
Unlike a workload dispatcher, this module knows only a matrix slice and shape.
"""
import triton
import triton.language as tl

@triton.jit
def matmul(A, B, C, stride_am: tl.constexpr, stride_bk: tl.constexpr,
           stride_cm: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, RELU: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    ap = A + m[:, None] * stride_am + k[None, :]
    bp = B + k[:, None] * stride_bk + n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(K // BLOCK_K):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += BLOCK_K
        bp += BLOCK_K * stride_bk
    if RELU:
        acc = tl.maximum(acc.to(tl.bfloat16), 0.0).to(tl.float32)
    tl.store(C + m[:, None] * stride_cm + n[None, :], acc)

@triton.jit
def matmul_accumulate(A, B, C, stride_am: tl.constexpr, stride_bk: tl.constexpr,
           stride_cm: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, INITIALIZE: tl.constexpr):
    """Sequential K MRE: one C BO, initialized once, reused between launches."""
    pm, pn = tl.program_id(0), tl.program_id(1)
    m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    cp = C + m[:, None] * stride_cm + n[None, :]
    if INITIALIZE:
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    else:
        acc = tl.load(cp)
    ap = A + m[:, None] * stride_am + k[None, :]
    bp = B + k[:, None] * stride_bk + n[None, :]
    for _ in range(K // BLOCK_K):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += BLOCK_K
        bp += BLOCK_K * stride_bk
    tl.store(cp, acc)

@triton.jit
def matmul_chunked(A, B, C, stride_am: tl.constexpr, stride_bk: tl.constexpr,
           stride_cm: tl.constexpr, CHUNKS: tl.constexpr, CHUNK_K: tl.constexpr,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """One device program with explicit semantic chunk and physical K loops."""
    pm, pn = tl.program_id(0), tl.program_id(1)
    m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for chunk in range(CHUNKS):
        ap = A + m[:, None] * stride_am + chunk * CHUNK_K + k[None, :]
        bp = B + (chunk * CHUNK_K + k[:, None]) * stride_bk + n[None, :]
        for inner in range(CHUNK_K // BLOCK_K):
            acc = tl.dot(tl.load(ap), tl.load(bp), acc)
            ap += BLOCK_K
            bp += BLOCK_K * stride_bk
    tl.store(C + m[:, None] * stride_cm + n[None, :], acc)

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# Kernel 1: Fused BatchNorm + Softmax over rows (each row has 48 elements)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
    ],
    key=['N_COLS'],
)
@triton.jit
def bn_softmax_kernel(
    M, N_COLS,
    eps,
    BLOCK_SIZE: tl.constexpr,
    N_COLS_CONST: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Load BN params
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < N_COLS

    mean = tl.load(mean_ptr + col_offs, mask=mask, other=0.0)
    var = tl.load(var_ptr + col_offs, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offs, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offs, mask=mask, other=0.0)

    # Load input row
    x = tl.load(input_ptr + row_idx * N_COLS + col_offs, mask=mask, other=0.0)

    # BatchNorm
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_bn = x_norm * weight + bias

    # Softmax: find max
    x_max = tl.max(tl.where(mask, x_bn, float('-inf')), axis=0)
    x_shifted = x_bn - x_max
    x_exp = libdevice.exp(x_shifted)
    x_exp_masked = tl.where(mask, x_exp, 0.0)
    x_sum = tl.sum(x_exp_masked, axis=0)
    x_softmax = x_exp_masked / x_sum

    tl.store(output_ptr + row_idx * N_COLS + col_offs, x_softmax, mask=mask)


# Kernel 2: Sum over dim=1 of [2048, 100, 32] -> [2048, 1, 32]
# Each program handles one (batch, cluster) pair
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32}, num_warps=2),
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
    ],
    key=['N_SAMPLES'],
)
@triton.jit
def sum_dim1_kernel(
    B, N_SAMPLES, K,
    input_stride_b, input_stride_n, input_stride_k,
    BLOCK_N: tl.constexpr,
):
    # program_id(0) = batch index, program_id(1) = cluster index
    b_idx = tl.program_id(0)
    k_idx = tl.program_id(1)

    if b_idx >= B or k_idx >= K:
        return

    acc = tl.zeros((), dtype=tl.float32)
    for n_start in range(0, N_SAMPLES, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        mask = n_offs < N_SAMPLES
        vals = tl.load(
            input_ptr + b_idx * input_stride_b + n_offs * input_stride_n + k_idx * input_stride_k,
            mask=mask, other=0.0
        )
        acc += tl.sum(vals, axis=0)

    tl.store(output_ptr + b_idx * K + k_idx, acc)


# Kernel 3: Compute vlad = assignment.T @ x - a_sum * clusters2, then normalize
# This is a complex fused kernel. Let's break it into steps.

# Kernel for: vlad[b, d, k] = sum_n(assignment[b, k, n] * x[b, n, d]) - a_sum[b, 0, k] * clusters2[0, d, k]
# Then normalize along dim=1 (D dimension)
# assignment: [B, K, N] with strides [4800, 1, 48] (from the slice/view/transpose)
# x: [B, N, D]
# a_sum: [B, 1, K]
# clusters2: [1, D, K]
# output: [B, D, K]

# We'll do the bmm in a separate step and fuse the subtract + normalize

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_D': 128, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_D': 64, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_D': 128, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['B', 'D', 'K'],
)
@triton.jit
def vlad_sub_normalize_dim1_kernel(
    a_ptr,       # [B, 1, K] = a_sum * clusters2 ... wait, a = a_sum * clusters2 shape [B, D, K]
    # Actually let's pass a_sum and clusters2 separately
    B, D, K,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles a tile of (D, K) for one batch
    b_idx = tl.program_id(2)
    d_start = tl.program_id(0) * BLOCK_D
    k_start = tl.program_id(1) * BLOCK_K

    d_offs = d_start + tl.arange(0, BLOCK_D)
    k_offs = k_start + tl.arange(0, BLOCK_K)

    d_mask = d_offs < D
    k_mask = k_offs < K

    # Load vlad[b, k, d] -> need to transpose to [b, d, k]
    # vlad from bmm is [B, K, D] with stride [K*D, D, 1]
    vlad_vals = tl.load(
        vlad_ptr + b_idx * K * D + k_offs[None, :] * D + d_offs[:, None],
        mask=d_mask[:, None] & k_mask[None, :], other=0.0
    )  # [BLOCK_D, BLOCK_K]

    # Load a_sum[b, 0, k]
    a_sum_vals = tl.load(
        a_sum_ptr + b_idx * K + k_offs,
        mask=k_mask, other=0.0
    )  # [BLOCK_K]

    # Load clusters2[0, d, k]
    c2_vals = tl.load(
        clusters2_ptr + d_offs[:, None] * K + k_offs[None, :],
        mask=d_mask[:, None] & k_mask[None, :], other=0.0
    )  # [BLOCK_D, BLOCK_K]

    # a = a_sum * clusters2 -> [B, D, K]
    a_vals = a_sum_vals[None, :] * c2_vals  # [BLOCK_D, BLOCK_K]

    # vlad = vlad - a
    vlad_sub = vlad_vals - a_vals  # [BLOCK_D, BLOCK_K]

    # Store to output (we'll normalize in a separate pass)
    tl.store(
        output_ptr + b_idx * D * K + d_offs[:, None] * K + k_offs[None, :],
        vlad_sub,
        mask=d_mask[:, None] & k_mask[None, :]
    )


# Kernel 4: Normalize vlad along dim=1 (D dimension) for each (b, k)
# input: [B, D, K], normalize over D for each (b, k)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 512}, num_warps=8),
        triton.Config({'BLOCK_D': 1024}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def normalize_dim1_kernel(
    B, D, K,
    clamp_min_val,
    BLOCK_D: tl.constexpr,
):
    # Each program handles one (b, k) pair
    bk_idx = tl.program_id(0)
    b_idx = bk_idx // K
    k_idx = bk_idx % K

    if b_idx >= B or k_idx >= K:
        return

    # Compute L2 norm over D
    norm_sq = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        vals = tl.load(
            input_ptr + b_idx * D * K + d_offs * K + k_idx,
            mask=mask, other=0.0
        )
        norm_sq += tl.sum(vals * vals, axis=0)

    norm = tl.sqrt(norm_sq)
    norm = tl.maximum(norm, clamp_min_val)

    # Normalize and store
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        vals = tl.load(
            input_ptr + b_idx * D * K + d_offs * K + k_idx,
            mask=mask, other=0.0
        )
        tl.store(
            output_ptr + b_idx * D * K + d_offs * K + k_idx,
            vals / norm,
            mask=mask
        )


# Kernel 5: Normalize along dim=1 (DK dimension) for each batch
# input: [B, D*K], normalize over D*K for each b
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['DK'],
)
@triton.jit
def normalize_rows_kernel(
    B, DK,
    clamp_min_val,
    BLOCK_SIZE: tl.constexpr,
):
    b_idx = tl.program_id(0)
    if b_idx >= B:
        return

    # Compute L2 norm
    norm_sq = tl.zeros((), dtype=tl.float32)
    for start in range(0, DK, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < DK
        vals = tl.load(input_ptr + b_idx * DK + offs, mask=mask, other=0.0)
        norm_sq += tl.sum(vals * vals, axis=0)

    norm = tl.sqrt(norm_sq)
    norm = tl.maximum(norm, clamp_min_val)

    # Normalize
    for start in range(0, DK, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < DK
        vals = tl.load(input_ptr + b_idx * DK + offs, mask=mask, other=0.0)
        tl.store(output_ptr + b_idx * DK + offs, vals / norm, mask=mask)


# Kernel for matmul: assignment [B, K, N] @ x [B, N, D] -> vlad [B, K, D]
# assignment has strides [4800, 1, 48] (non-contiguous due to slice/transpose)
# x has strides [51200, 512, 1]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['B', 'K_DIM', 'N_SAMPLES', 'D'],
)
@triton.jit
def bmm_assign_x_kernel(
    B, K_DIM, N_SAMPLES, D,
    stride_ab, stride_ak, stride_an,
    stride_xb, stride_xn, stride_xd,
):
    b_idx = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    k_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    k_mask = k_offs < K_DIM
    d_mask = d_offs < D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for n_start in range(0, N_SAMPLES, BLOCK_K):
        n_offs = n_start + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N_SAMPLES

        # Load assignment[b, k, n]: strides [4800, 1, 48]
        a_vals = tl.load(
            assign_ptr + b_idx * stride_ab + k_offs[:, None] * stride_ak + n_offs[None, :] * stride_an,
            mask=k_mask[:, None] & n_mask[None, :], other=0.0
        )  # [BLOCK_M, BLOCK_K]

        # Load x[b, n, d]: strides [51200, 512, 1]
        x_vals = tl.load(
            x_ptr + b_idx * stride_xb + n_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0
        )  # [BLOCK_K, BLOCK_N]

        acc = tl.dot(a_vals, x_vals, acc=acc)

    tl.store(
        out_ptr + b_idx * K_DIM * D + k_offs[:, None] * D + d_offs[None, :],
        acc,
        mask=k_mask[:, None] & d_mask[None, :]
    )


# Kernel for first matmul: x [M, D] @ clusters [D, C] -> out [M, C]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        a_vals = tl.load(
            a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0
        )
        b_vals = tl.load(
            b_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :], other=0.0
        )
        acc = tl.dot(a_vals, b_vals, acc=acc)

    tl.store(
        c_ptr + m_offs[:, None] * N + n_offs[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :]
    )


def kernel_function(p_clusters, p_batch_norm_weight, p_batch_norm_bias, p_clusters2,
    """
    """
    B, N, D = x.shape  # 2048, 100, 512
    C_total = p_clusters.shape[1]  # 48
    K = p_clusters2.shape[2]  # 32 (cluster_size)
    M = B * N  # 204800

    # Step 1: Reshape x to [M, D]
    x_flat = x.reshape(M, D)  # [204800, 512]

    # Step 2: Matmul x_flat @ p_clusters -> [M, C_total]
    assignment = torch.empty(M, C_total, device=x.device, dtype=x.dtype)
    grid_mm = (triton.cdiv(M, 64), triton.cdiv(C_total, 64))
    matmul_kernel[grid_mm](
        x_flat, p_clusters, assignment,
        M, C_total, D,
        D, 1,
        C_total, 1,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )

    # Step 3: BatchNorm + Softmax fused
    softmax_out = torch.empty(M, C_total, device=x.device, dtype=x.dtype)
    grid_bn_sm = (M,)
    bn_softmax_kernel[grid_bn_sm](
        assignment, softmax_out,
        b_batch_norm_running_mean, b_batch_norm_running_var,
        p_batch_norm_weight, p_batch_norm_bias,
        M, C_total,
        1e-5,
        N_COLS_CONST=C_total,
    )

    # Step 4: Slice first K columns -> [M, K], reshape to [B, N, K]
    # softmax_out has shape [M, C_total] with stride [C_total, 1]
    # slice gives [M, K] with stride [C_total, 1] (non-contiguous)
    # We need assignment_sliced as [B, N, K] for sum and [B, K, N] for matmul

    # Step 5: Sum over dim=1 -> [B, 1, K]
    a_sum = torch.empty(B, K, device=x.device, dtype=x.dtype)
    # softmax_out[b*N + n, k] with stride [C_total, 1]
    # We need sum over n for each (b, k)
    grid_sum = (B, K)
    sum_dim1_kernel[grid_sum](
        softmax_out,  # [M, C_total] = [B*N, C_total]
        B, N, K,
    )
    # a_sum shape: [B, K], reshape to [B, 1, K]
    a_sum = a_sum.view(B, 1, K)

    # Step 6: a = a_sum * clusters2 -> [B, 512, 32]
    # clusters2: [1, 512, 32]
    # a_sum: [B, 1, 32]
    # Result: [B, 512, 32]
    # We'll compute this in the vlad_sub kernel

    # Step 7: BMM: assignment[B, K, N] @ x[B, N, D] -> vlad[B, K, D]
    # assignment is softmax_out[:, :K] reshaped to [B, N, K] then transposed to [B, K, N]
    # softmax_out strides: [C_total, 1] = [48, 1]
    # As [B, N, K]: strides [N*C_total, C_total, 1] = [4800, 48, 1]
    # As [B, K, N] (transposed): strides [4800, 1, 48]

    vlad = torch.empty(B, K, D, device=x.device, dtype=x.dtype)
    grid_bmm = (
        triton.cdiv(K, 32),
        triton.cdiv(D, 64),
    )
    bmm_assign_x_kernel[grid_bmm](
        vlad,
        B, K, N, D,
        BLOCK_M=32, BLOCK_N=64, BLOCK_K=32
    )

    # Step 8: vlad = vlad.transpose(1,2) - a = [B, D, K] - [B, D, K]
    # vlad is [B, K, D], transpose to [B, D, K]
    # a = a_sum * clusters2: [B, 1, K] * [1, D, K] = [B, D, K]
    # Compute vlad_sub = vlad[b,d,k] - a_sum[b,0,k] * clusters2[0,d,k]
    vlad_sub = torch.empty(B, D, K, device=x.device, dtype=x.dtype)
    grid_sub = (
        triton.cdiv(D, 64),
        triton.cdiv(K, 32),
    )
    vlad_sub_normalize_dim1_kernel[grid_sub](
        p_clusters2.view(D, K),  # [D, K]
        vlad_sub,
        B, D, K,
        BLOCK_D=64, BLOCK_K=32
    )

    # Step 9: Normalize vlad_sub along dim=1 (D dimension) for each (b, k)
    vlad_norm1 = torch.empty(B, D, K, device=x.device, dtype=x.dtype)
    grid_norm1 = (B * K,)
    normalize_dim1_kernel[grid_norm1](
        vlad_sub, vlad_norm1,
        B, D, K,
        1e-12,
        BLOCK_D=512
    )

    # Step 10: Reshape to [B, D*K] and normalize along dim=1
    vlad_flat = vlad_norm1.reshape(B, D * K)
    vlad_out = torch.empty(B, D * K, device=x.device, dtype=x.dtype)
    grid_norm2 = (B,)
    normalize_rows_kernel[grid_norm2](
        vlad_flat, vlad_out,
        B, D * K,
        1e-12,
        BLOCK_SIZE=1024
    )

    return vlad_out


def get_inputs():
    import math
    feature_size = 512
    cluster_size = 32
    ghost_clusters = 16
    clusters_total = cluster_size + ghost_clusters  # 48

    init_sc = 1 / math.sqrt(feature_size)

    p_clusters = init_sc * torch.randn(feature_size, clusters_total, device='cuda')
    p_batch_norm_weight = torch.ones(clusters_total, device='cuda')
    p_batch_norm_bias = torch.zeros(clusters_total, device='cuda')
    p_clusters2 = init_sc * torch.randn(1, feature_size, cluster_size, device='cuda')
    b_batch_norm_running_mean = torch.zeros(clusters_total, device='cuda')
    b_batch_norm_running_var = torch.ones(clusters_total, device='cuda')
    x = torch.randn(2048, 100, feature_size, device='cuda')

    return [p_clusters, p_batch_norm_weight, p_batch_norm_bias, p_clusters2,


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs
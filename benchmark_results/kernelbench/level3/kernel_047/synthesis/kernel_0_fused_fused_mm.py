import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# Kernel 1: Fused BatchNorm + Softmax (per-row softmax over 32 elements)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_X': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_X': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_X': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_X': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_X': 32}, num_warps=8, num_stages=2),
    ],
    key=['N_ROWS'],
)
@triton.jit
def bn_softmax_kernel(
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,  # = 32
    eps: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Each program handles BLOCK_X rows
    pid = tl.program_id(0)
    row_start = pid * BLOCK_X
    
    # Load BN params (shared across rows)
    col_offs = tl.arange(0, 32)  # N_COLS = 32
    mean = tl.load(mean_ptr + col_offs)
    var = tl.load(var_ptr + col_offs)
    weight = tl.load(weight_ptr + col_offs)
    bias = tl.load(bias_ptr + col_offs)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = inv_std * weight
    
    for i in range(BLOCK_X):
        row_idx = row_start + i
        if row_idx < N_ROWS:
            # Load row
            row_ptr = input_ptr + row_idx * 32
            x = tl.load(row_ptr + col_offs, mask=col_offs < 32, other=0.0)
            
            # BatchNorm
            x_norm = (x - mean) * scale + bias
            
            # Softmax
            x_max = tl.max(x_norm, axis=0)
            x_shifted = x_norm - x_max
            x_exp = libdevice.exp(x_shifted)
            x_sum = tl.sum(x_exp, axis=0)
            x_softmax = x_exp / x_sum
            
            # Store
            out_ptr = output_ptr + row_idx * 32
            tl.store(out_ptr + col_offs, x_softmax, mask=col_offs < 32)


# Kernel 2: Fused VLAD computation
# Computes: vlad = (assignment.T @ x) - a_sum * clusters2, then L2 norm
# assignment: (2048, 100, 32), x: (2048, 100, 512), clusters2: (1, 512, 32)
# Output: (2048, 512, 32) normalized, then reshaped to (2048, 16384) normalized

# We'll do this in steps using separate kernels for clarity and correctness

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 32, 'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 64, 'BLOCK_D': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 64, 'BLOCK_D': 128}, num_warps=8, num_stages=2),
    ],
    key=['N', 'D'],
)
@triton.jit
def vlad_matmul_kernel(
    # assignment: (B, N, K) - softmax output reshaped
    assign_ptr,
    # x: (B, N, D)
    x_ptr,
    # clusters2: (1, D, K) - for subtraction
    clusters2_ptr,
    # a_sum: (B, 1, K)
    a_sum_ptr,
    # output: (B, D, K)
    out_ptr,
    B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Each program handles one (b, d_block, k_block) tile
    pid_b = tl.program_id(2)
    pid_d = tl.program_id(1)
    pid_k = tl.program_id(0)
    
    d_start = pid_d * BLOCK_D
    k_start = pid_k * BLOCK_K
    
    d_offs = d_start + tl.arange(0, BLOCK_D)
    k_offs = k_start + tl.arange(0, BLOCK_K)
    
    d_mask = d_offs < D
    k_mask = k_offs < K
    
    # Compute vlad[b, d, k] = sum_n(assignment[b, n, k] * x[b, n, d])
    acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
    
    for n in range(N):
        # Load x[b, n, d_offs]: shape (BLOCK_D,)
        x_base = pid_b * N * D + n * D
        x_vals = tl.load(x_ptr + x_base + d_offs, mask=d_mask, other=0.0)
        
        # Load assignment[b, n, k_offs]: shape (BLOCK_K,)
        a_base = pid_b * N * K + n * K
        a_vals = tl.load(assign_ptr + a_base + k_offs, mask=k_mask, other=0.0)
        
        # Outer product accumulation
        acc += x_vals[:, None] * a_vals[None, :]
    
    # Load a_sum[b, 0, k_offs] * clusters2[0, d_offs, k_offs]
    a_sum_base = pid_b * K
    a_sum_vals = tl.load(a_sum_ptr + a_sum_base + k_offs, mask=k_mask, other=0.0)
    
    # clusters2: (1, D, K) -> (D, K)
    # Load clusters2[d_offs, k_offs]
    c2_vals = tl.load(
        clusters2_ptr + d_offs[:, None] * K + k_offs[None, :],
        mask=d_mask[:, None] & k_mask[None, :],
        other=0.0
    )
    
    # vlad = acc - a_sum * clusters2
    a_broadcast = a_sum_vals[None, :] * c2_vals
    vlad = acc - a_broadcast
    
    # Store to output[b, d, k]
    out_base = pid_b * D * K
    tl.store(
        out_ptr + out_base + d_offs[:, None] * K + k_offs[None, :],
        vlad,
        mask=d_mask[:, None] & k_mask[None, :]
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
    ],
    key=['B', 'DK'],
)
@triton.jit
def sum_kernel(
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    DK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (b, k) pair
    pid = tl.program_id(0)
    b = pid // K
    k = pid % K
    
    if b >= B:
        return
    
    acc = tl.zeros((), dtype=tl.float32)
    for n in range(N):
        val = tl.load(input_ptr + b * N * K + n * K + k)
        acc += val
    
    tl.store(output_ptr + b * K + k, acc)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=['B', 'DK'],
)
@triton.jit
def l2norm_dim1_kernel(
    B: tl.constexpr,
    DK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one batch element)
    pid = tl.program_id(0)
    if pid >= B:
        return
    
    row_ptr = input_ptr + pid * DK
    out_row_ptr = output_ptr + pid * DK
    
    # Compute L2 norm
    norm_sq = tl.zeros((), dtype=tl.float32)
    for off in range(0, DK, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < DK
        vals = tl.load(row_ptr + offs, mask=mask, other=0.0)
        norm_sq += tl.sum(vals * vals, axis=0)
    
    norm = tl.sqrt(norm_sq)
    norm = tl.maximum(norm, 1e-12)
    
    # Normalize
    for off in range(0, DK, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < DK
        vals = tl.load(row_ptr + offs, mask=mask, other=0.0)
        tl.store(out_row_ptr + offs, vals / norm, mask=mask)


# Kernel for L2 norm along dim=1 of (B, D, K) tensor
# Normalize each (b, :, k) slice
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 512}, num_warps=8, num_stages=2),
    ],
    key=['B', 'D', 'K'],
)
@triton.jit
def l2norm_dim1_bdk_kernel(
    B: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Each program handles one (b, k) pair, normalizes over D
    pid = tl.program_id(0)
    b = pid // K
    k = pid % K
    
    if b >= B:
        return
    
    # Compute norm over D dimension
    norm_sq = tl.zeros((), dtype=tl.float32)
    for d_off in range(0, D, BLOCK_D):
        d_offs = d_off + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        vals = tl.load(input_ptr + b * D * K + d_offs * K + k, mask=mask, other=0.0)
        norm_sq += tl.sum(vals * vals, axis=0)
    
    norm = tl.sqrt(norm_sq)
    norm = tl.maximum(norm, 1e-12)
    
    # Normalize
    for d_off in range(0, D, BLOCK_D):
        d_offs = d_off + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        vals = tl.load(input_ptr + b * D * K + d_offs * K + k, mask=mask, other=0.0)
        tl.store(output_ptr + b * D * K + d_offs * K + k, vals / norm, mask=mask)


# Matmul kernel for (B, K, N) x (B, N, D) -> (B, K, D)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['B', 'M_DIM', 'N_DIM', 'K_DIM'],
)
@triton.jit
def bmm_kernel(
    B: tl.constexpr,
    M_DIM: tl.constexpr,
    N_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    m_mask = m_offs < M_DIM
    n_mask = n_offs < N_DIM
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K_DIM, BLOCK_K):
        k_offs = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_DIM
        
        # Load A[b, m, k]
        a_ptrs = A_ptr + pid_b * M_DIM * K_DIM + m_offs[:, None] * K_DIM + k_offs[None, :]
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B[b, k, n]
        b_ptrs = B_ptr + pid_b * K_DIM * N_DIM + k_offs[:, None] * N_DIM + n_offs[None, :]
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        acc = tl.dot(a, b, acc, allow_tf32=False)
    
    # Store C[b, m, n]
    c_ptrs = C_ptr + pid_b * M_DIM * N_DIM + m_offs[:, None] * N_DIM + n_offs[None, :]
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# Matmul kernel for (204800, 512) x (512, 32) -> (204800, 32)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def mm_kernel(
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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
    
    for k_off in range(0, K, BLOCK_K):
        k_offs = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        
        a_ptrs = A_ptr + m_offs[:, None] * K + k_offs[None, :]
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        b_ptrs = B_ptr + k_offs[:, None] * N + n_offs[None, :]
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        acc = tl.dot(a, b, acc, allow_tf32=False)
    
    c_ptrs = C_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def kernel_function(p_clusters, p_batch_norm_weight, p_batch_norm_bias,
    B_size = x.shape[0]   # 2048
    N_size = x.shape[1]   # 100
    D_size = x.shape[2]   # 512
    K_size = p_clusters.shape[1]  # 32
    
    # Step 1: view x to (B*N, D)
    x_flat = x.reshape(B_size * N_size, D_size)  # (204800, 512)
    
    # Step 2: matmul x_flat @ clusters -> (204800, 32)
    M_mm = B_size * N_size
    N_mm = K_size
    K_mm = D_size
    assignment = torch.empty((M_mm, N_mm), device=x.device, dtype=torch.float32)
    
    grid_mm = (triton.cdiv(M_mm, 128), triton.cdiv(N_mm, 32))
    mm_kernel[grid_mm](
        x_flat, p_clusters, assignment,
        M=M_mm, N=N_mm, K=K_mm,
        BLOCK_M=128, BLOCK_N=32, BLOCK_K=64,
    )
    
    # Step 3: BatchNorm + Softmax fused
    softmax_out = torch.empty_like(assignment)
    
    grid_bn = lambda meta: (triton.cdiv(M_mm, meta['BLOCK_X']),)
    bn_softmax_kernel[grid_bn](
        assignment,
        b_batch_norm_running_mean,
        b_batch_norm_running_var,
        p_batch_norm_weight,
        p_batch_norm_bias,
        softmax_out,
        N_ROWS=M_mm,
        N_COLS=N_mm,
        eps=1e-5,
    )
    
    # Step 4: view softmax_out to (B, N, K)
    assignment_3d = softmax_out.reshape(B_size, N_size, K_size)
    
    # Step 5: sum over dim=1 -> (B, 1, K)
    a_sum = torch.empty((B_size, K_size), device=x.device, dtype=torch.float32)
    
    grid_sum = (B_size * K_size,)
    sum_kernel[grid_sum](
        assignment_3d,
        a_sum,
        B=B_size, N=N_size, K=K_size,
        DK=B_size * K_size,
    )
    # a_sum shape: (B, K), will be used as (B, 1, K)
    
    # Step 6: Compute vlad = assignment.T @ x - a_sum * clusters2
    # assignment: (B, N, K), x: (B, N, D), clusters2: (1, D, K)
    # vlad[b, d, k] = sum_n(assignment[b,n,k] * x[b,n,d]) - a_sum[b,k] * clusters2[0,d,k]
    
    clusters2_flat = p_clusters2.reshape(D_size, K_size)  # (512, 32)
    vlad = torch.empty((B_size, D_size, K_size), device=x.device, dtype=torch.float32)
    
    grid_vlad = (
        triton.cdiv(K_size, 32),
        triton.cdiv(D_size, 64),
        B_size,
    )
    vlad_matmul_kernel[grid_vlad](
        assignment_3d,
        x_flat.reshape(B_size, N_size, D_size),
        clusters2_flat,
        a_sum,
        vlad,
        B=B_size, N=N_size, K=K_size, D=D_size,
        BLOCK_K=32, BLOCK_D=64,
    )
    
    # Step 7: L2 normalize vlad along dim=1 (D dimension)
    # vlad: (B, D, K), normalize each (b, :, k) slice
    vlad_norm = torch.empty_like(vlad)
    
    grid_l2_bdk = (B_size * K_size,)
    l2norm_dim1_bdk_kernel[grid_l2_bdk](
        vlad, vlad_norm,
        B=B_size, D=D_size, K=K_size,
    )
    
    # Step 8: reshape to (B, D*K) = (2048, 16384)
    vlad_flat = vlad_norm.reshape(B_size, D_size * K_size)
    
    # Step 9: L2 normalize along dim=1 (DK dimension)
    vlad_final = torch.empty_like(vlad_flat)
    
    grid_l2 = (B_size,)
    l2norm_dim1_kernel[grid_l2](
        vlad_flat, vlad_final,
        B=B_size, DK=D_size * K_size,
    )
    
    return vlad_final


def get_inputs():
    import math
    feature_size = 512
    cluster_size = 32
    ghost_clusters = 0
    init_sc = 1.0 / math.sqrt(feature_size)
    
    p_clusters = init_sc * torch.randn(feature_size, cluster_size + ghost_clusters, device='cuda')
    p_batch_norm_weight = torch.ones(cluster_size + ghost_clusters, device='cuda')
    p_batch_norm_bias = torch.zeros(cluster_size + ghost_clusters, device='cuda')
    p_clusters2 = init_sc * torch.randn(1, feature_size, cluster_size, device='cuda')
    b_batch_norm_running_mean = torch.zeros(cluster_size + ghost_clusters, device='cuda')
    b_batch_norm_running_var = torch.ones(cluster_size + ghost_clusters, device='cuda')
    x = torch.randn(2048, 100, 512, device='cuda')
    
    return [p_clusters, p_batch_norm_weight, p_batch_norm_bias, p_clusters2,


if __name__ == "__main__":
    import argparse
    import math
    import torch.nn as nn
    import torch.nn.functional as F
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        p_clusters, p_batch_norm_weight, p_batch_norm_bias, p_clusters2, \
            b_batch_norm_running_mean, b_batch_norm_running_var, x = inputs
        
        # Reference implementation
        feature_size = 512
        cluster_size = 32
        ghost_clusters = 0
        
        x_ref = x.clone()
        B_ref = x_ref.shape[0]
        N_ref = x_ref.shape[1]
        
        x_flat_ref = x_ref.view(-1, feature_size)
        
        # matmul
        assignment_ref = torch.matmul(x_flat_ref, p_clusters)
        
        # batch norm (inference mode)
        assignment_ref = F.batch_norm(
            assignment_ref,
            b_batch_norm_running_mean,
            b_batch_norm_running_var,
            p_batch_norm_weight,
            p_batch_norm_bias,
            training=False,
            eps=1e-5
        )
        
        # softmax
        assignment_ref = F.softmax(assignment_ref, dim=1)
        
        # take first cluster_size columns (all 32)
        assignment_ref = assignment_ref[:, :cluster_size]
        
        # view
        assignment_ref = assignment_ref.view(B_ref, N_ref, cluster_size)
        
        # sum
        a_sum_ref = torch.sum(assignment_ref, dim=1, keepdim=True)
        
        # mul with clusters2
        a_ref = a_sum_ref * p_clusters2
        
        # transpose assignment
        assignment_t_ref = assignment_ref.transpose(1, 2)
        
        # view x
        x_3d_ref = x_flat_ref.view(B_ref, N_ref, feature_size)
        
        # matmul
        vlad_ref = torch.matmul(assignment_t_ref, x_3d_ref)
        
        # transpose
        vlad_ref = vlad_ref.transpose(1, 2)
        
        # subtract
        vlad_ref = vlad_ref - a_ref
        
        # L2 norm along dim=1
        norm_ref = torch.linalg.vector_norm(vlad_ref, dim=1, keepdim=True)
        norm_ref = torch.clamp_min(norm_ref, 1e-12)
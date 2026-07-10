import torch
import triton
import triton.language as tl

# ─── GEMM kernel ────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (mask_k[None, :]), other=0.0)
        b = tl.load(B_ptrs, mask=(mask_k[:, None]) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    offs_m2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_out = (offs_m2[:, None] < M) & (offs_n2[None, :] < N)
    C_ptrs = C_ptr + offs_m2[:, None] * stride_cm + offs_n2[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask_out)


# ─── Fused: bias_add + hardtanh + mish + group_norm mean/var ────────────────

@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'RBLOCK': 32}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 2, 'RBLOCK': 32}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 4, 'RBLOCK': 32}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 8, 'RBLOCK': 32}, num_stages=2, num_warps=8),
    ],
    key=['XNUMEL', 'RNUMEL'],
)
@triton.jit
def fused_bias_act_groupnorm_stats_kernel(
    C_per_G,      # channels per group = N // G
    XNUMEL, RNUMEL,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    # Each program handles XBLOCK groups, each group has RBLOCK=C_per_G elements
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    rindex = tl.arange(0, RBLOCK)[None, :]             # [1, RBLOCK]

    xmask = xindex < XNUMEL  # [XBLOCK, 1]
    rmask = rindex < RNUMEL  # [1, RBLOCK]

    # xindex = batch * G + group_id
    batch = xindex // G       # [XBLOCK, 1]
    group = xindex % G        # [XBLOCK, 1]

    # channel index within group
    chan = group * C_per_G + rindex  # [XBLOCK, RBLOCK]

    flat_idx = batch * N + chan  # [XBLOCK, RBLOCK]

    combined_mask = xmask & rmask & (batch < M) & (chan < N)

    mm_val = tl.load(mm_ptr + flat_idx, mask=combined_mask, other=0.0)
    gb_val = tl.load(gemm_bias_ptr + chan, mask=combined_mask, other=0.0)
    b_val  = tl.load(bias_ptr + chan, mask=combined_mask, other=0.0)

    x = mm_val + gb_val + b_val

    # hardtanh
    x = tl.where(x < -1.0, -1.0, x)
    x = tl.where(x > 1.0, 1.0, x)

    # mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)) but for x > 20, softplus(x) ≈ x
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    mish_val = x * tl.tanh(sp)

    # compute mean and variance over the group (reduction over RBLOCK)
    mish_f = mish_val.to(tl.float32)
    sum_val = tl.sum(mish_f, axis=1)[:, None]  # [XBLOCK, 1]
    mean_val = sum_val / C_per_G.to(tl.float32)

    diff = mish_f - mean_val
    var_sum = tl.sum(diff * diff, axis=1)[:, None]  # [XBLOCK, 1]

    tl.store(mean_ptr + xindex, mean_val, mask=xmask)
    tl.store(var_ptr + xindex, var_sum, mask=xmask)


# ─── Fused: bias_add + hardtanh + mish + group_norm normalize ───────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256},  num_stages=3, num_warps=4),
    ],
    key=['TOTAL'],
)
@triton.jit
def fused_bias_act_groupnorm_norm_kernel(
    mm_ptr,
    gemm_bias_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    gn_weight_ptr,
    gn_bias_ptr,
    out_ptr,
    M, N, G,
    C_per_G,
    eps,
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    # offsets: flat index into [M, N]
    batch = offsets // N
    chan  = offsets % N
    group = chan // C_per_G

    group_idx = batch * G + group

    mm_val = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
    gb_val = tl.load(gemm_bias_ptr + chan, mask=mask, other=0.0)
    b_val  = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    x = mm_val + gb_val + b_val

    # hardtanh
    x = tl.where(x < -1.0, -1.0, x)
    x = tl.where(x > 1.0, 1.0, x)

    # mish
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    mish_val = x * tl.tanh(sp)

    mean_val = tl.load(mean_ptr + group_idx, mask=mask, other=0.0)
    var_sum  = tl.load(var_ptr + group_idx, mask=mask, other=0.0)

    var_val = var_sum / C_per_G.to(tl.float32)
    inv_std = tl.math.rsqrt(var_val + eps)

    x_norm = (mish_val - mean_val) * inv_std

    gn_w = tl.load(gn_weight_ptr + chan, mask=mask, other=1.0)
    gn_b = tl.load(gn_bias_ptr + chan, mask=mask, other=0.0)

    out = x_norm * gn_w + gn_b
    tl.store(out_ptr + offsets, out, mask=mask)


def kernel_function(p_gemm_weight, p_gemm_bias, p_bias, p_groupnorm_weight, p_groupnorm_bias, x):
    M, K = x.shape
    N_out, K2 = p_gemm_weight.shape
    assert K == K2, f"Shape mismatch: x has K={K}, weight has K2={K2}"
    N = N_out

    # Step 1: GEMM  x @ weight.T
    mm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid_gemm](
        x, p_gemm_weight, mm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_gemm_weight.stride(1), p_gemm_weight.stride(0),  # weight is [N, K], we want x @ W^T so stride_bk=stride(1), stride_bn=stride(0)
        mm_out.stride(0), mm_out.stride(1),
    )

    # Step 2: Fused bias_add + hardtanh + mish + group_norm stats
    G = 256
    C_per_G = N // G
    XNUMEL = M * G
    RNUMEL = C_per_G

    mean_buf = torch.empty((M * G,), device=x.device, dtype=torch.float32)
    var_buf  = torch.empty((M * G,), device=x.device, dtype=torch.float32)

    grid_stats = lambda meta: (triton.cdiv(XNUMEL, meta['XBLOCK']),)
    fused_bias_act_groupnorm_stats_kernel[grid_stats](
        mm_out, p_gemm_bias, p_bias,
        mean_buf, var_buf,
        M, N, G, C_per_G,
        XNUMEL, RNUMEL,
    )

    # Step 3: Fused normalize + scale + shift
    TOTAL = M * N
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    eps = 1e-5

    grid_norm = lambda meta: (triton.cdiv(TOTAL, meta['BLOCK_SIZE']),)
    fused_bias_act_groupnorm_norm_kernel[grid_norm](
        mm_out, p_gemm_bias, p_bias,
        mean_buf, var_buf,
        p_groupnorm_weight, p_groupnorm_bias,
        out,
        M, N, G, C_per_G, eps, TOTAL,
    )

    return out


def get_inputs():
    p_gemm_weight     = torch.randn(8192, 8192, dtype=torch.float32, device='cuda')
    p_gemm_bias       = torch.randn(8192, dtype=torch.float32, device='cuda')
    p_bias            = torch.randn(8192, dtype=torch.float32, device='cuda')
    p_groupnorm_weight = torch.randn(8192, dtype=torch.float32, device='cuda')
    p_groupnorm_bias   = torch.randn(8192, dtype=torch.float32, device='cuda')
    x                 = torch.rand(1024, 8192, dtype=torch.float32, device='cuda')
    return [p_gemm_weight, p_gemm_bias, p_bias, p_groupnorm_weight, p_groupnorm_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_gemm_weight, p_gemm_bias, p_bias, p_groupnorm_weight, p_groupnorm_bias, x = inputs

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn
        # Reference computation
        with torch.no_grad():
            ref = x @ p_gemm_weight.T + p_gemm_bias
            ref = ref + p_bias
            ref = torch.clamp(ref, -1.0, 1.0)
            # mish
            ref = ref * torch.tanh(torch.nn.functional.softplus(ref))
            # group norm
            ref_4d = ref.view(1024, 256, 32)
            mean_ref = ref_4d.mean(dim=2, keepdim=True)
            var_ref  = ref_4d.var(dim=2, keepdim=True, unbiased=False)
            ref_norm = (ref_4d - mean_ref) / torch.sqrt(var_ref + 1e-5)
            ref_norm = ref_norm.view(1024, 8192)
            ref_out  = ref_norm * p_groupnorm_weight + p_groupnorm_bias

        assert torch.allclose(output, ref_out, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref_out).abs().max().item()):.4e}"
        print("Kernel executed successfully.")
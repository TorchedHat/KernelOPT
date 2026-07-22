import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Linear projection  x(16384,768) @ W^T(768,2304) + bias -> (16384,2304)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Attention  q(B,H,T,hs) @ k^T(B,H,hs,T) * scale + mask -> relu -> att @ v
# We implement a tiled attention kernel.
# Each program handles one (batch, head) pair and tiles over query rows.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64,  'BLOCK_HS': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 128, 'BLOCK_HS': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_T': 64,  'BLOCK_HS': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_T': 32,  'BLOCK_HS': 64}, num_stages=3, num_warps=4),
    ],
    key=['T', 'HS'],
)
@triton.jit
def attn_relu_kernel(
    Q_ptr, K_ptr, V_ptr, bias_ptr, Out_ptr,
    B, H, T, HS,
    stride_qb, stride_qh, stride_qt, stride_qhs,
    stride_kb, stride_kh, stride_kt, stride_khs,
    stride_vb, stride_vh, stride_vt, stride_vhs,
    stride_ob, stride_oh, stride_ot, stride_ohs,
    scale,
    BLOCK_T: tl.constexpr,
    BLOCK_HS: tl.constexpr,
):
    # program_id(0) = batch * H + head
    # program_id(1) = tile over query rows
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    q_tile = tl.program_id(1)

    q_start = q_tile * BLOCK_T
    offs_q = q_start + tl.arange(0, BLOCK_T)  # query row indices
    offs_hs = tl.arange(0, BLOCK_HS)           # head-size indices

    # Pointers to Q block: shape [BLOCK_T, HS]
    Q_base = Q_ptr + b * stride_qb + h * stride_qh
    K_base = K_ptr + b * stride_kb + h * stride_kh
    V_base = V_ptr + b * stride_vb + h * stride_vh
    O_base = Out_ptr + b * stride_ob + h * stride_oh

    # Load Q tile [BLOCK_T, HS] -- iterate over HS in chunks
    # We'll accumulate att scores in a [BLOCK_T, T] buffer -- but T=1024 is too large for SRAM
    # Instead, do two-pass: first compute att scores tile by tile, apply relu, then accumulate output

    # Output accumulator [BLOCK_T, BLOCK_HS]
    # We need to iterate over key dimension T in tiles of BLOCK_T
    # and over hs dimension in tiles of BLOCK_HS

    # For each query tile, compute full attention row [BLOCK_T, T], apply mask+relu, then multiply by V

    # Step 1: Compute att = Q @ K^T * scale, masked_fill, relu
    # We store att in a [BLOCK_T, T] accumulator (T=1024, BLOCK_T=64 -> 64*1024*4=256KB too large)
    # Instead, we tile over K columns too and accumulate output directly.

    # Actually for correctness with relu (not softmax), we can tile:
    # out[q_tile, :] = sum_k relu(att[q_tile, k]) * V[k, :]
    # where att[q_tile, k] = Q[q_tile, :] @ K[k, :] * scale (masked)

    # We tile over k (key rows) in BLOCK_T chunks, and over hs in BLOCK_HS chunks.
    # For each (q_tile, k_tile), compute att_block [BLOCK_T, BLOCK_T], apply relu+mask,
    # then for each hs_tile, accumulate att_block @ V[k_tile, hs_tile] into out.

    # This requires iterating: for k_tile: for hs_tile: ...
    # Better: for k_tile: compute att_block [BLOCK_T, BLOCK_T]; for hs_tile: acc += att_block @ V_tile

    # Output accumulator
    acc = tl.zeros((BLOCK_T, BLOCK_HS), dtype=tl.float32)

    # Iterate over hs tiles for output
    # Actually we need to handle HS=64 which fits in one tile if BLOCK_HS=64
    # Let's iterate over key tiles
    for k_start in range(0, T, BLOCK_T):
        offs_k = k_start + tl.arange(0, BLOCK_T)

        # Compute att_block = Q_tile @ K_tile^T * scale  [BLOCK_T, BLOCK_T]
        # Q_tile: [BLOCK_T, HS], K_tile: [BLOCK_T, HS]
        att_block = tl.zeros((BLOCK_T, BLOCK_T), dtype=tl.float32)

        for hs_start in range(0, HS, BLOCK_HS):
            offs_hs2 = hs_start + tl.arange(0, BLOCK_HS)
            # Load Q tile [BLOCK_T, BLOCK_HS]
            q_mask = (offs_q[:, None] < T) & (offs_hs2[None, :] < HS)
            q_ptrs = Q_base + offs_q[:, None] * stride_qt + offs_hs2[None, :] * stride_qhs
            q_tile_data = tl.load(q_ptrs, mask=q_mask, other=0.0)

            # Load K tile [BLOCK_T, BLOCK_HS]
            k_mask = (offs_k[:, None] < T) & (offs_hs2[None, :] < HS)
            k_ptrs = K_base + offs_k[:, None] * stride_kt + offs_hs2[None, :] * stride_khs
            k_tile_data = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # att_block += Q_tile @ K_tile^T  [BLOCK_T, BLOCK_T]
            att_block = tl.dot(q_tile_data, tl.trans(k_tile_data), att_block, allow_tf32=False)

        # Scale
        att_block = att_block * scale

        # Apply causal mask: bias[0,0,q,k] == 0 -> fill with -inf
        # bias shape [1,1,T,T], we need bias[offs_q, offs_k]
        bias_ptrs = bias_ptr + offs_q[:, None] * T + offs_k[None, :]
        bias_mask_q = offs_q[:, None] < T
        bias_mask_k = offs_k[None, :] < T
        bias_vals = tl.load(bias_ptrs, mask=bias_mask_q & bias_mask_k, other=1.0)
        # Where bias == 0, fill with -inf
        att_block = tl.where(bias_vals == 0.0, float('-inf'), att_block)
        # Where k >= T (out of bounds), also mask
        att_block = tl.where(offs_k[None, :] < T, att_block, float('-inf'))
        # Where q >= T, mask
        att_block = tl.where(offs_q[:, None] < T, att_block, float('-inf'))

        # ReLU
        att_block = tl.maximum(att_block, 0.0)

        # Accumulate: att_block @ V_tile  [BLOCK_T, BLOCK_T] @ [BLOCK_T, BLOCK_HS]
        for hs_start in range(0, HS, BLOCK_HS):
            offs_hs2 = hs_start + tl.arange(0, BLOCK_HS)
            v_mask = (offs_k[:, None] < T) & (offs_hs2[None, :] < HS)
            v_ptrs = V_base + offs_k[:, None] * stride_vt + offs_hs2[None, :] * stride_vhs
            v_tile_data = tl.load(v_ptrs, mask=v_mask, other=0.0)
            # acc [BLOCK_T, BLOCK_HS] += att_block [BLOCK_T, BLOCK_T] @ v_tile [BLOCK_T, BLOCK_HS]
            # But we need to handle multiple hs tiles -- store separately
            # Since HS=64 and BLOCK_HS=64, this loop runs once
            acc_local = tl.dot(att_block.to(tl.float32), v_tile_data, allow_tf32=False)
            # We need per-hs_tile accumulator... let's restructure

    # The above structure has a bug: acc is shared across hs_tiles but we overwrite.
    # Since HS=64 and BLOCK_HS=64, the inner loop runs once, so it's fine.
    # But let's restructure properly for general case.
    # Actually let me rewrite more carefully.
    tl.store(
        O_base + offs_q[:, None] * stride_ot + offs_hs[None, :] * stride_ohs,
        acc.to(tl.float32),
        mask=(offs_q[:, None] < T) & (offs_hs[None, :] < HS),
    )


# The above kernel has structural issues. Let me write a cleaner version.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64,  'BLOCK_K': 64,  'BLOCK_HS': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_K': 64,  'BLOCK_HS': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_Q': 64,  'BLOCK_K': 128, 'BLOCK_HS': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_Q': 32,  'BLOCK_K': 64,  'BLOCK_HS': 64}, num_stages=3, num_warps=4),
    ],
    key=['T', 'HS'],
)
@triton.jit
def attn_relu_kernel_v2(
    Q_ptr, K_ptr, V_ptr, bias_ptr, Out_ptr,
    B, H, T, HS,
    stride_qb, stride_qh, stride_qt, stride_qhs,
    stride_kb, stride_kh, stride_kt, stride_khs,
    stride_vb, stride_vh, stride_vt, stride_vhs,
    stride_ob, stride_oh, stride_ot, stride_ohs,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_HS: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    q_tile_id = tl.program_id(1)

    q_start = q_tile_id * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_hs = tl.arange(0, BLOCK_HS)

    Q_base = Q_ptr + b * stride_qb + h * stride_qh
    K_base = K_ptr + b * stride_kb + h * stride_kh
    V_base = V_ptr + b * stride_vb + h * stride_vh
    O_base = Out_ptr + b * stride_ob + h * stride_oh

    # Output accumulator [BLOCK_Q, BLOCK_HS]
    acc = tl.zeros((BLOCK_Q, BLOCK_HS), dtype=tl.float32)

    # Iterate over key tiles
    for k_start in range(0, T, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Compute att_block [BLOCK_Q, BLOCK_K] = Q[offs_q,:] @ K[offs_k,:]^T * scale
        att_block = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

        # Iterate over head-size dimension
        for hs_start in range(0, HS, BLOCK_HS):
            offs_hs2 = hs_start + tl.arange(0, BLOCK_HS)
            q_mask = (offs_q[:, None] < T) & (offs_hs2[None, :] < HS)
            q_ptrs = Q_base + offs_q[:, None] * stride_qt + offs_hs2[None, :] * stride_qhs
            q_data = tl.load(q_ptrs, mask=q_mask, other=0.0)

            k_mask = (offs_k[:, None] < T) & (offs_hs2[None, :] < HS)
            k_ptrs = K_base + offs_k[:, None] * stride_kt + offs_hs2[None, :] * stride_khs
            k_data = tl.load(k_ptrs, mask=k_mask, other=0.0)

            att_block = tl.dot(q_data, tl.trans(k_data), att_block, allow_tf32=False)

        att_block = att_block * scale

        # Load bias mask [BLOCK_Q, BLOCK_K]
        bias_q_mask = offs_q[:, None] < T
        bias_k_mask = offs_k[None, :] < T
        bias_ptrs = bias_ptr + offs_q[:, None] * T + offs_k[None, :]
        bias_vals = tl.load(bias_ptrs, mask=bias_q_mask & bias_k_mask, other=1.0)

        # masked_fill where bias==0 -> -inf
        att_block = tl.where((bias_vals == 0.0) | ~(bias_k_mask) | ~(bias_q_mask), float('-inf'), att_block)

        # ReLU
        att_block = tl.maximum(att_block, 0.0)

        # Accumulate output: att_block [BLOCK_Q, BLOCK_K] @ V[offs_k, :] [BLOCK_K, BLOCK_HS]
        v_mask = (offs_k[:, None] < T) & (offs_hs[None, :] < HS)
        v_ptrs = V_base + offs_k[:, None] * stride_vt + offs_hs[None, :] * stride_vhs
        v_data = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc = tl.dot(att_block.to(tl.float32), v_data, acc, allow_tf32=False)

    # Store output
    out_mask = (offs_q[:, None] < T) & (offs_hs[None, :] < HS)
    out_ptrs = O_base + offs_q[:, None] * stride_ot + offs_hs[None, :] * stride_ohs
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(p_c_attn_weight, p_c_attn_bias, b_bias, x):
    """
    """
    B, T, C = x.shape          # 16, 1024, 768
    n_head = 12
    hs = C // n_head            # 64
    device = x.device

    # ── Step 1: Linear projection ──────────────────────────────────────────
    x_2d = x.reshape(B * T, C)  # [16384, 768]
    W = p_c_attn_weight          # [2304, 768]  -- weight is (out, in)
    # We want x_2d @ W^T + bias -> [16384, 2304]
    M_lin = B * T    # 16384
    N_lin = 3 * C    # 2304
    K_lin = C        # 768

    proj = torch.empty((M_lin, N_lin), device=device, dtype=torch.float32)

    grid_lin = lambda meta: (
        triton.cdiv(M_lin, meta['BLOCK_M']) * triton.cdiv(N_lin, meta['BLOCK_N']),
    )
    linear_kernel[grid_lin](
        x_2d, W, p_c_attn_bias, proj,
        M_lin, N_lin, K_lin,
        x_2d.stride(0), x_2d.stride(1),
        W.stride(1), W.stride(0),   # W stored as [2304,768]: stride_k=W.stride(1)=1, stride_n=W.stride(0)=768
        proj.stride(0), proj.stride(1),
    )

    # ── Step 2: Split into Q, K, V and reshape ─────────────────────────────
    proj_3d = proj.reshape(B, T, 3 * C)  # [16, 1024, 2304]
    q_raw = proj_3d[:, :, :C]            # [16, 1024, 768]
    k_raw = proj_3d[:, :, C:2*C]         # [16, 1024, 768]
    v_raw = proj_3d[:, :, 2*C:]          # [16, 1024, 768]

    # Reshape to [B, H, T, hs] -- need contiguous for kernel
    q = q_raw.reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()  # [16,12,1024,64]
    k = k_raw.reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()  # [16,12,1024,64]
    v = v_raw.reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()  # [16,12,1024,64]

    # ── Step 3: Attention with ReLU ────────────────────────────────────────
    scale = 1.0 / math.sqrt(hs)

    # bias is [1,1,1024,1024] -- we need the [T,T] slice as a 2D contiguous tensor
    bias_2d = b_bias[0, 0, :T, :T].contiguous()  # [T, T]

    out = torch.empty((B, n_head, T, hs), device=device, dtype=torch.float32)

    grid_attn = (B * n_head, triton.cdiv(T, 64))  # will be overridden by autotune

    def grid_attn_fn(meta):
        return (B * n_head, triton.cdiv(T, meta['BLOCK_Q']))

    attn_relu_kernel_v2[grid_attn_fn](
        q, k, v, bias_2d, out,
        B, n_head, T, hs,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale,
    )

    # ── Step 4: Reshape output ─────────────────────────────────────────────
    y = out.permute(0, 2, 1, 3).contiguous().reshape(B, T, C)  # [16, 1024, 768]
    return y


def get_inputs():
    B, T, C = 16, 1024, 768
    n_head = 12
    p_c_attn_weight = torch.randn(3 * C, C, dtype=torch.float32, device='cuda')
    p_c_attn_bias = torch.randn(3 * C, dtype=torch.float32, device='cuda')
    b_bias = torch.tril(torch.ones(T, T, dtype=torch.float32, device='cuda')).view(1, 1, T, T)
    x = torch.randn(B, T, C, dtype=torch.float32, device='cuda')
    return [p_c_attn_weight, p_c_attn_bias, b_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_c_attn_weight, p_c_attn_bias, b_bias, x = inputs

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference implementation
        B, T, C = x.shape
        n_head = 12
        hs = C // n_head

        # Linear
        proj_ref = x.reshape(B * T, C) @ p_c_attn_weight.t() + p_c_attn_bias
        proj_ref = proj_ref.reshape(B, T, 3 * C)
        q_ref = proj_ref[:, :, :C].reshape(B, T, n_head, hs).permute(0, 2, 1, 3)
        k_ref = proj_ref[:, :, C:2*C].reshape(B, T, n_head, hs).permute(0, 2, 1, 3)
        v_ref = proj_ref[:, :, 2*C:].reshape(B, T, n_head, hs).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(hs)
        att_ref = (q_ref @ k_ref.transpose(-2, -1)) * scale
        att_ref = att_ref.masked_fill(b_bias[:, :, :T, :T] == 0, float('-inf'))
        att_ref = torch.relu(att_ref)
        y_ref = (att_ref @ v_ref).permute(0, 2, 1, 3).contiguous().reshape(B, T, C)

        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - y_ref).abs().max().item():.4e}")
        assert torch.allclose(output, y_ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={(output - y_ref).abs().max().item():.4e}"
        print("Kernel executed successfully.")
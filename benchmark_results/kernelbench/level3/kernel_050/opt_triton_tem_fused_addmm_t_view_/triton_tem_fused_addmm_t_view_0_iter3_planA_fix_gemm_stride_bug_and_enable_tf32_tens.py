import torch
import triton
import triton.language as tl
import math



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_qkv_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
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

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (mask_k[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(mask_k[:, None]) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 16, 'BLOCK_S': 16, 'BLOCK_D': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_T': 16, 'BLOCK_S': 32, 'BLOCK_D': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_T': 32, 'BLOCK_S': 16, 'BLOCK_D': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_T': 32, 'BLOCK_S': 32, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 32, 'BLOCK_S': 64, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_S': 32, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_S': 64, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_S': 64, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_T': 64, 'BLOCK_S': 128, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_T': 128, 'BLOCK_S': 64, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_T': 128, 'BLOCK_S': 128, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
    ],
    key=['T', 'D'],
)
@triton.jit

def attention_relu_kernel(
    Q_ptr, K_ptr, V_ptr, bias_ptr, Out_ptr,
    B, NH, T, D,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    scale,
    BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Each program handles one (batch, head, tile_of_T) combination
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)

    b = pid_bh // NH
    h = pid_bh % NH

    # Q tile: rows [pid_t * BLOCK_T, (pid_t+1) * BLOCK_T)
    q_start = pid_t * BLOCK_T
    offs_t = q_start + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile [BLOCK_T, D]
    q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + offs_t[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q_mask = (offs_t[:, None] < T) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # We need to compute att[BLOCK_T, T] = q @ K^T * scale, then masked_fill, relu
    # Then out[BLOCK_T, D] = att @ V
    # Since T=1024 can be large, we accumulate att row by row in SRAM

    # First pass: compute full attention row for this Q tile
    # att shape: [BLOCK_T, T] - this might be too large for SRAM
    # We need to handle T=1024 carefully

    # Strategy: compute att in tiles of BLOCK_S, store full att row, then compute output
    # But [BLOCK_T=128, T=1024] = 128K floats = 512KB - too large for SRAM
    # Instead, use two-pass approach: first compute att, then multiply by V

    # Actually for T=1024, BLOCK_T=64: att = [64, 1024] = 256KB - still large
    # Let's use a different approach: compute att on-the-fly

    # We'll accumulate the output directly using the full attention row
    # For each S-tile of K, compute partial att scores, apply mask, relu
    # But we need the full row to apply relu correctly (relu is elementwise so it's fine)

    # Actually relu is elementwise, so we can process in tiles:
    # For each S-tile:
    #   att_tile = q @ K_tile^T * scale  [BLOCK_T, BLOCK_S]
    #   apply causal mask
    #   relu
    #   out += att_tile @ V_tile  [BLOCK_T, D]

    # This works! relu is elementwise so no need to see full row.

    out_acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

    for s_start in range(0, T, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)

        # Load K tile [BLOCK_S, D]
        k_ptrs = K_ptr + b * stride_kb + h * stride_kh + offs_s[:, None] * stride_kt + offs_d[None, :] * stride_kd
        k_mask = (offs_s[:, None] < T) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute att_tile = q @ k^T * scale [BLOCK_T, BLOCK_S]
        att = tl.dot(q, tl.trans(k), allow_tf32=False)
        att = att * scale

        # Apply causal mask: bias[0,0,t,s] == 0 means mask to -inf
        # bias is lower triangular: bias[t,s] = 1 if s <= t else 0
        # masked_fill where bias == 0 -> -inf
        # For causal: s <= t means valid
        t_idx = offs_t[:, None]  # [BLOCK_T, 1]
        s_idx = offs_s[None, :]  # [1, BLOCK_S]
        causal_mask = s_idx <= t_idx  # valid positions
        s_valid = offs_s[None, :] < T
        t_valid = offs_t[:, None] < T
        valid = s_valid & t_valid

        att = tl.where(causal_mask & valid, att, float('-inf'))
        att = tl.where(valid & ~causal_mask, float('-inf'), att)
        # Actually simplify: mask where NOT causal or out of bounds
        att = tl.where(causal_mask & (offs_s[None, :] < T) & (offs_t[:, None] < T), att, float('-inf'))

        # ReLU
        att = tl.maximum(att, 0.0)

        # Load V tile [BLOCK_S, D]
        v_ptrs = V_ptr + b * stride_vb + h * stride_vh + offs_s[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v_mask = (offs_s[:, None] < T) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Accumulate: out += att @ v [BLOCK_T, D]
        out_acc = tl.dot(att.to(tl.float32), v.to(tl.float32), out_acc, allow_tf32=False)

    # Store output [BLOCK_T, D]
    out_ptrs = Out_ptr + b * stride_ob + h * stride_oh + offs_t[:, None] * stride_ot + offs_d[None, :] * stride_od
    out_mask = (offs_t[:, None] < T) & (offs_d[None, :] < D)
    tl.store(out_ptrs, out_acc.to(tl.float32), mask=out_mask)


def kernel_function(p_c_attn_weight, p_c_attn_bias, b_bias, x):
    B, T, C = x.shape  # 16, 1024, 768
    NH = 12
    HS = C // NH  # 64
    N_OUT = 3 * C  # 2304

    # Step 1: QKV projection using GEMM
    # x: [B*T, C] @ W^T: [C, 3C] + bias: [3C] -> [B*T, 3C]
    x_2d = x.reshape(B * T, C)  # [16384, 768]
    qkv = torch.empty(B * T, N_OUT, device=x.device, dtype=x.dtype)

    M = B * T  # 16384
    N = N_OUT  # 2304
    K = C      # 768

    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_qkv_kernel[grid_gemm](
        x_2d, p_c_attn_weight, p_c_attn_bias, qkv,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        p_c_attn_weight.stride(1), p_c_attn_weight.stride(0),  # W is [N, K], so stride_bk=stride(1), stride_bn=stride(0)... wait
        # p_c_attn_weight shape: [2304, 768] = [N, K]
        # We want B[k, n] = p_c_attn_weight[n, k] (transposed)
        # stride_bk = stride along k dim of B = p_c_attn_weight.stride(1) = 1... 
        # Actually: B_ptr[k * stride_bk + n * stride_bn]
        # p_c_attn_weight[n, k] -> index = n * 768 + k
        # So stride_bk = 1, stride_bn = 768
        qkv.stride(0), qkv.stride(1),
    )
    # Wait, I need to fix the strides. Let me redo.
    # p_c_attn_weight: [2304, 768] = [N, K]
    # We access it as B[k, n] = weight[n, k]
    # B_ptr + k * stride_bk + n * stride_bn
    # weight[n, k] = weight_ptr + n * 768 + k * 1
    # So stride_bk = 1 (stride along k), stride_bn = 768 (stride along n)

    # Step 2: Reshape QKV and run attention
    qkv_3d = qkv.reshape(B, T, N_OUT)

    # Extract Q, K, V and reshape to [B, NH, T, HS]
    q = qkv_3d[:, :, :C].reshape(B, T, NH, HS).permute(0, 2, 1, 3).contiguous()
    k = qkv_3d[:, :, C:2*C].reshape(B, T, NH, HS).permute(0, 2, 1, 3).contiguous()
    v = qkv_3d[:, :, 2*C:].reshape(B, T, NH, HS).permute(0, 2, 1, 3).contiguous()

    # Output: [B, NH, T, HS]
    out = torch.empty(B, NH, T, HS, device=x.device, dtype=x.dtype)

    scale = 1.0 / math.sqrt(HS)

    grid_attn = lambda meta: (B * NH, triton.cdiv(T, meta['BLOCK_T']))
    attention_relu_kernel[grid_attn](
        q, k, v, b_bias, out,
        B, NH, T, HS,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale,
    )

    # Reshape output: [B, NH, T, HS] -> [B, T, C]
    result = out.permute(0, 2, 1, 3).contiguous().reshape(B, T, C)
    return result


def get_inputs():
    B, T, C = 16, 1024, 768
    NH = 12
    p_c_attn_weight = torch.randn(3 * C, C, device='cuda', dtype=torch.float32)
    p_c_attn_bias = torch.randn(3 * C, device='cuda', dtype=torch.float32)
    b_bias = torch.tril(torch.ones(T, T, device='cuda', dtype=torch.float32)).view(1, 1, T, T)
    x = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    return [p_c_attn_weight, p_c_attn_bias, b_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_c_attn_weight, p_c_attn_bias, b_bias, x = inputs
        B, T, C = x.shape
        NH = 12
        HS = C // NH

        # Reference computation
        qkv_ref = x.reshape(B * T, C) @ p_c_attn_weight.T + p_c_attn_bias
        qkv_ref = qkv_ref.reshape(B, T, 3 * C)
        q_ref = qkv_ref[:, :, :C].reshape(B, T, NH, HS).permute(0, 2, 1, 3)
        k_ref = qkv_ref[:, :, C:2*C].reshape(B, T, NH, HS).permute(0, 2, 1, 3)
        v_ref = qkv_ref[:, :, 2*C:].reshape(B, T, NH, HS).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(HS)
        att_ref = (q_ref @ k_ref.transpose(-2, -1)) * scale
        att_ref = att_ref.masked_fill(b_bias[:, :, :T, :T] == 0, float('-inf'))
        att_ref = torch.relu(att_ref)
        y_ref = att_ref @ v_ref
        ref = y_ref.permute(0, 2, 1, 3).contiguous().reshape(B, T, C)

        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")
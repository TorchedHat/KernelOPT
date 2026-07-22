import torch
import triton
import triton.language as tl
import math

# We'll implement this as multiple kernels:
# 1. Permute+copy kernel (x reshape)
# 2. GEMM for in_proj (32768 x 128) @ (128 x 384)
# 3. Bias add + QKV split
# 4. Attention QK^T BMM
# 5. Softmax
# 6. Attention AV BMM  
# 7. Output projection GEMM
# 8. Residual + LayerNorm

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(B_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def permute_kernel(
    x_ptr, out_ptr,
    B, C, HW,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # x: (B, C, HW) -> out: (HW, B, C)
    # x[b, c, hw] -> out[hw, b, c]
    # out index: hw * B * C + b * C + c
    # x index: b * C * HW + c * HW + hw
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # out linear index: offs
    # out[hw, b, c]: hw = offs // (B*C), b = (offs // C) % B, c = offs % C
    hw = offs // (B * C)
    b = (offs // C) % B
    c = offs % C
    x_idx = b * C * HW + c * HW + hw
    val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 128}, num_warps=2),
    ],
    key=['SEQ_LEN'],
)
@triton.jit
def softmax_kernel(
    inp_ptr, out_ptr,
    SEQ_LEN,
    BLOCK_N: tl.constexpr,
):
    # inp: (num_heads * batch * seq_len, seq_len) - each row is softmaxed
    row = tl.program_id(0)
    row_ptr = inp_ptr + row * SEQ_LEN
    out_row_ptr = out_ptr + row * SEQ_LEN

    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, SEQ_LEN, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < SEQ_LEN
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, SEQ_LEN, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < SEQ_LEN
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)

    # Normalize and store
    for off in range(0, SEQ_LEN, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < SEQ_LEN
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        out = tl.exp(x - row_max) / row_sum
        tl.store(out_row_ptr + cols, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    x_ptr, residual_ptr, weight_ptr, bias_ptr, out_ptr,
    N_ROWS, N_COLS,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * N_COLS
    res_row = residual_ptr + row * N_COLS
    out_row = out_ptr + row * N_COLS

    # Compute mean and variance
    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)

    for off in range(0, N_COLS, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        val = x + res
        mean += tl.sum(val, axis=0)

    mean = mean / N_COLS

    for off in range(0, N_COLS, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        val = x + res - mean
        var += tl.sum(val * val, axis=0)

    var = var / N_COLS
    inv_std = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N_COLS, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        val = (x + res - mean) * inv_std * w + b
        tl.store(out_row + cols, val.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(2)
    pid = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + batch * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + batch * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(B_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + batch * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=c_mask)


def kernel_function(p_attn_in_proj_bias, p_attn_in_proj_weight, p_attn_out_proj_weight, p_attn_out_proj_bias, p_norm_weight, p_norm_bias, x):
    B, C, H, W = x.shape
    HW = H * W
    SEQ_LEN = HW  # 16384
    BATCH = B     # 2
    EMBED = C     # 128
    NUM_HEADS = 4
    HEAD_DIM = EMBED // NUM_HEADS  # 32
    scale = 1.0 / math.sqrt(HEAD_DIM)  # 0.1767...

    # Step 1: Permute x from (B, C, H*W) to (H*W, B, C) = (16384, 2, 128)
    # x is (2, 128, 128, 128)
    x_flat = x.reshape(B, C, HW)  # (2, 128, 16384)
    N_perm = SEQ_LEN * BATCH * EMBED
    x_perm = torch.empty(SEQ_LEN, BATCH, EMBED, device=x.device, dtype=x.dtype)
    grid_perm = lambda meta: (triton.cdiv(N_perm, meta['BLOCK_SIZE']),)
    permute_kernel[grid_perm](
        x_flat, x_perm,
        BATCH, EMBED, HW,
        N=N_perm,
    )

    # Step 2: in_proj linear: (SEQ_LEN*BATCH, EMBED) @ (EMBED, 3*EMBED) + bias
    # x_perm: (16384, 2, 128) -> reshape to (32768, 128)
    x_2d = x_perm.reshape(SEQ_LEN * BATCH, EMBED)
    qkv_out = torch.empty(SEQ_LEN * BATCH, 3 * EMBED, device=x.device, dtype=x.dtype)
    M_mm = SEQ_LEN * BATCH  # 32768
    N_mm = 3 * EMBED         # 384
    K_mm = EMBED              # 128
    grid_mm = lambda meta: (triton.cdiv(M_mm, meta['BLOCK_M']) * triton.cdiv(N_mm, meta['BLOCK_N']),)
    matmul_kernel[grid_mm](
        x_2d, p_attn_in_proj_weight, qkv_out,
        M_mm, N_mm, K_mm,
        x_2d.stride(0), x_2d.stride(1),
        p_attn_in_proj_weight.stride(1), p_attn_in_proj_weight.stride(0),
        qkv_out.stride(0), qkv_out.stride(1),
    )
    # Add bias
    qkv_out = qkv_out + p_attn_in_proj_bias  # broadcast

    # Step 3: Reshape QKV and split
    # qkv_out: (32768, 384) -> (16384, 2, 3, 128)
    qkv = qkv_out.reshape(SEQ_LEN, BATCH, 3, EMBED)
    # Q, K, V: each (16384, 2, 128)
    Q = qkv[:, :, 0, :].contiguous()  # (16384, 2, 128)
    K = qkv[:, :, 1, :].contiguous()  # (16384, 2, 128)
    V = qkv[:, :, 2, :].contiguous()  # (16384, 2, 128)

    # Reshape for multi-head attention
    # Q: (16384, 2, 128) -> (16384, 2, 4, 32) -> (2*4, 16384, 32) = (8, 16384, 32)
    Q = Q.reshape(SEQ_LEN, BATCH, NUM_HEADS, HEAD_DIM).permute(1, 2, 0, 3).reshape(BATCH * NUM_HEADS, SEQ_LEN, HEAD_DIM).contiguous()
    K = K.reshape(SEQ_LEN, BATCH, NUM_HEADS, HEAD_DIM).permute(1, 2, 0, 3).reshape(BATCH * NUM_HEADS, SEQ_LEN, HEAD_DIM).contiguous()
    V = V.reshape(SEQ_LEN, BATCH, NUM_HEADS, HEAD_DIM).permute(1, 2, 0, 3).reshape(BATCH * NUM_HEADS, SEQ_LEN, HEAD_DIM).contiguous()

    # Scale Q
    Q = Q * scale

    # Step 4: Attention scores: (8, 16384, 32) @ (8, 32, 16384) -> (8, 16384, 16384)
    NUM_BATCHES = BATCH * NUM_HEADS  # 8
    attn_scores = torch.empty(NUM_BATCHES, SEQ_LEN, SEQ_LEN, device=x.device, dtype=x.dtype)

    # BMM: Q @ K^T
    # Q: (8, 16384, 32), K: (8, 16384, 32) -> K^T: (8, 32, 16384)
    # attn_scores: (8, 16384, 16384)
    grid_bmm = lambda meta: (
        triton.cdiv(SEQ_LEN, meta['BLOCK_M']) * triton.cdiv(SEQ_LEN, meta['BLOCK_N']),
        1,
        NUM_BATCHES,
    )
    # Actually need 2D grid for M and N
    def grid_bmm_fn(meta):
        return (triton.cdiv(SEQ_LEN, meta['BLOCK_M']), triton.cdiv(SEQ_LEN, meta['BLOCK_N']), NUM_BATCHES)

    bmm_kernel[grid_bmm_fn](
        Q, K, attn_scores,
        SEQ_LEN, SEQ_LEN, HEAD_DIM,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(2), K.stride(1),  # K transposed: stride_bk=K.stride(2), stride_bn=K.stride(1)
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2),
    )

    # Step 5: Softmax over last dim
    attn_probs = torch.empty_like(attn_scores)
    # Reshape to 2D for softmax: (8*16384, 16384)
    attn_scores_2d = attn_scores.reshape(NUM_BATCHES * SEQ_LEN, SEQ_LEN)
    attn_probs_2d = attn_probs.reshape(NUM_BATCHES * SEQ_LEN, SEQ_LEN)
    grid_softmax = (NUM_BATCHES * SEQ_LEN,)
    softmax_kernel[grid_softmax](
        attn_scores_2d, attn_probs_2d,
        SEQ_LEN,
    )

    # Step 6: Attention output: (8, 16384, 16384) @ (8, 16384, 32) -> (8, 16384, 32)
    attn_out = torch.empty(NUM_BATCHES, SEQ_LEN, HEAD_DIM, device=x.device, dtype=x.dtype)
    grid_bmm2_fn = lambda meta: (triton.cdiv(SEQ_LEN, meta['BLOCK_M']), triton.cdiv(HEAD_DIM, meta['BLOCK_N']), NUM_BATCHES)
    bmm_kernel[grid_bmm2_fn](
        attn_probs, V, attn_out,
        SEQ_LEN, HEAD_DIM, SEQ_LEN,
        attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
    )

    # Step 7: Reshape attn_out back
    # attn_out: (8, 16384, 32) -> (2, 4, 16384, 32) -> (16384, 2, 4, 32) -> (16384, 2, 128) -> (32768, 128)
    attn_out_reshaped = attn_out.reshape(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM).permute(2, 0, 1, 3).reshape(SEQ_LEN * BATCH, EMBED).contiguous()

    # Step 8: Output projection: (32768, 128) @ (128, 128) + bias
    out_proj_out = torch.empty(SEQ_LEN * BATCH, EMBED, device=x.device, dtype=x.dtype)
    M_out = SEQ_LEN * BATCH
    N_out = EMBED
    K_out = EMBED
    grid_out = lambda meta: (triton.cdiv(M_out, meta['BLOCK_M']) * triton.cdiv(N_out, meta['BLOCK_N']),)
    matmul_kernel[grid_out](
        attn_out_reshaped, p_attn_out_proj_weight, out_proj_out,
        M_out, N_out, K_out,
        attn_out_reshaped.stride(0), attn_out_reshaped.stride(1),
        p_attn_out_proj_weight.stride(1), p_attn_out_proj_weight.stride(0),
        out_proj_out.stride(0), out_proj_out.stride(1),
    )
    out_proj_out = out_proj_out + p_attn_out_proj_bias

    # Step 9: Reshape to (16384, 2, 128) for residual + layernorm
    attn_result = out_proj_out.reshape(SEQ_LEN, BATCH, EMBED)
    x_residual = x_perm  # (16384, 2, 128)

    # Step 10: Residual + LayerNorm
    # Both are (16384, 2, 128), layernorm over last dim
    # Reshape to (16384*2, 128) = (32768, 128)
    attn_result_2d = attn_result.reshape(SEQ_LEN * BATCH, EMBED)
    x_residual_2d = x_residual.reshape(SEQ_LEN * BATCH, EMBED)
    ln_out = torch.empty(SEQ_LEN * BATCH, EMBED, device=x.device, dtype=x.dtype)

    grid_ln = (SEQ_LEN * BATCH,)
    layernorm_kernel[grid_ln](
        attn_result_2d, x_residual_2d, p_norm_weight, p_norm_bias, ln_out,
        SEQ_LEN * BATCH, EMBED,
        N=EMBED,
        eps=1e-5,
    )

    # Step 11: Reshape back to (B, C, H, W)
    # ln_out: (32768, 128) -> (16384, 2, 128) -> permute(1,2,0) -> (2, 128, 16384) -> (2, 128, 128, 128)
    result = ln_out.reshape(SEQ_LEN, BATCH, EMBED).permute(1, 2, 0).reshape(B, C, H, W)
    return result


def get_inputs():
    return [torch.rand(2, 128, 128, 128)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    # Initialize model parameters
    embed_dim = 128
    num_heads = 4
    torch.manual_seed(42)

    p_attn_in_proj_bias = torch.randn(384, device='cuda')
    p_attn_in_proj_weight = torch.randn(384, 128, device='cuda') * 0.02
    p_attn_out_proj_weight = torch.randn(128, 128, device='cuda') * 0.02
    p_attn_out_proj_bias = torch.randn(128, device='cuda') * 0.02
    p_norm_weight = torch.ones(128, device='cuda')
    p_norm_bias = torch.zeros(128, device='cuda')
    x = get_inputs()[0].cuda()

    output = kernel_function(
        p_attn_in_proj_bias, p_attn_in_proj_weight,
        p_attn_out_proj_weight, p_attn_out_proj_bias,
    )
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self, embed_dim, num_heads):
                super(Model, self).__init__()
                self.attn = nn.MultiheadAttention(embed_dim, num_heads)
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, x):
                B, C, H, W = x.shape
                x_seq = x.view(B, C, H * W).permute(2, 0, 1)
                attn_output, _ = self.attn(x_seq, x_seq, x_seq)
                x_out = self.norm(attn_output + x_seq)
                x_out = x_out.permute(1, 2, 0).view(B, C, H, W)
                return x_out

        model = Model(embed_dim, num_heads).cuda()
        # Set weights to match our kernel
        with torch.no_grad():
            model.attn.in_proj_weight.copy_(p_attn_in_proj_weight)
            model.attn.in_proj_bias.copy_(p_attn_in_proj_bias)
            model.attn.out_proj.weight.copy_(p_attn_out_proj_weight)
            model.attn.out_proj.bias.copy_(p_attn_out_proj_bias)
            model.norm.weight.copy_(p_norm_weight)
            model.norm.bias.copy_(p_norm_bias)

        ref = model(x)
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")
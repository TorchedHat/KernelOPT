import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─── Kernel 1: permute + in_proj GEMM ───────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def in_proj_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Computes Out = X @ W^T + B
    X: (M, K) where M = seq_len * batch, K = embed_dim
    W: (N, K) where N = 3 * embed_dim
    
    i.e., stride_xm = K (embed_dim), stride_xk = 1 for contiguous
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        # Load X tile: X[offs_m, k_offs]
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load W tile: W[offs_n, k_offs] (we compute X @ W^T)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias
    b_offs = offs_n
    b_mask = b_offs < N
    bias = tl.load(B_ptr + b_offs, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]
    
    # Store
    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ─── Kernel 2: out_proj GEMM ─────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def out_proj_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Computes Out = X @ W^T + B
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        x_ptrs = X_ptr + offs_m[:, None] * K + k_offs[None, :]
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = W_ptr + offs_n[None, :] * K + k_offs[:, None]
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ─── Kernel 3: Softmax over last dim ─────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    Out_ptr, In_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    
    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + offs
        mask = cols < N
        x = tl.load(In_ptr + row * N + cols, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + offs
        mask = cols < N
        x = tl.load(In_ptr + row * N + cols, mask=mask, other=float('-inf'))
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)
    
    # Normalize
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + offs
        mask = cols < N
        x = tl.load(In_ptr + row * N + cols, mask=mask, other=float('-inf'))
        out = tl.exp(x - row_max) / row_sum
        tl.store(Out_ptr + row * N + cols, out, mask=mask)


# ─── Kernel 4: BMM A(batch, M, K) @ B(batch, K, N) -> C(batch, M, N) ─────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['BATCH', 'M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    BATCH, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    A_base = A_ptr + pid_b * stride_ab
    B_base = B_ptr + pid_b * stride_bb
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        a_ptrs = A_base + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_ptrs = B_base + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc = tl.dot(a_tile, b_tile, acc)
    
    C_base = C_ptr + pid_b * stride_cb
    c_ptrs = C_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


# ─── Kernel 5: LayerNorm ──────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    Out_ptr, In_ptr, Residual_ptr, W_ptr, B_ptr,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    """
    out[i] = layernorm(in[i] + residual[i]) * w + b
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    
    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + offs
        mask = cols < N
        x = tl.load(In_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(Residual_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x + r, axis=0)
    mean = mean / N
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + offs
        mask = cols < N
        x = tl.load(In_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(Residual_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        diff = (x + r) - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)
    
    # Normalize and scale
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + offs
        mask = cols < N
        x = tl.load(In_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(Residual_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = ((x + r) - mean) * rstd * w + b
        tl.store(Out_ptr + row * N + cols, out.to(tl.float32), mask=mask)


# ─── Kernel 6: Permute copy for output ───────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def permute_copy_kernel(
    In_ptr, Out_ptr,
    B, C, HW,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    """
    In[hw, b, c] -> Out[b, c, hw]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    
    # offs = b * C * HW + c * HW + hw
    hw = offs % HW
    tmp = offs // HW
    c = tmp % C
    b = tmp // C
    
    in_idx = hw * B * C + b * C + c
    in_val = tl.load(In_ptr + in_idx, mask=mask, other=0.0)
    tl.store(Out_ptr + offs, in_val, mask=mask)


def kernel_function(
    p_attn_in_proj_bias,
    p_attn_in_proj_weight,
    p_attn_out_proj_weight,
    p_attn_out_proj_bias,
    p_norm_weight,
    p_norm_bias,
    x,
):
    """
    """
    B, C, H, W = x.shape
    HW = H * W
    num_heads = 4  # from get_init_inputs
    head_dim = C // num_heads  # 128 // 4 = 32... wait, but inductor shows 8 heads with head_dim=32
    # Actually from the computation graph: view to [16384, 8, 32] means num_heads=8, head_dim=32
    # But get_init_inputs says num_heads=4... Let me re-check
    # embed_dim=128, num_heads=4 -> head_dim = 32, but graph shows 8 heads
    # Actually nn.MultiheadAttention with num_heads=4 and embed_dim=128 -> head_dim=32
    # But the graph shows [16384, 8, 32]... that's 8 heads with head_dim=32 = 256 != 128
    # Wait: 8 * 32 = 256 != 128. Let me re-read.
    # The graph shows view to [16384, 8, 32] from [16384, 2, 128]
    # 2 * 128 = 256 = 8 * 32. So it's treating batch=2 as part of the sequence!
    # Actually: select gives [16384, 2, 128], then view to [16384, 8, 32]
    # 16384 * 2 * 128 = 16384 * 256 = 4194304
    # [16384, 8, 32]: 16384 * 8 * 32 = 4194304. Yes!
    # So num_heads=8 (not 4), head_dim=32
    # But get_init_inputs says num_heads=4... 
    # nn.MultiheadAttention(128, 4) -> num_heads=4, head_dim=32
    # But the graph shows 8 heads... Let me look more carefully.
    # select gives [16384, 2, 128] (seq_len=16384, batch=2, embed=128)
    # view to [16384, 8, 32]: this is (seq_len, num_heads, head_dim) with num_heads=8? 
    # But 8*32=256 != 128. Unless it's (seq_len*batch, num_heads, head_dim)?
    # 16384 * 2 = 32768 tokens total, but select gives [16384, 2, 128]...
    # Actually view [16384, 2, 128] -> [16384, 8, 32]:
    # 2 * 128 = 256 = 8 * 32. So the last two dims are merged: (2*128) = (8*32)
    # This means num_heads=8 and head_dim=32, treating batch as part of heads!
    # That's how PyTorch's MHA works internally when batch_first=False
    # With num_heads=4, embed_dim=128: head_dim=32, but internally it processes
    # (seq_len, batch*num_heads, head_dim) = (16384, 2*4, 32) = (16384, 8, 32)
    # Yes! That's it. batch * num_heads = 2 * 4 = 8.
    
    num_heads_actual = 4  # from model
    head_dim = C // num_heads_actual  # 32
    num_heads_total = B * num_heads_actual  # 8 (batch * num_heads)
    seq_len = HW  # 16384
    
    # Step 1: in_proj: (seq_len, B, C) @ W_in^T + b_in -> (seq_len, B, 3*C)
    # x is (B, C, H, W), we need to treat it as (seq_len, B, C) with strides
    # x.view(B, C, HW).permute(2, 0, 1) -> (HW, B, C)
    # stride of x: (C*HW, HW, 1) after view(B, C, HW)
    # after permute(2,0,1): stride = (1, C*HW, HW) but we need contiguous
    
    # For the GEMM, we need X as (seq_len*B, C) = (32768, 128)
    # X[i*B+j, k] = x[j, k, i] (where i=hw index, j=batch, k=channel)
    # stride_xm for row (i*B+j): moving j by 1 -> stride C*HW... no
    # Let's think: row index = hw * B + b
    # x[b, c, hw] has linear index b*C*HW + c*HW + hw
    # For row (hw*B + b), col c: index = b*C*HW + c*HW + hw
    # stride_xm (increment row by 1, i.e., b by 1 within same hw): C*HW
    # But when hw increments (row += B): stride = 1 (hw increments by 1)
    # This is non-uniform stride, so we need to handle it carefully.
    # 
    # Actually, let's just do the permute+copy first, then GEMM.
    
    # Allocate contiguous (seq_len, B, C) = (16384, 2, 128)
    x_perm = torch.empty((seq_len, B, C), dtype=x.dtype, device=x.device)
    
    # Copy x (B, C, HW) -> x_perm (HW, B, C)
    total_perm = seq_len * B * C
    grid_perm = lambda meta: (triton.cdiv(total_perm, meta['BLOCK_SIZE']),)
    permute_copy_kernel[grid_perm](
        x, x_perm,
        B, C, HW,
        total_perm,
    )
    
    # in_proj: (seq_len*B, C) @ W_in^T + b -> (seq_len*B, 3*C)
    M_in = seq_len * B  # 32768
    N_in = 3 * C        # 384
    K_in = C            # 128
    
    qkv = torch.empty((M_in, N_in), dtype=x.dtype, device=x.device)
    
    grid_in = lambda meta: (
        triton.cdiv(M_in, meta['BLOCK_M']),
        triton.cdiv(N_in, meta['BLOCK_N']),
    )
    in_proj_kernel[grid_in](
        x_perm, p_attn_in_proj_weight, p_attn_in_proj_bias, qkv,
        M_in, N_in, K_in,
        K_in, 1,   # stride_xm=K, stride_xk=1 (contiguous)
        K_in, 1,   # stride_wn=K, stride_wk=1 (W is (N,K) contiguous)
    )
    # qkv: (32768, 384) = (seq_len*B, 3*C)
    # Reshape to (seq_len, B, 3, C) -> split Q,K,V each (seq_len, B, C)
    # Then reshape each to (seq_len, B*num_heads, head_dim) = (16384, 8, 32)
    # Then transpose to (B*num_heads, seq_len, head_dim) = (8, 16384, 32)
    
    qkv_3d = qkv.view(seq_len, B, 3, C)  # (16384, 2, 3, 128)
    # Q = qkv_3d[:, :, 0, :] -> (16384, 2, 128) -> (16384, 8, 32) -> (8, 16384, 32)
    # K = qkv_3d[:, :, 1, :] -> same
    # V = qkv_3d[:, :, 2, :] -> same
    
    Q = qkv_3d[:, :, 0, :].reshape(seq_len, num_heads_total, head_dim).permute(1, 0, 2).contiguous()
    K = qkv_3d[:, :, 1, :].reshape(seq_len, num_heads_total, head_dim).permute(1, 0, 2).contiguous()
    V = qkv_3d[:, :, 2, :].reshape(seq_len, num_heads_total, head_dim).permute(1, 0, 2).contiguous()
    # Q, K, V: (8, 16384, 32)
    
    scale = head_dim ** -0.5  # 1/sqrt(32) ≈ 0.1768
    
    # Scale Q
    Q_scaled = torch.empty_like(Q)
    total_q = num_heads_total * seq_len * head_dim
    
    @triton.jit
    def scale_kernel(In_ptr, Out_ptr, scale, total, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        x = tl.load(In_ptr + offs, mask=mask, other=0.0)
        tl.store(Out_ptr + offs, x * scale, mask=mask)
    
    # Actually let's just use a simple pointwise multiply inline
    Q_scaled = Q * scale
    
    # Attention scores: (8, 16384, 32) @ (8, 32, 16384) -> (8, 16384, 16384)
    # This is a huge matrix! 8 * 16384 * 16384 * 4 bytes = 8GB. That's too large.
    # But the spec says to implement it... let's proceed.
    
    attn_scores = torch.empty((num_heads_total, seq_len, seq_len), dtype=x.dtype, device=x.device)
    
    # BMM: Q_scaled (8, 16384, 32) @ K^T (8, 32, 16384) -> (8, 16384, 16384)
    grid_bmm1 = lambda meta: (
        triton.cdiv(seq_len, meta['BLOCK_M']),
        triton.cdiv(seq_len, meta['BLOCK_N']),
        num_heads_total,
    )
    bmm_kernel[grid_bmm1](
        Q_scaled, K, attn_scores,
        num_heads_total, seq_len, seq_len, head_dim,
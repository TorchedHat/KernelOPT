import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Fused LayerNorm kernel - single pass mean+var+normalize+scale
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=16, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_layer_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, eps,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    # Pointers to this row
    x_row = X_ptr + row * stride_xm
    y_row = Y_ptr + row * stride_ym

    # Pass 1: compute mean and variance
    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        # Welford online update
        count_new = count + tl.sum(mask.to(tl.float32), axis=0)
        delta = x - mean
        mean_new = mean + tl.sum(tl.where(mask, delta / tl.maximum(count_new, 1.0), 0.0), axis=0)
        delta2 = x - mean_new
        var = var + tl.sum(tl.where(mask, delta * delta2, 0.0), axis=0)
        mean = mean_new
        count = count_new

    var = var / N
    rstd = libdevice.rsqrt(var + eps)

    # Pass 2: normalize and scale
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(y_row + cols * stride_yn, y.to(tl.float32), mask=mask)


def fused_layer_norm(x, weight, bias, eps=1e-5):
    """Fused LayerNorm using Triton kernel."""
    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N
    x_2d = x.reshape(M, N)
    y = torch.empty_like(x_2d)

    grid = (M,)
    fused_layer_norm_kernel[grid](
        x_2d, weight, bias, y,
        M, N, eps,
        x_2d.stride(0), x_2d.stride(1),
        y.stride(0), y.stride(1),
    )
    return y.reshape(orig_shape)


# Tiled GEMM kernel for the QKV projection (M x K) @ (K x N) -> (M x N)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=c_mask)


def triton_linear(x, weight, bias=None):
    """Linear layer using Triton GEMM kernel."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]
    # weight is (N, K), we compute x @ weight.T
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    matmul_kernel[grid](
        x_2d, weight.t().contiguous(), out, bias if bias is not None else x_2d,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.t().contiguous().stride(0), weight.t().contiguous().stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return out.reshape(*orig_shape[:-1], N)


# Fused softmax kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    x_row = X_ptr + row * stride_xm
    y_row = Y_ptr + row * stride_ym

    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.where(mask, tl.exp(x - row_max), 0.0), axis=0)

    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=float('-inf')).to(tl.float32)
        y = tl.exp(x - row_max) / row_sum
        tl.store(y_row + cols * stride_yn, y.to(tl.float32), mask=mask)


def triton_softmax(x, dim=-1):
    """Softmax using Triton kernel."""
    assert dim == -1 or dim == x.dim() - 1
    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N
    x_2d = x.reshape(M, N).contiguous()
    y = torch.empty_like(x_2d)
    grid = (M,)
    softmax_kernel[grid](
        x_2d, y,
        M, N,
        x_2d.stride(0), x_2d.stride(1),
        y.stride(0), y.stride(1),
    )
    return y.reshape(orig_shape)


# Fused GELU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    X_ptr, Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    y = x * cdf
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


def triton_gelu(x):
    """GELU activation using Triton kernel."""
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, y, N)
    return y


# Fused sigmoid kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def sigmoid_kernel(
    X_ptr, Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


def triton_sigmoid(x):
    """Sigmoid activation using Triton kernel."""
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, y, N)
    return y


# Fused ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def relu_kernel(
    X_ptr, Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


def triton_relu(x):
    """ReLU activation using Triton kernel."""
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, y, N)
    return y


# ============================================================
# Swin Transformer V2 implementation using Triton primitives
# ============================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def swin_window_attention(
    x, qkv_weight, qkv_bias_full,
    proj_weight, proj_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    relative_coords_table, relative_position_index,
    attn_mask,
    num_heads, head_dim,
):
    """Window attention with cosine similarity."""
    B_, N, C = x.shape
    
    # QKV projection
    qkv = triton_linear(x, qkv_weight, qkv_bias_full)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # qkv: (3, B_, num_heads, N, head_dim) - wait, need to check
    # Actually: reshape to (B_, N, 3, num_heads, head_dim), permute to (3, B_, num_heads, N, head_dim)
    qkv = x.new_empty(0)  # placeholder
    
    # Redo properly
    qkv_out = triton_linear(x, qkv_weight, qkv_bias_full)
    qkv_r = qkv_out.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    # shape: (3, B_, num_heads, N, head_dim)
    q, k, v = qkv_r[0], qkv_r[1], qkv_r[2]
    
    # Normalize q and k
    q_norm = torch.nn.functional.normalize(q, dim=-1)
    k_norm = torch.nn.functional.normalize(k, dim=-1)
    
    # Logit scale
    scale = torch.clamp(logit_scale, max=torch.log(torch.tensor(1.0 / 0.01, device=x.device))).exp()
    
    # Attention weights
    attn = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * scale
    
    # Relative position bias via CPB MLP
    cpb_out = triton_linear(relative_coords_table.view(-1, 2), cpb_mlp_0_weight, cpb_mlp_0_bias)
    cpb_out = triton_relu(cpb_out)
    cpb_out = triton_linear(cpb_out, cpb_mlp_2_weight, None)
    
    rel_pos_bias_table = cpb_out.view(-1, num_heads)
    idx = relative_position_index.view(-1)
    rel_pos_bias = rel_pos_bias_table[idx].view(N, N, num_heads)
    rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
    rel_pos_bias = triton_sigmoid(rel_pos_bias) * 16
    
    attn = attn + rel_pos_bias.unsqueeze(0)
    
    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)
    
    attn = triton_softmax(attn, dim=-1)
    
    # Attention output
    x_out = torch.matmul(attn, v)
    x_out = x_out.transpose(1, 2).reshape(B_, N, C)
    x_out = triton_linear(x_out, proj_weight, proj_bias)
    
    return x_out


def swin_block(
    x, H, W, window_size, shift_size,
    # Attention params
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    relative_coords_table, relative_position_index,
    attn_mask,
    norm1_weight, norm1_bias,
    # MLP params
    mlp_fc1_weight, mlp_fc1_bias,
    mlp_fc2_weight, mlp_fc2_bias,
    norm2_weight, norm2_bias,
    num_heads,
):
    B, L, C = x.shape
    head_dim = C // num_heads
    
    # Build qkv_bias
    zeros = torch.zeros_like(v_bias)
    qkv_bias = torch.cat([q_bias, zeros, v_bias])
    
    # Norm1
    shortcut = x
    x_norm = fused_layer_norm(x, norm1_weight, norm1_bias)
    
    x_2d = x_norm.view(B, H, W, C)
    
    # Cyclic shift
    if shift_size > 0:
        shifted_x = torch.roll(x_2d, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x_2d
    
    # Window partition
    x_windows = window_partition(shifted_x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)
    
    # Window attention
    attn_windows = swin_window_attention(
        x_windows, qkv_weight, qkv_bias,
        proj_weight, proj_bias,
        logit_scale,
        cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
        relative_coords_table, relative_position_index,
        attn_mask,
        num_heads, head_dim,
    )
    
    # Reverse window
    attn_windows = attn_windows.view(-1, window_size, window_size, C)
    shifted_x = window_reverse(attn_windows, window_size, H, W)
    
    # Reverse cyclic shift
    if shift_size > 0:
        x_out = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x_out = shifted_x
    
    x_out = x_out.view(B, H * W, C)
    x = shortcut + x_out
    
    # MLP
    shortcut2 = x
    x_norm2 = fused_layer_norm(x, norm2_weight, norm2_bias)
    x_mlp = triton_linear(x_norm2, mlp_fc1_weight, mlp_fc1_bias)
    x_mlp = triton_gelu(x_mlp)
    x_mlp = triton_linear(x_mlp, mlp_fc2_weight, mlp_fc2_bias)
    x = shortcut2 + x_mlp
    
    return x


def patch_merging(x, H, W, reduction_weight, norm_weight, norm_bias):
    """Patch merging / downsampling."""
    B, L, C = x.shape
    x_2d = x.view(B, H, W, C)
    
    x0 = x_2d[:, 0::2, 0::2, :]
    x1 = x_2d[:, 1::2, 0::2, :]
    x2 = x_2d[:, 0::2, 1::2, :]
    x3 = x_2d[:, 1::2, 1::2, :]
    
    x_cat = torch.cat([x0, x1, x2, x3], dim=-1)
    x_cat = x_cat.view(B, -1, 4 * C)
    
    x_out = triton_linear(x_cat, reduction_weight, None)
    x_out = fused_layer_norm(x_out, norm_weight, norm_bias)
    
    return x_out


def kernel_function(
    p_patch_embed_proj_weight,
    p_patch_embed_proj_bias,
    p_patch_embed_norm_weight,
    p_patch_embed_norm_bias,
    p_layers_0_blocks_0_attn_q_bias,
    p_layers_0_blocks_0_attn_v_bias,
    p_layers_0_blocks_0_attn_qkv_weight,
    p_layers_0_blocks_0_attn_logit_scale,
    p_layers_0_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_0_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_0_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_0_blocks_0_attn_proj_weight,
    p_layers_0_blocks_0_attn_proj_bias,
    p_layers_0_blocks_0_norm1_weight,
    p_layers_0_blocks_0_norm1_bias,
    p_layers_0_blocks_0_mlp_fc1_weight,
    p_layers_0_blocks_0_mlp_fc1_bias,
    p_layers_0_blocks_0_mlp_fc2_weight,
    p_layers_0_blocks_0_mlp_fc2_bias,
    p_layers_0_blocks_0_norm2_weight,
    p_layers_0_blocks_0_norm2_bias,
    p_layers_0_blocks_1_attn_q_bias,
    p_layers_0_blocks_1_attn_v_bias,
    p_layers_0_blocks_1_attn_qkv_weight,
    p_layers_0_blocks_1_attn_logit_scale,
    p_layers_0_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_0_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_0_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_0_blocks_1_attn_proj_weight,
    p_layers_0_blocks_1_attn_proj_bias,
    p_layers_0_blocks_1_norm1_weight,
    p_layers_0_blocks_1_norm1_bias,
    p_layers_0_blocks_1_mlp_fc1_weight,
    p_layers_0_blocks_1_mlp_fc1_bias,
    p_layers_0_blocks_1_mlp_fc2_weight,
    p_layers_0_blocks_1_mlp_fc2_bias,
    p_layers_0_blocks_1_norm2_weight,
    p_layers_0_blocks_1_norm2_bias,
    p_layers_0_downsample_reduction_weight,
    p_layers_0_downsample_norm_weight,
    p_layers_0_downsample_norm_bias,
    p_layers_1_blocks_0_attn_q_bias,
    p_layers_1_blocks_0_attn_v_bias,
    p_layers_1_blocks_0_attn_qkv_weight,
    p_layers_1_blocks_0_attn_logit_scale,
    p_layers_1_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_1_blocks_0_attn_cpb_mlp_0_bias,
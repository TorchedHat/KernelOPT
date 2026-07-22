import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N,
    stride_m,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused LayerNorm kernel: computes mean, variance, normalize, scale, shift in one pass."""
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    row_start = row_idx * stride_m
    
    # First pass: compute mean
    mean = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)
    
    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x * mask.to(tl.float32), axis=0)
        count += tl.sum(mask.to(tl.float32), axis=0)
    
    mean = mean / count
    
    # Second pass: compute variance
    var = tl.zeros((), dtype=tl.float32)
    
    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        diff = (x - mean) * mask.to(tl.float32)
        var += tl.sum(diff * diff, axis=0)
    
    var = var / count
    rstd = 1.0 / libdevice.sqrt(var + eps)
    
    # Third pass: normalize and store
    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        x_norm = (x - mean) * rstd
        out = x_norm * w + b
        
        tl.store(out_ptr + row_start + cols, out.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Basic tiled matrix multiplication kernel."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=a_mask, other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=b_mask, other=0.0)
        
        acc = tl.dot(a, b, acc=acc)
    
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float32), mask=c_mask)


def layernorm_triton(x, weight, bias, eps=1e-5):
    """Apply LayerNorm using Triton kernel."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    
    out = torch.empty_like(x_2d)
    
    grid = (M,)
    fused_layernorm_kernel[grid](
        x_2d, weight, bias, out,
        M, N,
        x_2d.stride(0),
        eps,
    )
    
    return out.reshape(orig_shape)


def matmul_triton(a, b):
    """Matrix multiplication using Triton kernel."""
    assert a.shape[-1] == b.shape[0], f"Shape mismatch: {a.shape} vs {b.shape}"
    
    if a.dim() > 2:
        orig_shape = a.shape
        a_2d = a.reshape(-1, a.shape[-1])
        M, K = a_2d.shape
        N = b.shape[1] if b.dim() == 2 else b.shape[-1]
        
        out = torch.empty((M, N), device=a.device, dtype=torch.float32)
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        matmul_kernel[grid](
            a_2d, b, out,
            M, N, K,
            a_2d.stride(0), a_2d.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
        )
        
        return out.reshape(*orig_shape[:-1], N)
    else:
        M, K = a.shape
        N = b.shape[1]
        
        out = torch.empty((M, N), device=a.device, dtype=torch.float32)
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        matmul_kernel[grid](
            a, b, out,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
        )
        
        return out


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
    p_layers_1_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_1_blocks_0_attn_proj_weight,
    p_layers_1_blocks_0_attn_proj_bias,
    p_layers_1_blocks_0_norm1_weight,
    p_layers_1_blocks_0_norm1_bias,
    p_layers_1_blocks_0_mlp_fc1_weight,
    p_layers_1_blocks_0_mlp_fc1_bias,
    p_layers_1_blocks_0_mlp_fc2_weight,
    p_layers_1_blocks_0_mlp_fc2_bias,
    p_layers_1_blocks_0_norm2_weight,
    p_layers_1_blocks_0_norm2_bias,
    p_layers_1_blocks_1_attn_q_bias,
    p_layers_1_blocks_1_attn_v_bias,
    p_layers_1_blocks_1_attn_qkv_weight,
    p_layers_1_blocks_1_attn_logit_scale,
    p_layers_1_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_1_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_1_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_1_blocks_1_attn_proj_weight,
    p_layers_1_blocks_1_attn_proj_bias,
    p_layers_1_blocks_1_norm1_weight,
    p_layers_1_blocks_1_norm1_bias,
    p_layers_1_blocks_1_mlp_fc1_weight,
    p_layers_1_blocks_1_mlp_fc1_bias,
    p_layers_1_blocks_1_mlp_fc2_weight,
    p_layers_1_blocks_1_mlp_fc2_bias,
    p_layers_1_blocks_1_norm2_weight,
    p_layers_1_blocks_1_norm2_bias,
    p_layers_1_downsample_reduction_weight,
    p_layers_1_downsample_norm_weight,
    p_layers_1_downsample_norm_bias,
    p_layers_2_blocks_0_attn_q_bias,
    p_layers_2_blocks_0_attn_v_bias,
    p_layers_2_blocks_0_attn_qkv_weight,
    p_layers_2_blocks_0_attn_logit_scale,
    p_layers_2_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_0_attn_proj_weight,
    p_layers_2_blocks_0_attn_proj_bias,
    p_layers_2_blocks_0_norm1_weight,
    p_layers_2_blocks_0_norm1_bias,
    p_layers_2_blocks_0_mlp_fc1_weight,
    p_layers_2_blocks_0_mlp_fc1_bias,
    p_layers_2_blocks_0_mlp_fc2_weight,
    p_layers_2_blocks_0_mlp_fc2_bias,
    p_layers_2_blocks_0_norm2_weight,
    p_layers_2_blocks_0_norm2_bias,
    p_layers_2_blocks_1_attn_q_bias,
    p_layers_2_blocks_1_attn_v_bias,
    p_layers_2_blocks_1_attn_qkv_weight,
    p_layers_2_blocks_1_attn_logit_scale,
    p_layers_2_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_1_attn_proj_weight,
    p_layers_2_blocks_1_attn_proj_bias,
    p_layers_2_blocks_1_norm1_weight,
    p_layers_2_blocks_1_norm1_bias,
    p_layers_2_blocks_1_mlp_fc1_weight,
    p_layers_2_blocks_1_mlp_fc1_bias,
    p_layers_2_blocks_1_mlp_fc2_weight,
    p_layers_2_blocks_1_mlp_fc2_bias,
    p_layers_2_blocks_1_norm2_weight,
    p_layers_2_blocks_1_norm2_bias,
    p_layers_2_blocks_2_attn_q_bias,
    p_layers_2_blocks_2_attn_v_bias,
    p_layers_2_blocks_2_attn_qkv_weight,
    p_layers_2_blocks_2_attn_logit_scale,
    p_layers_2_blocks_2_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_2_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_2_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_2_attn_proj_weight,
    p_layers_2_blocks_2_attn_proj_bias,
    p_layers_2_blocks_2_norm1_weight,
    p_layers_2_blocks_2_norm1_bias,
    p_layers_2_blocks_2_mlp_fc1_weight,
    p_layers_2_blocks_2_mlp_fc1_bias,
    p_layers_2_blocks_2_mlp_fc2_weight,
    p_layers_2_blocks_2_mlp_fc2_bias,
    p_layers_2_blocks_2_norm2_weight,
    p_layers_2_blocks_2_norm2_bias,
    p_layers_2_blocks_3_attn_q_bias,
    p_layers_2_blocks_3_attn_v_bias,
    p_layers_2_blocks_3_attn_qkv_weight,
    p_layers_2_blocks_3_attn_logit_scale,
    p_layers_2_blocks_3_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_3_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_3_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_3_attn_proj_weight,
    p_layers_2_blocks_3_attn_proj_bias,
    p_layers_2_blocks_3_norm1_weight,
    p_layers_2_blocks_3_norm1_bias,
    p_layers_2_blocks_3_mlp_fc1_weight,
    p_layers_2_blocks_3_mlp_fc1_bias,
    p_layers_2_blocks_3_mlp_fc2_weight,
    p_layers_2_blocks_3_mlp_fc2_bias,
    p_layers_2_blocks_3_norm2_weight,
    p_layers_2_blocks_3_norm2_bias,
    p_layers_2_blocks_4_attn_q_bias,
    p_layers_2_blocks_4_attn_v_bias,
    p_layers_2_blocks_4_attn_qkv_weight,
    p_layers_2_blocks_4_attn_logit_scale,
    p_layers_2_blocks_4_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_4_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_4_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_4_attn_proj_weight,
    p_layers_2_blocks_4_attn_proj_bias,
    p_layers_2_blocks_4_norm1_weight,
    p_layers_2_blocks_4_norm1_bias,
    p_layers_2_blocks_4_mlp_fc1_weight,
    p_layers_2_blocks_4_mlp_fc1_bias,
    p_layers_2_blocks_4_mlp_fc2_weight,
    p_layers_2_blocks_4_mlp_fc2_bias,
    p_layers_2_blocks_4_norm2_weight,
    p_layers_2_blocks_4_norm2_bias,
    p_layers_2_blocks_5_attn_q_bias,
    p_layers_2_blocks_5_attn_v_bias,
    p_layers_2_blocks_5_attn_qkv_weight,
    p_layers_2_blocks_5_attn_logit_scale,
    p_layers_2_blocks_5_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_5_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_5_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_5_attn_proj_weight,
    p_layers_2_blocks_5_attn_proj_bias,
    p_layers_2_blocks_5_norm1_weight,
    p_layers_2_blocks_5_norm1_bias,
    p_layers_2_blocks_5_mlp_fc1_weight,
    p_layers_2_blocks_5_mlp_fc1_bias,
    p_layers_2_blocks_5_mlp_fc2_weight,
    p_layers_2_blocks_5_mlp_fc2_bias,
    p_layers_2_blocks_5_norm2_weight,
    p_layers_2_blocks_5_norm2_bias,
    p_layers_2_downsample_reduction_weight,
    p_layers_2_downsample_norm_weight,
    p_layers_2_downsample_norm_bias,
    p_layers_3_blocks_0_attn_q_bias,
    p_layers_3_blocks_0_attn_v_bias,
    p_layers_3_blocks_0_attn_qkv_weight,
    p_layers_3_blocks_0_attn_logit_scale,
    p_layers_3_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_3_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_3_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_3_blocks_0_attn_proj_weight,
    p_layers_3_blocks_0_attn_proj_bias,
    p_layers_3_blocks_0_norm1_weight,
    p_layers_3_blocks_0_norm1_bias,
    p_layers_3_blocks_0_mlp_fc1_weight,
    p_layers_3_blocks_0_mlp_fc1_bias,
    p_layers_3_blocks_0_mlp_fc2_weight,
    p_layers_3_blocks_0_mlp_fc2_bias,
    p_layers_3_blocks_0_norm2_weight,
    p_layers_3_blocks_0_norm2_bias,
    p_layers_3_blocks_1_attn_q_bias,
    p_layers_3_blocks_1_attn_v_bias,
    p_layers_3_blocks_1_attn_qkv_weight,
    p_layers_3_blocks_1_attn_logit_scale,
    p_layers_3_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_3_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_3_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_3_blocks_1_attn_proj_weight,
    p_layers_3_blocks_1_attn_proj_bias,
    p_layers_3_blocks_1_norm1_weight,
    p_layers_3_blocks_1_norm1_bias,
    p_layers_3_blocks_1_mlp_fc1_weight,
    p_layers_3_blocks_1_mlp_fc1_bias,
    p_layers_3_blocks_1_mlp_fc2_weight,
    p_layers_3_blocks_1_mlp_fc2_bias,
    p_layers_3_blocks_1_norm2_weight,
    p_layers_3_blocks_1_norm2_bias,
    p_norm_weight,
    p_norm_bias,
    p_head_weight,
    p_head_bias,
    b_layers_0_blocks_0_attn_relative_coords_table,
    b_layers_0_blocks_0_attn_relative_position_index,
    b_layers_0_blocks_1_attn_mask,
    b_layers_0_blocks_1_attn_relative_coords_table,
    b_layers_0_blocks_1_attn_relative_position_index,
    b_layers_1_blocks_0_attn_relative_coords_table,
    b_layers_1_blocks_0_attn_relative_position_index,
    b_layers_1_blocks_1_attn_mask,
    b_layers_1_blocks_1_attn_relative_coords_table,
    b_layers_1_blocks_1_attn_relative_position_index,
    b_layers_2_blocks_0_attn_relative_coords_table,
    b_layers_2_blocks_0_attn_relative_position_index,
    b_layers_2_blocks_1_attn_mask,
    b_layers_2_blocks_1_attn_relative_coords_table,
    b_layers_2_blocks_1_attn_relative_position_index,
    b_layers_2_blocks_2_attn_relative_coords_table,
    b_layers_2_blocks_2_attn_relative_position_index,
    b_layers_2_blocks_3_attn_mask,
    b_layers_2_blocks_3_attn_relative_coords_table,
    b_layers_2_blocks_3_attn_relative_position_index,
    b_layers_2_blocks_4_attn_relative_coords_table,
    b_layers_2_blocks_4_attn_relative_position_index,
    b_layers_2_blocks_5_attn_mask,
    b_layers_2_blocks_5_attn_relative_coords_table,
    b_layers_2_blocks_5_attn_relative_position_index,
    b_layers_3_blocks_0_attn_relative_coords_table,
    b_layers_3_blocks_0_attn_relative_position_index,
    b_layers_3_blocks_1_attn_relative_coords_table,
    b_layers_3_blocks_1_attn_relative_position_index,
    c_layers_0_blocks_0_attn_lifted_tensor_0,
    c_layers_0_blocks_1_attn_lifted_tensor_1,
    c_layers_1_blocks_0_attn_lifted_tensor_2,
    c_layers_1_blocks_1_attn_lifted_tensor_3,
    c_layers_2_blocks_0_attn_lifted_tensor_4,
    c_layers_2_blocks_1_attn_lifted_tensor_5,
    c_layers_2_blocks_2_attn_lifted_tensor_6,
    c_layers_2_blocks_3_attn_lifted_tensor_7,
    c_layers_2_blocks_4_attn_lifted_tensor_8,
    c_layers_2_blocks_5_attn_lifted_tensor_9,
    c_layers_3_blocks_0_attn_lifted_tensor_10,
    c_layers_3_blocks_1_attn_lifted_tensor_11,
    x,
):
    """
    """
    import torch.nn.functional as F
    
    device = x.device
    
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

    def swin_attn(
        x, num_heads, head_dim,
        qkv_weight, q_bias, v_bias,
        logit_scale,
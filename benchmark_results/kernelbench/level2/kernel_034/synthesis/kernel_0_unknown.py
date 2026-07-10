import torch
import triton
import triton.language as tl
import math

# ConvTranspose3d kernel: scatter approach
# Output shape: (N, Co, D', H', W') where D'=32, H'=64, W'=64
# Weight shape: (Ci, Co, kD, kH, kW) = (32, 64, 4, 4, 4)
# For ConvTranspose3d: output[n, co, od, oh, ow] += input[n, ci, id, ih, iw] * weight[ci, co, kd, kh, kw]
# where od = id*stride - pad + kd, oh = ih*stride - pad + kh, ow = iw*stride - pad + kw

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, Ci, Co,
    Di, Hi, Wi,
    Do, Ho, Wo,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    M, K,
    # X strides: (N, Ci, Di, Hi, Wi)
    sx_n, sx_c, sx_d, sx_h, sx_w,
    # W strides: (Ci, Co, kD, kH, kW)
    sw_ci, sw_co, sw_kd, sw_kh, sw_kw,
    # Y strides: (N, Co, Do, Ho, Wo)
    sy_n, sy_co, sy_d, sy_h, sy_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = N * Do * Ho * Wo (output spatial positions)
    # N_out = Co
    # K = Ci * kD * kH * kW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, od, oh, ow)
    ow = offs_m % Wo
    tmp = offs_m // Wo
    oh = tmp % Ho
    tmp2 = tmp // Ho
    od = tmp2 % Do
    batch = tmp2 // Do

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (ci, kd, kh, kw)
        kw_idx = offs_k % kW
        tmp_k = offs_k // kW
        kh_idx = tmp_k % kH
        tmp_k2 = tmp_k // kH
        kd_idx = tmp_k2 % kD
        ci_idx = tmp_k2 // kD

        # For ConvTranspose3d: od = id*stride - pad + kd
        # => id = (od + pad - kd) / stride (must be integer and in range)
        id_num_d = od[:, None] + pad_d - kd_idx[None, :]
        id_num_h = oh[:, None] + pad_h - kh_idx[None, :]
        id_num_w = ow[:, None] + pad_w - kw_idx[None, :]

        valid_d = (id_num_d % stride_d == 0)
        valid_h = (id_num_h % stride_h == 0)
        valid_w = (id_num_w % stride_w == 0)

        id_d = id_num_d // stride_d
        id_h = id_num_h // stride_h
        id_w = id_num_w // stride_w

        valid_range_d = (id_d >= 0) & (id_d < Di)
        valid_range_h = (id_h >= 0) & (id_h < Hi)
        valid_range_w = (id_w >= 0) & (id_w < Wi)
        valid_k = offs_k[None, :] < K
        valid_m = offs_m[:, None] < M

        valid = valid_d & valid_h & valid_w & valid_range_d & valid_range_h & valid_range_w & valid_k & valid_m

        # Load X: shape (BLOCK_M, BLOCK_K)
        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Load W: shape (BLOCK_K, BLOCK_N) - weight[ci, co, kd, kh, kw]
        valid_w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < Co)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=valid_w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias_mask = offs_n < Co
    bias = tl.load(B_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store Y
    valid_store = (offs_m[:, None] < M) & (offs_n[None, :] < Co)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_store)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['norm_size'],
)
@triton.jit
def layernorm_gelu_scale_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    num_rows, norm_size,
    row_stride,
    eps,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one set of norm_size elements)
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    row_start = row_idx * row_stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < norm_size

    # Load row
    x = tl.load(X_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=0) / norm_size

    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / norm_size

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std

    # Scale and shift
    gamma = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_out = x_norm * gamma + beta

    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c1 = 0.7978845608028654  # sqrt(2/pi)
    c2 = 0.044715
    x3 = x_out * x_out * x_out
    inner = c1 * (x_out + c2 * x3)
    # Use tanh approximation
    tanh_val = tl.extra.libdevice.tanh(inner)
    gelu_out = 0.5 * x_out * (1.0 + tanh_val)

    # Scale
    result = gelu_out * scaling_factor

    tl.store(Y_ptr + row_start + offs, result.to(tl.float32), mask=mask)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias,
    # x: (N, Ci, Di, Hi, Wi) = (32, 32, 16, 32, 32)
    # weight: (Ci, Co, kD, kH, kW) = (32, 64, 4, 4, 4)
    N, Ci, Di, Hi, Wi = x.shape
    Ci_w, Co, kD, kH, kW = p_conv_transpose_weight.shape

    stride_d = stride_h = stride_w = 2
    pad_d = pad_h = pad_w = 1

    Do = (Di - 1) * stride_d - 2 * pad_d + kD  # = 32
    Ho = (Hi - 1) * stride_h - 2 * pad_h + kH  # = 64
    Wo = (Wi - 1) * stride_w - 2 * pad_w + kW  # = 64

    M = N * Do * Ho * Wo
    K = Ci * kD * kH * kW

    # Allocate output
    y_conv = torch.empty((N, Co, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    # Launch ConvTranspose3d kernel
    def cdiv(a, b): return (a + b - 1) // b

    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(Co, meta['BLOCK_N']))

    conv_transpose3d_kernel[grid](
        x, p_conv_transpose_weight, p_conv_transpose_bias, y_conv,
        N, Ci, Co,
        Di, Hi, Wi,
        Do, Ho, Wo,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        M, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        p_conv_transpose_weight.stride(0), p_conv_transpose_weight.stride(1),
        p_conv_transpose_weight.stride(2), p_conv_transpose_weight.stride(3),
        p_conv_transpose_weight.stride(4),
        y_conv.stride(0), y_conv.stride(1), y_conv.stride(2), y_conv.stride(3), y_conv.stride(4),
    )

    # LayerNorm + GELU + scale
    # normalized_shape = [64] = last dim
    # Reshape to (num_rows, norm_size) where norm_size = 64 (last dim = Wo)
    norm_size = Wo  # = 64
    num_rows = N * Co * Do * Ho  # normalize over last dim

    # y_conv is (N, Co, Do, Ho, Wo) - last dim is Wo=64
    y_flat = y_conv.reshape(num_rows, norm_size)
    y_out = torch.empty_like(y_flat)

    grid2 = (num_rows,)
    layernorm_gelu_scale_kernel[grid2](
        y_flat, p_layer_norm_weight, p_layer_norm_bias, y_out,
        num_rows, norm_size,
        1e-5,
        1.0,  # scaling_factor
    )

    return y_out.reshape(N, Co, Do, Ho, Wo)


def get_inputs():
    p_conv_transpose_weight = torch.randn(32, 64, 4, 4, 4)
    p_conv_transpose_bias = torch.randn(64)
    p_layer_norm_weight = torch.randn(64)
    p_layer_norm_bias = torch.randn(64)
    x = torch.rand(32, 32, 16, 32, 32)
    return [p_conv_transpose_weight, p_conv_transpose_bias,


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn
        # Reference computation
        p_conv_transpose_weight, p_conv_transpose_bias, p_layer_norm_weight, p_layer_norm_bias, x = inputs
        x_cuda = x.cuda()
        w_cuda = p_conv_transpose_weight.cuda()
        b_cuda = p_conv_transpose_bias.cuda()
        ln_w = p_layer_norm_weight.cuda()
        ln_b = p_layer_norm_bias.cuda()

        # ConvTranspose3d reference
        conv = nn.ConvTranspose3d(32, 64, 4, stride=2, padding=1, bias=True).cuda()
        conv.weight.data = w_cuda
        conv.bias.data = b_cuda

        with torch.no_grad():
            ref_conv = conv(x_cuda)
            # LayerNorm with normalized_shape=[64]
            ln = nn.LayerNorm(64, eps=1e-5).cuda()
            ln.weight.data = ln_w
            ln.bias.data = ln_b
            ref_ln = ln(ref_conv)
            ref_gelu = torch.nn.functional.gelu(ref_ln)
            ref_out = ref_gelu * 1.0

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref_out.shape}")
        print(f"Max diff: {(output - ref_out).abs().max().item():.4e}")
        assert torch.allclose(output, ref_out, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={(output - ref_out).abs().max().item():.4e}"
        print("Kernel executed successfully.")
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_epilogue_kernel(
    X_ptr, W_ptr, conv_bias_ptr, bias_ptr,
    OUT_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    constant_value: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KH * KW
    # X is in NHWC layout: shape (BATCH, IN_H, IN_W, IN_C)
    # W is in (OUT_C, KH, KW, IN_C) layout - Co outermost, Ci innermost
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = BATCH * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # X in NHWC: stride_n = IN_H*IN_W*IN_C, stride_h = IN_W*IN_C, stride_w = IN_C, stride_c = 1
    # W in (OUT_C, KH, KW, IN_C): stride_co = KH*KW*IN_C, stride_kh = KW*IN_C, stride_kw = IN_C, stride_ci = 1
    # K dimension: k = ci + IN_C * (kw + KW * kh)
    # So: ci = k % IN_C, kw = (k // IN_C) % KW, kh = k // (IN_C * KW)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (ci, kw, kh)
        ci = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        # Input coordinates
        ih = out_h_idx[:, None] + kh[None, :]  # no padding, stride=1
        iw = out_w_idx[:, None] + kw[None, :]

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < BATCH

        x_mask = valid_m & valid_k & valid_ih & valid_iw & valid_batch

        # X NHWC: index = batch*IN_H*IN_W*IN_C + ih*IN_W*IN_C + iw*IN_C + ci
        x_idx = (batch_idx[:, None] * (IN_H * IN_W * IN_C) +
                 ci[None, :])
        x_tile = tl.load(X_ptr + x_idx, mask=x_mask, other=0.0)

        # W (OUT_C, KH, KW, IN_C): index = co*KH*KW*IN_C + kh*KW*IN_C + kw*IN_C + ci
        valid_n = offs_n[None, :] < OUT_C
        w_mask = valid_k[:, None] & valid_n
        w_idx = (offs_n[None, :] * (KH * KW * IN_C) +
                 kh[:, None] * (KW * IN_C) +
                 kw[:, None] * IN_C +
                 ci[:, None])
        w_tile = tl.load(W_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Epilogue: add conv_bias, min with constant, add bias, mul scaling
    valid_out_m = offs_m[:, None] < M
    valid_out_n = offs_n[None, :] < OUT_C

    # conv_bias: shape (OUT_C,)
    conv_b = tl.load(conv_bias_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + conv_b[None, :]

    # min with constant
    acc = tl.minimum(acc, constant_value)

    # bias: shape (OUT_C, 1, 1) -> broadcast over spatial, indexed by out_c
    bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias_vals[None, :]

    # mul scaling
    acc = acc * scaling_factor

    # Store output in NCHW layout: (BATCH, OUT_C, OUT_H, OUT_W)
    # out index = batch * OUT_C*OUT_H*OUT_W + co * OUT_H*OUT_W + oh * OUT_W + ow
    out_idx = (batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
               offs_n[None, :] * (OUT_H * OUT_W) +
               out_h_idx[:, None] * OUT_W +
               out_w_idx[:, None])

    out_mask = valid_out_m & valid_out_n
    tl.store(OUT_ptr + out_idx, acc.to(tl.float32), mask=out_mask)


def kernel_function(p_conv_weight, p_conv_bias, p_bias, c_lifted_tensor_0, x):
    # x: (128, 64, 128, 128) NCHW
    # p_conv_weight: (128, 64, 3, 3)
    # p_conv_bias: (128,)
    # p_bias: (128, 1, 1)
    # c_lifted_tensor_0: scalar (constant_value = 0.5)

    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = p_conv_weight.shape
    OUT_H = IN_H - KH + 1  # no padding, stride=1
    OUT_W = IN_W - KW + 1

    constant_value = float(c_lifted_tensor_0.item())
    scaling_factor = 2.0

    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)

    # Convert weight to (OUT_C, KH, KW, IN_C) for coalesced access
    w_nhwc = p_conv_weight.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)

    # bias: (128, 1, 1) -> flatten to (128,)
    bias_flat = p_bias.reshape(OUT_C)

    output = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M = BATCH * OUT_H * OUT_W

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv2d_epilogue_kernel[grid](
        x_nhwc, w_nhwc, p_conv_bias, bias_flat,
        output,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        constant_value,
        scaling_factor,
    )

    return output


def get_inputs():
    p_conv_weight = torch.randn(128, 64, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(128, device='cuda', dtype=torch.float32)
    p_bias = torch.randn(128, 1, 1, device='cuda', dtype=torch.float32)
    c_lifted_tensor_0 = torch.tensor(0.5, dtype=torch.float32)
    x = torch.rand(128, 64, 128, 128, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, p_bias, c_lifted_tensor_0, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_conv_weight, p_conv_bias, p_bias, c_lifted_tensor_0, x = inputs
        constant_value = float(c_lifted_tensor_0.item())
        scaling_factor = 2.0

        import torch.nn.functional as F
        conv_out = F.conv2d(x, p_conv_weight, p_conv_bias)
        ref = torch.min(conv_out, torch.tensor(constant_value, device=x.device))
        ref = ref + p_bias
        ref = ref * scaling_factor

        max_diff = (output - ref).abs().max().item()
        print(f"max_diff={max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")
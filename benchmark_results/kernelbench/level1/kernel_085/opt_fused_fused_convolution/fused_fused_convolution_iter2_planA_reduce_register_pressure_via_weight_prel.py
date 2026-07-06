import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256}, num_warps=4,  num_stages=3, maxnreg=32),
        triton.Config({'BLOCK_M': 128}, num_warps=4,  num_stages=3, maxnreg=32),
        triton.Config({'BLOCK_M': 512}, num_warps=8,  num_stages=3, maxnreg=32),
        triton.Config({'BLOCK_M': 64},  num_warps=2,  num_stages=4, maxnreg=32),
        triton.Config({'BLOCK_M': 256}, num_warps=8,  num_stages=2, maxnreg=32),
        triton.Config({'BLOCK_M': 128}, num_warps=8,  num_stages=4, maxnreg=32),
        triton.Config({'BLOCK_M': 1024}, num_warps=8, num_stages=3, maxnreg=32),
        triton.Config({'BLOCK_M': 512}, num_warps=4,  num_stages=4, maxnreg=32),
        triton.Config({'BLOCK_M': 256}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 512}, num_warps=8,  num_stages=3),
    ],
    key=['M', 'C'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    M,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Decode spatial position
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    # Channel index
    c = pid_c

    # Accumulator
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Weight base pointer for this channel: W[c, 0, :, :]
    # Weight shape: [C, 1, KH, KW], strides: [KH*KW, KH*KW, KW, 1]
    w_base = W_ptr + c * KH * KW

    # Input base: X[batch, c, :, :], strides: [C*IN_H*IN_W, IN_H*IN_W, IN_W, 1]
    # X shape: [N, C, IN_H, IN_W]
    x_batch_c_base = batch * (C * IN_H * IN_W) + c * (IN_H * IN_W)

    # Vectorized preload of all KH*KW weights into a register vector
    # KH*KW=21, pad to 32 (next power-of-2) for vectorized load
    w_offs = tl.arange(0, 32)
    w_mask = w_offs < KH * KW
    w_smem_buf = tl.load(w_base + w_offs, mask=w_mask, other=0.0)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            ih = out_h * STRIDE_H + kh - PAD_H
            iw = out_w * STRIDE_W + kw - PAD_W

            valid = mask_m & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)

            x_ptrs = X_ptr + x_batch_c_base + ih * IN_W + iw
            x_val = tl.load(x_ptrs, mask=valid, other=0.0)

            # Extract scalar weight from preloaded vector using one-hot sum
            # kh * KW + kw is a compile-time constant (tl.static_range)
            w_val = tl.sum(tl.where(w_offs == (kh * KW + kw), w_smem_buf, 0.0))

            acc += x_val * w_val

    # Store output: Y[batch, c, out_h, out_w]
    # Y shape: [N, C, OUT_H, OUT_W], strides: [C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1]
    y_ptrs = Y_ptr + batch * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + out_h * OUT_W + out_w
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m)

def kernel_function(weight, x):
    # weight: [128, 1, 3, 7], x: [32, 128, 128, 256]
    N, C, IN_H, IN_W = x.shape
    C_out, C_in_per_group, KH, KW = weight.shape
    # depthwise: groups = C, C_in_per_group = 1
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0

    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M = N * OUT_H * OUT_W

    # Make weight contiguous
    w = weight.contiguous()
    x_cont = x.contiguous()

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), C)

    depthwise_conv2d_kernel[grid](
        x_cont, w, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M,
    )

    return y


def get_inputs():
    weight = torch.randn(128, 1, 3, 7, device='cuda', dtype=torch.float32)
    x = torch.randn(32, 128, 128, 256, device='cuda', dtype=torch.float32)
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify correctness
        weight, x = inputs
        import torch.nn.functional as F
        ref = F.conv2d(x, weight, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=128)
        print(f"Max diff: {(output - ref).abs().max().item()}")
        print(f"Output shape: {output.shape}")
        assert torch.allclose(output, ref, atol=1e-3), "Results don't match!"
        print("Kernel executed successfully.")
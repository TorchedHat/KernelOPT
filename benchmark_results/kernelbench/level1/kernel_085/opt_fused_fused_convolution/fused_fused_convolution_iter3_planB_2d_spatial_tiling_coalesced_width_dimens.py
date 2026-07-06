import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64},  num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 32},  num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 32},  num_warps=4, num_stages=4),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W'],
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
    # 2D spatial tiling: pid_w tiles OUT_W (BLOCK_M wide), pid_bh tiles N*OUT_H (1 row at a time)
    # Grid is 3D: (ceil(OUT_W/BLOCK_M), N*OUT_H, C)
    pid_w  = tl.program_id(0)  # tiles OUT_W dimension (contiguous in memory)
    pid_bh = tl.program_id(1)  # indexes a single (batch, out_h) row
    pid_c  = tl.program_id(2)  # channel

    # Single row per pid_bh (BLOCK_H=1)
    bh_idx  = pid_bh
    mask_bh = bh_idx < N * OUT_H
    out_h   = bh_idx % OUT_H
    batch   = bh_idx // OUT_H

    # BLOCK_M contiguous out_w values
    offs_w  = pid_w * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_w  = offs_w < OUT_W

    c = pid_c

    # Weight base pointer for this channel
    w_base = W_ptr + c * KH * KW

    # 1D accumulator over BLOCK_M width positions
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            ih = out_h * STRIDE_H + kh - PAD_H   # scalar
            iw = offs_w * STRIDE_W + kw - PAD_W  # shape [BLOCK_M]

            valid_h = mask_bh & (ih >= 0) & (ih < IN_H)
            valid_w = mask_w  & (iw >= 0) & (iw < IN_W)
            valid   = valid_h & valid_w  # [BLOCK_M] broadcast

            x_base  = batch * (C * IN_H * IN_W) + c * (IN_H * IN_W)
            x_ptrs  = X_ptr + x_base + ih * IN_W + iw
            x_val   = tl.load(x_ptrs, mask=valid, other=0.0)

            w_val   = tl.load(w_base + kh * KW + kw)

            acc    += x_val * w_val

    # Store output: Y[batch, c, out_h, out_w]
    y_base  = batch * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + out_h * OUT_W
    y_ptrs  = Y_ptr + y_base + offs_w
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_w & mask_bh)

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
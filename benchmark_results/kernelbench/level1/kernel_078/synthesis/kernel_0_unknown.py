import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv_transpose2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for Y (NHWC layout)
    s_yn, s_yh, s_yw, s_yc,
    M_TOTAL, TOTAL_K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    out_h_idx = (offs_m // OUT_W) % OUT_H
    batch_idx = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For conv_transpose2d with stride=1:
    # out[n, oc, oh, ow] = sum over (ic, kh, kw) of x[n, ic, oh - kh + PAD_H, ow - kw + PAD_W] * w[ic, oc, kh, kw]
    # where w[ic, oc, kh, kw] is the weight (note: ConvTranspose2d weight shape is [in_c, out_c, kh, kw])
    # TOTAL_K = IN_C * KH * KW

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        in_c_idx = offs_k // (KW * KH)

        # For conv_transpose2d: input position = (oh - kh + PAD_H, ow - kw + PAD_W)
        ih = out_h_idx[:, None] - kh_idx[None, :] + PAD_H
        iw = out_w_idx[:, None] - kw_idx[None, :] + PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < N
        x_mask = valid_m & valid_k & valid_ih & valid_iw & valid_batch

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Weight layout: [IN_C, OUT_C, KH, KW]
        # w[in_c, out_c, kh, kw]
        # offs_k encodes (in_c, kh, kw), offs_n encodes out_c
        w_ptrs = (W_ptr
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Store output (NHWC)
    y_ptrs = (Y_ptr
    y_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    tl.store(y_ptrs, acc.to(tl.float32), mask=y_mask)


def kernel_function(weight, x):
    # weight shape: [in_c, out_c, kh, kw] = [32, 32, 3, 7]
    # x shape: [N, in_c, H, W] = [8, 32, 512, 1024]
    N, IN_C, IN_H, IN_W = x.shape
    IN_C_w, OUT_C, KH, KW = weight.shape
    assert IN_C == IN_C_w

    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 1, 3

    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KH
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KW
    # With stride=1, pad=(1,3), kernel=(3,7): OUT_H = 512-1-2+3=512, OUT_W=1024-1-6+7=1024

    # Convert x to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Output in NHWC
    y_nhwc = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x.device, dtype=x.dtype)

    M_TOTAL = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    s_xn, s_xh, s_xw, s_xc = x_nhwc.stride()
    s_yn, s_yh, s_yw, s_yc = y_nhwc.stride()

    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    conv_transpose2d_kernel[grid](
        x_nhwc, weight, y_nhwc,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xh, s_xw, s_xc,
        s_yn, s_yh, s_yw, s_yc,
        M_TOTAL, TOTAL_K,
    )

    # Convert back to NCHW
    return y_nhwc.permute(0, 3, 1, 2).contiguous()


def get_inputs():
    weight = torch.randn(32, 32, 3, 7, dtype=torch.float32, device='cuda')
    x = torch.randn(8, 32, 512, 1024, dtype=torch.float32, device='cuda')
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, x = inputs

    # Verify correctness
    if not args.profile_only:
        import torch.nn as nn
        # Reference: ConvTranspose2d with weight shape [in_c, out_c, kh, kw]
        conv_ref = nn.ConvTranspose2d(32, 32, (3, 7), stride=(1, 1), padding=(1, 3), bias=False)
        conv_ref.weight.data = weight.cpu()
        conv_ref = conv_ref.cuda()
        with torch.no_grad():
            ref_out = conv_ref(x)
        
        our_out = kernel_function(weight, x)
        
        max_diff = (our_out - ref_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-2, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")
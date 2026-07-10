import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_CO': 32, 'BLOCK_H': 16}, num_warps=4),
        triton.Config({'BLOCK_CO': 64, 'BLOCK_H': 8}, num_warps=4),
        triton.Config({'BLOCK_CO': 128, 'BLOCK_H': 4}, num_warps=8),
        triton.Config({'BLOCK_CO': 32, 'BLOCK_H': 32}, num_warps=8),
    ],
    key=['OUT_H', 'OUT_W', 'OUT_C', 'IN_C'],
)
@triton.jit
def fused_kernel(
    x_ptr, weight_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # x strides
    x_sN, x_sC, x_sH, x_sW,
    # weight strides: weight[in_c, out_c, kh, kw]
    w_sIC, w_sOC, w_sKH, w_sKW,
    # output strides
    out_sN, out_sW,
    BLOCK_CO: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one (batch, w_out) pair
    pid = tl.program_id(0)
    n = pid // OUT_W
    w_out = pid % OUT_W

    # We'll accumulate sum over h_out of min over co
    total_sum = tl.zeros((), dtype=tl.float32)

    # Iterate over h_out in blocks
    for h_base in range(0, OUT_H, BLOCK_H):
        h_offs = h_base + tl.arange(0, BLOCK_H)
        h_mask = h_offs < OUT_H

        # For each h_out, compute min over all co channels
        # Initialize min values to +inf
        min_vals = tl.full([BLOCK_H], float('inf'), dtype=tl.float32)

        # Iterate over co in blocks
        for co_base in range(0, OUT_C, BLOCK_CO):
            co_offs = co_base + tl.arange(0, BLOCK_CO)
            co_mask = co_offs < OUT_C

            # Accumulate conv_transpose output for each (h_out, co) pair
            # acc shape: [BLOCK_H, BLOCK_CO]
            acc = tl.zeros([BLOCK_H, BLOCK_CO], dtype=tl.float32)

            # Add conv bias
            cb = tl.load(conv_bias_ptr + co_offs, mask=co_mask, other=0.0)
            acc = acc + cb[None, :]

            # Iterate over kernel positions and input channels
            for kh in range(KH):
                for kw in range(KW):
                    # For output position (h_out, w_out) with kernel (kh, kw):
                    # ih = (h_out + PAD_H - kh) / STRIDE_H (must be integer and in range)
                    # iw = (w_out + PAD_W - kw) / STRIDE_W (must be integer and in range)

                    # Compute ih for each h_out
                    h_num = h_offs + PAD_H - kh  # [BLOCK_H]
                    w_num = w_out + PAD_W - kw    # scalar

                    # Check divisibility and bounds
                    h_valid_div = (h_num % STRIDE_H) == 0
                    w_valid_div = (w_num % STRIDE_W) == 0

                    ih = h_num // STRIDE_H  # [BLOCK_H]
                    iw = w_num // STRIDE_W  # scalar

                    h_in_bounds = (ih >= 0) & (ih < IN_H)
                    w_in_bounds = (iw >= 0) & (iw < IN_W)

                    valid_hw = h_valid_div & w_valid_div & h_in_bounds & w_in_bounds & h_mask

                    if tl.sum(valid_hw.to(tl.int32), axis=0) > 0:
                        # Load input for all in_channels: x[n, :, ih, iw] shape [IN_C, BLOCK_H]
                        # Then load weight[:, co_offs, kh, kw] shape [IN_C, BLOCK_CO]
                        # acc += x_tile @ w_tile

                        # We need to iterate over IN_C
                        for ci in range(IN_C):
                            # Load x[n, ci, ih, iw] for each h in BLOCK_H
                            x_ptrs = x_ptr + n * x_sN + ci * x_sC + ih * x_sH + iw * x_sW
                            x_vals = tl.load(x_ptrs, mask=valid_hw, other=0.0)  # [BLOCK_H]

                            # Load weight[ci, co_offs, kh, kw] for each co in BLOCK_CO
                            w_ptrs = weight_ptr + ci * w_sIC + co_offs * w_sOC + kh * w_sKH + kw * w_sKW
                            w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)  # [BLOCK_CO]

                            # Outer product accumulation
                            acc = acc + x_vals[:, None] * w_vals[None, :]

            # Compute min over co dimension for valid h positions
            # acc: [BLOCK_H, BLOCK_CO], take min over BLOCK_CO axis
            # But we need global min over all co_base iterations
            local_min = tl.min(acc, axis=1)  # [BLOCK_H]
            min_vals = tl.minimum(min_vals, local_min)

        # Sum min_vals over valid h positions
        total_sum = total_sum + tl.sum(tl.where(h_mask, min_vals, 0.0), axis=0)

    # Apply GELU to total_sum
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    val = total_sum
    val_gelu = 0.5 * val * (1.0 + libdevice.tanh(0.7978845608028654 * (val + 0.044715 * val * val * val)))

    # Add bias (scalar since bias_shape=(1,1,1))
    b = tl.load(bias_ptr)
    result = val_gelu + b

    # Store output: out[n, 0, 0, w_out]
    out_offset = n * out_sN + w_out * out_sW
    tl.store(out_ptr + out_offset, result)


def kernel_function(weight, conv_bias, bias_param, x):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape  # weight for ConvTranspose2d is [in_c, out_c, kh, kw]
    # Actually PyTorch ConvTranspose2d weight shape is [in_channels, out_channels, kH, kW]
    # But nn.ConvTranspose2d stores weight as [in_channels, out_channels/groups, kH, kW]

    STRIDE_H, STRIDE_W = 2, 2
    PAD_H, PAD_W = 1, 1

    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KH + 1  # output_padding=1
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KW + 1

    # weight shape from nn.ConvTranspose2d: [in_channels, out_channels, kH, kW]
    IN_C_w, OUT_C_w = weight.shape[0], weight.shape[1]

    out = torch.empty(N, 1, 1, OUT_W, device=x.device, dtype=x.dtype)

    grid = (N * OUT_W,)

    fused_kernel[grid](
        x, weight, conv_bias, bias_param, out,
        N, IN_C, IN_H, IN_W,
        OUT_C_w, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(3),
    )

    return out


def get_inputs():
    torch.manual_seed(42)
    weight = torch.randn(64, 128, 3, 3)
    conv_bias = torch.randn(128)
    bias_param = torch.randn(1, 1, 1)
    x = torch.rand(16, 64, 128, 128)
    return [weight, conv_bias, bias_param, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, conv_bias, bias_param, x = inputs

    # Move to GPU
    weight = weight.cuda()
    conv_bias = conv_bias.cuda()
    bias_param = bias_param.cuda()
    x = x.cuda()

    output = kernel_function(weight, conv_bias, bias_param, x)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F
        # Reference computation
        ref = F.conv_transpose2d(x, weight, conv_bias, stride=2, padding=1, output_padding=1)
        ref = torch.min(ref, dim=1, keepdim=True)[0]
        ref = torch.sum(ref, dim=2, keepdim=True)
        ref = F.gelu(ref)
        ref = ref + bias_param

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")
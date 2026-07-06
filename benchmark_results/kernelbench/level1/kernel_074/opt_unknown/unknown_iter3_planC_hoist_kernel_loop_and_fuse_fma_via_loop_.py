import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        # BLOCK_IC=1: minimal register footprint
        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_L': 1024, 'BLOCK_IC': 1}, num_warps=4, num_stages=3),
        # BLOCK_IC=2: minimal register footprint, highest occupancy potential
        triton.Config({'BLOCK_L': 256,  'BLOCK_IC': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_L': 1024, 'BLOCK_IC': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_L': 256,  'BLOCK_IC': 2}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 2}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_L': 1024, 'BLOCK_IC': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_L': 256,  'BLOCK_IC': 2}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 2}, num_warps=4, num_stages=4),
        # BLOCK_IC=4: balanced register pressure vs arithmetic intensity
        triton.Config({'BLOCK_L': 256,  'BLOCK_IC': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_L': 1024, 'BLOCK_IC': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_L': 256,  'BLOCK_IC': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_L': 256,  'BLOCK_IC': 4}, num_warps=4, num_stages=4),
        # Original configs kept as fallback
        triton.Config({'BLOCK_L': 128,  'BLOCK_IC': 8},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_L': 256,  'BLOCK_IC': 8},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 4},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_L': 256,  'BLOCK_IC': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_L': 128,  'BLOCK_IC': 16}, num_warps=4, num_stages=3),
    ],
    key=['IN_C', 'OUT_L', 'KERNEL_SIZE'],
)
@triton.jit
def conv_transpose1d_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH, IN_C, OUT_C, IN_L, OUT_L,
    STRIDE, PADDING, DILATION,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Grid: (BATCH * OUT_C, ceil(OUT_L / BLOCK_L))
    pid_boc = tl.program_id(0)
    pid_l = tl.program_id(1)

    batch_idx = pid_boc // OUT_C
    oc = pid_boc % OUT_C

    # Output positions this block handles
    out_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    out_mask = out_offs < OUT_L

    # Accumulator
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # Loop over kernel positions (outer) - hoists index computation out of ic_start loop
    for k in tl.static_range(KERNEL_SIZE):
        # For output position o, kernel position k:
        # input position i = (o - k*dilation + padding) / stride
        # STRIDE==1 fast-path: num%1==0 always, num//1==num
        k_dil = k * DILATION
        # numerator for input index
        num = out_offs - k_dil + PADDING
        # STRIDE=1 fast-path: i_idx = num (no division/modulo needed)
        i_idx = num
        valid_i = (i_idx >= 0) & (i_idx < IN_L)
        valid = out_mask & valid_i  # valid_stride dropped (always True for STRIDE=1)

        # Loop over in_channels in blocks (inner)
        for ic_start in range(0, IN_C, BLOCK_IC):
            # Load weights for this kernel position, for BLOCK_IC channels
            ic_offs = ic_start + tl.arange(0, BLOCK_IC)
            ic_mask = ic_offs < IN_C

            # Weight shape: (IN_C, OUT_C, KERNEL_SIZE)
            # weight[ic, oc, k]
            w_ptrs = W_ptr + ic_offs * (OUT_C * KERNEL_SIZE) + oc * KERNEL_SIZE + k
            w_vals = tl.load(w_ptrs, mask=ic_mask, other=0.0)  # [BLOCK_IC]

            # Load input for all BLOCK_IC channels and all BLOCK_L output positions at once
            # X shape: (BATCH, IN_C, IN_L)
            # x_ptrs_2d[ic_local, l] = X_ptr + batch_idx*(IN_C*IN_L) + ic_offs[ic_local]*IN_L + i_idx[l]
            x_ptrs_2d = X_ptr + batch_idx * (IN_C * IN_L) + ic_offs[:, None] * IN_L + i_idx[None, :]  # [BLOCK_IC, BLOCK_L]
            x_mask_2d = ic_mask[:, None] & valid[None, :]  # [BLOCK_IC, BLOCK_L]
            x_block = tl.load(x_ptrs_2d, mask=x_mask_2d, other=0.0)  # [BLOCK_IC, BLOCK_L]

            # Accumulate: weight each channel's x values by its weight, then sum over IC dimension
            acc += tl.sum(x_block * w_vals[:, None], axis=0)  # [BLOCK_L]

    # Store output
    # Y shape: (BATCH, OUT_C, OUT_L)
    y_ptrs = Y_ptr + batch_idx * (OUT_C * OUT_L) + oc * OUT_L + out_offs
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)

def kernel_function(weight, x):
    # weight: (IN_C, OUT_C, KERNEL_SIZE) - ConvTranspose1d weight is (in_channels, out_channels, kernel_size)
    # x: (BATCH, IN_C, IN_L)
    BATCH, IN_C, IN_L = x.shape
    IN_C_w, OUT_C, KERNEL_SIZE = weight.shape
    
    STRIDE = 1
    PADDING = 0
    DILATION = 3
    
    OUT_L = (IN_L - 1) * STRIDE - 2 * PADDING + DILATION * (KERNEL_SIZE - 1) + 1
    
    y = torch.zeros(BATCH, OUT_C, OUT_L, device=x.device, dtype=torch.float32)
    
    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    
    grid = lambda meta: (BATCH * OUT_C, triton.cdiv(OUT_L, meta['BLOCK_L']))
    
    conv_transpose1d_kernel[grid](
        x_contig, w_contig, y,
        BATCH, IN_C, OUT_C, IN_L, OUT_L,
        STRIDE, PADDING, DILATION,
        KERNEL_SIZE=KERNEL_SIZE,
    )
    
    return y


def get_inputs():
    weight = torch.randn(32, 64, 5, dtype=torch.float32, device='cuda')
    x = torch.rand(32, 32, 131072, dtype=torch.float32, device='cuda')
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    weight, x = inputs
    
    # Reference implementation
    import torch.nn as nn
    conv = nn.ConvTranspose1d(32, 64, 5, stride=1, padding=0, dilation=3, bias=False)
    conv.weight.data = weight
    conv = conv.cuda()
    
    with torch.no_grad():
        ref_out = conv(x)
    
    triton_out = kernel_function(weight, x)
    
    torch.cuda.synchronize()
    
    if not args.profile_only:
        max_diff = (triton_out - ref_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        print(f"Output shape: {triton_out.shape}")
        assert max_diff < 1e-3, f"Max diff {max_diff} exceeds tolerance"
        print("Kernel executed successfully.")
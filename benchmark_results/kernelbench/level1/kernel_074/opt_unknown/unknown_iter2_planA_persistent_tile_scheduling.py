import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 128, 'BLOCK_IC': 8},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_L': 256, 'BLOCK_IC': 8},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_L': 512, 'BLOCK_IC': 4},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_L': 128, 'BLOCK_IC': 16}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_L': 256, 'BLOCK_IC': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_L': 64,  'BLOCK_IC': 8},  num_warps=2, num_stages=2),
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
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)  # = 1056

    # Compute tile counts from runtime args (BLOCK_L is constexpr so cdiv is compile-time)
    tiles_per_boc = tl.cdiv(OUT_L, BLOCK_L)
    num_tiles = BATCH * OUT_C * tiles_per_boc

    for tile_id in tl.range(pid, num_tiles, grid_size):
        pid_boc  = tile_id // tiles_per_boc
        pid_l    = tile_id %  tiles_per_boc
        batch_idx = pid_boc // OUT_C
        oc        = pid_boc %  OUT_C

        # Output positions this block handles
        out_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        out_mask = out_offs < OUT_L

        # Accumulator
        acc = tl.zeros([BLOCK_L], dtype=tl.float32)

        # Base pointer for input
        x_base = X_ptr + batch_idx * (IN_C * IN_L)

        # Loop over in_channels in blocks
        for ic_start in range(0, IN_C, BLOCK_IC):
            # Hoist ic_offs and ic_mask outside the k-loop (they don't depend on k)
            ic_offs = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
            ic_mask = ic_offs < IN_C                       # [BLOCK_IC]

            # Loop over kernel positions
            for k in range(KERNEL_SIZE):
                # For output position o, kernel position k:
                # input position i = (o - k*dilation + padding) / stride
                k_dil = k * DILATION
                # numerator for input index
                num = out_offs - k_dil + PADDING
                i_idx = num // STRIDE

                # Compute validity mask
                valid_i = (i_idx >= 0) & (i_idx < IN_L)
                # STRIDE==1 fast-path: skip modulo check (always True when STRIDE==1)
                if STRIDE == 1:
                    valid = out_mask & valid_i
                else:
                    valid_stride = (num % STRIDE) == 0
                    valid = out_mask & valid_stride & valid_i

                # Load weights for this kernel position, for BLOCK_IC channels
                # Weight shape: (IN_C, OUT_C, KERNEL_SIZE)
                # weight[ic, oc, k]
                w_ptrs = W_ptr + ic_offs * (OUT_C * KERNEL_SIZE) + oc * KERNEL_SIZE + k
                w_vals = tl.load(w_ptrs, mask=ic_mask, other=0.0)  # [BLOCK_IC]

                # Load input as 2D tile [BLOCK_IC, BLOCK_L]
                # X shape: (BATCH, IN_C, IN_L)
                # x[batch, ic, i_idx]
                x_ptrs = (x_base
                          + ic_offs[:, None] * IN_L   # [BLOCK_IC, 1]
                          + i_idx[None, :])            # [1, BLOCK_L]
                x_mask = ic_mask[:, None] & valid[None, :]  # [BLOCK_IC, BLOCK_L]
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_IC, BLOCK_L]

                # Accumulate: sum over ic dimension using already-loaded w_vals
                acc += tl.sum(w_vals[:, None] * x_tile, axis=0)  # [BLOCK_L]

        # Store output
        # Y shape: (BATCH, OUT_C, OUT_L)
        y_ptrs = Y_ptr + batch_idx * (OUT_C * OUT_L) + oc * OUT_L + out_offs
        tl.store(y_ptrs, acc, mask=out_mask)

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
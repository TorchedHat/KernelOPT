import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_clone_permute_view_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Square-tiled coalesced transpose using tl.trans()
    # XBLOCK == YBLOCK (square tile) for coalesced access
    pid_x = tl.program_id(0)  # tile index along x-dimension (256)
    pid_y = tl.program_id(1)  # tile index along y-dimension (16384)
    x_base = pid_x * XBLOCK
    y_base = pid_y * YBLOCK
    row = tl.arange(0, YBLOCK)   # y-local offsets
    col = tl.arange(0, XBLOCK)   # x-local offsets
    y_idx = y_base + row[:, None]   # shape [YBLOCK, 1]
    x_idx = x_base + col[None, :]   # shape [1, XBLOCK]
    y_mask = y_idx < ynumel
    x_mask = x_idx < xnumel
    mask = y_mask & x_mask
    # COALESCED LOAD: in_ptr0 offset = y + 16384*x
    # For fixed x, consecutive y values are stride-1 -> coalesced
    vals = tl.load(in_ptr0 + (y_idx + 16384 * x_idx), mask=mask, other=0.0)
    # Transpose in registers: shape [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK]
    vals_t = tl.trans(vals)
    # COALESCED STORE: out_ptr0 offset = x + 256*y
    # After transpose: row dim = x_local, col dim = y_local
    x_out = x_base + tl.arange(0, XBLOCK)[:, None]   # x indices, shape [XBLOCK, 1]
    y_out = y_base + tl.arange(0, YBLOCK)[None, :]   # y indices, shape [1, YBLOCK]
    out_mask = (x_out < xnumel) & (y_out < ynumel)
    tl.store(out_ptr0 + (x_out + 256 * y_out), vals_t, mask=out_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 16384
    xnumel = 256
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_clone_permute_view_0[grid](arg_0, arg_1, 16384, 256)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((2, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384, 2, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        print("Kernel executed successfully.")

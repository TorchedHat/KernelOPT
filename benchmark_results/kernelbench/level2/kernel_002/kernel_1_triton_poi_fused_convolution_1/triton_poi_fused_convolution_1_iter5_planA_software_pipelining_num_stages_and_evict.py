import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # TENSOR LAYOUT:
    # Input shape (32,128,126,250), strides (4032000,1,32000,128) = NHWC-like
    #   elem[n,c,h,w] = n*4032000 + c*1 + hw*128  where hw = h*250+w
    # Output shape (32,128,126,250), strides (4032000,31500,250,1) = NCHW
    #   elem[n,c,h,w] = n*4032000 + c*31500 + hw*1
    # NC = 32*128 = 4096, HW = 126*250 = 31500
    #
    # COALESCING STRATEGY:
    # Use XBLOCK as TILE_HW (hw outer dim) and YBLOCK as TILE_C (nc inner dim).
    # 2D tile shape: [XBLOCK, YBLOCK] = [TILE_HW, TILE_NC]
    # Load: inner dim (YBLOCK) varies in nc -> c varies (stride=1 in input) -> coalesced
    # Store: outer dim (XBLOCK) varies in hw (stride=1 in output) -> coalesced
    C = 128
    HW = 31500
    NC = 4096

    pid_hw = tl.program_id(0)   # drives hw dimension
    pid_nc = tl.program_id(1)   # drives nc dimension

    # hw_off: outer dimension, shape [XBLOCK]
    hw_off = pid_hw * XBLOCK + tl.arange(0, XBLOCK)
    hw_mask = hw_off < HW

    # nc_off: inner dimension, shape [YBLOCK]
    nc_off = pid_nc * YBLOCK + tl.arange(0, YBLOCK)
    nc_mask = nc_off < NC

    # Decompose nc into n and c
    n_idx = nc_off // C   # shape [YBLOCK]
    c_idx = nc_off % C    # shape [YBLOCK]

    # 2D tile: [XBLOCK, YBLOCK] = [TILE_HW, TILE_NC]
    # Load index: n*4032000 + c*1 + hw*128
    # Inner dim (YBLOCK) varies in c (stride=1 in input) -> coalesced load
    load_idx = n_idx[None, :] * 4032000 + c_idx[None, :] * 1 + hw_off[:, None] * 128
    mask2d = hw_mask[:, None] & nc_mask[None, :]
    vals = tl.load(in_ptr0 + load_idx, mask=mask2d, eviction_policy='evict_first')  # free L2 for store traffic

    # Store index: n*4032000 + c*31500 + hw*1
    # Inner dim (YBLOCK) varies in c (stride=31500 in output)
    # Outer dim (XBLOCK) varies in hw (stride=1 in output) -> coalesced store
    store_idx = n_idx[None, :] * 4032000 + c_idx[None, :] * 31500 + hw_off[:, None] * 1
    tl.store(out_ptr0 + store_idx, vals, mask=mask2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_x = tl.program_id(0)  # covers x2 dimension
    pid_y = tl.program_id(1)  # covers y dimension
    yoffset = pid_y * YBLOCK
    yrange  = tl.arange(0, YBLOCK)       # [YBLOCK]
    yindex  = yoffset + yrange            # [YBLOCK]
    ymask   = yindex < ynumel
    xoffset = pid_x * XBLOCK
    xrange  = tl.arange(0, XBLOCK)       # [XBLOCK]
    xindex  = xoffset + xrange            # [XBLOCK]
    xmask   = xindex < xnumel
    y0 = yindex % 128   # stride-1 in source
    y1 = yindex // 128  # batch dim
    # COALESCED LOAD: Y varies fast (stride-1 in source)
    load_mask = ymask[:, None] & xmask[None, :]
    src_idx   = y0[:, None] + 128 * xindex[None, :] + 4032000 * y1[:, None]
    tile = tl.load(in_ptr0 + src_idx, mask=load_mask, other=0.0,
                   eviction_policy='evict_first')  # [YBLOCK, XBLOCK]
    # SMEM TRANSPOSE
    tile_T = tl.trans(tile)  # [XBLOCK, YBLOCK]
    # COALESCED STORE: Y varies fast (stride-1 in dest)
    store_mask = xmask[:, None] & ymask[None, :]
    dst_idx    = xindex[:, None] + 31500 * yindex[None, :]  # [XBLOCK, YBLOCK]
    tl.store(out_ptr0 + dst_idx, tile_T, mask=store_mask,
             eviction_policy='evict_first')

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

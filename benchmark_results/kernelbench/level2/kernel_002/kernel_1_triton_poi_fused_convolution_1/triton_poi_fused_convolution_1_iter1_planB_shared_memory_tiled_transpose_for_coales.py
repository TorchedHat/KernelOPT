import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Constants for this transpose: N=32, C=128, HW=31500
    # in_ptr0: NHWC layout, strides (4032000, 1, 32000, 128) => src[n, hw, c] = in_ptr0[n*4032000 + hw*128 + c]
    # out_ptr0: NCHW layout, strides (4032000, 31500, 250, 1) => dst[n, c, hw] = out_ptr0[n*4032000 + c*31500 + hw]
    # XBLOCK = TILE_HW (tile size along HW dimension)
    # YBLOCK = TILE_C  (tile size along C dimension)
    N = 32
    C = 128
    HW = 31500

    pid_hw = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    hw_off = pid_hw * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK] = TILE_HW indices
    c_off  = pid_c  * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK] = TILE_C indices

    hw_mask = hw_off < HW
    c_mask  = c_off  < C

    # Phase 1: coalesced load — c_off varies fastest (stride=1 in src NHWC)
    # src index: c_off + C * hw_off + C * HW * pid_n
    # Shape: [YBLOCK, XBLOCK] where YBLOCK=TILE_C (inner/fast), XBLOCK=TILE_HW (outer/slow)
    src_idx = c_off[:, None] + C * hw_off[None, :] + C * HW * pid_n  # [YBLOCK, XBLOCK]
    mask2d = c_mask[:, None] & hw_mask[None, :]
    tile = tl.load(in_ptr0 + src_idx, mask=mask2d, other=0.0)

    # Phase 2: coalesced store — hw_off varies fastest (stride=1 in dst NCHW)
    # dst index: hw_off + HW * c_off + HW * C * pid_n
    dst_idx = hw_off[None, :] + HW * c_off[:, None] + HW * C * pid_n  # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + dst_idx, tile, mask=mask2d)

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

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
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Tensor shape: (N=32, C=128, H=126, W=250)
    # src strides: (4032000, 1, 32000, 128)  => element [n,c,hw] at n*4032000 + c*1 + hw*128
    # dst strides: (4032000, 31500, 250, 1)  => element [n,c,hw] at n*4032000 + c*31500 + hw*1
    # ynumel = 4096 = N*C, xnumel = 31500 = HW
    # We use XBLOCK as TILE_C (covers C dim, stride=1 in src => coalesced load)
    # and YBLOCK as TILE_HW (covers HW dim, stride=1 in dst => coalesced store)
    # Grid: axis0 = cdiv(C=128, XBLOCK), axis1 = cdiv(HW=31500, YBLOCK), axis2 = N=32

    C = 128
    HW = 31500
    N = 32

    pid_c  = tl.program_id(0)   # tile over C
    pid_hw = tl.program_id(1)   # tile over HW
    pid_n  = tl.program_id(2)   # tile over N

    # C offsets: these vary fastest in src (stride=1) => coalesced load
    c_off = pid_c * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK]
    # HW offsets: these vary fastest in dst (stride=1) => coalesced store
    hw_off = pid_hw * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]

    c_mask  = c_off  < C
    hw_mask = hw_off < HW

    # 2D mask: [XBLOCK, YBLOCK] = [TILE_C, TILE_HW]
    mask2d = c_mask[:, None] & hw_mask[None, :]

    # Phase 1: coalesced load from src
    # src layout: element [n, c, hw] at n*4032000 + c*1 + hw*128
    # src_idx shape: [XBLOCK, YBLOCK] = [TILE_C, TILE_HW]
    # c_off varies along axis0 (stride=1 in src) => coalesced
    src_idx = c_off[:, None] + 128 * hw_off[None, :] + 4032000 * pid_n
    tile = tl.load(in_ptr0 + src_idx, mask=mask2d, other=0.0)

    # Phase 2: coalesced store to dst
    # dst layout: element [n, c, hw] at n*4032000 + c*31500 + hw*1
    # dst_idx shape: [XBLOCK, YBLOCK] = [TILE_C, TILE_HW]
    # hw_off varies along axis1 (stride=1 in dst) => coalesced
    dst_idx = hw_off[None, :] + 31500 * c_off[:, None] + 4032000 * pid_n
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

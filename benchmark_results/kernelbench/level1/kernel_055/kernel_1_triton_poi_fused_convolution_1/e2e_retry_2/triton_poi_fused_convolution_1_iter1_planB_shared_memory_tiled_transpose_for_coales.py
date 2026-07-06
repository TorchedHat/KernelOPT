import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8, num_stages=2),
    triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=8, num_stages=2),
    triton.Config({'YBLOCK': 128, 'XBLOCK': 8}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 8,  'XBLOCK': 128}, num_warps=4, num_stages=3),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # ynumel = 4096 = B*C = 32*128, xnumel = 31500 = H*W = 126*250
    # YBLOCK = TILE_C (channel/batch tile, stride-1 in source)
    # XBLOCK = TILE_X (spatial tile, stride-1 in dest)
    # Source strides: (4032000, 1, 32000, 128) -> channel dim is stride-1
    # Dest strides:   (4032000, 31500, 250, 1) -> spatial dim is stride-1
    pid_x  = tl.program_id(0)  # spatial tile
    pid_bc = tl.program_id(1)  # batch*channel tile
    x_off  = pid_x  * XBLOCK
    bc_off = pid_bc * YBLOCK
    tc = tl.arange(0, YBLOCK)  # channel axis - stride-1 in source
    tx = tl.arange(0, XBLOCK)  # spatial axis - stride-1 in dest
    bc_idx = bc_off + tc        # [YBLOCK]
    x_idx  = x_off  + tx        # [XBLOCK]
    b_idx  = bc_idx // 128      # batch index
    c_idx  = bc_idx % 128       # channel index
    # Source address: c_idx is stride-1, so loading along YBLOCK dim is coalesced
    src_addr = c_idx[:, None] + 128 * x_idx[None, :] + 4032000 * b_idx[:, None]  # [YBLOCK, XBLOCK]
    bc_mask = bc_idx < ynumel   # [YBLOCK]
    x_mask  = x_idx  < xnumel  # [XBLOCK]
    mask2d  = bc_mask[:, None] & x_mask[None, :]  # [YBLOCK, XBLOCK]
    tile = tl.load(in_ptr0 + src_addr, mask=mask2d, other=0.0)  # [YBLOCK, XBLOCK]
    tile_T = tl.trans(tile)  # [XBLOCK, YBLOCK] - in-register transpose
    # Dest address: x_idx is stride-1, so storing along XBLOCK dim is coalesced
    dst_addr = x_idx[:, None] + 31500 * bc_idx[None, :]  # [XBLOCK, YBLOCK]
    tl.store(out_ptr0 + dst_addr, tile_T, mask=tl.trans(mask2d))

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

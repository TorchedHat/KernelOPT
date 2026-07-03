import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # XBLOCK = TILE_C (channel tile, fast axis in source, stride=1)
    # YBLOCK = TILE_HW (spatial tile, fast axis in dest, stride=1)
    pid_c  = tl.program_id(0)   # tile over channel dim (128 channels)
    pid_hw = tl.program_id(1)   # tile over spatial dim (31500 = 126*250)
    pid_b  = tl.program_id(2)   # tile over batch dim (32 batches)
    c_offs  = pid_c  * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK] channel offsets
    hw_offs = pid_hw * YBLOCK + tl.arange(0, YBLOCK)   # [YBLOCK] spatial offsets
    c_mask  = c_offs  < 128
    hw_mask = hw_offs < 31500
    h_offs = hw_offs // 250
    w_offs = hw_offs  % 250
    # Coalesced LOAD: channel is fast axis (stride=1 in source)
    # src shape: [YBLOCK, XBLOCK]
    src = (pid_b * 4032000
           + c_offs[None, :]          # [1, XBLOCK] stride-1 -> coalesced
           + h_offs[:, None] * 32000  # [YBLOCK, 1]
           + w_offs[:, None] * 128)   # [YBLOCK, 1]
    mask2d = hw_mask[:, None] & c_mask[None, :]
    tile = tl.load(in_ptr0 + src, mask=mask2d, other=0.0)  # [YBLOCK, XBLOCK]
    # Coalesced STORE: width is fast axis (stride=1 in dest)
    # dst shape: [YBLOCK, XBLOCK]
    dst = (pid_b * 4032000
           + c_offs[None, :] * 31500  # [1, XBLOCK]
           + h_offs[:, None] * 250    # [YBLOCK, 1]
           + w_offs[:, None])         # [YBLOCK, 1] stride-1 -> coalesced
    tl.store(out_ptr0 + dst, tile, mask=mask2d)

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

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
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Reinterpret: XBLOCK = CBLOCK (channel tile, stride-1 in in_ptr0)
    #              YBLOCK = SBLOCK (spatial tile, stride-1 in out_ptr0)
    # Fixed problem dimensions:
    C = 96
    HW = 3025
    N = 256
    # Grid axis 0: cdiv(C, XBLOCK) * cdiv(HW, YBLOCK)
    # Grid axis 1: N
    pid_x = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_s_tiles = tl.cdiv(HW, YBLOCK)
    tile_c = pid_x // num_s_tiles
    tile_s = pid_x % num_s_tiles
    # Channel offsets (XBLOCK): stride-1 in in_ptr0 => coalesced load
    c_off = tile_c * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    c_mask = c_off < C
    # Spatial offsets (YBLOCK): stride-1 in out_ptr0 => coalesced store
    s_off = tile_s * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
    s_mask = s_off < HW
    # Load from in_ptr0: strides (290400, 1, 5280, 96)
    # in_ptr0[n, c, hw] = n*290400 + c*1 + hw*96
    # Shape [YBLOCK, XBLOCK]: inner dim XBLOCK varies along channel (stride-1) => coalesced
    in_addr = pid_n * 290400 + s_off[:, None] * 96 + c_off[None, :]
    in_mask = s_mask[:, None] & c_mask[None, :]
    data = tl.load(in_ptr0 + in_addr, mask=in_mask, other=0.0,
                   eviction_policy='evict_first')  # [YBLOCK, XBLOCK]
    # Load bias: shape [XBLOCK] = [CBLOCK], broadcast over YBLOCK
    bias = tl.load(in_ptr1 + c_off, mask=c_mask, other=0.0,
                   eviction_policy='evict_last')  # [XBLOCK]
    # Add bias: broadcast [XBLOCK] -> [YBLOCK, XBLOCK]
    result = data + bias[None, :]  # [YBLOCK, XBLOCK]
    # Store to out_ptr0: strides (290400, 3025, 55, 1)
    # out_ptr0[n, c, hw] = n*290400 + c*3025 + hw*1
    # Shape [YBLOCK, XBLOCK]: outer dim YBLOCK varies along spatial (stride-1) => coalesced
    out_addr = pid_n * 290400 + c_off[None, :] * 3025 + s_off[:, None]
    tl.store(out_ptr0 + out_addr, result, mask=in_mask,
             eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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

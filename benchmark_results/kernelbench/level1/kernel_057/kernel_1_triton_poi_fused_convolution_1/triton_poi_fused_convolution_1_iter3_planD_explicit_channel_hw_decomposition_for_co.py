import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 2},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # 3D grid: pid(0)=HW tiles (YBLOCK=HWBLOCK), pid(1)=C tiles (XBLOCK=CBLOCK), pid(2)=N
    # XBLOCK = CBLOCK (innermost, stride=1 in input -> coalesced)
    # YBLOCK = HWBLOCK (outer)
    hw_offset = tl.program_id(0) * YBLOCK
    c_offset  = tl.program_id(1) * XBLOCK
    n_idx     = tl.program_id(2)

    # hw_idx: [YBLOCK, 1], c_idx: [1, XBLOCK]
    hw_idx = hw_offset + tl.arange(0, YBLOCK)[:, None]
    c_idx  = c_offset  + tl.arange(0, XBLOCK)[None, :]

    # HW=31500, C=128
    hw_mask = hw_idx < 31500
    c_mask  = c_idx  < 128
    mask    = hw_mask & c_mask

    # Decompose hw_idx -> (h, w): H=126, W=250
    h_idx = hw_idx // 250
    w_idx = hw_idx % 250

    # Load: input strides (N=4032000, C=1, H=32000, W=128)
    # addr = n*4032000 + c*1 + h*32000 + w*128
    # Consecutive threads differ in c_idx (stride=1) -> COALESCED
    load_addr = n_idx * 4032000 + c_idx + h_idx * 32000 + w_idx * 128
    tmp = tl.load(in_ptr0 + load_addr, mask=mask)

    # Store: output strides (N=4032000, C=31500, H=250, W=1)
    # addr = n*4032000 + c*31500 + h*250 + w
    store_addr = n_idx * 4032000 + c_idx * 31500 + h_idx * 250 + w_idx
    tl.store(out_ptr0 + store_addr, tmp, mask=mask)

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

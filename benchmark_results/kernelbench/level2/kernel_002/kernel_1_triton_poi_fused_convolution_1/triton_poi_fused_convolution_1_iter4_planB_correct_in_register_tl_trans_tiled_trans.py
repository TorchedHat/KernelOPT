import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500

    pid_x = tl.program_id(0)  # spatial (x2) tile
    pid_y = tl.program_id(1)  # channel*batch (y3) tile

    x2_range = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    y3_range = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]

    y0_range = y3_range % 128   # C dim, stride=1 in input
    y1_range = y3_range // 128  # N dim, stride=4032000 in input

    # LOAD tile shape [XBLOCK, YBLOCK]: last axis = y0 (stride=1 in input) -> COALESCED
    load_x2 = x2_range[:, None]    # [XBLOCK, 1]
    load_y0 = y0_range[None, :]    # [1, YBLOCK]  <- FAST axis, stride=1 in input
    load_y1 = y1_range[None, :]    # [1, YBLOCK]
    load_mask = (x2_range[:, None] < xnumel) & (y3_range[None, :] < ynumel)
    load_idx = load_y0 + 128 * load_x2 + 4032000 * load_y1  # [XBLOCK, YBLOCK]
    tmp = tl.load(in_ptr0 + load_idx, mask=load_mask, other=0.0, eviction_policy='evict_first')
    # tmp shape: [XBLOCK, YBLOCK]

    # Transpose in-register: [XBLOCK, YBLOCK] -> [YBLOCK, XBLOCK]
    tmp_T = tl.trans(tmp)  # [YBLOCK, XBLOCK]

    # STORE tile shape [YBLOCK, XBLOCK]: last axis = x2 (stride=1 in output) -> COALESCED
    store_y3 = y3_range[:, None]   # [YBLOCK, 1]
    store_x2 = x2_range[None, :]  # [1, XBLOCK]  <- FAST axis, stride=1 in output
    store_mask = (y3_range[:, None] < ynumel) & (x2_range[None, :] < xnumel)
    store_idx = store_x2 + 31500 * store_y3  # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + store_idx, tmp_T, mask=store_mask)

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

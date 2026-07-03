import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # ynumel = 4096 (= 32 * 128), xnumel = 31500
    # XBLOCK covers y0 in [0, 128) — stride-1 dimension in in_ptr0 (coalesced loads)
    # YBLOCK covers outer = (x2, y1) pairs, where x2 in [0, xnumel) and y1 in [0, ynumel//128)
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    # y0: channel index within group of 128 — stride-1 in in_ptr0
    # Apply contiguity hints on 1D before broadcasting to 2D
    y0_1d = pid_x * XBLOCK + tl.arange(0, XBLOCK)
    y0_1d = tl.max_contiguous(tl.multiple_of(y0_1d, XBLOCK), XBLOCK)
    y0mask_1d = y0_1d < 128
    y0 = y0_1d[None, :]  # shape [1, XBLOCK]

    # outer: covers (x2, y1) pairs — x2 in [0, 31500), y1 in [0, 32)
    outer_1d = pid_y * YBLOCK + tl.arange(0, YBLOCK)
    outer_mask_1d = outer_1d < (xnumel * (ynumel // 128))
    outer = outer_1d[:, None]  # shape [YBLOCK, 1]

    x2 = outer % xnumel   # shape [YBLOCK, 1]
    y1 = outer // xnumel  # shape [YBLOCK, 1]

    # Combined mask: [YBLOCK, XBLOCK]
    mask = outer_mask_1d[:, None] & y0mask_1d[None, :]

    # Load: in_ptr0 stride pattern is (batch=4032000, channel=1, H=32000, W=128)
    # in_ptr0[y1, y0, x2//250, x2%250] => flat index: y0 + 128*x2 + 4032000*y1
    tmp0 = tl.load(in_ptr0 + (y0 + 128 * x2 + 4032000 * y1), mask=mask, eviction_policy='evict_last')

    # Store: out_ptr0 stride pattern is (batch=4032000, channel=31500, H=250, W=1)
    # out_ptr0[y1, y0, x2//250, x2%250] => flat index: x2 + 31500*(y1*128 + y0)
    tl.store(out_ptr0 + (x2 + 31500 * (y1 * 128 + y0)), tmp0, mask=mask)

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

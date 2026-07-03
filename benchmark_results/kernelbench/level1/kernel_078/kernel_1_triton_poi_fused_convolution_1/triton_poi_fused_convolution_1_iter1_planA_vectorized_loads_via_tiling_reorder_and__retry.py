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
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # ynumel = 4096 = 32 * 128  (batch * channels)
    # xnumel = 31500 = 126 * 250 (spatial)
    # XBLOCK covers y0 in [0, 128)  -- stride-1 dimension in in_ptr0 (coalesced loads)
    # YBLOCK covers outer in [0, xnumel * (ynumel // 128)) = [0, 31500 * 32)
    ynumel = 4096
    xnumel = 31500

    # y0: channel index within batch, stride=1 in in_ptr0
    pid_x = tl.program_id(0)
    y0 = pid_x * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    y0 = tl.max_contiguous(tl.multiple_of(y0, XBLOCK), XBLOCK)
    y0mask = y0 < 128

    # outer: encodes (x2, y1) pair — x2 is spatial index, y1 is batch index
    pid_y = tl.program_id(1)
    outer = pid_y * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    outer_mask = outer < (xnumel * (ynumel // 128))  # < 31500 * 32 = 1008000

    x2 = outer % xnumel   # spatial index [0, 31500)
    y1 = outer // xnumel  # batch index   [0, 32)

    mask = outer_mask & y0mask

    # Load: in_ptr0 strides are (4032000, 1, 32000, 128)
    # flat index = y0 + 128*x2 + 4032000*y1
    # consecutive threads (varying y0) differ by 1 -> coalesced
    tmp0 = tl.load(in_ptr0 + (y0 + 128 * x2 + 4032000 * y1), mask=mask, other=0.0)

    # Store: out_ptr0 strides are (4032000, 31500, 250, 1)
    # y3 = y1*128 + y0  (full channel*batch index)
    # flat index = x2 + 31500 * (y1*128 + y0)
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

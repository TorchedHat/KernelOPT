import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 4, 'YBLOCK': 32},   num_warps=4),
    triton.Config({'XBLOCK': 4, 'YBLOCK': 64},   num_warps=4),
    triton.Config({'XBLOCK': 4, 'YBLOCK': 128},  num_warps=8),
    triton.Config({'XBLOCK': 4, 'YBLOCK': 256},  num_warps=8),
    triton.Config({'XBLOCK': 4, 'YBLOCK': 512},  num_warps=8),
    triton.Config({'XBLOCK': 4, 'YBLOCK': 1024}, num_warps=8),
    triton.Config({'XBLOCK': 1, 'YBLOCK': 128},  num_warps=4),
    triton.Config({'XBLOCK': 2, 'YBLOCK': 128},  num_warps=8),
    triton.Config({'XBLOCK': 2, 'YBLOCK': 256},  num_warps=8),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4
    GROUP_SIZE_Y = 8
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    num_pid_y = tl.cdiv(ynumel, YBLOCK)
    group_id = pid_y // GROUP_SIZE_Y
    group_size = tl.minimum(num_pid_y - group_id * GROUP_SIZE_Y, GROUP_SIZE_Y)
    pid_y_in_group = pid_y % GROUP_SIZE_Y
    swizzled_pid_y = group_id * GROUP_SIZE_Y + (pid_y_in_group % group_size)
    yoffset = swizzled_pid_y * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = pid_x * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    mask2d = xmask & ymask
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), mask=mask2d, eviction_policy='evict_first')
    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, mask=mask2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 8192
    xnumel = 4
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_sub_36[grid](arg_0, arg_1, 8192, 4)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cuda:0', dtype=torch.float32)
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

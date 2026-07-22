import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 2, 'YBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 4, 'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 8, 'YBLOCK': 128}, num_warps=8),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 2048
    yoffset = tl.program_id(0) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 128)
    x3 = xindex // 128
    y0 = (yindex % 128)
    y1 = yindex // 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask & ymask)
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask & ymask)
    tmp3 = tl.load(in_ptr0 + (256 + 2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask & ymask)
    tmp5 = tl.load(in_ptr0 + (257 + 2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask & ymask)
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 128*x4 + 262144*y1), tmp6, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 1024
    xnumel = 2048
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10[grid](arg_0, arg_1, 1024, 2048)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 128, 32, 256), (2097152, 8192, 256, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 128, 16, 128), (262144, 1, 16384, 128), device='cuda:0', dtype=torch.float32)
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

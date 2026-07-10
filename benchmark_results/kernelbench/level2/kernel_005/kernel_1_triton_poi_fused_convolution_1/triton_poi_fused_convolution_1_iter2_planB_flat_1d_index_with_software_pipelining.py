import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # 1D flat-index kernel: each thread owns one output element
    # Total elements: ynumel * xnumel = 4096 * 31500 = 129,024,000
    # Output layout contiguous: stride (4032000, 31500, 250, 1)
    # Input layout channels-last: stride (4032000, 1, 32000, 128)
    numel = ynumel * xnumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < numel
    # Decompose flat output index -> (n, c, h, w)
    # Output stride: (4032000, 31500, 250, 1)
    # w = xindex % 250
    # h = (xindex // 250) % 126
    # c = (xindex // 31500) % 128
    # n = xindex // 4032000
    w = xindex % 250
    tmp = xindex // 250
    h = tmp % 126
    tmp = tmp // 126
    c = tmp % 128
    n = tmp // 128
    # Input address: n*4032000 + c*1 + h*32000 + w*128
    in_idx = n * 4032000 + c + h * 32000 + w * 128
    tmp0 = tl.load(in_ptr0 + in_idx, xmask)
    tl.store(out_ptr0 + xindex, tmp0, xmask)

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

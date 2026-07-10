import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10
    xnumel = 150528
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    x_flat = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = x_flat[None, :]
    xmask = xindex < xnumel
    n_idx = yindex
    c_idx = xindex % 3
    hw_idx = xindex // 3
    in_offset = n_idx * 150528 + c_idx * 50176 + hw_idx
    tmp0 = tl.load(in_ptr0 + in_offset, mask=xmask & ymask, eviction_policy='evict_last')
    out_offset = n_idx * 150528 + xindex
    tl.store(out_ptr0 + out_offset, tmp0, mask=xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
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

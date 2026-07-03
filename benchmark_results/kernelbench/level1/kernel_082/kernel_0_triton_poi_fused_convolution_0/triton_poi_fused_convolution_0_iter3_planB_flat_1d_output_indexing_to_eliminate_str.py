import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=16, num_stages=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=16, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # 1D flat output indexing: numel = 1505280 = 10*3*224*224
    # Output layout: stride (150528, 1, 672, 3) -> flat out = batch*150528 + ch + spatial*3
    # Input layout:  stride (150528, 50176, 224, 1) -> flat in = batch*150528 + ch*50176 + spatial
    numel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < numel
    # Decompose flat output index into logical coords
    batch   = xindex // 150528
    rem     = xindex % 150528
    channel = rem % 3
    spatial = rem // 3
    # Compute flat input index
    in_idx = batch * 150528 + channel * 50176 + spatial
    tmp0 = tl.load(in_ptr0 + in_idx, xmask, eviction_policy='evict_first')
    tl.store(out_ptr0 + xindex, tmp0, xmask)

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

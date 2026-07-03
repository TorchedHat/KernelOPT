import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 4,  'YBLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2,  'YBLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2,  'YBLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 3       # channel dimension — stride=1 in NHWC output
    ynumel = 501760  # batch * spatial = 10 * 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # fast dim: channel c in [0,3)
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # slow dim: n*H*W pixel index
    ymask = yindex < ynumel
    n  = yindex // 50176   # batch index [0, 10)
    hw = yindex % 50176    # spatial pixel index [0, 50176)
    c  = xindex            # channel index [0, 3)
    # Load from NCHW input: in_ptr0[n*150528 + c*50176 + hw]
    tmp0 = tl.load(in_ptr0 + (hw + 50176*c + 150528*n), xmask & ymask, eviction_policy='evict_first')
    # Store to NHWC output: out_ptr0[n*150528 + hw*3 + c]  <- consecutive c = stride-1 COALESCED
    tl.store(out_ptr0 + (c + 3*hw + 150528*n), tmp0, xmask & ymask)

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

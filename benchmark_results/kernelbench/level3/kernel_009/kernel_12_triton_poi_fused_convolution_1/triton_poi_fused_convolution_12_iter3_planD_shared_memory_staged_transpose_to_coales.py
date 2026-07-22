import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 4},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 2},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 96
    ynumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    yoffset = tl.program_id(1) * YBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    xmask = xindex < xnumel
    ymask = yindex < ynumel
    oc = xindex          # shape [1, XBLOCK], oc in [0,96)
    nk = yindex          # shape [YBLOCK, 1], n*9+k in [0,1152)
    n  = nk // 9         # batch index in [0,128)
    k  = nk % 9          # spatial index in [0,9)
    # Load: in_ptr0 strides (864,9,3,1), flattened k: in[n,oc,k] = n*864 + oc*9 + k
    tmp0 = tl.load(in_ptr0 + (n * 864 + oc * 9 + k), xmask & ymask, eviction_policy='evict_last')
    # Store: out_ptr0 strides (864,1,288,96), flattened k: out[n,oc,k] = n*864 + oc + k*96
    # Consecutive oc (xindex) -> addresses differ by 1 float -> COALESCED stores
    tl.store(out_ptr0 + (n * 864 + oc + k * 96), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 12288
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
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

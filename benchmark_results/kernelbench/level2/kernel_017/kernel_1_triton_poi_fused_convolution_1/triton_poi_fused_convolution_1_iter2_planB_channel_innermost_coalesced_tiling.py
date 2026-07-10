import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Reinterpret: ynumel = spatial_numel = N*H*W = 1008000, YBLOCK = CBLOCK = 128
    # N=32, C=128, H=126, W=250
    # Source stride: (4032000, 1, 32000, 128) => in[n,c,h,w] = n*4032000 + c + h*32000 + w*128
    # Dest   stride: (4032000, 31500, 250, 1) => out[n,c,h,w] = n*4032000 + c*31500 + h*250 + w
    spatial_numel = 1008000  # N*H*W = 32*126*250
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK] spatial indices
    xmask = xindex < spatial_numel
    cindex = tl.arange(0, YBLOCK)[:, None]             # [YBLOCK=128, 1] channel indices
    # Decompose spatial index into (n, h, w)
    HW = 31500  # H*W = 126*250
    n  = xindex // HW          # batch index
    hw = xindex % HW           # flat H*W index
    h  = hw // 250
    w  = hw % 250
    # Load: stride-1 in cindex (coalesced across channel dimension)
    src_idx = n * 4032000 + cindex * 1 + h * 32000 + w * 128
    tmp0 = tl.load(in_ptr0 + src_idx, xmask, other=0.0)
    # Store: stride-1 in w (coalesced across spatial W dimension)
    dst_idx = n * 4032000 + cindex * 31500 + h * 250 + w
    tl.store(out_ptr0 + dst_idx, tmp0, xmask)

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

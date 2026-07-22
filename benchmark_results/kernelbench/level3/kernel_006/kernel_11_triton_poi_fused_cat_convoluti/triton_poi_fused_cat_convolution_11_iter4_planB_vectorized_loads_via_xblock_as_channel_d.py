import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, maxnreg=32),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, maxnreg=32),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, maxnreg=32),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, maxnreg=32),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=8, maxnreg=32),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, maxnreg=32),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, maxnreg=32),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, maxnreg=32),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8, maxnreg=32),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_cat_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # NEW: xnumel=512 (channel C, stride-1 in source), ynumel=501760 (N*H*W)
    xnumel = 512
    ynumel = 501760

    # XBLOCK covers channel dimension (stride-1 in source) -> COALESCED loads
    xoffset = tl.program_id(0) * XBLOCK
    x_arange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = xoffset + x_arange[None, :]  # shape [1, XBLOCK], channel index c
    xmask = xindex < xnumel  # xnumel=512

    # YBLOCK covers batch*spatial dimension
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = yindex < ynumel  # ynumel=501760

    # Decompose yindex into batch (b) and spatial flat index (s)
    b = yindex // 50176   # batch index [0..9]
    s = yindex % 50176    # spatial flat index [0..50175] = h*224 + w
    c = xindex            # channel index [0..511]

    # LOAD: source in_ptr0 stride=(25690112, 1, 114688, 512)
    # offset[b,c,h,w] = c*1 + (h*224+w)*512 + b*25690112 = c + 512*s + 25690112*b
    # Consecutive warp threads vary c (stride-1) -> COALESCED
    tmp0 = tl.load(in_ptr0 + (c + 512*s + 25690112*b), xmask & ymask,
                   eviction_policy='evict_last')

    # STORE: output out_ptr0 stride=(25690112, 50176, 224, 1) (NCHW contiguous)
    # offset[b,c,h,w] = (h*224+w)*1 + c*50176 + b*25690112 = s + 50176*c + 25690112*b
    tl.store(out_ptr0 + (s + 50176*c + 25690112*b), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 5120
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_cat_convolution_11[grid](arg_0, arg_1, 5120, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 512, 224, 224), (25690112, 1, 114688, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 512, 224, 224), (25690112, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
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

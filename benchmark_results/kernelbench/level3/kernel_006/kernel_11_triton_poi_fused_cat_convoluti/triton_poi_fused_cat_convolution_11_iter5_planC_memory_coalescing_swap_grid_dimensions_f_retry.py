import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 32},  num_warps=8),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 16},  num_warps=8),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 128}, num_warps=8),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 64},  num_warps=8),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 32},  num_warps=16),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_cat_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # XBLOCK tiles channel dim C=512 (stride-1 in source -> coalesced loads)
    # YBLOCK tiles spatial*batch dim N*HW=501760
    C = 512
    HW = 50176   # 224*224
    N_HW = 501760  # 10*50176

    xoffset = tl.program_id(0) * XBLOCK
    # xindex is 1D [XBLOCK], represents channel indices
    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xmask = xindex < C  # [XBLOCK]

    yoffset = tl.program_id(1) * YBLOCK
    # yindex is [YBLOCK, 1], represents N*HW indices
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < N_HW  # [YBLOCK, 1]

    y_hw = yindex % HW   # spatial position [YBLOCK, 1]
    y_n  = yindex // HW  # batch index      [YBLOCK, 1]

    # LOAD: source NHWC strides (25690112, 1, 114688, 512)
    # addr[n, hw, c] = c + 512*hw + 25690112*n  <- stride-1 in c (xindex) -> COALESCED
    # xindex[None, :] has shape [1, XBLOCK], broadcasts with [YBLOCK, 1] -> [YBLOCK, XBLOCK]
    load_idx = xindex[None, :] + 512 * y_hw + 25690112 * y_n
    tmp0 = tl.load(in_ptr0 + load_idx, mask=xmask[None, :] & ymask, eviction_policy='evict_first')

    # STORE: output NCHW strides (25690112, 50176, 224, 1)
    # addr[n, c, hw] = hw + 50176*c + 25690112*n  <- stride-1 in hw (y_hw) -> COALESCED
    store_idx = y_hw + HW * xindex[None, :] + 25690112 * y_n
    tl.store(out_ptr0 + store_idx, tmp0, mask=xmask[None, :] & ymask)

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

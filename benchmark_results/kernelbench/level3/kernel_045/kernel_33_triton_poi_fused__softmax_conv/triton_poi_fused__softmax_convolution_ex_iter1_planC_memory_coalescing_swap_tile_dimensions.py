import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_33(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # pid(0) covers ynumel=1024 (channel+batch), XBLOCK is fast-varying -> y0 (stride=1 in in_ptr0)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK]
    xmask = xindex < 1024
    y0 = xindex % 128   # channel: stride=1 in in_ptr0 -- NOW FAST (GOOD)
    y1 = xindex // 128  # batch index
    # pid(1) covers xnumel=8192 (spatial), YBLOCK is slow-varying -> x2 (stride=128 in in_ptr0)
    yoffset = tl.program_id(1) * YBLOCK
    x2 = yoffset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1]
    ymask = x2 < 8192
    # Load: y0 stride=1 is fast-varying -> COALESCED
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 1048576*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # Store: x2 stride=1 in out_ptr0 is fast-varying for the store
    tl.store(out_ptr0 + (x2 + 8192*y0 + 2097152*y1), tmp2, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 1024
    xnumel = 8192
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_sub_33[grid](arg_0, arg_1, arg_2, 1024, 8192)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((8, 128, 32, 256), (1048576, 1, 32768, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 128, 32, 256), (2097152, 8192, 256, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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

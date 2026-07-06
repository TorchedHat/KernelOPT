import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # X-dim = channel*batch (24576), stride-1 in input -> coalesced loads
    # Y-dim = spatial (3025)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    xmask = xindex < 24576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = yindex < 3025
    c = xindex % 96   # channel index, stride-1 in input, shape [1, XBLOCK]
    b = xindex // 96  # batch index, shape [1, XBLOCK]
    sp = yindex       # spatial index 0..3024, shape [YBLOCK, 1]
    # Apply vectorization hints on the 2D tensor: [1, XBLOCK] -> hint on last dim
    c_cont = tl.max_contiguous(tl.multiple_of(c, [1, 16]), [1, 16])
    # COALESCED load: consecutive threads read consecutive channels (stride=1)
    tmp0 = tl.load(in_ptr0 + (c_cont + 96*sp + 290400*b), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + c_cont, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # Store: out_ptr0 strides (290400, 3025, 55, 1) -> index = sp + 3025*(c + 96*b)
    tl.store(out_ptr0 + (sp + 3025*(c + 96*b)), tmp2, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
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

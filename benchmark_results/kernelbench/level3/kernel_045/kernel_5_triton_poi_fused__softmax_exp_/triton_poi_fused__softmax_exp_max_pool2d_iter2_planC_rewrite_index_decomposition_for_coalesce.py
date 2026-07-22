import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 4, 'XBLOCK': 64}, num_warps=4),
        triton.Config({'YBLOCK': 8, 'XBLOCK': 64}, num_warps=4),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=8),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 64}, num_warps=8),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 64}, num_warps=16),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Use a flat 1D grid: program_id(0) covers all output elements
    # BLOCK_SIZE = YBLOCK * XBLOCK elements per program
    # Total output elements = 8*64*32*256 = 4194304
    # Output strides: (524288, 1, 16384, 64) for (batch, chan, out_h, out_w)
    # Decompose flat index so chan (stride-1) varies fastest
    BLOCK_SIZE: tl.constexpr = YBLOCK * XBLOCK
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    numel = ynumel * xnumel  # 512 * 8192 = 4194304
    mask = offsets < numel
    chan  = offsets % 64
    rest  = offsets // 64
    out_w = rest % 256
    rest2 = rest // 256
    out_h = rest2 % 32
    batch = rest2 // 32
    # Input: (8,64,64,512) strides (4194304,32768,512,1)
    # Pool window: in_h=2*out_h, in_w=2*out_w
    base = batch * 4194304 + chan * 32768 + (2 * out_h) * 512 + (2 * out_w)
    tmp0 = tl.load(in_ptr0 + base,       mask=mask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + base + 1,   mask=mask, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + base + 512, mask=mask, eviction_policy='evict_first')
    tmp5 = tl.load(in_ptr0 + base + 513, mask=mask, eviction_policy='evict_first')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    # Output index: batch*524288 + chan*1 + out_h*16384 + out_w*64
    out_idx = batch * 524288 + chan + out_h * 16384 + out_w * 64
    tl.store(out_ptr0 + out_idx, tmp6, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 512
    xnumel = 8192
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5[grid](arg_0, arg_1, 512, 8192)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 64, 64, 512), (4194304, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 64, 32, 256), (524288, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
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

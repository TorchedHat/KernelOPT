import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80494592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_chan  = xindex // 39304
    x_group = x_chan // 16
    x2      = x_chan % 128
    tmp0     = tl.load(in_ptr0 + xindex, xmask)
    tmp_relu = tl.maximum(tl.zeros([XBLOCK], dtype=tl.float32), tmp0)
    tmp_mean     = tl.load(in_ptr1 + x_group, xmask, eviction_policy='evict_last')
    tmp_centered = tmp_relu - tmp_mean
    tmp_var     = tl.load(in_ptr2 + x_group, xmask, eviction_policy='evict_last')
    tmp_inv_std = libdevice.rsqrt(tmp_var * 1.5908420085906982e-06 + 1e-05)
    tmp_weight = tl.load(in_ptr3 + x2, xmask, eviction_policy='evict_last')
    tmp_bias   = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp_out    = tmp_centered * tmp_inv_std * tmp_weight + tmp_bias
    tl.store(out_ptr0 + xindex, tmp_out, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 80494592
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_native_group_norm_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 80494592)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 8, 1, 1), (8, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 8, 1, 1), (8, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5]

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

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=2),
    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
], key=['xnumel'], restore_value=['in_out_ptr0'])
@triton.jit
def triton_poi_fused_add_addmm_gelu_hardtanh_mul_sigmoid_tanh_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8192)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp5 * tmp4
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tl.full([1], 0.5, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp11 = tmp7 * tmp10
    tmp12 = libdevice.erf(tmp11)
    tmp13 = tl.full([1], 1.0, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp9 * tmp14
    tmp16 = tl.full([1], -1.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.minimum(tmp17, tmp13)
    tl.store(in_out_ptr0 + (x2), tmp18, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 8388608
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_addmm_gelu_hardtanh_mul_sigmoid_tanh_1[grid](arg_0, arg_1, arg_2, 8388608)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
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

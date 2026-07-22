import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(xnumel, XBLOCK)
    GROUP_SIZE = 8
    group_id = pid // GROUP_SIZE
    group_offset = pid % GROUP_SIZE
    swizzled_pid = group_offset * tl.cdiv(num_blocks, GROUP_SIZE) + group_id
    swizzled_pid = tl.minimum(swizzled_pid, num_blocks - 1)
    xoffset = swizzled_pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    # Coalesced index decomposition: channel (stride=1) is innermost dimension
    x0 = xindex % 64          # channel index (stride-1 in memory)
    sp = xindex // 64          # spatial position index
    batch = sp // 196          # batch index
    hw = sp % 196              # spatial (h*w) index
    h = hw // 14               # height index
    w = hw % 14                # width index
    # Reconstruct flat offset matching strides (12544, 1, 896, 64)
    flat = batch * 12544 + x0 * 1 + h * 896 + w * 64
    tmp0 = tl.load(in_out_ptr0 + flat, xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    # rsqrt(var+eps): single HW instruction (rsqrt.approx.f32 PTX)
    inv_std = tl.math.rsqrt(tmp3 + 1e-05)
    # normalize: (x - mean) * inv_std
    x_norm = (tmp0 - tmp1) * inv_std
    # scale + bias as single FMA: x_norm * weight + bias (fma.rn.f32 PTX)
    result = tl.math.fma(x_norm, tmp12, tmp14)
    tl.store(in_out_ptr0 + flat, result, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 125440
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 125440)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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

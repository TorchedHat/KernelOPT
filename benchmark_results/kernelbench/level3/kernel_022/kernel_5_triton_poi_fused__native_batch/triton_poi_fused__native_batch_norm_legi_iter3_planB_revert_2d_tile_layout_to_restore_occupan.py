import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=4, maxnreg=64),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=3, maxnreg=64),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # 1D flat layout: treat total elements as ynumel * xnumel = 1920 * 12544 = 24084480
    # YBLOCK is fixed to 1 in all configs, so we use only XBLOCK for tiling
    xnumel_flat = 1920 * 12544  # = 24084480
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)   # shape [XBLOCK] - 1D flat
    xmask   = xindex < xnumel_flat
    # Decompose flat index into spatial, channel, batch
    # xindex = batch * 2408448 + channel * 12544 + spatial
    # but we want: spatial = xindex % 12544, channel = (xindex // 12544) % 192, batch = xindex // 2408448
    spatial  = xindex % 12544
    tmp_bc   = xindex // 12544
    channel  = tmp_bc % 192
    batch    = tmp_bc // 192
    # Load per-channel BN parameters (shape [XBLOCK])
    tmp_mean   = tl.load(in_ptr1 + channel, mask=xmask, eviction_policy='evict_last')
    tmp_var    = tl.load(in_ptr2 + channel, mask=xmask, eviction_policy='evict_last')
    tmp_weight = tl.load(in_ptr3 + channel, mask=xmask, eviction_policy='evict_last')
    tmp_bias   = tl.load(in_ptr4 + channel, mask=xmask, eviction_policy='evict_last')
    # Precompute per-channel scale and bias (all shape [XBLOCK])
    tmp_inv_std    = libdevice.rsqrt(tmp_var + 1e-05)
    combined_scale = tmp_weight * tmp_inv_std
    combined_bias  = tmp_bias - tmp_mean * combined_scale
    # Load main input: in_ptr0 has shape (125440, 192), strides (192, 1)
    # Address: channel + 192*spatial + 2408448*batch
    tmp0 = tl.load(in_ptr0 + (channel + 192*spatial + 2408448*batch), mask=xmask, eviction_policy='evict_first')
    # Apply BN: (x - mean) / sqrt(var + eps) * weight + bias
    tmp15 = tmp0 * combined_scale + combined_bias
    # Apply hardtanh clamp to [-1, 1]
    tmp15 = tl.maximum(tmp15, -1.0)
    tmp15 = tl.minimum(tmp15, 1.0)
    # Store: out_ptr0 has shape (10, 192, 12544), strides (2408448, 12544, 1)
    # Address: spatial + 12544*channel + 2408448*batch
    tl.store(out_ptr0 + (spatial + 12544*channel + 2408448*batch), tmp15, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    ynumel = 1920
    xnumel = 12544
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1920, 12544)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((125440, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 192, 112, 112), (2408448, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
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

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8, num_stages=2),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # After swap: xnumel=1920 (channel*batch, stride-1 in in_ptr0 — COALESCED)
    #             ynumel=12544 (spatial, stride-192 in in_ptr0 — slow axis)
    ynumel = 12544
    xnumel = 1920
    # X dimension: channel*batch (fast, stride-1 in in_ptr0)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    # Y dimension: spatial (slow, stride-192 in in_ptr0)
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    # Decompose x dimension: x0=channel (0..191), x1=batch (0..9)
    x0 = xindex % 192
    x1 = xindex // 192
    # y2 = spatial index
    y2 = yindex
    # Channel-only scalar loads (indexed by x0, broadcast over y)
    tmp_mean   = tl.load(in_ptr1 + x0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    tmp_var    = tl.load(in_ptr2 + x0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    tmp_weight = tl.load(in_ptr3 + x0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    tmp_bias   = tl.load(in_ptr4 + x0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    # Precompute per-channel scale and bias
    tmp_eps     = tl.full([1, 1], 1e-05, tl.float32)
    tmp_inv_std = libdevice.rsqrt(tmp_var + tmp_eps)        # rsqrt(var + eps)
    combined_scale = tmp_weight * tmp_inv_std               # weight / sqrt(var+eps)  [1, XBLOCK]
    combined_bias  = tmp_bias - tmp_mean * combined_scale   # bias - mean*scale        [1, XBLOCK]
    # Load main input element — COALESCED: x0 is stride-1 in in_ptr0
    # in_ptr0 address: x0 + 192*y2 + 2408448*x1
    tmp0 = tl.load(in_ptr0 + (x0 + 192 * y2 + 2408448 * x1), mask=xmask & ymask, eviction_policy='evict_first')
    # BN: x * scale + bias, then hardtanh clamp [-1, 1]
    tmp15 = tmp0 * combined_scale + combined_bias
    # Store — out_ptr0 address: y2 + 12544*x0 + 2408448*x1
    tl.store(out_ptr0 + (y2 + 12544 * x0 + 2408448 * x1), tmp15, mask=xmask & ymask)

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

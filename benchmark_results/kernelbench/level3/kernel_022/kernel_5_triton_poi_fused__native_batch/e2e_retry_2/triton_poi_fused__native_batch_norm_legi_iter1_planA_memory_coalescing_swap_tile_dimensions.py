import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 32}, num_warps=8, num_stages=3),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # NEW: xnumel=192 (channels, stride-1 in in_ptr0), ynumel=125440 (batch*spatial)
    xnumel = 192
    ynumel = 125440
    # xindex covers channels (stride-1 dimension) — COALESCED
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    # yindex covers batch*spatial
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    # Channel index (for BN params)
    c = xindex   # shape [1, XBLOCK]
    # Spatial-batch index
    sp = yindex  # shape [YBLOCK, 1]
    # Load BN parameters indexed by channel only — coalesced (stride-1 in c)
    tmp_mean   = tl.load(in_ptr1 + c, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    tmp_var    = tl.load(in_ptr2 + c, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    tmp_weight = tl.load(in_ptr3 + c, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    tmp_bias   = tl.load(in_ptr4 + c, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    # Precompute per-channel scale and shift
    tmp_eps     = tl.full([1, 1], 1e-05, tl.float32)
    tmp_inv_std = libdevice.rsqrt(tmp_var + tmp_eps)         # [1, XBLOCK]
    combined_scale = tmp_weight * tmp_inv_std                # [1, XBLOCK]
    combined_bias  = tmp_bias - tmp_mean * combined_scale    # [1, XBLOCK]
    # Load main input: in_ptr0 has shape (125440, 192) with strides (192, 1)
    # Address: sp * 192 + c — stride-1 in c (COALESCED)
    tmp0 = tl.load(in_ptr0 + (sp * 192 + c), mask=xmask & ymask, eviction_policy='evict_first')
    # Batch-norm + hardtanh(0, 6)
    tmp15 = tmp0 * combined_scale + combined_bias
    tmp15 = tl.maximum(tmp15, 0.0)
    tmp15 = tl.minimum(tmp15, 6.0)
    # Output tensor has shape (10, 192, 112, 112) with strides (2408448, 12544, 112, 1)
    # batch = sp // 12544, spatial = sp % 12544
    batch   = sp // 12544   # [YBLOCK, 1]
    spatial = sp % 12544    # [YBLOCK, 1]
    # out_ptr0 offset: batch*2408448 + c*12544 + spatial
    tl.store(out_ptr0 + (batch * 2408448 + c * 12544 + spatial), tmp15, mask=xmask & ymask)

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

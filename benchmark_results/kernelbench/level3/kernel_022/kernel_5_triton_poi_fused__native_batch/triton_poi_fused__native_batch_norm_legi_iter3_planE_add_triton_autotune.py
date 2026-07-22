import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 16}, num_warps=8, num_stages=3),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 12544
    # XBLOCK covers channel+batch dimension (ynumel=1920, stride-1 in in_ptr0)
    # Consecutive warp threads differ in channel (y0, stride-1) -> coalesced
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]   # shape [1, XBLOCK]
    xmask = xindex < ynumel
    # YBLOCK covers spatial dimension (xnumel=12544, stride-192 in in_ptr0)
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # shape [YBLOCK, 1]
    ymask = yindex < xnumel
    # Index decomposition
    y0 = xindex % 192       # channel index [0, 192), shape [1, XBLOCK]
    y1 = xindex // 192      # batch index [0, 10), shape [1, XBLOCK]
    x2 = yindex             # spatial index [0, 12544), shape [YBLOCK, 1]
    y3 = xindex             # channel+batch combined index, shape [1, XBLOCK]
    # Channel parameter loads: y0 varies across XBLOCK (warp threads) -> coalesced
    tmp_mean   = tl.load(in_ptr1 + y0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    tmp_var    = tl.load(in_ptr2 + y0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    tmp_weight = tl.load(in_ptr3 + y0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    tmp_bias   = tl.load(in_ptr4 + y0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
    # Precompute per-channel scale and bias
    tmp_eps     = tl.full([1, 1], 1e-05, tl.float32)
    tmp_inv_std = libdevice.rsqrt(tmp_var + tmp_eps)        # rsqrt(var + eps)
    combined_scale = tmp_weight * tmp_inv_std               # weight / sqrt(var+eps)  [1, XBLOCK]
    combined_bias  = tmp_bias - tmp_mean * combined_scale   # bias - mean*scale        [1, XBLOCK]
    # Load main input element: address formula y0 + 192*x2 + 2408448*y1 UNCHANGED
    # Now consecutive warp threads differ in y0 (stride-1) -> coalesced
    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 2408448*y1), mask=xmask & ymask, eviction_policy='evict_first')
    # FMA: broadcasts [1, XBLOCK] over [YBLOCK, 1]
    tmp15 = tmp0 * combined_scale + combined_bias
    # Store: address formula x2 + 12544*y3 UNCHANGED
    tl.store(out_ptr0 + (x2 + 12544*y3), tmp15, mask=xmask & ymask)

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

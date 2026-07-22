import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 16384)
    x2 = xindex // 524288
    x3 = xindex
    # x2 in [0..7], x0 in [0..31]: h = x0+32*x2 in [0..255]
    # h//128 = x2//4 (compiler reduces to >>2)
    # h%128  = x0 + 32*(x2%4) (x2%4 reduces to &3)
    # h//256 = 0 always (since h <= 255), so drop 768*(h//256) term
    x2_div4 = x2 // 4           # 0 or 1
    x2_mod4 = x2 % 4            # 0..3
    h_mod128 = x0 + 32 * x2_mod4
    in_offset = 384 * x2_div4 + 768 * x1 + h_mod128
    tmp0 = tl.load(in_ptr0 + in_offset, xmask)
    tmp1 = tl.load(in_ptr1 + h_mod128, xmask, eviction_policy='evict_last')
    SCALE = 0.1767766952966369
    # Fuse FADD+FMUL: (tmp0+tmp1)*SCALE = tmp0*SCALE + tmp1*SCALE
    tmp4 = tl.math.fma(tmp0, SCALE, tmp1 * SCALE)
    tl.store(out_ptr0 + (x3), tmp4, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 4194304
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2[grid](arg_0, arg_1, arg_2, 4194304, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
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

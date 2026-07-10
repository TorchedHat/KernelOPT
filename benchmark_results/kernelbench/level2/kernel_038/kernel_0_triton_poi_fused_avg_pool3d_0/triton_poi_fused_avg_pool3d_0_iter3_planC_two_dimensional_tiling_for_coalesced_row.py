import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_avg_pool3d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: pid_w covers output W tiles (WBLOCK=32), pid_rest covers H*D*C*batch tiles (RBLOCK=4)
    # XBLOCK = WBLOCK * RBLOCK = 32 * 4 = 128 threads per block
    WBLOCK: tl.constexpr = 32
    RBLOCK: tl.constexpr = XBLOCK // WBLOCK  # = 4

    pid_w    = tl.program_id(0)  # W tile
    pid_rest = tl.program_id(1)  # rest tile

    # Output W indices: contiguous [0..31]
    x0 = pid_w * WBLOCK + tl.arange(0, WBLOCK)  # [WBLOCK]

    # Rest indices
    r_idx = pid_rest * RBLOCK + tl.arange(0, RBLOCK)  # [RBLOCK]
    rest_numel = xnumel // 32  # = 524288
    rmask = r_idx < rest_numel  # boundary guard

    # Decode rest index into output H and batch*C*D
    x1 = (r_idx % 32)   # output H in [0,32)
    x2 = r_idx // 32    # batch*C*D

    # Apply contiguity hints on 1D x0 before broadcasting
    iw0_1d = tl.max_contiguous(tl.multiple_of(2 * x0, 2), WBLOCK)       # [WBLOCK]
    iw1_1d = tl.max_contiguous(tl.multiple_of(2 * x0 + 1, 1), WBLOCK)   # [WBLOCK]

    # Broadcast to 2D [RBLOCK, WBLOCK]
    iw0    = iw0_1d[None, :]   # [1, WBLOCK]
    iw1    = iw1_1d[None, :]   # [1, WBLOCK]
    x1_2d  = x1[:, None]       # [RBLOCK, 1]
    x2_2d  = x2[:, None]       # [RBLOCK, 1]
    mask2d = rmask[:, None]    # [RBLOCK, 1]

    # Input row base address
    row_base = 128 * x1_2d + 8192 * x2_2d  # [RBLOCK, 1]

    tmp0  = tl.load(in_ptr0 + iw0 + row_base,        mask=mask2d)
    tmp1  = tl.load(in_ptr0 + iw1 + row_base,        mask=mask2d)
    tmp3  = tl.load(in_ptr0 + iw0 + row_base + 64,   mask=mask2d)
    tmp5  = tl.load(in_ptr0 + iw1 + row_base + 64,   mask=mask2d)
    tmp7  = tl.load(in_ptr0 + iw0 + row_base + 4096, mask=mask2d)
    tmp9  = tl.load(in_ptr0 + iw1 + row_base + 4096, mask=mask2d)
    tmp11 = tl.load(in_ptr0 + iw0 + row_base + 4160, mask=mask2d)
    tmp13 = tl.load(in_ptr0 + iw1 + row_base + 4160, mask=mask2d)

    tmp14 = tmp0 + tmp1 + tmp3 + tmp5 + tmp7 + tmp9 + tmp11 + tmp13
    tmp16 = tmp14 / 8.0

    # Output index: x2*1024 + x1*32 + x0
    x0_2d   = x0[None, :]  # [1, WBLOCK]
    out_idx = x2_2d * 1024 + x1_2d * 32 + x0_2d  # [RBLOCK, WBLOCK]
    tl.store(out_ptr0 + out_idx, tmp16, mask=mask2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 16777216
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_avg_pool3d_0[grid](arg_0, arg_1, 16777216, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 32, 32, 64, 64), (4194304, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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

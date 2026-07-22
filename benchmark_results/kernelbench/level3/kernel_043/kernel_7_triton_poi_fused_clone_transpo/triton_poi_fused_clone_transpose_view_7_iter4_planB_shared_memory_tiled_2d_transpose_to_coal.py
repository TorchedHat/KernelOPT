import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8}, num_warps=4),
        triton.Config({'XBLOCK': 4}, num_warps=4),
        triton.Config({'XBLOCK': 8}, num_warps=8),
        triton.Config({'XBLOCK': 4}, num_warps=8),
        triton.Config({'XBLOCK': 2}, num_warps=4),
        triton.Config({'XBLOCK': 1}, num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_clone_transpose_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # XBLOCK = TILE_X1 (number of heads to process per CTA)
    # We always process 128 features (TILE_X0=128) with masking for the 96-element boundary
    # in_ptr0:  (1024, 512, 96) strides (49152, 96, 1)  [batch*heads, seq, feat]
    # out_ptr0: (128, 512, 8, 96) strides (393216, 768, 96, 1)  [batch, seq, heads, feat]
    TILE_X0: tl.constexpr = 128
    pid = tl.program_id(0)
    num_tiles_x1 = tl.cdiv(8, XBLOCK)
    # Decompose pid -> (x3, x2, tile_x1)
    tile_x1_id = pid % num_tiles_x1
    pid_rem = pid // num_tiles_x1
    x2 = pid_rem % 512
    x3 = pid_rem // 512
    x1_base = tile_x1_id * XBLOCK
    x1_range = x1_base + tl.arange(0, XBLOCK)   # [XBLOCK]
    x0_range = tl.arange(0, TILE_X0)             # [TILE_X0]
    x1_idx = x1_range[:, None]   # [XBLOCK, 1]
    x0_idx = x0_range[None, :]   # [1, TILE_X0]
    # Mask for x0 boundary (96 elements, TILE_X0=128)
    x0_mask = x0_idx < 96
    x1_mask = x1_idx < 8
    mask = x1_mask & x0_mask
    # Coalesced load: innermost dim is x0 (stride=1), x1 stride=49152
    src_idx = (x3 * 8 + x1_idx) * 49152 + x2 * 96 + x0_idx
    tile = tl.load(in_ptr0 + src_idx, mask=mask, other=0.0)  # [XBLOCK, TILE_X0]
    # Coalesced store: innermost dim is x0 (stride=1)
    dst_idx = x3 * 393216 + x2 * 768 + x1_idx * 96 + x0_idx
    tl.store(out_ptr0 + dst_idx, tile, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_view_7[grid](arg_0, arg_1, 50331648)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 8, 96), (393216, 768, 96, 1), device='cuda:0', dtype=torch.float32)
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

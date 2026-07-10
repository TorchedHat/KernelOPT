import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 1,  'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 2,  'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 4,  'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 8,  'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Repurposed: ynumel=spatial_numel=31500, xnumel=batch_numel=32
    # YBLOCK=BLOCK_S (spatial tile), XBLOCK=BLOCK_C=128 (channel tile, covers all C)
    # W=250, H=126, C=128, stride_src=(4032000,1,32000,128), stride_dst=(4032000,31500,250,1)
    W = 250
    pid_s = tl.program_id(0)   # spatial tile index
    pid_n = tl.program_id(1)   # batch index
    s_off = pid_s * YBLOCK + tl.arange(0, YBLOCK)   # [YBLOCK] spatial offsets
    c_off = tl.arange(0, XBLOCK)                     # [XBLOCK=128] channel offsets 0..127
    s_mask = s_off < ynumel
    # Clamp s_off to avoid out-of-bounds pointer arithmetic on masked lanes
    s_off_safe = tl.where(s_mask, s_off, 0)
    h_idx = s_off_safe // W    # [YBLOCK]
    w_idx = s_off_safe % W     # [YBLOCK]
    # Source: stride (4032000,1,32000,128) -> flat = n*4032000 + c + h*32000 + w*128
    src_base = pid_n * 4032000 + h_idx * 32000 + w_idx * 128   # [YBLOCK]
    src_idx = src_base[:, None] + c_off[None, :]                # [YBLOCK, XBLOCK]
    tile = tl.load(in_ptr0 + src_idx, mask=s_mask[:, None], other=0.0)
    # Dest: stride (4032000,31500,250,1) -> flat = n*4032000 + c*31500 + s
    dst_base = pid_n * 4032000 + s_off_safe                     # [YBLOCK]
    dst_idx = dst_base[:, None] + c_off[None, :] * 31500        # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + dst_idx, tile, mask=s_mask[:, None])

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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

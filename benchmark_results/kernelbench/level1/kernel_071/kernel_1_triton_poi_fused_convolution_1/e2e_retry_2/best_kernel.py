import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=4, num_stages=4),
    triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8, num_stages=3),
    triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=8, num_stages=4),
    triton.Config({'YBLOCK': 8,  'XBLOCK': 128}, num_warps=4, num_stages=2),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    # rows: flat (b*128+c) index, size YBLOCK
    rows = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
    # cols: flat spatial (h*250+w) index, size XBLOCK
    cols = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    row_mask = rows < ynumel
    col_mask = cols < xnumel
    c = rows % 128   # channel index (stride-1 in src)
    b = rows // 128  # batch index
    # Load [XBLOCK, YBLOCK]: threads span YBLOCK (channel, stride 1 in src) => COALESCED
    src = c[None, :] + 128 * cols[:, None] + 4032000 * b[None, :]  # [XBLOCK, YBLOCK]
    load_mask = col_mask[:, None] & row_mask[None, :]
    tile = tl.load(in_ptr0 + src, mask=load_mask, other=0.0)  # [XBLOCK, YBLOCK]
    # Store [YBLOCK, XBLOCK]: threads span XBLOCK (cols, stride 1 in dst) => COALESCED
    dst = cols[None, :] + 31500 * rows[:, None]  # [YBLOCK, XBLOCK]
    store_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out_ptr0 + dst, tl.trans(tile), mask=store_mask)  # tl.trans: [XBLOCK,YBLOCK]->[YBLOCK,XBLOCK]

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

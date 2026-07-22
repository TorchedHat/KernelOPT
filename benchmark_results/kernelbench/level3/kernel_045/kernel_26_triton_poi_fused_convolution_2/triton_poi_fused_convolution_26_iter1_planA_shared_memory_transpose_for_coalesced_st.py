import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 16, 'XBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=8, num_stages=2),
    triton.Config({'YBLOCK': 64, 'XBLOCK': 64}, num_warps=8, num_stages=2),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 512
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)
    y_base = pid_y * YBLOCK
    x_base = pid_x * XBLOCK
    row = tl.arange(0, YBLOCK)
    col = tl.arange(0, XBLOCK)
    yindex = y_base + row[:, None]   # [YBLOCK, 1]
    xindex = x_base + col[None, :]   # [1, XBLOCK]
    ymask = yindex < ynumel
    xmask = xindex < xnumel
    # COALESCED LOAD: xindex is fast dim (stride 1 in input)
    tmp0 = tl.load(in_ptr0 + (xindex + 512 * yindex), mask=ymask & xmask, eviction_policy='evict_last')
    # TRANSPOSE the [YBLOCK, XBLOCK] register tile -> [XBLOCK, YBLOCK]
    tmp0_T = tl.trans(tmp0)
    # Transposed store indices: y varies in inner dim (stride 1 in output)
    yindex_T = y_base + col[:, None]   # [XBLOCK, 1]
    xindex_T = x_base + row[None, :]   # [1, YBLOCK]
    ymask_T = yindex_T < ynumel
    xmask_T = xindex_T < xnumel
    y0_T = yindex_T % 1024
    y1_T = yindex_T // 1024
    # COALESCED STORE: yindex_T (stride 1) is now the fast dim
    tl.store(out_ptr0 + (y0_T + 1024 * xindex_T + 524288 * y1_T), tmp0_T, mask=ymask_T & xmask_T)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 8192
    xnumel = 512
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_26[grid](arg_0, arg_1, 8192, 512)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 1024, 8, 64), (524288, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 1024, 8, 64), (524288, 1, 65536, 1024), device='cuda:0', dtype=torch.float32)
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

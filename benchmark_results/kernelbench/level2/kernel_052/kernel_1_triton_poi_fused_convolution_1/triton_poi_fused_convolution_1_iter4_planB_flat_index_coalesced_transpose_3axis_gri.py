import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # 3-axis grid: pid(0)=hw tile, pid(1)=channel tile, pid(2)=batch
    # XBLOCK = HWBLOCK (spatial), YBLOCK = CBLOCK (channel)
    pid_hw = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    # HW offsets: [XBLOCK, 1]
    hw_off = pid_hw * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    hw_mask = hw_off < 31500  # HW=31500, not divisible by XBLOCK=32

    # Channel offsets: [1, YBLOCK]
    c_off = pid_c * YBLOCK + tl.arange(0, YBLOCK)[None, :]
    # C=128 is exactly divisible by YBLOCK=32, no c_mask needed

    # Decompose hw_off into (h, w) for input stride computation
    # Input shape: (32, 128, 126, 250), strides: (4032000, 1, 32000, 128)
    # W=250, stride_h=32000, stride_w=128
    h = hw_off // 250   # [XBLOCK, 1]
    w = hw_off % 250    # [XBLOCK, 1]

    # Input index: n*4032000 + c*1 + h*32000 + w*128
    # c_off varies along innermost warp dim -> COALESCED LOADS (stride-1 in input)
    in_idx = pid_n * 4032000 + c_off + h * 32000 + w * 128  # [XBLOCK, YBLOCK]
    tmp = tl.load(in_ptr0 + in_idx, mask=hw_mask)

    # Output index: n*4032000 + c*31500 + h*250 + w*1
    # Note: h*250 + w == hw_off (algebraic identity)
    # hw_off varies along outer dim -> COALESCED STORES (stride-1 in output)
    out_idx = pid_n * 4032000 + c_off * 31500 + hw_off  # [XBLOCK, YBLOCK]
    tl.store(out_ptr0 + out_idx, tmp, mask=hw_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
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

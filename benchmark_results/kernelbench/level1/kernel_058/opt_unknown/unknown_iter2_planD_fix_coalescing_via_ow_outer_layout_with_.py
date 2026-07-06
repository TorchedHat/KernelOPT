import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 32,  'BLOCK_OH': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64,  'BLOCK_OH': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 16,  'BLOCK_OH': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_OH': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 256, 'BLOCK_OH': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 32,  'BLOCK_OH': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OW': 64,  'BLOCK_OH': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_OH': 2}, num_warps=8, num_stages=3),
    ],
    key=['OD', 'OH', 'OW', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    s_ib, s_iic, s_iid, s_iih, s_iiw,
    s_wic, s_woc, s_wkd, s_wkh, s_wkw,
    s_ob, s_ooc, s_ood, s_ooh, s_oow,
    BLOCK_OW: tl.constexpr,
    BLOCK_OH: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    b_oc_od = pid0
    od = b_oc_od % OD
    b_oc = b_oc_od // OD
    oc = b_oc % OC
    b = b_oc // OC

    oh_start = pid1 * BLOCK_OH
    ow_start = pid2 * BLOCK_OW

    oh_offs = oh_start + tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    ow_offs = ow_start + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]

    oh_mask = oh_offs < OH  # [BLOCK_OH]
    ow_mask = ow_offs < OW  # [BLOCK_OW]

    acc = tl.zeros([BLOCK_OW, BLOCK_OH], dtype=tl.float32)

    for ic in range(IC):
        for kd in range(KD):
            # od = id * SD - PD + kd  =>  id = (od + PD - kd) / SD
            od_src = od + PD - kd
            id_val = od_src // SD
            id_valid = (od_src >= 0) & (od_src % SD == 0) & (id_val >= 0) & (id_val < ID)

            for kh in range(KH):
                for kw in range(KW):
                    w_val = tl.load(weight_ptr + ic * s_wic + oc * s_woc + kd * s_wkd + kh * s_wkh + kw * s_wkw)

                    # oh = ih * SH - PH + kh => ih = (oh + PH - kh) / SH
                    oh_src = oh_offs + PH - kh  # [BLOCK_OH]
                    ih_vals = oh_src // SH       # [BLOCK_OH]
                    ih_valid = (oh_src >= 0) & (oh_src % SH == 0) & (ih_vals >= 0) & (ih_vals < IH) & oh_mask  # [BLOCK_OH]

                    ow_src = ow_offs + PW - kw  # [BLOCK_OW]
                    iw_vals = ow_src // SW       # [BLOCK_OW]
                    iw_valid = (ow_src >= 0) & (ow_src % SW == 0) & (iw_vals >= 0) & (iw_vals < IW) & ow_mask  # [BLOCK_OW]

                    inp_base = b * s_ib + ic * s_iic + id_val * s_iid
                    ih_ptrs = ih_vals * s_iih   # [BLOCK_OH]
                    iw_ptrs = iw_vals * s_iiw   # [BLOCK_OW]

                    full_mask = id_valid & iw_valid[:, None] & ih_valid[None, :]  # [BLOCK_OW, BLOCK_OH]
                    inp_ptrs = input_ptr + inp_base + iw_ptrs[:, None] + ih_ptrs[None, :]
                    inp_vals = tl.load(inp_ptrs, mask=full_mask, other=0.0)

                    acc += inp_vals * w_val

    out_base = b * s_ob + oc * s_ooc + od * s_ood
    out_oh_ptrs = oh_offs * s_ooh  # [BLOCK_OH]
    out_ow_ptrs = ow_offs * s_oow  # [BLOCK_OW]
    out_ptrs = output_ptr + out_base + out_ow_ptrs[:, None] + out_oh_ptrs[None, :]
    out_mask = ow_mask[:, None] & oh_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)

def kernel_function(weight, x):
    # weight: [IC, OC, KD, KH, KW]
    # x: [B, IC, ID, IH, IW]
    B, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    SD, SH, SW = 1, 1, 1
    PD, PH, PW = 0, 0, 0

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    output = torch.zeros(B, OC, OD, OH, OW, device=x.device, dtype=torch.float32)

    s_ib, s_iic, s_iid, s_iih, s_iiw = x.stride()
    s_wic, s_woc, s_wkd, s_wkh, s_wkw = weight.stride()
    s_ob, s_ooc, s_ood, s_ooh, s_oow = output.stride()

    grid = lambda meta: (B * OC * OD, triton.cdiv(OH, meta['BLOCK_OH']), triton.cdiv(OW, meta['BLOCK_OW']))

    conv_transpose3d_kernel[grid](
        x, weight, output,
        B, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        s_ib, s_iic, s_iid, s_iih, s_iiw,
        s_wic, s_woc, s_wkd, s_wkh, s_wkw,
        s_ob, s_ooc, s_ood, s_ooh, s_oow,
    )

    return output


def get_inputs():
    weight = torch.randn(32, 16, 3, 5, 7)
    x = torch.randn(16, 32, 16, 32, 64)
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, x = inputs
    weight_gpu = weight.cuda()
    x_gpu = x.cuda()

    out = kernel_function(weight_gpu, x_gpu)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F
        ref = F.conv_transpose3d(x_gpu, weight_gpu, bias=None, stride=(1, 1, 1), padding=(0, 0, 0))
        max_diff = (out - ref).abs().max().item()
        print(f"Max diff: {max_diff}")
        assert max_diff < 1e-3, f"Max diff too large: {max_diff}"
        print("Kernel executed successfully.")
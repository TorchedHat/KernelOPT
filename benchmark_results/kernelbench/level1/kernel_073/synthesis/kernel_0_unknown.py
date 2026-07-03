import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 4}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_D, IN_H, IN_W,
    OUT_C, OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    GROUPS, IN_C_PER_GROUP, OUT_C_PER_GROUP,
    M, N_out, K,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    w_stride_ic, w_stride_oc, w_stride_kd, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: batch * OUT_D * OUT_H * OUT_W
    OUT_DHW = OUT_D * OUT_H * OUT_W
    OUT_HW = OUT_H * OUT_W

    batch = offs_m // OUT_DHW
    rem = offs_m % OUT_DHW
    od = rem // OUT_HW
    rem2 = rem % OUT_HW
    oh = rem2 // OUT_W
    ow = rem2 % OUT_W

    # Decode N index: group * OUT_C_PER_GROUP + oc_in_group
    group = offs_n // OUT_C_PER_GROUP
    oc_in_group = offs_n % OUT_C_PER_GROUP

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = IN_C_PER_GROUP * KD * KH * KW
    KHW = KH * KW
    KDHW = KD * KH * KW

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Decode K: ic_in_group, kd, kh, kw
        ic_in_group = offs_k // KDHW
        kd = (offs_k % KDHW) // KHW
        kh = (offs_k % KHW) // KW
        kw = offs_k % KW

        # For ConvTranspose3d: output[od] += input[id] * weight[ic, oc, kd, kh, kw]
        # where od = id * stride - pad + kd => id = (od + pad - kd) / stride
        # id must be integer and in [0, IN_D)

        id_num_d = od[:, None] + PAD_D - kd[None, :]
        id_num_h = oh[:, None] + PAD_H - kh[None, :]
        id_num_w = ow[:, None] + PAD_W - kw[None, :]

        valid_d = (id_num_d % STRIDE_D == 0)
        valid_h = (id_num_h % STRIDE_H == 0)
        valid_w = (id_num_w % STRIDE_W == 0)

        id_d = id_num_d // STRIDE_D
        id_h = id_num_h // STRIDE_H
        id_w = id_num_w // STRIDE_W

        valid_range_d = (id_d >= 0) & (id_d < IN_D)
        valid_range_h = (id_h >= 0) & (id_h < IN_H)
        valid_range_w = (id_w >= 0) & (id_w < IN_W)
        valid_k = offs_k[None, :] < K
        valid_m = offs_m[:, None] < M

        valid_x = valid_d & valid_h & valid_w & valid_range_d & valid_range_h & valid_range_w & valid_k & valid_m

        ic_global = group[:, None] * IN_C_PER_GROUP + ic_in_group[None, :]
        # Wait - group is per output (N dim), ic_in_group is per K dim
        # Need to be careful: group is [BLOCK_N], ic_in_group is [BLOCK_K]
        # For X load: we need group from M side? No...
        # Actually for ConvTranspose3d, the input channel determines the group
        # X is indexed by (batch, ic_global, id, ih, iw)
        # ic_global = group_of_output * IN_C_PER_GROUP + ic_in_group
        # But group_of_output comes from N (output channel), not M
        # So X load needs group from N side

        # X ptr: batch[M], ic_global[N,K], id[M,K], ih[M,K], iw[M,K]
        # This is tricky because ic_global depends on both N (group) and K (ic_in_group)
        # We need to handle this differently

        # Let's restructure: for each (m, n, k) triple:
        # m -> (batch, od, oh, ow)
        # n -> (group, oc_in_group)  
        # k -> (ic_in_group, kd, kh, kw)
        # X[batch, group*IN_C_PER_GROUP + ic_in_group, id, ih, iw]
        # W[group*IN_C_PER_GROUP + ic_in_group, oc_in_group, kd, kh, kw]

        # For X: batch from M, ic from K (ic_in_group) + N (group)
        # This means X tile depends on N, so we can't do a simple GEMM
        # We need to loop over groups separately or handle per-group

        # Since groups=4, IN_C_PER_GROUP=8, OUT_C_PER_GROUP=8
        # Let's just compute with the full indexing

        ic_from_n = group[None, :] * IN_C_PER_GROUP  # [1, BLOCK_N]
        ic_from_k = ic_in_group  # [BLOCK_K]

        # X needs: batch[M], (ic_from_n + ic_from_k), id[M,K], ih[M,K], iw[M,K]
        # The problem: ic depends on both M-tile (via batch) and N-tile (via group) and K-tile
        # This breaks the standard GEMM structure

        # For correctness with groups, we need to ensure that when loading X,
        # we use the group from the N dimension
        # One approach: iterate over groups in the outer loop

        # Actually let me reconsider the tiling strategy
        # Since this is getting complex, let me use a simpler approach:
        # tile over (batch*OUT_DHW) for M, and (OUT_C) for N
        # For each (m,n) pair, accumulate over K = IN_C_PER_GROUP * KD * KH * KW
        # where the group is determined by n

        # The issue is that X[batch, ic, ...] where ic = group*IN_C_PER_GROUP + ic_in_group
        # and group is determined by n (output channel)
        # So X_tile[BLOCK_M, BLOCK_K] actually varies with n (the group changes)

        # This means we can't do a simple matrix multiply
        # We need to either:
        # 1. Process one group at a time (loop over groups)
        # 2. Accept that X must be reloaded for each group

        # For simplicity and correctness, let's restructure to loop over groups
        # and within each group do the GEMM

        # I'll restructure the kernel to be group-aware
        pass

    # This approach is getting too complex, let me restart with a cleaner design
    pass
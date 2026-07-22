# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_18` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 45020 ns — **Best latency**: 25730 ns (**1.7497×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`reduce_register_pressure_with_maxnreg_to_improve_warp_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_18`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_18_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7,1,1), Waves/SM: 0.01
- **Registers/thread**: 53
- **Duration**: 45020 ns
- **SM Throughput**: 0.44 % | **Memory**: 4.67 % | **DRAM**: 0.34 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.35 inst/cycle | **No Eligible**: 91.29 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_kernel_with_autotune_and_bank_conflict_fix | 1.744× | 1.774× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 45020 ns | 25380 ns | -19640 ns (-43.6%) |
| SM Throughput (%) | 0.44 % | 0.26 % | -0.18 pp |
| Memory Throughput (%) | 4.67 % | 0.76 % | -3.91 pp |
| DRAM Throughput (%) | 0.34 % | 0.61 % | +0.27 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp |
| Registers/Thread | 53.00 register/thread | 91.00 register/thread | +38 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 91.29 % | 90.73 % | -0.56 pp |
| IPC Active | 0.35 inst/cycle | 0.19 inst/cycle | -0.16 |

---

## Iteration 2

**Best so far**: 1.7738× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_maxnreg_and_block_n_tuning | 1.745× | 1.636× | ✓ |  |
| B | reduce_register_pressure_with_maxnreg_to_improve_warp_occupancy | 1.748× | 1.750× | ✓ | **✓ best** |
| D | increase_block_n_and_autotune_for_latency_hiding | 1.634× | 1.827× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | increase_block_n_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45020 ns | 27520 ns | -17500 ns (-38.9%) | 25730 ns | -19290 ns (-42.8%) | 24640 ns | -20380 ns (-45.3%) |
| SM Throughput (%) | 0.44 % | 0.31 % | -0.13 pp | 0.25 % | -0.19 pp | 0.19 % | -0.25 pp |
| Memory Throughput (%) | 4.67 % | 0.80 % | -3.87 pp | 0.75 % | -3.92 pp | 0.63 % | -4.04 pp |
| DRAM Throughput (%) | 0.34 % | 0.56 % | +0.22 pp | 0.60 % | +0.26 pp | 0.63 % | +0.29 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 53.00 register/thread | 64.00 register/thread | +11 | 91.00 register/thread | +38 | 178.00 register/thread | +125 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 91.29 % | 89.53 % | -1.76 pp | 90.58 % | -0.71 pp | 86.06 % | -5.23 pp |
| IPC Active | 0.35 inst/cycle | 0.21 inst/cycle | -0.14 | 0.19 inst/cycle | -0.16 | 0.28 inst/cycle | -0.07 |

---

## Iteration 3

**Best so far**: 1.7738× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_block_k_and_block_n_autotune | 0.958× | 0.988× | ✓ |  |
| B | increase_block_n_and_num_stages_for_l1tex_latency_hiding | 1.643× | 1.597× | ✓ |  |
| C | replace_broadcast_mac_with_tl_dot_padded_block_m | 1.657× | 0.808× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | increase_block_n_... | Δ | replace_broadcast... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45020 ns | 45570 ns | +550 ns (+1.2%) | 28190 ns | -16830 ns (-37.4%) | 55740 ns | +10720 ns (+23.8%) |
| SM Throughput (%) | 0.44 % | 0.11 % | -0.33 pp | 0.16 % | -0.28 pp | 0.33 % | -0.11 pp |
| Memory Throughput (%) | 4.67 % | 0.36 % | -4.31 pp | 0.55 % | -4.12 pp | 3.81 % | -0.86 pp |
| DRAM Throughput (%) | 0.34 % | 0.34 % | +0.00 pp | 0.55 % | +0.21 pp | 0.28 % | -0.06 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 128.00 register/thread | +75 | 255.00 register/thread | +202 | 62.00 register/thread | +9 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 91.29 % | 84.95 % | -6.34 pp | 77.58 % | -13.71 pp | 93.47 % | +2.18 pp |
| IPC Active | 0.35 inst/cycle | 0.15 inst/cycle | -0.20 | 0.22 inst/cycle | -0.13 | 0.26 inst/cycle | -0.09 |


### Key Learnings

#### ❌ Plan C: `replace_broadcast_mac_with_tl_dot_padded_block_m` (0.808×)

Padding BLOCK_M from 2 to 16 to satisfy the tl.dot minimum tile requirement (16x16x16 for FP32 MMA) introduced severe shared memory bank conflicts (17-way, 82% of wavefronts excessive) that did not exist with the broadcast MAC pattern. The MIO stall fraction rose from ~0% to 62.8% of all warp cycles, and Warp Cycles Per Issued Instruction more than doubled (7.18 → 15.48), causing duration to increase from 24.64µs to 55.74µs. The instruction count also exploded from 38,776 to 144,900 — a 3.7x increase — indicating the padded tl.dot is doing far more work than the broadcast MAC on this tiny (M=2, N=216, K=864) problem. When the actual M dimension is far below the tl.dot minimum (M=2 vs minimum 16), the 87.5% wasted computation on padded rows combined with the shared memory access pattern mismatch makes tl.dot strictly worse than broadcast MAC.

> **Applicable when**: Never apply this specific pattern (BLOCK_M padding to 16 for tl.dot) when actual M << 16 and the kernel is already using broadcast MAC without bank conflicts. The strategy is theoretically sound for large M but empirically harmful for M=2.

> **Avoid when**: Actual M dimension is far below the tl.dot minimum (M < 8); the existing broadcast MAC pattern does not exhibit shared memory bank conflicts; the kernel is already achieving reasonable latency with broadcast MAC; BLOCK_N is large enough to amortize the broadcast MAC overhead. Padding BLOCK_M to 16 when M=2 wastes 87.5% of compute, introduces 17-way shared memory bank conflicts (MIO stall 62.8%), and triples instruction count — all causing severe regression.

---

## Iteration 4

**Best so far**: 1.7738× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_with_block_k_autotune_for_latency_hiding | 1.632× | 1.606× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 45020 ns | 28030 ns | -16990 ns (-37.7%) |
| SM Throughput (%) | 0.44 % | 0.16 % | -0.28 pp |
| Memory Throughput (%) | 4.67 % | 0.65 % | -4.02 pp |
| DRAM Throughput (%) | 0.34 % | 0.65 % | +0.31 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp |
| Registers/Thread | 53.00 register/thread | 255.00 register/thread | +202 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 91.29 % | 77.74 % | -13.55 pp |
| IPC Active | 0.35 inst/cycle | 0.22 inst/cycle | -0.13 |

---

## Cumulative Progress

```
Baseline:  45020 ns  (1.000×)
Iter 1     25380 ns  (1.774×)  — persistent_kernel_with_autotune_and_bank_conflict_fix

Final best: 25730 ns  (1.7497×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,15 +5,42 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({}, num_warps=1, num_stages=2),
+    triton.Config({}, num_warps=2, num_stages=2),
+    triton.Config({}, num_warps=4, num_stages=2),
+    triton.Config({}, num_warps=8, num_stages=2),
+    triton.Config({}, num_warps=1, num_stages=3),
+    triton.Config({}, num_warps=2, num_stages=3),
+    triton.Config({}, num_warps=4, num_stages=3),
+    triton.Config({}, num_warps=8, num_stages=3),
+    triton.Config({}, num_warps=1, num_stages=4),
+    triton.Config({}, num_warps=2, num_stages=4),
+    triton.Config({}, num_warps=4, num_stages=4),
+    triton.Config({}, num_warps=8, num_stages=4),
+    triton.Config({'maxnreg': 64}, num_warps=1, num_stages=2),
+    triton.Config({'maxnreg': 64}, num_warps=2, num_stages=2),
+    triton.Config({'maxnreg': 64}, num_warps=4, num_stages=2),
+    triton.Config({'maxnreg': 64}, num_warps=8, num_stages=2),
+    triton.Config({'maxnreg': 64}, num_warps=1, num_stages=3),
+    triton.Config({'maxnreg': 64}, num_warps=2, num_stages=3),
+    triton.Config({'maxnreg': 64}, num_warps=4, num_stages=3),
+    triton.Config({'maxnreg': 64}, num_warps=8, num_stages=3),
+    triton.Config({'maxnreg': 64}, num_warps=1, num_stages=4),
+    triton.Config({'maxnreg': 64}, num_warps=2, num_stages=4),
+    triton.Config({'maxnreg': 64}, num_warps=4, num_stages=4),
+    triton.Config({'maxnreg': 64}, num_warps=8, num_stages=4),
+], key=[])
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_18(arg_A, arg_B, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
-    BLOCK_N : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 2
+    BLOCK_N : tl.constexpr = 64
     BLOCK_K : tl.constexpr = 32
-    GROUP_M : tl.constexpr = 8
+    GROUP_M : tl.constexpr = 1
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
     A = arg_A
@@ -30,66 +57,59 @@

     stride_bk = 1
     stride_bn = 864
 
-    # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    num_programs = tl.num_programs(0)
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    num_tiles_m = tl.cdiv(M, BLOCK_M)
+    num_tiles_n = tl.cdiv(N, BLOCK_N)
+    total_tiles = num_tiles_m * num_tiles_n
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
+    for tile_id in tl.range(pid, total_tiles, num_programs):
+        pid_m = (tile_id // num_tiles_n).to(INDEX_DTYPE)
+        pid_n = (tile_id % num_tiles_n).to(INDEX_DTYPE)
+
+        tl.assume(pid_m >= 0)
+        tl.assume(pid_n >= 0)
+
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+
         offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        offs_k = tl.arange(0, BLOCK_K)
+        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 864*idx_m
-        a = tl.load(A + (xindex))
+            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
+            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 216*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 864*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+            idx_m = offs_a_m[:, None]
+            idx_n = a_k_idx_vals
+            xindex = idx_n + 864 * idx_m
+            a = tl.load(A + xindex)
 
+            idx_m = b_k_idx_vals
+            idx_n = offs_b_n[None, :]
+            xindex = idx_n + 216 * idx_m
+            b = tl.load(B + (tl.broadcast_to(idx_m + 864 * idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape))
 
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+            # Use broadcast MAC instead of tl.dot to support BLOCK_M < 16
+            acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)
 
+        # rematerialize rm and rn to save registers
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+        idx_m = rm[:, None]
+        idx_n = rn[None, :]
+        mask = (idx_m < M) & (idx_n < N)
 
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
-
-    # inductor generates a suffix
-    xindex = idx_n + 216*idx_m
-    tmp0 = tl.full([1], 0, tl.int32)
-    tmp1 = tl.maximum(tmp0, acc)
-    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
+        # inductor generates a suffix
+        xindex = idx_n + 216 * idx_m
+        tmp0 = tl.full([1], 0, tl.int32)
+        tmp1 = tl.maximum(tmp0, acc)
+        tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

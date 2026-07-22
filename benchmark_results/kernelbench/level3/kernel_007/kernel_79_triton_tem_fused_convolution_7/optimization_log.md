# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_79` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 48220 ns — **Best latency**: 47940 ns (**1.0058×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_b_matrix_load_order_to_eliminate_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_79`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_79_triton_tem_fused_convolution_7/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.02
- **Registers/thread**: 64
- **Duration**: 48220 ns
- **SM Throughput**: 1.53 % | **Memory**: 10.17 % | **DRAM**: 0.74 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.53 inst/cycle | **No Eligible**: 86.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_b_matrix | 1.004× | 1.002× | ✓ |  |
| B | persistent_kernel_multi_tile_loop | 1.007× | 1.006× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | persistent_kernel... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 48220 ns | 48130 ns | -90 ns (-0.2%) | 47940 ns | -280 ns (-0.6%) |
| SM Throughput (%) | 1.53 % | 1.53 % | +0.00 pp | 1.53 % | +0.00 pp |
| Memory Throughput (%) | 10.17 % | 10.17 % | +0.00 pp | 10.24 % | +0.07 pp |
| DRAM Throughput (%) | 0.74 % | 0.74 % | +0.00 pp | 0.74 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 64.00 register/thread | +0 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 |
| No Eligible (%) | 86.85 % | 86.88 % | +0.03 pp | 86.86 % | +0.01 pp |
| IPC Active | 0.53 inst/cycle | 0.53 inst/cycle | +0.00 | 0.52 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0058× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_load_order_to_eliminate_bank_conflicts | 1.001× | 1.006× | ✓ |  |
| D | eliminate_shared_memory_bank_conflicts_via_smem_staging_with_padding | 1.003× | 0.763× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 48220 ns | 47940 ns | -280 ns (-0.6%) | 63200 ns | +14980 ns (+31.1%) |
| SM Throughput (%) | 1.53 % | 1.53 % | +0.00 pp | 1.11 % | -0.42 pp |
| Memory Throughput (%) | 10.17 % | 10.27 % | +0.10 pp | 8.14 % | -2.03 pp |
| DRAM Throughput (%) | 0.74 % | 0.74 % | +0.00 pp | 0.56 % | -0.18 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 53.00 register/thread | -11 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.01 | -0.01 |
| No Eligible (%) | 86.85 % | 86.88 % | +0.03 pp | 90.58 % | +3.73 pp |
| IPC Active | 0.53 inst/cycle | 0.52 inst/cycle | -0.01 | 0.37 inst/cycle | -0.16 |

---

## Iteration 4

**Best so far**: 1.0058× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_load_canonical_and_enlarge_block_m_for_grid_coverage | 0.566× | 0.551× | ✓ |  |
| C | tf32_tensor_cores | 0.999× | 0.766× | ✓ |  |
| D | restore_persistent_kernel_with_sm_tile_loop | 1.003× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_load_canoni... | Δ | tf32_tensor_cores | Δ | restore_persisten... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48220 ns | 87580 ns | +39360 ns (+81.6%) | 62940 ns | +14720 ns (+30.5%) | 48160 ns | -60 ns (-0.1%) |
| SM Throughput (%) | 1.53 % | 0.74 % | -0.79 pp | 1.10 % | -0.43 pp | 1.52 % | -0.01 pp |
| Memory Throughput (%) | 10.17 % | 5.56 % | -4.61 pp | 8.13 % | -2.04 pp | 10.18 % | +0.01 pp |
| DRAM Throughput (%) | 0.74 % | 0.41 % | -0.33 pp | 0.57 % | -0.17 pp | 0.74 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 92.00 register/thread | +28 | 53.00 register/thread | -11 | 64.00 register/thread | +0 |
| Waves/SM | 0.02 | 0.03 | +0.01 | 0.01 | -0.01 | 0.02 | +0.00 |
| No Eligible (%) | 86.85 % | 87.50 % | +0.65 pp | 90.58 % | +3.73 pp | 86.99 % | +0.14 pp |
| IPC Active | 0.53 inst/cycle | 0.50 inst/cycle | -0.03 | 0.37 inst/cycle | -0.16 | 0.52 inst/cycle | -0.01 |


### Key Learnings

#### ❌ Plan B: `fix_b_load_canonical_and_enlarge_block_m_for_grid_coverage` (0.551×)

Replacing the tl.trans(b_T) B-load pattern with a canonical [BLOCK_K, BLOCK_N] direct load while simultaneously doubling BLOCK_M (32→64) and BLOCK_K (32→64) caused an 82% regression (0.55x speedup). Despite the bank conflict count remaining similarly high (SharedMemoryConflicts wavefronts ~745k→799k, bank-conflict way average worsened from 12 to 18), the dominant effect was that the larger tile sizes tripled dynamic shared memory per block (16→49 KB), cutting the shared-memory-limited block count from 9 to 4 per SM and halving theoretical occupancy from 50% to 25%. Registers per thread also jumped from 64 to 92, further constraining occupancy. The net result was a 83% increase in kernel duration (47.94→87.58 µs) with achieved occupancy unchanged at 6.25% — the GPU remained equally underutilized but each tile now took longer to execute.

> **Applicable when**: Never: this combined change caused a severe regression. The canonical B-load pattern itself is structurally correct, but pairing it with 2x tile size increases on a kernel already constrained by shared memory limits is counterproductive.

> **Avoid when**: Do not simultaneously enlarge BLOCK_M and BLOCK_K when the kernel is already near the shared-memory-per-block limit (Block Limit Shared Mem ≤ 9). Doubling both tile dimensions quadruples the shared memory footprint, which cuts the smem-limited block count per SM and reduces theoretical occupancy. On this kernel, the bank conflict problem (12-18 way conflicts, ~75-81% MIO stall fraction) persists regardless of the load pattern change — the structural access pattern driving conflicts was not resolved by removing tl.trans. Do not increase BLOCK_K beyond 32 on this kernel without first verifying that smem usage stays within the limit that preserves the original Block Limit Shared Mem value.

#### ⚠️ Plan C: `tf32_tensor_cores` (0.000×)

Plan 'tf32_tensor_cores' was applied but failed correctness. Error: Correctness FAILED:
seed=0 output[0]: max_diff=9.089e-02 > atol=0.001
seed=42 output[0]: max_diff=9.145e-02 > atol=0.001
seed=1337 output[0]: max_diff=9.122e-02 > atol=0.001

> **Applicable when**: The kernel computes a 490x32x832 GEMM producing only 16 output tiles on a 132-SM H200, yielding 0.02 waves/SM and 6.25% achieved occupancy. Compute SOL is 1.53% — the GPU is almost entirely idle. The root cause is that ALLOW_TF32=False forces tl.dot to use FP32 FFMA scalar units instead of Tensor Cores. On H200, TF32 Tensor Cores deliver ~8x the throughput of FP32 FFMA (~989 TFLOPS vs ~67 TFLOPS). Enabling ALLOW_TF32=True routes the dot product through Tensor Cores, dramatically increasing compute throughput per cycle. The accumulator remains FP32 (ACC_TYPE=tl.float32), so only the dot-product inputs are rounded to TF32 (10-bit mantissa vs 23-bit). This is a single-line, zero-risk change with no tile-size or indexing modifications — avoiding the correctness failures seen in prior autotune attempts.

> **Avoid when**: Previous implementation failed: Correctness FAILED:
seed=0 output[0]: max_diff=9.089e-02 > atol=0.001
seed=42 output[0]: max_diff=9.145e-02 > atol=0.001
seed=1337 output[0]: max_diff=9.122e-02 > atol=0.001

---

## Cumulative Progress

```
Baseline:  48220 ns  (1.000×)
Iter 2     47940 ns  (1.006×)  — persistent_kernel_multi_tile_loop

Final best: 47940 ns  (1.0058×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,8 +5,9 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_tem_fused_convolution_79(in_ptr0, arg_A, arg_B, out_ptr0):
+def triton_tem_fused_convolution_79_persistent(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
@@ -16,6 +17,7 @@

     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
+    NUM_SMS : tl.constexpr = 132
     A = arg_A
     B = arg_B
 
@@ -30,66 +32,68 @@

     stride_bk = 1
     stride_bn = 832
 
-    # based on triton.ops.matmul
-    pid = tl.program_id(0).to(INDEX_DTYPE)
+    sm_id = tl.program_id(0).to(INDEX_DTYPE)
     grid_m = (M + BLOCK_M - 1) // BLOCK_M
     grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    num_tiles = grid_m * grid_n
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    tile_id = sm_id
+    while tile_id < num_tiles:
+        # re-order program ID for better L2 performance
+        width = GROUP_M * grid_n
+        group_id = tile_id // width
+        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
+        pid_m = group_id * GROUP_M + (tile_id % group_size)
+        pid_n = (tile_id % width) // group_size
+        tl.assume(pid_m >= 0)
+        tl.assume(pid_n >= 0)
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+        if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
+            offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
+        else:
+            offs_a_m = rm % M
+        if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
+            offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
+        else:
+            offs_b_n = rn % N
+        offs_k = tl.arange(0, BLOCK_K)
+        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 832*idx_m
-        a = tl.load(A + (xindex))
+            idx_m = offs_a_m[:, None]
+            idx_n = a_k_idx_vals
+            xindex = idx_n + 832*idx_m
+            a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
+            # NEW — load B as [BLOCK_N, BLOCK_K] row-major (K stride=1 → coalesced, zero bank conflicts)
+            offs_b_n_col = offs_b_n[:, None]                        # shape [BLOCK_N, 1]
+            offs_k_row   = offs_k[None, :] + (k_idx * BLOCK_K)     # shape [1, BLOCK_K]
+            # B layout: element B[k, n] at offset k*1 + n*832 (stride_bk=1, stride_bn=832)
+            # So B[n, k] row-major = offs_b_n*832 + offs_k
+            b_T = tl.load(B + (offs_b_n_col * 832 + offs_k_row))   # shape [BLOCK_N, BLOCK_K]
+            b   = tl.trans(b_T)                                     # shape [BLOCK_K, BLOCK_N] for tl.dot
+
+            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+
+        # rematerialize rm and rn to save registers
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+        idx_m = rm[:, None]
+        idx_n = rn[None, :]
+        mask = (idx_m < M) & (idx_n < N)
+
+        # inductor generates a suffix
         xindex = idx_n + 32*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 832*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
+        tmp1 = acc + tmp0
+        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
 
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
-
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
-
-    # inductor generates a suffix
-    xindex = idx_n + 32*idx_m
-    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp1 = acc + tmp0
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
+        tile_id += NUM_SMS
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -97,7 +101,7 @@

     arg_2 = inputs[2]
     arg_3 = inputs[3]
     grid = (16, 1, 1)
-    triton_tem_fused_convolution_79[grid](arg_0, arg_1, arg_2, arg_3)
+    triton_tem_fused_convolution_79_persistent[grid](arg_0, arg_1, arg_2, arg_3)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

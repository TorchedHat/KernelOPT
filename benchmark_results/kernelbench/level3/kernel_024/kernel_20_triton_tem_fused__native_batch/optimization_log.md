# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_20` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 45250 ns — **Best latency**: 23780 ns (**1.9029×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_shared_memory_bank_conflicts_direct_strided_b_load_block_m2`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_20`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_20_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6,1,1), Waves/SM: 0.01
- **Registers/thread**: 53
- **Duration**: 45250 ns
- **SM Throughput**: 0.38 % | **Memory**: 3.99 % | **DRAM**: 0.31 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 91.36 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_tiny_gemm | 1.394× | 0.851× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 45250 ns | 53150 ns | +7900 ns (+17.5%) |
| SM Throughput (%) | 0.38 % | 0.26 % | -0.12 pp |
| Memory Throughput (%) | 3.99 % | 1.94 % | -2.05 pp |
| DRAM Throughput (%) | 0.31 % | 0.26 % | -0.05 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp |
| Registers/Thread | 53.00 register/thread | 64.00 register/thread | +11 |
| Waves/SM | 0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.36 % | 88.14 % | -3.22 pp |
| IPC Active | 0.34 inst/cycle | 0.24 inst/cycle | -0.10 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_kernel_one_block_per_sm | 1.025× | 1.025× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 45250 ns | 44130 ns | -1120 ns (-2.5%) |
| SM Throughput (%) | 0.38 % | 0.38 % | +0.00 pp |
| Memory Throughput (%) | 3.99 % | 4.08 % | +0.09 pp |
| DRAM Throughput (%) | 0.31 % | 0.31 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 48.00 register/thread | -5 |
| Waves/SM | 0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.36 % | 91.19 % | -0.17 pp |
| IPC Active | 0.34 inst/cycle | 0.35 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_transposed_load_bank_conflicts | 1.024× | 1.027× | ✓ |  |
| B | fix_block_m_waste_and_shared_memory_bank_conflicts | 1.814× | 1.898× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ | fix_block_m_waste... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 45250 ns | 44060 ns | -1190 ns (-2.6%) | 23840 ns | -21410 ns (-47.3%) |
| SM Throughput (%) | 0.38 % | 0.38 % | +0.00 pp | 0.34 % | -0.04 pp |
| Memory Throughput (%) | 3.99 % | 4.08 % | +0.09 pp | 3.82 % | -0.17 pp |
| DRAM Throughput (%) | 0.31 % | 0.31 % | +0.00 pp | 0.60 % | +0.29 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 48.00 register/thread | -5 | 40.00 register/thread | -13 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.36 % | 91.20 % | -0.16 pp | 94.85 % | +3.49 pp |
| IPC Active | 0.34 inst/cycle | 0.35 inst/cycle | +0.01 | 0.20 inst/cycle | -0.14 |

---

## Iteration 4

**Best so far**: 1.8981× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | full_problem_single_tile_eliminate_k_loop_and_shared_mem_staging | 0.940× | 0.935× | ✓ |  |
| C | eliminate_shared_memory_bank_conflicts_direct_strided_b_load_block_m2 | 1.811× | 1.903× | ✓ |  |
| D | eliminate_shared_memory_bank_conflicts_via_direct_strided_b_load_and_correct_block_m | 1.815× | 1.903× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | full_problem_sing... | Δ | eliminate_shared_... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45250 ns | 48380 ns | +3130 ns (+6.9%) | 23780 ns | -21470 ns (-47.4%) | 23780 ns | -21470 ns (-47.4%) |
| SM Throughput (%) | 0.38 % | 0.07 % | -0.31 pp | 0.34 % | -0.04 pp | 0.34 % | -0.04 pp |
| Memory Throughput (%) | 3.99 % | 0.64 % | -3.35 pp | 3.80 % | -0.19 pp | 3.83 % | -0.16 pp |
| DRAM Throughput (%) | 0.31 % | 0.29 % | -0.02 pp | 0.60 % | +0.29 pp | 0.60 % | +0.29 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 43.00 register/thread | -10 | 40.00 register/thread | -13 | 40.00 register/thread | -13 |
| Waves/SM | 0.01 | 0.02 | +0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.36 % | 90.93 % | -0.43 pp | 94.84 % | +3.48 pp | 94.86 % | +3.50 pp |
| IPC Active | 0.34 inst/cycle | 0.36 inst/cycle | +0.02 | 0.21 inst/cycle | -0.13 | 0.21 inst/cycle | -0.13 |

---

## Iteration 5

**Best so far**: 1.9029× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  45250 ns  (1.000×)
Iter 3     23840 ns  (1.898×)  — fix_block_m_waste_and_shared_memory_bank_conflicts
Iter 4     23780 ns  (1.903×)  — eliminate_shared_memory_bank_conflicts_via_direct_strided_b_load_and_correct_block_m

Final best: 23780 ns  (1.9029×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,17 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_20(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
+    NUM_SMS : tl.constexpr = 132
     A = arg_A
     B = arg_B
 
@@ -30,64 +32,67 @@

     stride_bk = 1
     stride_bn = 864
 
-    # based on triton.ops.matmul
-    pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    sm_id = tl.program_id(0).to(INDEX_DTYPE)
+    grid_m = tl.cdiv(M, BLOCK_M)
+    grid_n = tl.cdiv(N, BLOCK_N)
+    num_tiles = grid_m * grid_n
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    for tile_id in range(sm_id, num_tiles, NUM_SMS):
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
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        # Reset accumulator for each tile
+        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 864*idx_m
-        a = tl.load(A + (xindex))
+            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
+            idx_m = offs_a_m[:, None]
+            idx_n = a_k_idx_vals
+            xindex = idx_n + 864*idx_m
+            a = tl.load(A + (xindex))
+
+            # Direct [BLOCK_K, BLOCK_N] load — no shared memory staging, no bank conflicts
+            k_base = k_idx * BLOCK_K
+            offs_k_col = tl.arange(0, BLOCK_K)[:, None]  # [BLOCK_K, 1]
+            offs_n_row = offs_b_n[None, :]                # [1, BLOCK_N]
+            b = tl.load(
+                B + (k_base + offs_k_col) * stride_bk + offs_n_row * stride_bn,
+                mask=((k_base + offs_k_col) < K) & (offs_n_row < N),
+                other=0.0
+            )  # shape [BLOCK_K, BLOCK_N], loaded directly into registers
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
         xindex = idx_n + 192*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 864*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
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
-    xindex = idx_n + 192*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

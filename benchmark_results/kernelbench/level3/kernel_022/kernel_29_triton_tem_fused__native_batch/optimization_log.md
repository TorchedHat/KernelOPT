# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 48640 ns — **Best latency**: 48540 ns (**1.0021×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`pipeline_stages_to_hide_shared_memory_bank_conflict_stalls`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_29_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.19
- **Registers/thread**: 94
- **Duration**: 48640 ns
- **SM Throughput**: 12.47 % | **Memory**: 84.93 % | **DRAM**: 1.67 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.56 inst/cycle | **No Eligible**: 86.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_fix_bank_conflicts | 1.415× | 0.996× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_gemm_occupancy_and_bank_conflict_reduction | 0.893× | 0.887× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 48640 ns | 48830 ns | +190 ns (+0.4%) | 54850 ns | +6210 ns (+12.8%) |
| SM Throughput (%) | 12.47 % | 12.46 % | -0.01 pp | 20.59 % | +8.12 pp |
| Memory Throughput (%) | 84.93 % | 84.86 % | -0.07 pp | 79.94 % | -4.99 pp |
| DRAM Throughput (%) | 1.67 % | 1.67 % | +0.00 pp | 1.49 % | -0.18 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 94.00 register/thread | +0 | 128.00 register/thread | +34 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.23 | +0.04 |
| No Eligible (%) | 86.12 % | 86.13 % | +0.01 pp | 76.94 % | -9.18 pp |
| IPC Active | 0.56 inst/cycle | 0.56 inst/cycle | +0.00 | 0.92 inst/cycle | +0.36 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | pipeline_stages_to_hide_shared_memory_bank_conflict_stalls | 1.425× | 1.002× | ✓ | **✓ best** |
| B | tma_loads_via_block_ptr | 1.365× | 0.986× | ✓ |  |
| C | fix_b_matrix_transposed_load_to_eliminate_bank_conflicts | 0.894× | 0.887× | ✓ |  |
| D | tma_loads | 0.896× | 0.892× | ✓ |  |
| E | add_triton_autotune | 1.407× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | pipeline_stages_t... | Δ | tma_loads_via_blo... | Δ | fix_b_matrix_tran... | Δ | tma_loads | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48640 ns | 48540 ns | -100 ns (-0.2%) | 49340 ns | +700 ns (+1.4%) | 54850 ns | +6210 ns (+12.8%) | 54500 ns | +5860 ns (+12.0%) | 48700 ns | +60 ns (+0.1%) |
| SM Throughput (%) | 12.47 % | 12.44 % | -0.03 pp | 12.56 % | +0.09 pp | 20.69 % | +8.22 pp | 21.17 % | +8.70 pp | 12.49 % | +0.02 pp |
| Memory Throughput (%) | 84.93 % | 84.71 % | -0.22 pp | 83.91 % | -1.02 pp | 80.34 % | -4.59 pp | 81.22 % | -3.71 pp | 85.06 % | +0.13 pp |
| DRAM Throughput (%) | 1.67 % | 1.68 % | +0.01 pp | 1.65 % | -0.02 pp | 1.49 % | -0.18 pp | 1.50 % | -0.17 pp | 1.67 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 94.00 register/thread | +0 | 92.00 register/thread | -2 | 128.00 register/thread | +34 | 128.00 register/thread | +34 | 94.00 register/thread | +0 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.19 | +0.00 | 0.23 | +0.04 | 0.23 | +0.04 | 0.19 | +0.00 |
| No Eligible (%) | 86.12 % | 86.12 % | +0.00 pp | 85.99 % | -0.13 pp | 76.98 % | -9.14 pp | 76.31 % | -9.81 pp | 86.09 % | -0.03 pp |
| IPC Active | 0.56 inst/cycle | 0.56 inst/cycle | +0.00 | 0.56 inst/cycle | +0.00 | 0.92 inst/cycle | +0.36 | 0.95 inst/cycle | +0.39 | 0.55 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0021× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_block_ptr_order_to_eliminate_bank_conflicts | 0.892× | 0.895× | ✓ |  |
| C | fix_b_matrix_load_coalescing_to_eliminate_bank_conflicts | 1.325× | 0.976× | ✓ |  |
| D | fix_b_matrix_shared_memory_layout_to_eliminate_bank_conflicts | 1.356× | 0.987× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | fix_b_matrix_load... | Δ | fix_b_matrix_shar... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48640 ns | 54370 ns | +5730 ns (+11.8%) | 49820 ns | +1180 ns (+2.4%) | 49280 ns | +640 ns (+1.3%) |
| SM Throughput (%) | 12.47 % | 21.14 % | +8.67 pp | 12.43 % | -0.04 pp | 12.57 % | +0.10 pp |
| Memory Throughput (%) | 84.93 % | 81.10 % | -3.83 pp | 83.20 % | -1.73 pp | 83.92 % | -1.01 pp |
| DRAM Throughput (%) | 1.67 % | 1.50 % | -0.17 pp | 1.63 % | -0.04 pp | 1.65 % | -0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 | 96.00 register/thread | +2 | 92.00 register/thread | -2 |
| Waves/SM | 0.19 | 0.23 | +0.04 | 0.19 | +0.00 | 0.19 | +0.00 |
| No Eligible (%) | 86.12 % | 76.32 % | -9.80 pp | 86.14 % | +0.02 pp | 85.98 % | -0.14 pp |
| IPC Active | 0.56 inst/cycle | 0.95 inst/cycle | +0.39 | 0.55 inst/cycle | -0.01 | 0.56 inst/cycle | +0.00 |


### Key Learnings

#### ⚠️ Plan C: `fix_b_matrix_load_coalescing_to_eliminate_bank_conflicts` (0.000×)

Plan 'fix_b_matrix_load_coalescing_to_eliminate_bank_conflicts' was applied but failed correctness. Error: Correctness FAILED:
seed=0 output[0]: max_diff=1.329e+02 > atol=0.001
seed=42 output[0]: max_diff=1.375e+02 > atol=0.001
seed=1337 output[0]: max_diff=1.502e+02 > atol=0.001

> **Applicable when**: The B matrix (arg_B) has shape (112, 480, 1, 1) stored with stride (480, 1, 1, 1), meaning it is laid out as (N=112, K=480) in row-major — i.e., N is the outer dimension and K is contiguous. The current kernel hardcodes stride_bk=1 and stride_bn=480, treating B as if it were (K, N) layout, but then computes xindex = idx_n + 112*idx_m where idx_m is the K-tile index and idx_n is the N-tile index. This produces a gather with stride 480 in K, causing 16.9-way bank conflicts when the (BLOCK_K=32, BLOCK_N=64) tile is staged through shared memory. The fix is to correct the B load to use the actual (N, K) layout: stride_bk=112 (K is inner, stride=N=112 between K rows) and stride_bn=1 (N is contiguous). With this correction, the N-dimension load becomes contiguous (stride 1), enabling coalesced global loads and conflict-free shared memory access. Additionally, applying tl.max_contiguous and tl.multiple_of hints on offs_b_n will allow the compiler to emit vectorized 128-bit loads. This directly attacks the #1 bottleneck (77.97% estimated speedup) and the CPIStall (76.5% of stall cycles).

> **Avoid when**: Previous implementation failed: Correctness FAILED:
seed=0 output[0]: max_diff=1.329e+02 > atol=0.001
seed=42 output[0]: max_diff=1.375e+02 > atol=0.001
seed=1337 output[0]: max_diff=1.502e+02 > atol=0.001

---

## Iteration 4

**Best so far**: 1.0021× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_smem_order_column_major_to_eliminate_bank_conflicts | 1.359× | 0.987× | ✓ |  |
| B | fix_b_block_ptr_order_to_eliminate_shared_memory_bank_conflicts | 0.897× | 0.891× | ✓ |  |
| C | fix_b_matrix_load_via_transposed_coalesced_tile | 1.324× | 0.976× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_s... | Δ | fix_b_block_ptr_o... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48640 ns | 49280 ns | +640 ns (+1.3%) | 54590 ns | +5950 ns (+12.2%) | 49820 ns | +1180 ns (+2.4%) |
| SM Throughput (%) | 12.47 % | 12.58 % | +0.11 pp | 21.20 % | +8.73 pp | 12.43 % | -0.04 pp |
| Memory Throughput (%) | 84.93 % | 83.98 % | -0.95 pp | 81.36 % | -3.57 pp | 83.12 % | -1.81 pp |
| DRAM Throughput (%) | 1.67 % | 1.65 % | -0.02 pp | 1.49 % | -0.18 pp | 1.63 % | -0.04 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 92.00 register/thread | -2 | 128.00 register/thread | +34 | 96.00 register/thread | +2 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.23 | +0.04 | 0.19 | +0.00 |
| No Eligible (%) | 86.12 % | 86.00 % | -0.12 pp | 76.37 % | -9.75 pp | 86.16 % | +0.04 pp |
| IPC Active | 0.56 inst/cycle | 0.56 inst/cycle | +0.00 | 0.95 inst/cycle | +0.39 | 0.55 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  48640 ns  (1.000×)
Iter 2     48540 ns  (1.002×)  — pipeline_stages_to_hide_shared_memory_bank_conflict_stalls

Final best: 48540 ns  (1.0021×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=5),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -46,35 +62,37 @@

 
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Construct TMA block pointers OUTSIDE the k-loop
+    # A: shape (M=1960, K=480), strides=(480, 1), row-major
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B stored as (N=112, K=480): B[n,k] at n*480+k.
+    # As a (K, N) tile: strides=(stride_bk=1, stride_bn=480)
+    # order=(1, 0) = row-major layout in shared memory, eliminates bank conflicts
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 480*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 112*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +104,6 @@

     # inductor generates a suffix
     xindex = idx_n + 112*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

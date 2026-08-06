# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_27` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 18300 ns — **Best latency**: 10720 ns (**1.7071×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_block_m_to_match_m_dimension_eliminate_bank_conflicts_via_block_n_padding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_27`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_27_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (54,1,1), Waves/SM: 0.05
- **Registers/thread**: 56
- **Duration**: 18300 ns
- **SM Throughput**: 3.12 % | **Memory**: 29.60 % | **DRAM**: 2.27 %
- **Achieved Occupancy**: 5.94 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 90.92 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_to_bypass_shared_memory_bank_conflicts | 1.003× | 1.007× | ✓ |  |
| C | fix_b_matrix_load_index_for_contiguous_memory_access | 1.004× | 1.003× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_to_bypa... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 18300 ns | 18180 ns | -120 ns (-0.7%) | 18240 ns | -60 ns (-0.3%) |
| SM Throughput (%) | 3.12 % | 3.31 % | +0.19 pp | 3.16 % | +0.04 pp |
| Memory Throughput (%) | 29.60 % | 30.32 % | +0.72 pp | 29.96 % | +0.36 pp |
| DRAM Throughput (%) | 2.27 % | 2.28 % | +0.01 pp | 2.27 % | +0.00 pp |
| Achieved Occupancy (%) | 5.94 % | 6.01 % | +0.07 pp | 5.94 % | +0.00 pp |
| Registers/Thread | 56.00 register/thread | 54.00 register/thread | -2 | 56.00 register/thread | +0 |
| Waves/SM | 0.05 | 0.05 | +0.00 | 0.05 | +0.00 |
| No Eligible (%) | 90.92 % | 90.60 % | -0.32 pp | 90.94 % | +0.02 pp |
| IPC Active | 0.34 inst/cycle | 0.36 inst/cycle | +0.02 | 0.34 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0033× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_for_column_major_layout | 1.007× | 1.010× | ✓ |  |
| B | fix_b_block_ptr_order_to_eliminate_shared_memory_bank_conflicts | 1.006× | 1.012× | ✓ |  |
| C | eliminate_tl_trans_to_fix_shared_memory_bank_conflicts | 1.007× | 0.998× | ✓ |  |
| D | tma_loads_via_block_ptr_to_eliminate_bank_conflicts_and_address_overhead | 1.008× | 1.018× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | fix_b_block_ptr_o... | Δ | eliminate_tl_tran... | Δ | tma_loads_via_blo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18300 ns | 18110 ns | -190 ns (-1.0%) | 18080 ns | -220 ns (-1.2%) | 18340 ns | +40 ns (+0.2%) | 17980 ns | -320 ns (-1.7%) |
| SM Throughput (%) | 3.12 % | 3.31 % | +0.19 pp | 3.31 % | +0.19 pp | 3.13 % | +0.01 pp | 3.29 % | +0.17 pp |
| Memory Throughput (%) | 29.60 % | 30.33 % | +0.73 pp | 30.34 % | +0.74 pp | 29.72 % | +0.12 pp | 30.12 % | +0.52 pp |
| DRAM Throughput (%) | 2.27 % | 2.29 % | +0.02 pp | 2.29 % | +0.02 pp | 2.26 % | -0.01 pp | 2.30 % | +0.03 pp |
| Achieved Occupancy (%) | 5.94 % | 6.04 % | +0.10 pp | 6.04 % | +0.10 pp | 5.92 % | -0.02 pp | 6.04 % | +0.10 pp |
| Registers/Thread | 56.00 register/thread | 54.00 register/thread | -2 | 54.00 register/thread | -2 | 56.00 register/thread | +0 | 54.00 register/thread | -2 |
| Waves/SM | 0.05 | 0.05 | +0.00 | 0.05 | +0.00 | 0.05 | +0.00 | 0.05 | +0.00 |
| No Eligible (%) | 90.92 % | 90.59 % | -0.33 pp | 90.60 % | -0.32 pp | 90.91 % | -0.01 pp | 90.58 % | -0.34 pp |
| IPC Active | 0.34 inst/cycle | 0.36 inst/cycle | +0.02 | 0.36 inst/cycle | +0.02 | 0.34 inst/cycle | +0.00 | 0.36 inst/cycle | +0.02 |

---

## Iteration 4

**Best so far**: 1.0178× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_with_correct_b_shared_memory_order | 1.003× | 1.009× | ✓ |  |
| B | tf32_tensor_cores_surgical_constexpr_only | 0.912× | 0.909× | ✓ |  |
| D | reduce_block_m_to_match_m_dimension_and_increase_block_n | 1.538× | 1.707× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ | tf32_tensor_cores... | Δ | reduce_block_m_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18300 ns | 18140 ns | -160 ns (-0.9%) | 20130 ns | +1830 ns (+10.0%) | 10720 ns | -7580 ns (-41.4%) |
| SM Throughput (%) | 3.12 % | 3.09 % | -0.03 pp | 4.91 % | +1.79 pp | 2.61 % | -0.51 pp |
| Memory Throughput (%) | 29.60 % | 29.30 % | -0.30 pp | 28.99 % | -0.61 pp | 25.83 % | -3.77 pp |
| DRAM Throughput (%) | 2.27 % | 2.28 % | +0.01 pp | 2.06 % | -0.21 pp | 3.88 % | +1.61 pp |
| Achieved Occupancy (%) | 5.94 % | 5.92 % | -0.02 pp | 6.15 % | +0.21 pp | 6.25 % | +0.31 pp |
| Registers/Thread | 56.00 register/thread | 56.00 register/thread | +0 | 72.00 register/thread | +16 | 48.00 register/thread | -8 |
| Waves/SM | 0.05 | 0.05 | +0.00 | 0.06 | +0.01 | 0.04 | -0.01 |
| No Eligible (%) | 90.92 % | 90.91 % | -0.01 pp | 85.95 % | -4.97 pp | 93.85 % | +2.93 pp |
| IPC Active | 0.34 inst/cycle | 0.34 inst/cycle | +0.00 | 0.55 inst/cycle | +0.21 | 0.25 inst/cycle | -0.09 |


### Key Learnings

#### ✅ Plan D: `reduce_block_m_to_match_m_dimension_and_increase_block_n` (1.707×)

When a GEMM has a very small M dimension (e.g., M=2) but BLOCK_M is set much larger (e.g., 16), the vast majority of M-threads are masked/wasted, inflating register usage and instruction count without doing useful work. Switching from tl.make_block_ptr with tl.advance to explicit index arithmetic (offs_k, offs_a_m, offs_b_n) while setting BLOCK_M=2 and BLOCK_N=64 reduced executed instructions from 472K to 171K (63.8% reduction) and cut elapsed cycles from 26937 to 16021 (40.5% reduction), yielding a 1.71x speedup. The L2 hit rate improved dramatically from 12.1% to 41.9%, indicating better cache reuse with the restructured access pattern.

> **Applicable when**: GEMM where M is very small (M <= 4) but BLOCK_M is set to a much larger value (e.g., 16), causing >75% of M-threads to be masked. SharedMemoryConflicts rule fires with 17-way bank conflicts (>80% excess wavefronts). LaunchConfiguration rule fires because grid_size < sm_count. Executed instructions are dominated by masked/predicated-off work. Registers per thread are high (>50) due to large tile accumulator.

> **Avoid when**: M is large enough that BLOCK_M is not wasteful (M >= BLOCK_M/2). When tl.dot requires BLOCK_M >= 16 in the target Triton version (use fallback BLOCK_M=16, BLOCK_N=128 in that case). Do not enable ALLOW_TF32 for this kernel — it causes correctness failures (max_diff up to 370) due to K=288 accumulation error exceeding atol=0.001.

---

## Iteration 5

**Best so far**: 1.7071× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_k_and_block_n_to_reduce_smem_bank_conflicts | 1.277× | 1.282× | ✓ |  |
| B | increase_block_n_with_block_m2_explicit_index_arithmetic | 1.429× | 1.575× | ✓ |  |
| C | fix_block_m_to_match_m_dimension_eliminate_bank_conflicts_via_block_n_padding | 1.548× | 1.707× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_block_k_... | Δ | increase_block_n_... | Δ | fix_block_m_to_ma... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18300 ns | 14270 ns | -4030 ns (-22.0%) | 11620 ns | -6680 ns (-36.5%) | 10720 ns | -7580 ns (-41.4%) |
| SM Throughput (%) | 3.12 % | 3.97 % | +0.85 pp | 3.51 % | +0.39 pp | 2.62 % | -0.50 pp |
| Memory Throughput (%) | 29.60 % | 23.33 % | -6.27 pp | 25.46 % | -4.14 pp | 25.88 % | -3.72 pp |
| DRAM Throughput (%) | 2.27 % | 2.91 % | +0.64 pp | 3.60 % | +1.33 pp | 3.89 % | +1.62 pp |
| Achieved Occupancy (%) | 5.94 % | 6.24 % | +0.30 pp | 6.24 % | +0.30 pp | 6.25 % | +0.31 pp |
| Registers/Thread | 56.00 register/thread | 65.00 register/thread | +9 | 54.00 register/thread | -2 | 48.00 register/thread | -8 |
| Waves/SM | 0.05 | 0.14 | +0.09 | 0.07 | +0.02 | 0.04 | -0.01 |
| No Eligible (%) | 90.92 % | 87.66 % | -3.26 pp | 90.43 % | -0.49 pp | 93.84 % | +2.92 pp |
| IPC Active | 0.34 inst/cycle | 0.49 inst/cycle | +0.15 | 0.38 inst/cycle | +0.04 | 0.25 inst/cycle | -0.09 |

---

## Cumulative Progress

```
Baseline:  18300 ns  (1.000×)
Iter 2     18240 ns  (1.003×)  — fix_b_matrix_load_index_for_contiguous_memory_access
Iter 3     17980 ns  (1.018×)  — tma_loads_via_block_ptr_to_eliminate_bank_conflicts_and_address_overhead
Iter 4     10720 ns  (1.707×)  — reduce_block_m_to_match_m_dimension_and_increase_block_n

Final best: 10720 ns  (1.7071×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,14 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_27(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
-    BLOCK_N : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 2
+    BLOCK_N : tl.constexpr = 64
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
@@ -60,18 +61,16 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 288*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1728*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 288*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # Load B directly as [BLOCK_K, BLOCK_N] - no transpose needed, no bank conflicts
+        k_offset = offs_k[:, None] + (k_idx * BLOCK_K)   # shape [BLOCK_K, 1]
+        n_offset = offs_b_n[None, :]                       # shape [1, BLOCK_N]
+        b = tl.load(B + (k_offset + 288 * n_offset))      # [BLOCK_K, BLOCK_N] directly
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -103,8 +102,6 @@

     tmp15 = tl.full([1], 0, tl.int32)
     tmp16 = tl.maximum(tmp15, tmp14)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1728*idx_m, [BLOCK_M, BLOCK_N])), tmp16, mask)
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

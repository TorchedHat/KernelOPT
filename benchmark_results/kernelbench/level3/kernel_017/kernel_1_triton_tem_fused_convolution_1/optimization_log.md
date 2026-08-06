# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 825250 ns — **Best latency**: 82370 ns (**10.0188×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`hoist_bias_load_to_1d_coalesced_load_with_register_broadcast`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_017/kernel_1_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 165.49
- **Registers/thread**: 160
- **Duration**: 825250 ns
- **SM Throughput**: 25.61 % | **Memory**: 83.25 % | **DRAM**: 6.87 %
- **Achieved Occupancy**: 18.03 %
- **IPC Active**: 1.03 inst/cycle | **No Eligible**: 74.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.999× | ✓ |  |
| D | autotune_tile_sizes_to_fix_overprovisioned_block_nk_and_register_pressure | 1.816× | 1.813× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 825250 ns | 825760 ns | +510 ns (+0.1%) | 455070 ns | -370180 ns (-44.9%) |
| SM Throughput (%) | 25.61 % | 25.62 % | +0.01 pp | 36.73 % | +11.12 pp |
| Memory Throughput (%) | 83.25 % | 83.29 % | +0.04 pp | 97.59 % | +14.34 pp |
| DRAM Throughput (%) | 6.87 % | 6.86 % | -0.01 pp | 12.42 % | +5.55 pp |
| Achieved Occupancy (%) | 18.03 % | 18.04 % | +0.01 pp | 54.69 % | +36.66 pp |
| Registers/Thread | 160.00 register/thread | 160.00 register/thread | +0 | 54.00 register/thread | -106 |
| Waves/SM | 165.49 | 165.49 | +0.00 | 55.16 | -110.33 |
| No Eligible (%) | 74.27 % | 74.27 % | +0.00 pp | 63.07 % | -11.20 pp |
| IPC Active | 1.03 inst/cycle | 1.03 inst/cycle | +0.00 | 1.48 inst/cycle | +0.45 |

---

## Iteration 2

**Best so far**: 1.8135× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_tl_dot_with_scalar_fma_to_eliminate_shared_memory_bank_conflicts | 7.383× | 7.293× | ✓ | **✓ best** |
| B | tma_loads_to_bypass_shared_memory_bank_conflicts | 1.815× | 1.816× | ✓ |  |
| D | tma_loads | 1.815× | 1.817× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_tl_dot_wi... | Δ | tma_loads_to_bypa... | Δ | tma_loads | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 825250 ns | 113150 ns | -712100 ns (-86.3%) | 454370 ns | -370880 ns (-44.9%) | 454300 ns | -370950 ns (-45.0%) |
| SM Throughput (%) | 25.61 % | 84.75 % | +59.14 pp | 36.74 % | +11.13 pp | 32.42 % | +6.81 pp |
| Memory Throughput (%) | 83.25 % | 82.76 % | -0.49 pp | 97.60 % | +14.35 pp | 97.54 % | +14.29 pp |
| DRAM Throughput (%) | 6.87 % | 49.79 % | +42.92 pp | 12.45 % | +5.58 pp | 12.44 % | +5.57 pp |
| Achieved Occupancy (%) | 18.03 % | 94.33 % | +76.30 pp | 54.69 % | +36.66 pp | 54.69 % | +36.66 pp |
| Registers/Thread | 160.00 register/thread | 32.00 register/thread | -128 | 54.00 register/thread | -106 | 54.00 register/thread | -106 |
| Waves/SM | 165.49 | 31.03 | -134.46 | 55.16 | -110.33 | 55.16 | -110.33 |
| No Eligible (%) | 74.27 % | 13.93 % | -60.34 pp | 63.07 % | -11.20 pp | 67.40 % | -6.87 pp |
| IPC Active | 1.03 inst/cycle | 3.44 inst/cycle | +2.41 | 1.48 inst/cycle | +0.45 | 1.30 inst/cycle | +0.27 |


### Key Learnings

#### ✅ Plan A: `replace_tl_dot_with_scalar_fma_to_eliminate_shared_memory_bank_conflicts` (7.421×)

When a Triton GEMM kernel uses tl.dot for a very small K dimension (K≤4), the shared memory staging required by tl.dot introduces severe bank conflicts (10.2-way average, 72.82% of shared load wavefronts) that saturate the L1/TEX pipeline (98.12%) and stall the MIO queue. Replacing tl.dot with an explicit tl.static_range loop over K that loads column vectors from A and row vectors from B and accumulates outer products entirely in registers eliminates all shared memory usage for the accumulation, dropping bank conflicts to zero, reducing registers/thread from 54 to 32, and raising achieved occupancy from 54.69% to 94.33%, yielding a 7.42x speedup. This pattern applies whenever NCU reports SharedMemoryConflicts with high bank-conflict-way (>5x) alongside a K dimension small enough to fully unroll in registers.

> **Applicable when**: Triton GEMM kernel where K is very small (K≤8, ideally K≤4); NCU reports SharedMemoryConflicts with average bank-conflict-way > 5 and conflict wavefronts > 50% of total shared load wavefronts; L1/TEX Cache Throughput > 90%; MIO stall fraction > 30% of cycles; the tl.dot tile is mostly masked (BLOCK_K >> actual K).

> **Avoid when**: K is large (K > 16) — the register file cannot hold the unrolled accumulation and register spilling will cause worse performance than shared memory bank conflicts; kernel is already compute-bound on tensor cores with high SM throughput and low bank conflict rate; BLOCK_K is already tight to actual K and bank conflicts are low.

---

## Iteration 3

**Best so far**: 7.2934× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_bias_load_to_eliminate_uncoalesced_broadcast_global_access | 1.887× | 1.886× | ✓ |  |
| B | replace_tl_dot_with_scalar_fma_outer_product_for_tiny_k | 7.384× | 7.421× | ✓ | **✓ best** |
| C | disable_software_pipeline_on_single_iteration_k_loop | 1.816× | 1.815× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_bias_load_t... | Δ | replace_tl_dot_wi... | Δ | disable_software_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 825250 ns | 437500 ns | -387750 ns (-47.0%) | 111200 ns | -714050 ns (-86.5%) | 454720 ns | -370530 ns (-44.9%) |
| SM Throughput (%) | 25.61 % | 34.01 % | +8.40 pp | 84.26 % | +58.65 pp | 36.72 % | +11.11 pp |
| Memory Throughput (%) | 83.25 % | 97.73 % | +14.48 pp | 82.25 % | -1.00 pp | 97.57 % | +14.32 pp |
| DRAM Throughput (%) | 6.87 % | 12.94 % | +6.07 pp | 50.72 % | +43.85 pp | 12.43 % | +5.56 pp |
| Achieved Occupancy (%) | 18.03 % | 54.65 % | +36.62 pp | 94.36 % | +76.33 pp | 54.69 % | +36.66 pp |
| Registers/Thread | 160.00 register/thread | 56.00 register/thread | -104 | 32.00 register/thread | -128 | 54.00 register/thread | -106 |
| Waves/SM | 165.49 | 55.16 | -110.33 | 31.03 | -134.46 | 55.16 | -110.33 |
| No Eligible (%) | 74.27 % | 65.77 % | -8.50 pp | 14.30 % | -59.97 pp | 63.07 % | -11.20 pp |
| IPC Active | 1.03 inst/cycle | 1.37 inst/cycle | +0.34 | 3.44 inst/cycle | +2.41 | 1.48 inst/cycle | +0.45 |

---

## Iteration 4

**Best so far**: 7.4213× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_bias_load_to_1d_coalesced_load_with_register_broadcast | 9.754× | 10.019× | ✓ | **✓ best** |
| B | increase_num_warps_to_hide_mio_stall_latency | 1.059× | 1.058× | ✓ |  |
| C | eliminate_tl_dot_via_register_tiled_row_matmul_with_b_broadcast | 5.139× | 5.022× | ✓ |  |
| D | add_triton_autotune | 7.383× | 7.389× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_bias_load_t... | Δ | increase_num_warp... | Δ | eliminate_tl_dot_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 825250 ns | 82370 ns | -742880 ns (-90.0%) | 779740 ns | -45510 ns (-5.5%) | 164320 ns | -660930 ns (-80.1%) | 111680 ns | -713570 ns (-86.5%) |
| SM Throughput (%) | 25.61 % | 78.32 % | +52.71 pp | 35.10 % | +9.49 pp | 78.23 % | +52.62 pp | 84.15 % | +58.54 pp |
| Memory Throughput (%) | 83.25 % | 68.42 % | -14.83 pp | 81.51 % | -1.74 pp | 82.61 % | -0.64 pp | 82.16 % | -1.09 pp |
| DRAM Throughput (%) | 6.87 % | 68.42 % | +61.55 pp | 9.03 % | +2.16 pp | 42.63 % | +35.76 pp | 50.52 % | +43.65 pp |
| Achieved Occupancy (%) | 18.03 % | 92.42 % | +74.39 pp | 18.10 % | +0.07 pp | 59.03 % | +41.00 pp | 94.33 % | +76.30 pp |
| Registers/Thread | 160.00 register/thread | 32.00 register/thread | -128 | 162.00 register/thread | +2 | 42.00 register/thread | -118 | 32.00 register/thread | -128 |
| Waves/SM | 165.49 | 31.03 | -134.46 | 165.49 | +0.00 | 49.65 | -115.84 | 31.03 | -134.46 |
| No Eligible (%) | 74.27 % | 18.84 % | -55.43 pp | 64.75 % | -9.52 pp | 20.99 % | -53.28 pp | 14.07 % | -60.20 pp |
| IPC Active | 1.03 inst/cycle | 3.27 inst/cycle | +2.24 | 1.41 inst/cycle | +0.38 | 3.16 inst/cycle | +2.13 | 3.44 inst/cycle | +2.41 |

---

## Iteration 5

**Best so far**: 10.0188× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_a_load_via_2d_block_with_streaming_eviction_and_transposed_k_loop | 5.679× | 5.731× | ✓ |  |
| B | hoist_bias_load_to_1d_coalesced_load_with_register_broadcast_block_m_256 | 1.382× | 1.378× | ✓ |  |
| C | vectorized_coalesced_a_load_with_1d_bias_broadcast | 9.736× | 9.874× | ✓ |  |
| D | restore_scalar_fma_static_range_with_block_m_128_and_coalesced_bias_load | 9.635× | 9.750× | ✓ |  |
| E | add_triton_autotune | 7.365× | 7.337× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_a_load_... | Δ | hoist_bias_load_t... | Δ | vectorized_coales... | Δ | restore_scalar_fm... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 825250 ns | 144000 ns | -681250 ns (-82.6%) | 598720 ns | -226530 ns (-27.4%) | 83580 ns | -741670 ns (-89.9%) | 84640 ns | -740610 ns (-89.7%) | 112480 ns | -712770 ns (-86.4%) |
| SM Throughput (%) | 25.61 % | 82.34 % | +56.73 pp | 42.89 % | +17.28 pp | 78.09 % | +52.48 pp | 81.06 % | +55.45 pp | 84.69 % | +59.08 pp |
| Memory Throughput (%) | 83.25 % | 84.82 % | +1.57 pp | 97.68 % | +14.43 pp | 68.07 % | -15.18 pp | 67.07 % | -16.18 pp | 82.72 % | -0.53 pp |
| DRAM Throughput (%) | 6.87 % | 39.06 % | +32.19 pp | 11.76 % | +4.89 pp | 67.44 % | +60.57 pp | 66.62 % | +59.75 pp | 50.11 % | +43.24 pp |
| Achieved Occupancy (%) | 18.03 % | 95.40 % | +77.37 pp | 48.65 % | +30.62 pp | 92.42 % | +74.39 pp | 92.48 % | +74.45 pp | 94.30 % | +76.27 pp |
| Registers/Thread | 160.00 register/thread | 32.00 register/thread | -128 | 64.00 register/thread | -96 | 32.00 register/thread | -128 | 31.00 register/thread | -129 | 32.00 register/thread | -128 |
| Waves/SM | 165.49 | 31.03 | -134.46 | 62.06 | -103.43 | 31.03 | -134.46 | 31.03 | -134.46 | 31.03 | -134.46 |
| No Eligible (%) | 74.27 % | 16.44 % | -57.83 pp | 56.91 % | -17.36 pp | 19.15 % | -55.12 pp | 17.58 % | -56.69 pp | 13.85 % | -60.42 pp |
| IPC Active | 1.03 inst/cycle | 3.34 inst/cycle | +2.31 | 1.72 inst/cycle | +0.69 | 3.26 inst/cycle | +2.23 | 3.34 inst/cycle | +2.31 | 3.43 inst/cycle | +2.40 |

---

## Cumulative Progress

```
Baseline:  825250 ns  (1.000×)
Iter 1     455070 ns  (1.813×)  — autotune_tile_sizes_to_fix_overprovisioned_block_nk_and_register_pressure
Iter 2     113150 ns  (7.293×)  — replace_tl_dot_with_scalar_fma_to_eliminate_shared_memory_bank_conflicts
Iter 3     111200 ns  (7.421×)  — replace_tl_dot_with_scalar_fma_outer_product_for_tiny_k
Iter 4     82370 ns  (10.019×)  — hoist_bias_load_to_1d_coalesced_load_with_register_broadcast

Final best: 82370 ns  (10.0188×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,14 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused_convolution_1(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
-    BLOCK_N : tl.constexpr = 16
-    BLOCK_K : tl.constexpr = 16
+    BLOCK_N : tl.constexpr = 8
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -54,25 +54,15 @@

         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 3*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        b = tl.load(B + b_k_idx_vals * 1 + offs_b_n[None, :] * 3, mask=b_mask, other=0.0)
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+    # Scalar FMA outer product: K=3, fully unrolled at compile time
+    # Eliminates shared memory staging and bank conflicts
+    for k in tl.static_range(3):  # K=3, fully unrolled at compile time
+        a_col = tl.load(A + offs_a_m * stride_am + k * stride_ak, mask=offs_a_m < M, other=0.0)  # shape (BLOCK_M,)
+        b_row = tl.load(B + k * stride_bk + offs_b_n * stride_bn, mask=offs_b_n < N, other=0.0)  # shape (BLOCK_N,)
+        acc = acc + a_col[:, None] * b_row[None, :]  # outer product, all in registers
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -83,10 +73,12 @@

 
     # inductor generates a suffix
     xindex = idx_n + 6*idx_m
-    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
+    # 1D coalesced bias load: fetch BLOCK_N bias values once per CTA, broadcast in registers
+    bias_mask = rn < N
+    tmp0_1d = tl.load(in_ptr0 + rn, mask=bias_mask, other=0.0)
+    tmp0 = tmp0_1d[None, :]
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

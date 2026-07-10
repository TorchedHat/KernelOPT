# Kernel Optimization Log

## Summary

- **Kernel**: `fused_linear_swish_scale_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 15910000 ns — **Best latency**: 1210000 ns (**13.1488×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`increase_block_k_and_num_stages_for_arithmetic_intensity_and_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `fused_linear_swish_scale_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_059/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 255
- **Duration**: 15910000 ns
- **SM Throughput**: 41.34 % | **Memory**: 56.10 % | **DRAM**: 5.60 %
- **Achieved Occupancy**: 12.24 %
- **IPC Active**: 1.75 inst/cycle | **No Eligible**: 56.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 13.265× | 13.041× | ✓ | **✓ best** |
| B | reduce_register_pressure_via_tma_loads_and_smaller_tiles | 0.958× | 0.298× | ✓ |  |
| C | eviction_policies_streaming_weight_loads | 1.000× | 0.997× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tile_autotune_configs | 1.000× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_register_p... | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15910000 ns | 1220000 ns | -14690000 ns (-92.3%) | 53430000 ns | +37520000 ns (+235.8%) | 15960000 ns | +50000 ns (+0.3%) | 15940000 ns | +30000 ns (+0.2%) |
| SM Throughput (%) | 41.34 % | 64.12 % | +22.78 pp | 14.57 % | -26.77 pp | 41.24 % | -0.10 pp | 41.26 % | -0.08 pp |
| Memory Throughput (%) | 56.10 % | 72.39 % | +16.29 pp | 91.88 % | +35.78 pp | 55.96 % | -0.14 pp | 55.99 % | -0.11 pp |
| DRAM Throughput (%) | 5.60 % | 72.39 % | +66.79 pp | 1.80 % | -3.80 pp | 5.59 % | -0.01 pp | 5.60 % | +0.00 pp |
| Achieved Occupancy (%) | 12.24 % | 12.05 % | -0.19 pp | 12.50 % | +0.26 pp | 12.23 % | -0.01 pp | 12.23 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 96.00 register/thread | -159 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 15.52 | +14.55 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 56.27 % | 78.23 % | +21.96 pp | 84.97 % | +28.70 pp | 56.29 % | +0.02 pp | 56.29 % | +0.02 pp |
| IPC Active | 1.75 inst/cycle | 0.88 inst/cycle | -0.87 | 0.60 inst/cycle | -1.15 | 1.75 inst/cycle | +0.00 | 1.75 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 13.041× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_k_and_num_stages_for_arithmetic_intensity_and_latency_hiding | 13.323× | 13.149× | ✓ | **✓ best** |
| B | fix_weight_load_transpose_to_eliminate_shared_memory_bank_conflicts | 1.000× | 0.998× | ✓ |  |
| C | fix_weight_load_coalescing_via_transposed_tile | 1.000× | 0.998× | ✓ |  |
| D | revert_block_ptr_to_scalar_ptrs_reduce_smem_and_num_stages | 1.080× | 0.913× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_k_... | Δ | fix_weight_load_t... | Δ | fix_weight_load_c... | Δ | revert_block_ptr_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15910000 ns | 1210000 ns | -14700000 ns (-92.4%) | 15940000 ns | +30000 ns (+0.2%) | 15950000 ns | +40000 ns (+0.3%) | 17430000 ns | +1520000 ns (+9.6%) |
| SM Throughput (%) | 41.34 % | 61.92 % | +20.58 pp | 41.26 % | -0.08 pp | 41.23 % | -0.11 pp | 35.39 % | -5.95 pp |
| Memory Throughput (%) | 56.10 % | 72.57 % | +16.47 pp | 55.98 % | -0.12 pp | 55.94 % | -0.16 pp | 78.50 % | +22.40 pp |
| DRAM Throughput (%) | 5.60 % | 72.57 % | +66.97 pp | 5.60 % | +0.00 pp | 5.59 % | -0.01 pp | 5.22 % | -0.38 pp |
| Achieved Occupancy (%) | 12.24 % | 12.06 % | -0.18 pp | 12.24 % | +0.00 pp | 12.23 % | -0.01 pp | 12.18 % | -0.06 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 | 1.94 | +0.97 |
| No Eligible (%) | 56.27 % | 77.76 % | +21.49 pp | 56.29 % | +0.02 pp | 56.28 % | +0.01 pp | 63.51 % | +7.24 pp |
| IPC Active | 1.75 inst/cycle | 0.89 inst/cycle | -0.86 | 1.75 inst/cycle | +0.00 | 1.75 inst/cycle | +0.00 | 1.46 inst/cycle | -0.29 |

---

## Iteration 3

**Best so far**: 13.1488× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_tl_trans_by_direct_k_n_weight_load | 0.996× | 1.001× | ✓ |  |
| C | eliminate_tl_trans_to_fix_shared_memory_bank_conflicts | 0.997× | 1.002× | ✓ |  |
| D | cap_registers_per_thread_via_maxnreg_to_improve_occupancy | 0.878× | 0.893× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_tl_tran... | Δ | eliminate_tl_tran... | Δ | cap_registers_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15910000 ns | 15900000 ns | -10000 ns (-0.1%) | 15880000 ns | -30000 ns (-0.2%) | 17810000 ns | +1900000 ns (+11.9%) |
| SM Throughput (%) | 41.34 % | 41.29 % | -0.05 pp | 41.37 % | +0.03 pp | 36.88 % | -4.46 pp |
| Memory Throughput (%) | 56.10 % | 56.03 % | -0.07 pp | 56.14 % | +0.04 pp | 85.70 % | +29.60 pp |
| DRAM Throughput (%) | 5.60 % | 5.61 % | +0.01 pp | 5.61 % | +0.01 pp | 6.55 % | +0.95 pp |
| Achieved Occupancy (%) | 12.24 % | 12.24 % | +0.00 pp | 12.24 % | +0.00 pp | 22.13 % | +9.89 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 128.00 register/thread | -127 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 56.27 % | 56.28 % | +0.01 pp | 56.26 % | -0.01 pp | 61.75 % | +5.48 pp |
| IPC Active | 1.75 inst/cycle | 1.75 inst/cycle | +0.00 | 1.75 inst/cycle | +0.00 | 1.53 inst/cycle | -0.22 |

---

## Iteration 4

**Best so far**: 13.1488× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cast_dot_inputs_to_fp16_to_unlock_fp16_tensor_core_path | 12.328× | 12.528× | ✓ |  |
| B | cast_dot_inputs_to_fp16_for_tensor_core_mma_path | 12.321× | 12.528× | ✓ |  |
| C | eliminate_register_spill_via_smaller_accumulator_tiles_with_maxnreg64 | 0.880× | 0.892× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cast_dot_inputs_t... | Δ | cast_dot_inputs_t... | Δ | eliminate_registe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15910000 ns | 1270000 ns | -14640000 ns (-92.0%) | 1270000 ns | -14640000 ns (-92.0%) | 17830000 ns | +1920000 ns (+12.1%) |
| SM Throughput (%) | 41.34 % | 31.20 % | -10.14 pp | 31.13 % | -10.21 pp | 36.87 % | -4.47 pp |
| Memory Throughput (%) | 56.10 % | 69.34 % | +13.24 pp | 69.84 % | +13.74 pp | 85.67 % | +29.57 pp |
| DRAM Throughput (%) | 5.60 % | 69.18 % | +63.58 pp | 69.65 % | +64.05 pp | 6.62 % | +1.02 pp |
| Achieved Occupancy (%) | 12.24 % | 12.16 % | -0.08 pp | 12.15 % | -0.09 pp | 22.13 % | +9.89 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 128.00 register/thread | -127 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 56.27 % | 74.17 % | +17.90 pp | 74.18 % | +17.91 pp | 61.74 % | +5.47 pp |
| IPC Active | 1.75 inst/cycle | 1.04 inst/cycle | -0.71 | 1.03 inst/cycle | -0.72 | 1.53 inst/cycle | -0.22 |


### Key Learnings

#### ✅ Plan A: `cast_dot_inputs_to_fp16_to_unlock_fp16_tensor_core_path` (12.528×)

When a Triton GEMM kernel uses `allow_tf32=False` with FP32 inputs, the compiler emits scalar FFMA instructions that saturate the register file at 255 registers/thread, capping occupancy at ~12.5% and generating uncoalesced local memory traffic from register spilling (MemoryCacheAccessPattern rule, ~54% potential speedup). Casting the dot-product inputs to FP16 immediately before `tl.dot` (while keeping the accumulator in FP32) routes computation through FP16 Tensor Core MMA instructions (HMMA.16816.F32.F16 on Hopper), which use a dedicated accumulator register file and dramatically reduce register pressure — DRAM throughput jumps from 5.6% to 69.2% and total executed instructions drop by 23x, yielding ~12.5x speedup. This approach is mechanistically distinct from `allow_tf32=True` (which uses TF32 MMA) and achieves similar speedup (~12.5x vs ~13x for TF32), making it a valid alternative when TF32 is not available or when FP16 precision is acceptable.

> **Applicable when**: Triton kernel has registers_per_thread at or near 255; theoretical occupancy ≤ 12.5% limited by Block Limit Registers; kernel contains tl.dot with allow_tf32=False and FP32 inputs; MemoryCacheAccessPattern rule is triggered for uncoalesced local loads/stores (register spill traffic); correctness tolerance is ≥ 1e-3 (compatible with FP16 7-bit mantissa for typical input ranges); GPU is Volta or newer (FP16 Tensor Core support); allow_tf32=True is not available or not preferred

> **Avoid when**: Kernel requires full FP32 precision (correctness tolerance < 1e-3 or large dynamic range inputs where FP16 overflows); inputs are already FP16/BF16 (cast is redundant); kernel is not register-bound (registers_per_thread well below 255); allow_tf32=True is already enabled and working (TF32 achieves similar speedup with less precision loss); K dimension is very large with high dynamic range where FP16 accumulation errors compound

---

## Cumulative Progress

```
Baseline:  15910000 ns  (1.000×)
Iter 1     1220000 ns  (13.041×)  — tf32_tensor_cores
Iter 2     1210000 ns  (13.149×)  — increase_block_k_and_num_stages_for_arithmetic_intensity_and_latency_hiding

Final best: 1210000 ns  (13.1488×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -1,6 +1,7 @@

 import torch
 import triton
 import triton.language as tl
+
 
 
 @triton.autotune(
@@ -13,6 +14,8 @@

         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
     ],
     key=['M', 'N', 'K'],
 )
@@ -54,30 +57,32 @@

     # Pointers to first tiles
     x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
     # Weight is stored as (N, K) i.e. w[out_feature, in_feature]
-    # We want w.T so we load w[offs_n, offs_k] and treat as (K, N) for dot
-    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
+    # Load w directly as [BLOCK_K, BLOCK_N] to avoid tl.trans() and shared memory bank conflicts
+    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
 
     # Accumulate
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
     for k in range(0, tl.cdiv(K, BLOCK_K)):
         k_remaining = K - k * BLOCK_K
-        # Load x tile [BLOCK_M, BLOCK_K]
+        # Load x tile [BLOCK_M, BLOCK_K] — reused across N-tiles, keep in L2
         x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
-        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
+        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0, eviction_policy='evict_last')
 
-        # Load w tile [BLOCK_K, BLOCK_N]
+        # Load w tile [BLOCK_K, BLOCK_N] — directly in the layout needed for tl.dot
+        # streaming, evict immediately after use
         w_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
-        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
+        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0, eviction_policy='evict_first')
 
+        # No tl.trans needed: w_tile is already [BLOCK_K, BLOCK_N]
         acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
 
         x_ptrs += BLOCK_K * stride_xk
         w_ptrs += BLOCK_K * stride_wk
 
-    # Load bias and add
+    # Load bias and add — bias (128 KB) fits in L2, keep it there
     bias_mask = offs_n < N
-    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
+    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0, eviction_policy='evict_last')
     acc = acc + bias[None, :]
 
     # Swish: x * sigmoid(x)
@@ -88,11 +93,10 @@

     # Scale
     acc = acc * scaling_factor
 
-    # Store output
+    # Store output — write-only, evict immediately to free L2 for reused data
     out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
     out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
-    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)
-
+    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask, eviction_policy='evict_first')
 
 def kernel_function(w, bias, x):
     # x: [M, K], w: [N, K] (Linear weight), bias: [N]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

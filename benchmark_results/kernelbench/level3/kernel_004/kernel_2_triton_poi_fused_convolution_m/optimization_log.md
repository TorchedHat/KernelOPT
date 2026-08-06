# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_max_pool2d_with_indices_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2370 ns — **Best latency**: 2270 ns (**1.0441×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_grid_lambda_and_coalesced_store_access_pattern`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_max_pool2d_with_indices_relu_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_004/kernel_2_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (12,1,1), Waves/SM: 0.01
- **Registers/thread**: 30
- **Duration**: 2370 ns
- **SM Throughput**: 0.30 % | **Memory**: 0.57 % | **DRAM**: 0.12 %
- **Achieved Occupancy**: 6.20 %
- **IPC Active**: 0.39 inst/cycle | **No Eligible**: 89.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune_smaller_tiles | 1.006× | 1.058× | ✓ |  |
| C | reduce_block_size_to_increase_grid_coverage | 1.025× | 1.044× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_grid_coverage | 0.983× | 0.937× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | reduce_block_size... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2370 ns | 2240 ns | -130 ns (-5.5%) | 2270 ns | -100 ns (-4.2%) | 2530 ns | +160 ns (+6.8%) |
| SM Throughput (%) | 0.30 % | 0.04 % | -0.26 pp | 0.03 % | -0.27 pp | 0.36 % | +0.06 pp |
| Memory Throughput (%) | 0.57 % | 0.45 % | -0.12 pp | 0.47 % | -0.10 pp | 0.50 % | -0.07 pp |
| DRAM Throughput (%) | 0.12 % | 0.04 % | -0.08 pp | 0.04 % | -0.08 pp | 0.12 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 1.56 % | -4.64 pp | 1.56 % | -4.64 pp | 6.19 % | -0.01 pp |
| Registers/Thread | 30.00 register/thread | 16.00 register/thread | -14 | 16.00 register/thread | -14 | 24.00 register/thread | -6 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.01 | +0.00 |
| No Eligible (%) | 89.01 % | 95.42 % | +6.41 pp | 95.48 % | +6.47 pp | 81.54 % | -7.47 pp |
| IPC Active | 0.39 inst/cycle | 0.04 inst/cycle | -0.35 | 0.04 inst/cycle | -0.35 | 0.71 inst/cycle | +0.32 |

---

## Iteration 2

**Best so far**: 1.0441× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_add_persistent_tile_scheduling | 0.981× | 1.058× | ✓ |  |
| B | expand_autotune_configs_with_minimal_tiles_for_maximum_grid_coverage | 0.984× | 0.948× | ✓ |  |
| C | fix_autotune_grid_lambda_and_coalesced_store_layout | 0.979× | 0.891× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | expand_autotune_c... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2370 ns | 2240 ns | -130 ns (-5.5%) | 2500 ns | +130 ns (+5.5%) | 2660 ns | +290 ns (+12.2%) |
| SM Throughput (%) | 0.30 % | 0.03 % | -0.27 pp | 0.03 % | -0.27 pp | 0.36 % | +0.06 pp |
| Memory Throughput (%) | 0.57 % | 0.58 % | +0.01 pp | 0.42 % | -0.15 pp | 0.46 % | -0.11 pp |
| DRAM Throughput (%) | 0.12 % | 0.04 % | -0.08 pp | 0.04 % | -0.08 pp | 0.11 % | -0.01 pp |
| Achieved Occupancy (%) | 6.20 % | 1.56 % | -4.64 pp | 1.56 % | -4.64 pp | 6.20 % | +0.00 pp |
| Registers/Thread | 30.00 register/thread | 16.00 register/thread | -14 | 16.00 register/thread | -14 | 24.00 register/thread | -6 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.01 | +0.00 |
| No Eligible (%) | 89.01 % | 95.53 % | +6.52 pp | 95.31 % | +6.30 pp | 84.64 % | -4.37 pp |
| IPC Active | 0.39 inst/cycle | 0.04 inst/cycle | -0.35 | 0.04 inst/cycle | -0.35 | 0.67 inst/cycle | +0.28 |

---

## Iteration 3

**Best so far**: 1.0441× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_lambda_and_coalesced_store_access_pattern | 0.976× | 1.044× | ✓ |  |
| C | fix_hardcoded_grid_lambda_and_add_store_eviction_hint | 0.980× | 0.975× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2370 ns | 2270 ns | -100 ns (-4.2%) | 2430 ns | +60 ns (+2.5%) |
| SM Throughput (%) | 0.30 % | 0.03 % | -0.27 pp | 0.04 % | -0.26 pp |
| Memory Throughput (%) | 0.57 % | 0.44 % | -0.13 pp | 0.37 % | -0.20 pp |
| DRAM Throughput (%) | 0.12 % | 0.08 % | -0.04 pp | 0.03 % | -0.09 pp |
| Achieved Occupancy (%) | 6.20 % | 1.56 % | -4.64 pp | 1.56 % | -4.64 pp |
| Registers/Thread | 30.00 register/thread | 16.00 register/thread | -14 | 16.00 register/thread | -14 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 89.01 % | 95.15 % | +6.14 pp | 95.05 % | +6.04 pp |
| IPC Active | 0.39 inst/cycle | 0.04 inst/cycle | -0.35 | 0.04 inst/cycle | -0.35 |

---

## Cumulative Progress

```
Baseline:  2370 ns  (1.000×)
Iter 1     2270 ns  (1.044×)  — reduce_block_size_to_increase_grid_coverage

Final best: 2270 ns  (1.0441×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 1}, num_warps=1),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 4}, num_warps=1),
+        triton.Config({'XBLOCK': 8, 'YBLOCK': 4}, num_warps=1),
+        triton.Config({'XBLOCK': 8, 'YBLOCK': 8}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 8}, num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 4}, num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 8}, num_warps=4),
+        # NEW persistent configs: power-of-2 YBLOCK values only (tl.arange requirement)
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4),  # 1 CTA covers all 96x25
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4),   # 2 CTAs
+        triton.Config({'XBLOCK': 8, 'YBLOCK': 128}, num_warps=4),   # 1 CTA full Y
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=2),   # 2 CTAs
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_max_pool2d_with_indices_relu_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 96
@@ -22,15 +41,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 6*x2 + 150*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 96
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_max_pool2d_with_indices_relu_2[grid](arg_0, arg_1, 96, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_max_pool2d_with_indices_relu_2[grid](arg_0, arg_1, 96, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

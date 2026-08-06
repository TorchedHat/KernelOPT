# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_26` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10780 ns — **Best latency**: 10850 ns (**0.9935×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_26`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_26_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 28
- **Duration**: 10780 ns
- **SM Throughput**: 16.68 % | **Memory**: 65.43 % | **DRAM**: 34.75 %
- **Achieved Occupancy**: 74.94 %
- **IPC Active**: 0.88 inst/cycle | **No Eligible**: 77.54 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | software_pipelining_with_autotune | — | 0.994× | ✓ | **✓ best** |
| C | increase_tile_sizes_eliminate_partial_wave | — | 2.118× | ✓ |  |
| D | autotune_tile_sizes_and_fix_eviction_policy | — | 2.130× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | increase_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10780 ns | 10850 ns | +70 ns (+0.6%) | 5090 ns | -5690 ns (-52.8%) | 5060 ns | -5720 ns (-53.1%) |
| SM Throughput (%) | 16.68 % | 16.24 % | -0.44 pp | 26.78 % | +10.10 pp | 26.62 % | +9.94 pp |
| Memory Throughput (%) | 65.43 % | 65.04 % | -0.39 pp | 17.50 % | -47.93 pp | 17.48 % | -47.95 pp |
| DRAM Throughput (%) | 34.75 % | 34.42 % | -0.33 pp | 8.49 % | -26.26 pp | 8.47 % | -26.28 pp |
| Achieved Occupancy (%) | 74.94 % | 75.46 % | +0.52 pp | 58.44 % | -16.50 pp | 59.19 % | -15.75 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 18.00 register/thread | -10 | 18.00 register/thread | -10 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 77.54 % | 77.67 % | +0.13 pp | 38.34 % | -39.20 pp | 40.19 % | -37.35 pp |
| IPC Active | 0.88 inst/cycle | 0.88 inst/cycle | +0.00 | 2.39 inst/cycle | +1.51 | 2.35 inst/cycle | +1.47 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_add_num_stages_to_autotune | — | 2.093× | ✓ |  |
| B | software_pipelining_num_stages_fixed_configs | — | 2.118× | ✓ |  |
| C | fix_autotune_grid_lambda_to_enable_tile_selection | — | 2.118× | ✓ |  |
| D | wave_aligned_large_tile_autotune_for_transpose | — | 2.093× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | software_pipelini... | Δ | fix_autotune_grid... | Δ | wave_aligned_larg... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10780 ns | 5150 ns | -5630 ns (-52.2%) | 5090 ns | -5690 ns (-52.8%) | 5090 ns | -5690 ns (-52.8%) | 5150 ns | -5630 ns (-52.2%) |
| SM Throughput (%) | 16.68 % | 25.75 % | +9.07 pp | 27.10 % | +10.42 pp | 26.60 % | +9.92 pp | 26.75 % | +10.07 pp |
| Memory Throughput (%) | 65.43 % | 16.77 % | -48.66 pp | 17.47 % | -47.96 pp | 17.31 % | -48.12 pp | 17.43 % | -48.00 pp |
| DRAM Throughput (%) | 34.75 % | 8.32 % | -26.43 pp | 8.46 % | -26.29 pp | 8.47 % | -26.28 pp | 8.33 % | -26.42 pp |
| Achieved Occupancy (%) | 74.94 % | 62.04 % | -12.90 pp | 59.72 % | -15.22 pp | 57.75 % | -17.19 pp | 63.28 % | -11.66 pp |
| Registers/Thread | 28.00 register/thread | 18.00 register/thread | -10 | 18.00 register/thread | -10 | 18.00 register/thread | -10 | 18.00 register/thread | -10 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 77.54 % | 38.82 % | -38.72 pp | 37.32 % | -40.22 pp | 38.76 % | -38.78 pp | 38.75 % | -38.79 pp |
| IPC Active | 0.88 inst/cycle | 2.15 inst/cycle | +1.27 | 2.33 inst/cycle | +1.45 | 2.38 inst/cycle | +1.50 | 2.36 inst/cycle | +1.48 |

---

## Cumulative Progress

```
Baseline:  10780 ns  (1.000×)

Final best: 10850 ns  (0.9935×)
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
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
@@ -20,9 +36,8 @@

     y3 = yindex
     y0 = (yindex % 1024)
     y1 = yindex // 1024
-    tmp0 = tl.load(in_ptr0 + (x2 + 512*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 512*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 1024*x2 + 524288*y1), tmp0, xmask & ymask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +45,7 @@

     ynumel = 8192
     xnumel = 512
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_26[grid](arg_0, arg_1, 8192, 512, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_26[grid](arg_0, arg_1, 8192, 512)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

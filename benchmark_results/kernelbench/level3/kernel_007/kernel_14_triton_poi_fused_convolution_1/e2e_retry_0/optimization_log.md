# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2400 ns — **Best latency**: 2300 ns (**1.0435×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_small_grid_utilization`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_14_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.03
- **Registers/thread**: 28
- **Duration**: 2400 ns
- **SM Throughput**: 1.28 % | **Memory**: 1.22 % | **DRAM**: 0.48 %
- **Achieved Occupancy**: 6.10 %
- **IPC Active**: 0.36 inst/cycle | **No Eligible**: 89.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_small_grid_utilization | 1.024× | 1.044× | ✓ |  |
| B | reduce_xblock_increase_grid_coverage_with_autotune | 1.067× | 1.044× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.030× | 0.927× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | reduce_xblock_inc... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2400 ns | 2300 ns | -100 ns (-4.2%) | 2300 ns | -100 ns (-4.2%) | 2590 ns | +190 ns (+7.9%) |
| SM Throughput (%) | 1.28 % | 0.13 % | -1.15 pp | 0.12 % | -1.16 pp | 1.33 % | +0.05 pp |
| Memory Throughput (%) | 1.22 % | 0.61 % | -0.61 pp | 0.61 % | -0.61 pp | 1.27 % | +0.05 pp |
| DRAM Throughput (%) | 0.48 % | 0.33 % | -0.15 pp | 0.32 % | -0.16 pp | 0.45 % | -0.03 pp |
| Achieved Occupancy (%) | 6.10 % | 1.56 % | -4.54 pp | 1.56 % | -4.54 pp | 6.13 % | +0.03 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.02 | -0.01 | 0.02 | -0.01 | 0.03 | +0.00 |
| No Eligible (%) | 89.88 % | 96.63 % | +6.75 pp | 93.89 % | +4.01 pp | 91.63 % | +1.75 pp |
| IPC Active | 0.36 inst/cycle | 0.03 inst/cycle | -0.33 | 0.05 inst/cycle | -0.31 | 0.31 inst/cycle | -0.05 |


### Key Learnings

#### ✅ Plan A: `autotune_tile_sizes_for_small_grid_utilization` (1.044×)

For very small kernels where the problem dimensions are not multiples of the default tile size, oversized tiles waste lanes and produce far fewer CTAs than the GPU has SMs, causing near-zero occupancy and waves-per-SM well below 1. Adding @triton.autotune over a range of smaller XBLOCK/YBLOCK values allows the runtime to select tiles that reduce per-block waste and increase CTA count. In this case, reducing block size from 128 to 32 threads cut register pressure (28→16 registers/thread) and eliminated dynamic shared memory (4KB→0), yielding a modest 4.4% speedup despite the grid remaining at 64 CTAs because the autotuner selected a config that reduced overhead rather than expanding the grid as predicted.

> **Applicable when**: Triton kernel has a 2D tile decomposition where one or both problem dimensions are small (e.g., xnumel=25) and not multiples of the default tile size; grid size is less than SM count (waves_per_sm < 0.5); achieved occupancy is very low (<10%); LaunchConfiguration NCU rule fires; kernel is latency-bound rather than memory/compute-bound.

> **Avoid when**: Problem dimensions are large and already produce many CTAs (waves_per_sm >= 1); kernel is already memory or compute bound (SOL > 50%); tile size reduction would cause excessive boundary-check overhead that outweighs the grid expansion benefit; kernel relies on shared memory tiling that requires minimum block size.

---

## Cumulative Progress

```
Baseline:  2400 ns  (1.000×)
Iter 1     2300 ns  (1.044×)  — reduce_xblock_increase_grid_coverage_with_autotune

Final best: 2300 ns  (1.0435×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 16},  num_warps=1),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 16},  num_warps=1),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 8},   num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16},  num_warps=4),
+    ],
+    key=['xnumel', 'ynumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 512
@@ -22,15 +39,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 16*x2 + 400*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 512
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_14[grid](arg_0, arg_1, 512, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_14[grid](arg_0, arg_1, 512, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

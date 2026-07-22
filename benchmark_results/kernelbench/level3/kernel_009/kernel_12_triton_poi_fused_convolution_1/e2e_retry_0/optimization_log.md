# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3580 ns — **Best latency**: 2880 ns (**1.2431×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_small_xnumel_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_12_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1536,1,1), Waves/SM: 0.73
- **Registers/thread**: 28
- **Duration**: 3580 ns
- **SM Throughput**: 22.91 % | **Memory**: 21.12 % | **DRAM**: 2.56 %
- **Achieved Occupancy**: 53.19 %
- **IPC Active**: 2.08 inst/cycle | **No Eligible**: 47.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_small_xnumel_kernel | 1.133× | 1.243× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3580 ns | 2880 ns | -700 ns (-19.6%) |
| SM Throughput (%) | 22.91 % | 8.11 % | -14.80 pp |
| Memory Throughput (%) | 21.12 % | 3.17 % | -17.95 pp |
| DRAM Throughput (%) | 2.56 % | 3.17 % | +0.61 pp |
| Achieved Occupancy (%) | 53.19 % | 42.55 % | -10.64 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 |
| Waves/SM | 0.73 | 0.73 | +0.00 |
| No Eligible (%) | 47.67 % | 66.47 % | +18.80 pp |
| IPC Active | 2.08 inst/cycle | 0.82 inst/cycle | -1.26 |


### Key Learnings

#### ✅ Plan A: `autotune_tile_sizes_for_small_xnumel_kernel` (1.243×)

When a 2D Triton kernel has a very small xnumel (e.g., 9) but uses a large XBLOCK (e.g., 128), the vast majority of threads per block are masked out and do no useful work, inflating instruction count and wasting register/shared-memory resources. Adding @triton.autotune with configs that sweep XBLOCK down to 1 or 16 and YBLOCK up to 128–256 allows the runtime to find a tile shape that eliminates wasted threads and reduces total instruction count (638K → 173K here), cutting duration from 3.58 µs to 2.88 µs (1.24× speedup). The key profiling signals are: waves_per_sm < 1, achieved occupancy well below theoretical, and a large gap between xnumel and XBLOCK visible in the launch statistics.

> **Applicable when**: 2D Triton pointwise/permutation kernel where one tile dimension (xnumel) is much smaller than the fixed XBLOCK (e.g., xnumel <= 16 but XBLOCK=128), waves_per_sm < 1.0, achieved occupancy < 60% of theoretical, and IssueSlotUtilization/SOLBottleneck NCU rules are triggered. Instruction count is inflated relative to actual work due to masked threads.

> **Avoid when**: xnumel is close to or larger than XBLOCK (masking overhead is minimal); kernel is already autotuned; grid is already large enough to saturate SMs (waves_per_sm >= 2); the kernel is memory-bandwidth bound at near-peak DRAM utilization where tile shape changes won't help.

---

## Cumulative Progress

```
Baseline:  3580 ns  (1.000×)
Iter 1     2880 ns  (1.243×)  — autotune_tile_sizes_for_small_xnumel_kernel

Final best: 2880 ns  (1.2431×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 1},  num_warps=4),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 1},  num_warps=4),
+        triton.Config({'YBLOCK': 32,  'XBLOCK': 1},  num_warps=2),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 16}, num_warps=4),
+        triton.Config({'YBLOCK': 32,  'XBLOCK': 16}, num_warps=2),
+        triton.Config({'YBLOCK': 256, 'XBLOCK': 1},  num_warps=8),
+        triton.Config({'YBLOCK': 256, 'XBLOCK': 16}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 12288
@@ -22,14 +36,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 12288
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

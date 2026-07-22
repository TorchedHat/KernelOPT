# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2720 ns — **Best latency**: 2370 ns (**1.1477×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_small_pointwise_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_36`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_36_triton_poi_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (96,1,1), Waves/SM: 0.05
- **Registers/thread**: 28
- **Duration**: 2720 ns
- **SM Throughput**: 1.83 % | **Memory**: 1.75 % | **DRAM**: 0.61 %
- **Achieved Occupancy**: 6.12 %
- **IPC Active**: 0.33 inst/cycle | **No Eligible**: 90.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_small_pointwise_kernel | 1.024× | 1.148× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2720 ns | 2370 ns | -350 ns (-12.9%) |
| SM Throughput (%) | 1.83 % | 0.16 % | -1.67 pp |
| Memory Throughput (%) | 1.75 % | 0.66 % | -1.09 pp |
| DRAM Throughput (%) | 0.61 % | 0.46 % | -0.15 pp |
| Achieved Occupancy (%) | 6.12 % | 1.56 % | -4.56 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 |
| Waves/SM | 0.05 | 0.02 | -0.03 |
| No Eligible (%) | 90.82 % | 92.09 % | +1.27 pp |
| IPC Active | 0.33 inst/cycle | 0.06 inst/cycle | -0.27 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_small_pointwise_kernel` (1.148×)

When a pointwise kernel has very few total elements (e.g., <100K) and the default XBLOCK is much larger than xnumel (e.g., XBLOCK=128 for xnumel=25), the majority of threads per block are masked out as wasted work, and the grid may be smaller than the SM count yielding <1 wave. Adding @triton.autotune with smaller XBLOCK values (1, 4, 8, 16, 32) and varied YBLOCK values allows the runtime to select a configuration that reduces per-thread waste and register pressure (28→16 registers/thread), cutting kernel duration even when occupancy metrics remain low. The primary gain comes from reduced instruction count (38784→2760 executed instructions) and lower register/shared-memory overhead rather than improved occupancy.

> **Applicable when**: Pointwise or transpose kernel with total elements < 100K; default XBLOCK >> xnumel causing >50% thread masking; grid_size < SM count (waves_per_sm < 0.5); AchievedOccupancy < 10%; IssueSlotUtilization rule triggered with No Eligible > 85%; registers_per_thread > 20 suggesting over-provisioned block size

> **Avoid when**: Kernel has large total elements (>1M) where default tile sizes already fill the GPU; xnumel is close to or larger than XBLOCK (low thread waste); kernel is compute-bound rather than launch-overhead-bound; autotune overhead would dominate (very frequent re-compilation scenarios)

---

## Cumulative Progress

```
Baseline:  2720 ns  (1.000×)
Iter 1     2370 ns  (1.148×)  — autotune_tile_sizes_for_small_pointwise_kernel

Final best: 2370 ns  (1.1477×)
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
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=1),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 16},  num_warps=1),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 8},   num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 768
@@ -22,15 +36,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 16*x2 + 400*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 768
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_36[grid](arg_0, arg_1, 768, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_36[grid](arg_0, arg_1, 768, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

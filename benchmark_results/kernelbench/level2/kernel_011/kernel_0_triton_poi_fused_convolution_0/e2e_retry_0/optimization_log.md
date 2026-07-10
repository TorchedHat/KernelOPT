# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6620 ns — **Best latency**: 6110 ns (**1.0835×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_011/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6620 ns
- **SM Throughput**: 16.66 % | **Memory**: 37.40 % | **DRAM**: 18.58 %
- **Achieved Occupancy**: 55.67 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel | 1.440× | 1.083× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 6620 ns | 6110 ns | -510 ns (-7.7%) |
| SM Throughput (%) | 16.66 % | 15.41 % | -1.25 pp |
| Memory Throughput (%) | 37.40 % | 40.84 % | +3.44 pp |
| DRAM Throughput (%) | 18.58 % | 20.17 % | +1.59 pp |
| Achieved Occupancy (%) | 55.67 % | 55.47 % | -0.20 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.19 % | 70.82 % | -0.37 pp |
| IPC Active | 1.14 inst/cycle | 1.13 inst/cycle | -0.01 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel` (1.083×)

For memory-bound transpose kernels with fewer than 1 full wave across SMs (waves_per_sm < 1.0) and high scheduler idle rate (No Eligible > 70%), adding @triton.autotune with a sweep of XBLOCK/YBLOCK/num_warps configurations allows the runtime to select the tile shape that best hides L1TEX latency and maximizes memory throughput. The autotuner achieved a modest 8.4% speedup (6.62 us → 6.11 us) primarily through improved memory throughput (909 → 986 GB/s), though the fundamental bottlenecks (low waves, high scheduler idle) persisted because the grid size and occupancy were unchanged by the winning config. The strategy is most impactful when the winning config meaningfully reduces CTA count or increases warps-per-SM.

> **Applicable when**: Triton pointwise/transpose kernel with waves_per_sm < 1.0, Scheduler No Eligible > 65%, CPIStall on L1TEX > 40%, and fixed hardcoded XBLOCK/YBLOCK tile parameters. Memory throughput is the primary bottleneck (DRAM or L2 throughput < 50%).

> **Avoid when**: Kernel already uses @triton.autotune, or waves_per_sm > 2.0 (grid is already large enough to fill SMs). Also less effective when the dominant bottleneck is compute (SM Throughput > 60%) rather than memory latency. Note: if the grid size does not change with the winning config (same XBLOCK/YBLOCK as baseline), the speedup will be minimal.

---

## Cumulative Progress

```
Baseline:  6620 ns  (1.000×)
Iter 1     6110 ns  (1.083×)  — autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel

Final best: 6110 ns  (1.0835×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=2, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +40,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

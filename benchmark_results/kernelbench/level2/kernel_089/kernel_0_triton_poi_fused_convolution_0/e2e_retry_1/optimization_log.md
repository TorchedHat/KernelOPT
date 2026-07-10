# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 3300 ns (**1.8424×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_089/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6080 ns
- **SM Throughput**: 16.22 % | **Memory**: 40.85 % | **DRAM**: 20.29 %
- **Achieved Occupancy**: 54.92 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.280× | 1.842× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_warps_to_hide_l1tex_latency | 1.257× | 1.045× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 3300 ns | -2780 ns (-45.7%) | 5820 ns | -260 ns (-4.3%) |
| SM Throughput (%) | 16.22 % | 17.31 % | +1.09 pp | 16.74 % | +0.52 pp |
| Memory Throughput (%) | 40.85 % | 20.73 % | -20.12 pp | 41.87 % | +1.02 pp |
| DRAM Throughput (%) | 20.29 % | 5.02 % | -15.27 pp | 21.16 % | +0.87 pp |
| Achieved Occupancy (%) | 54.92 % | 43.32 % | -11.60 pp | 55.04 % | +0.12 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.90 % | 54.85 % | -16.05 pp | 72.32 % | +1.42 pp |
| IPC Active | 1.15 inst/cycle | 1.66 inst/cycle | +0.51 | 1.13 inst/cycle | -0.02 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_and_warps_to_hide_l1tex_latency` (1.045×)

When a Triton kernel shows high L1TEX scoreboard stall cycles (CPIStall ~14 cycles, ~47% of total warp cycles) and low achieved occupancy (~55%) with many idle scheduler cycles (~71% No Eligible), adding @triton.autotune across XBLOCK/YBLOCK/num_warps/num_stages configurations allows the autotuner to empirically select the tile shape that best hides memory latency. In this case the autotuner found a configuration that reduced elapsed cycles from 9050 to 8693 (~4.4% improvement) and increased memory throughput from 992 GB/s to 1030 GB/s, primarily by enabling more in-flight memory requests per CTA. The improvement is modest when the kernel is already wave-count limited (0.74 waves/SM) and the dominant bottleneck (uncoalesced stores, 23% excessive sectors) is not addressed by tile-size changes alone.

> **Applicable when**: Triton pointwise/permutation kernel with: CPIStall L1TEX scoreboard stalls >10 cycles avg, No Eligible scheduler cycles >60%, achieved occupancy <70% of theoretical, waves/SM <1.5, and fixed tile sizes (XBLOCK, YBLOCK, num_warps) not previously autotuned.

> **Avoid when**: Kernel is already autotuned; kernel is compute-bound rather than memory-latency-bound; dominant bottleneck is uncoalesced access pattern (23%+ excessive sectors) that tile-size changes cannot fix — in that case address the memory layout first; kernel has very small grid where autotuning overhead dominates.

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)
Iter 1     3300 ns  (1.842×)  — vectorized_loads_with_autotune

Final best: 3300 ns  (1.8424×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +26,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints to the 1D arange before broadcasting to 2D
+    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +37,13 @@

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

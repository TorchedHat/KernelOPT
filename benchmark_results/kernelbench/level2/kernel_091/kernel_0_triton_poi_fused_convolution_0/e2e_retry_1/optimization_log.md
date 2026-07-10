# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 3710 ns (**1.5957×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_to_improve_occupancy_and_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_091/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5920 ns
- **SM Throughput**: 16.26 % | **Memory**: 42.11 % | **DRAM**: 20.81 %
- **Achieved Occupancy**: 55.55 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 70.94 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.243× | 1.423× | ✓ |  |
| B | autotune_tile_sizes_to_improve_occupancy_and_coalescing | 1.281× | 1.596× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 4160 ns | -1760 ns (-29.7%) | 3710 ns | -2210 ns (-37.3%) |
| SM Throughput (%) | 16.26 % | 18.32 % | +2.06 pp | 25.09 % | +8.83 pp |
| Memory Throughput (%) | 42.11 % | 21.16 % | -20.95 pp | 18.66 % | -23.45 pp |
| DRAM Throughput (%) | 20.81 % | 3.99 % | -16.82 pp | 4.46 % | -16.35 pp |
| Achieved Occupancy (%) | 55.55 % | 49.05 % | -6.50 pp | 68.33 % | +12.78 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.94 % | 52.45 % | -18.49 pp | 41.36 % | -29.58 pp |
| IPC Active | 1.12 inst/cycle | 1.86 inst/cycle | +0.74 | 2.16 inst/cycle | +1.04 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_to_improve_occupancy_and_coalescing` (1.596×)

When a Triton pointwise kernel has fewer than ~1 wave per SM (waves_per_sm < 1.0), the GPU is mostly idle and IssueSlotUtilization is flagged. Adding @triton.autotune with a range of XBLOCK/YBLOCK configs and num_warps values allows the runtime to select the tile shape that maximizes wave count and hides L1TEX latency, reducing Warp Cycles Per Issued Instruction and raising achieved occupancy. In this case, doubling the block size from 128 to 256 threads (selected by autotune) raised waves_per_sm from 0.74 to 1.48, eligible warps per scheduler from 0.94 to 3.23, and cut kernel duration from 5.92 µs to 3.71 µs (1.60x speedup).

> **Applicable when**: waves_per_sm < 1.0; IssueSlotUtilization or AchievedOccupancy NCU rules triggered; Eligible Warps Per Scheduler < 1.5; Warp Cycles Per Issued Instruction > 20; kernel uses fixed hardcoded XBLOCK/YBLOCK tile sizes; 2D pointwise or layout-permutation kernel with separable x/y tile dimensions

> **Avoid when**: Kernel already has waves_per_sm >> 1 and is bottlenecked on compute or DRAM bandwidth rather than latency/occupancy; autotune overhead is unacceptable (e.g., very short-lived kernels called once); tile dimensions are not independent (e.g., shared memory tiling with cross-tile dependencies)

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)
Iter 1     3710 ns  (1.596×)  — autotune_tile_sizes_to_improve_occupancy_and_coalescing

Final best: 3710 ns  (1.5957×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +36,13 @@

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

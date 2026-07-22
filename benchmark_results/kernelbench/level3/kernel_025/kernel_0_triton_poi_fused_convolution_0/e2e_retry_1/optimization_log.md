# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6180 ns — **Best latency**: 3680 ns (**1.6793×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_025/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6180 ns
- **SM Throughput**: 16.90 % | **Memory**: 39.97 % | **DRAM**: 19.89 %
- **Achieved Occupancy**: 55.32 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 70.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.280× | 1.398× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.411× | 1.679× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6180 ns | 4420 ns | -1760 ns (-28.5%) | 3680 ns | -2500 ns (-40.5%) |
| SM Throughput (%) | 16.90 % | 21.16 % | +4.26 pp | 24.99 % | +8.09 pp |
| Memory Throughput (%) | 39.97 % | 32.02 % | -7.95 pp | 18.67 % | -21.30 pp |
| DRAM Throughput (%) | 19.89 % | 14.98 % | -4.91 pp | 4.50 % | -15.39 pp |
| Achieved Occupancy (%) | 55.32 % | 46.05 % | -9.27 pp | 67.40 % | +12.08 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.85 % | 55.54 % | -15.31 pp | 43.45 % | -27.40 pp |
| IPC Active | 1.10 inst/cycle | 1.70 inst/cycle | +0.60 | 2.25 inst/cycle | +1.15 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalescing_and_occupancy` (1.679×)

When a Triton pointwise kernel has fewer than 1.0 waves/SM (grid too small to saturate the GPU), high 'No Eligible' warp percentage (>60%), and significant L1TEX scoreboard stalls (CPIStall >40%), adding a @triton.autotune decorator with configs spanning XBLOCK in {64,128,256,512}, YBLOCK in {1,2,4,8}, and num_warps in {4,8,16} allows the runtime to select the tile shape that maximizes SM utilization and hides memory latency. In this case, autotuning doubled waves/SM (0.74→1.48), raised achieved occupancy (55%→67%), halved warp stall cycles per instruction (32→18), and reduced kernel duration by 40% (6.18→3.68 µs). The key mechanism is that larger XBLOCK increases the grid block count and memory-level parallelism, while more num_warps provides additional in-flight warps to hide L1TEX latency.

> **Applicable when**: Triton pointwise/elementwise kernel with: waves_per_sm < 1.0, achieved_occupancy < 60%, No Eligible warp % > 60%, CPIStall (L1TEX scoreboard) > 30% of cycles, and fixed XBLOCK/YBLOCK tile sizes hardcoded at launch. Especially effective when the kernel has a 2D grid (ynumel × xnumel) and the y-dimension is small (e.g., channels=3 or 30).

> **Avoid when**: Kernel already uses @triton.autotune, or when the grid is already large enough to saturate all SMs (waves_per_sm > 2.0). Also avoid when correctness depends on a specific tile shape (e.g., shared memory tiling with fixed sizes). Note that autotuning adds first-run overhead; avoid for kernels called only once.

---

## Cumulative Progress

```
Baseline:  6180 ns  (1.000×)
Iter 1     3680 ns  (1.679×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3680 ns  (1.6793×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=16, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +35,13 @@

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

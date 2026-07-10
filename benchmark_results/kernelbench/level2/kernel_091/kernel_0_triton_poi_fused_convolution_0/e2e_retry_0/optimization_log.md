# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6050 ns — **Best latency**: 4450 ns (**1.3596×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps_for_underutilized_layout_kernel`** beats baseline latency.

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
- **Duration**: 6050 ns
- **SM Throughput**: 16.31 % | **Memory**: 41.54 % | **DRAM**: 20.40 %
- **Achieved Occupancy**: 55.19 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 70.81 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.314× | 1.296× | ✓ |  |
| B | autotune_tile_sizes_and_warps_for_underutilized_layout_kernel | 1.331× | 1.360× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 4670 ns | -1380 ns (-22.8%) | 4450 ns | -1600 ns (-26.4%) |
| SM Throughput (%) | 16.31 % | 20.15 % | +3.84 pp | 20.71 % | +4.40 pp |
| Memory Throughput (%) | 41.54 % | 30.01 % | -11.53 pp | 31.74 % | -9.80 pp |
| DRAM Throughput (%) | 20.40 % | 14.07 % | -6.33 pp | 14.91 % | -5.49 pp |
| Achieved Occupancy (%) | 55.19 % | 47.17 % | -8.02 pp | 46.65 % | -8.54 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 27.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.81 % | 55.46 % | -15.35 pp | 56.20 % | -14.61 pp |
| IPC Active | 1.12 inst/cycle | 1.70 inst/cycle | +0.58 | 1.68 inst/cycle | +0.56 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_and_warps_for_underutilized_layout_kernel` (1.360×)

When a Triton kernel shows high idle issue slots (>60% No Eligible) and low eligible warps per scheduler (<1.0), the fixed tile configuration chosen by the compiler heuristic is likely suboptimal. Adding @triton.autotune with a sweep of XBLOCK, YBLOCK, and num_warps configs allows the runtime to select a tile shape that better hides memory latency, reducing Warp Cycles Per Issued Instruction from ~31 to ~17 and improving Issue Slots Busy from 28% to 43%. The grid size and waves/SM may remain similar, but the autotuner finds a configuration that increases eligible warps per scheduler (0.95 → 1.72) and reduces kernel duration significantly.

> **Applicable when**: Triton kernel with: Scheduler 'No Eligible' > 60%, Eligible Warps Per Scheduler < 1.0, Warp Cycles Per Issued Instruction > 25, IssueSlotUtilization NCU rule triggered with >50% speedup estimate, kernel is a pointwise/layout-permutation op with 2D grid (XBLOCK x YBLOCK tiling), and the compiler used a fixed tile heuristic.

> **Avoid when**: Kernel is already autotuned or uses dynamic tile selection; kernel is compute-bound (high SM throughput >80%); grid is already large enough to saturate SMs (waves_per_sm > 2); kernel has complex shared memory usage that constrains tile sizes; autotuning overhead is unacceptable (e.g., single-use kernels in latency-critical paths).

---

## Cumulative Progress

```
Baseline:  6050 ns  (1.000×)
Iter 1     4450 ns  (1.360×)  — autotune_tile_sizes_and_warps_for_underutilized_layout_kernel

Final best: 4450 ns  (1.3596×)
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
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8),
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

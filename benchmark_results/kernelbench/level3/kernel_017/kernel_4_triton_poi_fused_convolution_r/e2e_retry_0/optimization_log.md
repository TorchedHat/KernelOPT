# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2620 ns — **Best latency**: 2240 ns (**1.1696×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`increase_tile_sizes_and_fix_grid_underutilization`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_017/kernel_4_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (48,1,1), Waves/SM: 0.02
- **Registers/thread**: 30
- **Duration**: 2620 ns
- **SM Throughput**: 1.09 % | **Memory**: 1.19 % | **DRAM**: 0.14 %
- **Achieved Occupancy**: 6.18 %
- **IPC Active**: 0.37 inst/cycle | **No Eligible**: 88.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_small_workload | 1.027× | 0.963× | ✓ |  |
| B | increase_tile_sizes_and_fix_grid_underutilization | 1.036× | 1.170× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 2720 ns | +100 ns (+3.8%) | 2240 ns | -380 ns (-14.5%) |
| SM Throughput (%) | 1.09 % | 2.14 % | +1.05 pp | 0.08 % | -1.01 pp |
| Memory Throughput (%) | 1.19 % | 1.80 % | +0.61 pp | 0.49 % | -0.70 pp |
| DRAM Throughput (%) | 0.14 % | 0.15 % | +0.01 pp | 0.16 % | +0.02 pp |
| Achieved Occupancy (%) | 6.18 % | 6.19 % | +0.01 pp | 1.56 % | -4.62 pp |
| Registers/Thread | 30.00 register/thread | 32.00 register/thread | +2 | 16.00 register/thread | -14 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.01 | -0.01 |
| No Eligible (%) | 88.82 % | 72.19 % | -16.63 pp | 92.77 % | +3.95 pp |
| IPC Active | 0.37 inst/cycle | 1.08 inst/cycle | +0.71 | 0.05 inst/cycle | -0.32 |

---

## Cumulative Progress

```
Baseline:  2620 ns  (1.000×)
Iter 1     2240 ns  (1.170×)  — increase_tile_sizes_and_fix_grid_underutilization

Final best: 2240 ns  (1.1696×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 32}, num_warps=1),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 64}, num_warps=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     # Simple 2D tiled kernel -- NO persistent loop
@@ -20,14 +32,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9 * yindex), mask=ymask & xmask, other=0.0)
     tl.store(out_ptr0 + (y0 + 6 * x2 + 54 * y1), tmp0, mask=ymask & xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 384
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_relu_4[grid](arg_0, arg_1, 384, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_relu_4[grid](arg_0, arg_1, 384, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

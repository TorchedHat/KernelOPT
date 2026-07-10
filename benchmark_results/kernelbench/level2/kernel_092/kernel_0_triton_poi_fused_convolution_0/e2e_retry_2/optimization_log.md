# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 4190 ns (**1.4511×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_fix_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_092/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6080 ns
- **SM Throughput**: 15.95 % | **Memory**: 40.27 % | **DRAM**: 20.24 %
- **Achieved Occupancy**: 55.29 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.05 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.269× | 1.310× | ✓ |  |
| B | autotune_tile_sizes_and_fix_coalescing | 1.417× | 1.451× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.005× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 4640 ns | -1440 ns (-23.7%) | 4190 ns | -1890 ns (-31.1%) | 6020 ns | -60 ns (-1.0%) |
| SM Throughput (%) | 15.95 % | 22.83 % | +6.88 pp | 24.69 % | +8.74 pp | 16.19 % | +0.24 pp |
| Memory Throughput (%) | 40.27 % | 30.15 % | -10.12 pp | 19.65 % | -20.62 pp | 42.16 % | +1.89 pp |
| DRAM Throughput (%) | 20.24 % | 14.19 % | -6.05 pp | 3.95 % | -16.29 pp | 20.53 % | +0.29 pp |
| Achieved Occupancy (%) | 55.29 % | 47.34 % | -7.95 pp | 72.29 % | +17.00 pp | 55.63 % | +0.34 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.05 % | 52.34 % | -18.71 pp | 37.36 % | -33.69 pp | 71.51 % | +0.46 pp |
| IPC Active | 1.14 inst/cycle | 1.74 inst/cycle | +0.60 | 2.14 inst/cycle | +1.00 | 1.13 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)
Iter 1     4190 ns  (1.451×)  — autotune_tile_sizes_and_fix_coalescing

Final best: 4190 ns  (1.4511×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8}, num_warps=2),
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

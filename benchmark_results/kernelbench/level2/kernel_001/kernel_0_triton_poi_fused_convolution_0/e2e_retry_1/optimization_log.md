# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 3330 ns (**1.7868×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_fix_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_001/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 16.37 % | **Memory**: 41.41 % | **DRAM**: 20.77 %
- **Achieved Occupancy**: 54.51 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.220× | 1.010× | ✓ |  |
| B | autotune_tile_sizes_and_fix_coalescing | 1.361× | 1.787× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 5890 ns | -60 ns (-1.0%) | 3330 ns | -2620 ns (-44.0%) |
| SM Throughput (%) | 16.37 % | 16.88 % | +0.51 pp | 17.26 % | +0.89 pp |
| Memory Throughput (%) | 41.41 % | 42.18 % | +0.77 pp | 20.84 % | -20.57 pp |
| DRAM Throughput (%) | 20.77 % | 20.93 % | +0.16 pp | 5.01 % | -15.76 pp |
| Achieved Occupancy (%) | 54.51 % | 54.87 % | +0.36 pp | 44.19 % | -10.32 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.95 % | 72.47 % | +0.52 pp | 55.46 % | -16.49 pp |
| IPC Active | 1.15 inst/cycle | 1.14 inst/cycle | -0.01 | 1.69 inst/cycle | +0.54 |

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 1     3330 ns  (1.787×)  — autotune_tile_sizes_and_fix_coalescing

Final best: 3330 ns  (1.7868×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1024,'YBLOCK': 1}, num_warps=8, num_stages=2),
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

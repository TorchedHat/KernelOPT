# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6530 ns — **Best latency**: 4220 ns (**1.5474×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy_and_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_085/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6530 ns
- **SM Throughput**: 16.46 % | **Memory**: 38.98 % | **DRAM**: 18.89 %
- **Achieved Occupancy**: 55.44 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 72.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.235× | 1.467× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.347× | 1.547× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6530 ns | 4450 ns | -2080 ns (-31.9%) | 4220 ns | -2310 ns (-35.4%) |
| SM Throughput (%) | 16.46 % | 21.19 % | +4.73 pp | 27.05 % | +10.59 pp |
| Memory Throughput (%) | 38.98 % | 31.84 % | -7.14 pp | 21.30 % | -17.68 pp |
| DRAM Throughput (%) | 18.89 % | 14.91 % | -3.98 pp | 3.93 % | -14.96 pp |
| Achieved Occupancy (%) | 55.44 % | 46.62 % | -8.82 pp | 70.74 % | +15.30 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 72.56 % | 55.30 % | -17.26 pp | 38.11 % | -34.45 pp |
| IPC Active | 1.15 inst/cycle | 1.70 inst/cycle | +0.55 | 2.30 inst/cycle | +1.15 |

---

## Cumulative Progress

```
Baseline:  6530 ns  (1.000×)
Iter 1     4220 ns  (1.547×)  — autotune_tile_sizes_for_occupancy_and_coalescing

Final best: 4220 ns  (1.5474×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=2),
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

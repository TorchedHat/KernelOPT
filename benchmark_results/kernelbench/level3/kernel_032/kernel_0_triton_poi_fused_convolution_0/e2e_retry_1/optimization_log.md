# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 3200 ns (**1.9000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_032/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6080 ns
- **SM Throughput**: 16.64 % | **Memory**: 41.69 % | **DRAM**: 20.27 %
- **Achieved Occupancy**: 55.88 %
- **IPC Active**: 1.09 inst/cycle | **No Eligible**: 71.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.003× | 0.995× | ✓ |  |
| B | autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel | 1.433× | 1.900× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 6110 ns | +30 ns (+0.5%) | 3200 ns | -2880 ns (-47.4%) |
| SM Throughput (%) | 16.64 % | 16.10 % | -0.54 pp | 5.36 % | -11.28 pp |
| Memory Throughput (%) | 41.69 % | 40.84 % | -0.85 pp | 11.29 % | -30.40 pp |
| DRAM Throughput (%) | 20.27 % | 20.23 % | -0.04 pp | 2.61 % | -17.66 pp |
| Achieved Occupancy (%) | 55.88 % | 54.49 % | -1.39 pp | 27.53 % | -28.35 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 71.90 % | 70.49 % | -1.41 pp | 82.49 % | +10.59 pp |
| IPC Active | 1.09 inst/cycle | 1.15 inst/cycle | +0.06 | 0.70 inst/cycle | -0.39 |

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)
Iter 1     3200 ns  (1.900×)  — autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel

Final best: 3200 ns  (1.9000×)
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
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 1},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 2},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 32,   'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=8, num_stages=4),
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

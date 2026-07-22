# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2940 ns — **Best latency**: 2720 ns (**1.0809×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy_and_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_12_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (768,1,1), Waves/SM: 0.36
- **Registers/thread**: 28
- **Duration**: 2940 ns
- **SM Throughput**: 13.53 % | **Memory**: 12.34 % | **DRAM**: 1.59 %
- **Achieved Occupancy**: 29.75 %
- **IPC Active**: 1.45 inst/cycle | **No Eligible**: 61.52 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_occupancy_and_coalescing | 1.001× | 1.081× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2940 ns | 2720 ns | -220 ns (-7.5%) |
| SM Throughput (%) | 13.53 % | 8.57 % | -4.96 pp |
| Memory Throughput (%) | 12.34 % | 7.71 % | -4.63 pp |
| DRAM Throughput (%) | 1.59 % | 1.70 % | +0.11 pp |
| Achieved Occupancy (%) | 29.75 % | 12.63 % | -17.12 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 |
| Waves/SM | 0.36 | 0.18 | -0.18 |
| No Eligible (%) | 61.52 % | 66.56 % | +5.04 pp |
| IPC Active | 1.45 inst/cycle | 1.08 inst/cycle | -0.37 |

---

## Cumulative Progress

```
Baseline:  2940 ns  (1.000×)
Iter 1     2720 ns  (1.081×)  — autotune_tile_sizes_for_occupancy_and_coalescing

Final best: 2720 ns  (1.0809×)
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
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 6144
@@ -22,14 +36,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 6144
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12[grid](arg_0, arg_1, 6144, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12[grid](arg_0, arg_1, 6144, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

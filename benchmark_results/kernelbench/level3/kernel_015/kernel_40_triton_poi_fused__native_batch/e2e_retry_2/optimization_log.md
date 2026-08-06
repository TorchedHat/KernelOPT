# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3940 ns — **Best latency**: 3420 ns (**1.1520×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_small_xnumel_transpose`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_40_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1920,1,1), Waves/SM: 0.91
- **Registers/thread**: 28
- **Duration**: 3940 ns
- **SM Throughput**: 26.23 % | **Memory**: 24.63 % | **DRAM**: 2.91 %
- **Achieved Occupancy**: 61.85 %
- **IPC Active**: 2.23 inst/cycle | **No Eligible**: 42.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_small_xnumel_transpose | 1.103× | 1.152× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.020× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 3420 ns | -520 ns (-13.2%) | 3940 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 26.23 % | 16.98 % | -9.25 pp | 26.45 % | +0.22 pp |
| Memory Throughput (%) | 24.63 % | 15.41 % | -9.22 pp | 24.54 % | -0.09 pp |
| DRAM Throughput (%) | 2.91 % | 3.36 % | +0.45 pp | 2.92 % | +0.01 pp |
| Achieved Occupancy (%) | 61.85 % | 33.87 % | -27.98 pp | 61.32 % | -0.53 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 28.00 register/thread | +0 |
| Waves/SM | 0.91 | 0.45 | -0.46 | 0.91 | +0.00 |
| No Eligible (%) | 42.65 % | 44.11 % | +1.46 pp | 41.79 % | -0.86 pp |
| IPC Active | 2.23 inst/cycle | 2.05 inst/cycle | -0.18 | 2.16 inst/cycle | -0.07 |

---

## Cumulative Progress

```
Baseline:  3940 ns  (1.000×)
Iter 1     3420 ns  (1.152×)  — autotune_tile_sizes_for_small_xnumel_transpose

Final best: 3420 ns  (1.1520×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 15360
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +33,8 @@

     y3 = yindex
     y0 = (yindex % 480)
     y1 = yindex // 480
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 480*x2 + 4320*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tl.store(out_ptr0 + (y0 + 480*x2 + 4320*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +42,7 @@

     ynumel = 15360
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40[grid](arg_0, arg_1, 15360, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40[grid](arg_0, arg_1, 15360, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

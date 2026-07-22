# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7260 ns — **Best latency**: 6820 ns (**1.0645×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_shape_for_tiny_xdim_and_improve_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_228_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7296,1,1), Waves/SM: 3.45
- **Registers/thread**: 28
- **Duration**: 7260 ns
- **SM Throughput**: 54.39 % | **Memory**: 51.38 % | **DRAM**: 5.94 %
- **Achieved Occupancy**: 72.36 %
- **IPC Active**: 3.14 inst/cycle | **No Eligible**: 20.06 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_tile_shape_for_tiny_xdim_and_improve_coalescing | 1.038× | 1.065× | ✓ | **✓ best** |
| C | add_triton_autotune | 0.999× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_shape_fo... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7260 ns | 6820 ns | -440 ns (-6.1%) | 7200 ns | -60 ns (-0.8%) |
| SM Throughput (%) | 54.39 % | 22.57 % | -31.82 pp | 55.03 % | +0.64 pp |
| Memory Throughput (%) | 51.38 % | 6.33 % | -45.05 pp | 51.98 % | +0.60 pp |
| DRAM Throughput (%) | 5.94 % | 6.33 % | +0.39 pp | 6.00 % | +0.06 pp |
| Achieved Occupancy (%) | 72.36 % | 29.97 % | -42.39 pp | 70.75 % | -1.61 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 3.45 | 3.45 | +0.00 | 3.45 | +0.00 |
| No Eligible (%) | 20.06 % | 55.69 % | +35.63 pp | 20.79 % | +0.73 pp |
| IPC Active | 3.14 inst/cycle | 1.63 inst/cycle | -1.51 | 3.17 inst/cycle | +0.03 |

---

## Cumulative Progress

```
Baseline:  7260 ns  (1.000×)
Iter 1     6820 ns  (1.065×)  — fix_tile_shape_for_tiny_xdim_and_improve_coalescing

Final best: 6820 ns  (1.0645×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 58368
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +32,8 @@

     y3 = yindex
     y0 = (yindex % 1824)
     y1 = yindex // 1824
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 1824*x2 + 16416*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 1824*x2 + 16416*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +41,7 @@

     ynumel = 58368
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228[grid](arg_0, arg_1, 58368, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228[grid](arg_0, arg_1, 58368, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

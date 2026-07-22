# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2880 ns — **Best latency**: 2660 ns (**1.0827×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_shape_and_autotune_for_underutilized_transpose_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (640,1,1), Waves/SM: 0.30
- **Registers/thread**: 28
- **Duration**: 2880 ns
- **SM Throughput**: 12.06 % | **Memory**: 11.10 % | **DRAM**: 1.36 %
- **Achieved Occupancy**: 25.67 %
- **IPC Active**: 1.28 inst/cycle | **No Eligible**: 65.21 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_tile_shape_and_autotune_for_underutilized_transpose_kernel | — | 1.083× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_shape_an... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2880 ns | 2660 ns | -220 ns (-7.6%) |
| SM Throughput (%) | 12.06 % | 6.90 % | -5.16 pp |
| Memory Throughput (%) | 11.10 % | 6.21 % | -4.89 pp |
| DRAM Throughput (%) | 1.36 % | 1.48 % | +0.12 pp |
| Achieved Occupancy (%) | 25.67 % | 10.86 % | -14.81 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 |
| Waves/SM | 0.30 | 0.15 | -0.15 |
| No Eligible (%) | 65.21 % | 77.69 % | +12.48 pp |
| IPC Active | 1.28 inst/cycle | 0.90 inst/cycle | -0.38 |

---

## Cumulative Progress

```
Baseline:  2880 ns  (1.000×)
Iter 1     2660 ns  (1.083×)  — fix_tile_shape_and_autotune_for_underutilized_transpose_kernel

Final best: 2660 ns  (1.0827×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 5120
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +30,8 @@

     y3 = yindex
     y0 = (yindex % 160)
     y1 = yindex // 160
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +39,7 @@

     ynumel = 5120
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9[grid](arg_0, arg_1, 5120, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9[grid](arg_0, arg_1, 5120, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

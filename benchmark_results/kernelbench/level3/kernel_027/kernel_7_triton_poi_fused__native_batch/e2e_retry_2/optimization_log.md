# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3870 ns — **Best latency**: 3140 ns (**1.2325×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`persistent_tile_scheduling_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_7_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 0.97
- **Registers/thread**: 28
- **Duration**: 3870 ns
- **SM Throughput**: 27.06 % | **Memory**: 26.17 % | **DRAM**: 3.15 %
- **Achieved Occupancy**: 65.25 %
- **IPC Active**: 2.24 inst/cycle | **No Eligible**: 41.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 1.131× | 1.232× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3870 ns | 3140 ns | -730 ns (-18.9%) |
| SM Throughput (%) | 27.06 % | 8.64 % | -18.42 pp |
| Memory Throughput (%) | 26.17 % | 6.48 % | -19.69 pp |
| DRAM Throughput (%) | 3.15 % | 3.94 % | +0.79 pp |
| Achieved Occupancy (%) | 65.25 % | 22.58 % | -42.67 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 |
| Waves/SM | 0.97 | 0.97 | +0.00 |
| No Eligible (%) | 41.80 % | 80.87 % | +39.07 pp |
| IPC Active | 2.24 inst/cycle | 0.86 inst/cycle | -1.38 |

---

## Cumulative Progress

```
Baseline:  3870 ns  (1.000×)
Iter 1     3140 ns  (1.232×)  — persistent_tile_scheduling_with_autotune

Final best: 3140 ns  (1.2325×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 16384
     xnumel = 9
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
+    pid_y = tl.program_id(1)
+    pid_x = tl.program_id(0)
+    num_pid_y = tl.num_programs(1)
+    xoffset = pid_x * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
-    y3 = yindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
-
-
+    # Persistent loop: each CTA handles multiple y-tiles
+    yoffset = pid_y * YBLOCK
+    while yoffset < ynumel:
+        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+        ymask = yindex < ynumel
+        y3 = yindex
+        y0 = (yindex % 128)
+        y1 = yindex // 128
+        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+        tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
+        yoffset += num_pid_y * YBLOCK
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +42,7 @@

     ynumel = 16384
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7[grid](arg_0, arg_1, 16384, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7[grid](arg_0, arg_1, 16384, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

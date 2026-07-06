# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174820 ns — **Best latency**: 52450 ns (**3.3331×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`add_autotune_fix_memory_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_055/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174820 ns
- **SM Throughput**: 27.97 % | **Memory**: 81.42 % | **DRAM**: 67.19 %
- **Achieved Occupancy**: 94.31 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | add_autotune_fix_memory_coalescing | 3.195× | 3.333× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_autotune_fix_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 174820 ns | 52450 ns | -122370 ns (-70.0%) |
| SM Throughput (%) | 27.97 % | 23.48 % | -4.49 pp |
| Memory Throughput (%) | 81.42 % | 14.70 % | -66.72 pp |
| DRAM Throughput (%) | 67.19 % | 1.37 % | -65.82 pp |
| Achieved Occupancy (%) | 94.31 % | 24.78 % | -69.53 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 70.99 % | 69.75 % | -1.24 pp |
| IPC Active | 1.16 inst/cycle | 1.14 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  174820 ns  (1.000×)
Iter 1     52450 ns  (3.333×)  — add_autotune_fix_memory_coalescing

Final best: 52450 ns  (3.3331×)
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
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 256}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},   num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 24576
     xnumel = 3025
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,12 +33,10 @@

     y0 = (yindex % 96)
     y1 = yindex // 96
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask)
-
-
+    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +45,7 @@

     ynumel = 24576
     xnumel = 3025
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
     return arg_2
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

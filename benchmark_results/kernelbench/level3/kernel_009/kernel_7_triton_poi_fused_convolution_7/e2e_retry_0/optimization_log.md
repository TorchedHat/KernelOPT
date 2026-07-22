# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3360 ns — **Best latency**: 2910 ns (**1.1546×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_and_autotune_for_underutilized_transpose_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_7_triton_poi_fused_convolution_7/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 0.48
- **Registers/thread**: 28
- **Duration**: 3360 ns
- **SM Throughput**: 16.02 % | **Memory**: 15.18 % | **DRAM**: 1.82 %
- **Achieved Occupancy**: 37.25 %
- **IPC Active**: 1.70 inst/cycle | **No Eligible**: 53.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_and_autotune_for_underutilized_transpose_kernel | 1.012× | 1.155× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_an... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3360 ns | 2910 ns | -450 ns (-13.4%) |
| SM Throughput (%) | 16.02 % | 22.37 % | +6.35 pp |
| Memory Throughput (%) | 15.18 % | 15.19 % | +0.01 pp |
| DRAM Throughput (%) | 1.82 % | 2.12 % | +0.30 pp |
| Achieved Occupancy (%) | 37.25 % | 37.24 % | -0.01 pp |
| Registers/Thread | 28.00 register/thread | 27.00 register/thread | -1 |
| Waves/SM | 0.48 | 0.48 | +0.00 |
| No Eligible (%) | 53.86 % | 38.20 % | -15.66 pp |
| IPC Active | 1.70 inst/cycle | 2.37 inst/cycle | +0.67 |

---

## Cumulative Progress

```
Baseline:  3360 ns  (1.000×)
Iter 1     2910 ns  (1.155×)  — fix_tile_sizes_and_autotune_for_underutilized_transpose_kernel

Final best: 2910 ns  (1.1546×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +31,8 @@

     y3 = yindex
     y0 = (yindex % 64)
     y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +40,7 @@

     ynumel = 8192
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_7[grid](arg_0, arg_1, 8192, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_7[grid](arg_0, arg_1, 8192, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3040 ns — **Best latency**: 2820 ns (**1.0780×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_shape_and_autotune_for_full_sm_utilization`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_36`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_36_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 0.48
- **Registers/thread**: 27
- **Duration**: 3040 ns
- **SM Throughput**: 14.56 % | **Memory**: 14.48 % | **DRAM**: 0.91 %
- **Achieved Occupancy**: 40.34 %
- **IPC Active**: 1.49 inst/cycle | **No Eligible**: 59.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_shape_and_autotune_for_full_sm_utilization | — | 1.078× | ✓ | **✓ best** |
| B | fix_tile_dimensions_to_eliminate_wasted_threads | — | 1.160× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_shape_an... | Δ | fix_tile_dimensio... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3040 ns | 2820 ns | -220 ns (-7.2%) | 2620 ns | -420 ns (-13.8%) |
| SM Throughput (%) | 14.56 % | 1.66 % | -12.90 pp | 9.34 % | -5.22 pp |
| Memory Throughput (%) | 14.48 % | 3.20 % | -11.28 pp | 2.87 % | -11.61 pp |
| DRAM Throughput (%) | 0.91 % | 0.14 % | -0.77 pp | 1.05 % | +0.14 pp |
| Achieved Occupancy (%) | 40.34 % | 10.36 % | -29.98 pp | 31.41 % | -8.93 pp |
| Registers/Thread | 27.00 register/thread | 16.00 register/thread | -11 | 16.00 register/thread | -11 |
| Waves/SM | 0.48 | 0.24 | -0.24 | 0.48 | +0.00 |
| No Eligible (%) | 59.25 % | 93.90 % | +34.65 pp | 72.90 % | +13.65 pp |
| IPC Active | 1.49 inst/cycle | 0.21 inst/cycle | -1.28 | 1.27 inst/cycle | -0.22 |

---

## Cumulative Progress

```
Baseline:  3040 ns  (1.000×)
Iter 1     2820 ns  (1.078×)  — fix_tile_shape_and_autotune_for_full_sm_utilization

Final best: 2820 ns  (1.0780×)
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
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 1}, num_warps=1),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 2}, num_warps=1),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 4}, num_warps=1),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 8}, num_warps=1),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 16}, num_warps=2),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 64}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_sub_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
     xnumel = 4
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +32,8 @@

     y3 = yindex
     y0 = (yindex % 64)
     y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +41,7 @@

     ynumel = 8192
     xnumel = 4
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_sub_36[grid](arg_0, arg_1, 8192, 4, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_sub_36[grid](arg_0, arg_1, 8192, 4)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

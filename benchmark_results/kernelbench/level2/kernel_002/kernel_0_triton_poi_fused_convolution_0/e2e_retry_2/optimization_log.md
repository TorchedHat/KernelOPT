# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 3680 ns (**1.6168×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_002/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 16.37 % | **Memory**: 41.69 % | **DRAM**: 20.77 %
- **Achieved Occupancy**: 55.09 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.29 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.327× | 1.617× | ✓ | **✓ best** |
| B | fix_uncoalesced_stores_via_autotune_tile_reshape | 1.188× | 1.051× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_uncoalesced_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3680 ns | -2270 ns (-38.2%) | 5660 ns | -290 ns (-4.9%) |
| SM Throughput (%) | 16.37 % | 25.64 % | +9.27 pp | 17.68 % | +1.31 pp |
| Memory Throughput (%) | 41.69 % | 18.69 % | -23.00 pp | 41.94 % | +0.25 pp |
| DRAM Throughput (%) | 20.77 % | 4.51 % | -16.26 pp | 21.76 % | +0.99 pp |
| Achieved Occupancy (%) | 55.09 % | 65.29 % | +10.20 pp | 55.14 % | +0.05 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.29 % | 42.03 % | -29.26 pp | 68.91 % | -2.38 pp |
| IPC Active | 1.12 inst/cycle | 2.18 inst/cycle | +1.06 | 1.21 inst/cycle | +0.09 |

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 1     3680 ns  (1.617×)  — vectorized_loads_with_autotune

Final best: 3680 ns  (1.6168×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +27,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
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

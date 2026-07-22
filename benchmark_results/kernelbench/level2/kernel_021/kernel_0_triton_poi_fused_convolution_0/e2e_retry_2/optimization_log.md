# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6340 ns — **Best latency**: 3330 ns (**1.9039×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_multiple_of_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_021/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6340 ns
- **SM Throughput**: 16.43 % | **Memory**: 40.06 % | **DRAM**: 19.39 %
- **Achieved Occupancy**: 54.81 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_multiple_of_hints | 1.404× | 1.904× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalesced_access_and_occupancy | 1.371× | 1.585× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6340 ns | 3330 ns | -3010 ns (-47.5%) | 4000 ns | -2340 ns (-36.9%) |
| SM Throughput (%) | 16.43 % | 16.64 % | +0.21 pp | 27.65 % | +11.22 pp |
| Memory Throughput (%) | 40.06 % | 20.57 % | -19.49 pp | 21.94 % | -18.12 pp |
| DRAM Throughput (%) | 19.39 % | 4.97 % | -14.42 pp | 4.13 % | -15.26 pp |
| Achieved Occupancy (%) | 54.81 % | 44.52 % | -10.29 pp | 68.60 % | +13.79 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.49 % | 59.36 % | -11.13 pp | 36.92 % | -33.57 pp |
| IPC Active | 1.15 inst/cycle | 1.64 inst/cycle | +0.49 | 2.38 inst/cycle | +1.23 |

---

## Cumulative Progress

```
Baseline:  6340 ns  (1.000×)
Iter 1     3330 ns  (1.904×)  — vectorized_loads_with_contiguous_multiple_of_hints

Final best: 3330 ns  (1.9039×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +26,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(
+                 tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK),
+                 XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +37,13 @@

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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

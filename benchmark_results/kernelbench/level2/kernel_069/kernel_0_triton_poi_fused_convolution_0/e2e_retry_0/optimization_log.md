# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6180 ns — **Best latency**: 4000 ns (**1.5450×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_069/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6180 ns
- **SM Throughput**: 15.47 % | **Memory**: 39.87 % | **DRAM**: 20.01 %
- **Achieved Occupancy**: 55.22 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.366× | 1.545× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.312× | 1.033× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6180 ns | 4000 ns | -2180 ns (-35.3%) | 5980 ns | -200 ns (-3.2%) |
| SM Throughput (%) | 15.47 % | 27.00 % | +11.53 pp | 13.88 % | -1.59 pp |
| Memory Throughput (%) | 39.87 % | 21.44 % | -18.43 pp | 25.38 % | -14.49 pp |
| DRAM Throughput (%) | 20.01 % | 4.15 % | -15.86 pp | 5.15 % | -14.86 pp |
| Achieved Occupancy (%) | 55.22 % | 68.15 % | +12.93 pp | 43.46 % | -11.76 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 24.00 register/thread | -2 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.76 % | 38.31 % | -33.45 pp | 67.19 % | -4.57 pp |
| IPC Active | 1.15 inst/cycle | 2.41 inst/cycle | +1.26 | 1.21 inst/cycle | +0.06 |

---

## Cumulative Progress

```
Baseline:  6180 ns  (1.000×)
Iter 1     4000 ns  (1.545×)  — vectorized_loads_with_autotune

Final best: 4000 ns  (1.5450×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4,  num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8,  num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +25,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xarange = tl.arange(0, XBLOCK)
+    xarange = tl.max_contiguous(tl.multiple_of(xarange, 16), 16)
+    xindex = xoffset + xarange[None, :]
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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 5920 ns (**1.0270×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_057/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6080 ns
- **SM Throughput**: 15.99 % | **Memory**: 41.72 % | **DRAM**: 20.36 %
- **Achieved Occupancy**: 55.71 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 71.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.313× | 1.027× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_underutilized_transpose_kernel | 1.180× | 1.398× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 5920 ns | -160 ns (-2.6%) | 4350 ns | -1730 ns (-28.5%) |
| SM Throughput (%) | 15.99 % | 16.20 % | +0.21 pp | 21.01 % | +5.02 pp |
| Memory Throughput (%) | 41.72 % | 42.21 % | +0.49 pp | 32.33 % | -9.39 pp |
| DRAM Throughput (%) | 20.36 % | 20.79 % | +0.43 pp | 15.17 % | -5.19 pp |
| Achieved Occupancy (%) | 55.71 % | 54.85 % | -0.86 pp | 48.97 % | -6.74 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 27.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.53 % | 70.18 % | -1.35 pp | 55.59 % | -15.94 pp |
| IPC Active | 1.13 inst/cycle | 1.16 inst/cycle | +0.03 | 1.62 inst/cycle | +0.49 |

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)
Iter 1     5920 ns  (1.027×)  — vectorized_loads_with_contiguous_hints

Final best: 5920 ns  (1.0270×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8),
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
+    # Apply hints to 1D arange first, then reshape to 2D to avoid shape mismatch
+    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xarange[None, :]
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

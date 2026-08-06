# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6050 ns — **Best latency**: 6110 ns (**0.9902×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_021/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6050 ns
- **SM Throughput**: 16.36 % | **Memory**: 41.19 % | **DRAM**: 20.34 %
- **Achieved Occupancy**: 55.25 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.152× | 0.990× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_wave_saturation_and_coalescing | 1.115× | 1.000× | ✓ |  |
| C | add_triton_autotune | 1.013× | 0.979× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 6110 ns | +60 ns (+1.0%) | 6050 ns | +0 ns (+0.0%) | 6180 ns | +130 ns (+2.1%) |
| SM Throughput (%) | 16.36 % | 15.87 % | -0.49 pp | 16.25 % | -0.11 pp | 16.21 % | -0.15 pp |
| Memory Throughput (%) | 41.19 % | 41.84 % | +0.65 pp | 40.79 % | -0.40 pp | 39.69 % | -1.50 pp |
| DRAM Throughput (%) | 20.34 % | 20.17 % | -0.17 pp | 20.45 % | +0.11 pp | 19.92 % | -0.42 pp |
| Achieved Occupancy (%) | 55.25 % | 54.98 % | -0.27 pp | 55.34 % | +0.09 pp | 55.86 % | +0.61 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.99 % | 71.59 % | +0.60 pp | 70.84 % | -0.15 pp | 70.77 % | -0.22 pp |
| IPC Active | 1.15 inst/cycle | 1.16 inst/cycle | +0.01 | 1.15 inst/cycle | +0.00 | 1.12 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  6050 ns  (1.000×)

Final best: 6110 ns  (0.9902×)
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
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8},  num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2},  num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4},  num_warps=16),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +27,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorized load hints on the 1D arange before broadcasting
+    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +38,13 @@

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

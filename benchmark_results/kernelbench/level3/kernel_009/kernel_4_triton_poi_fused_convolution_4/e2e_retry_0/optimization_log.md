# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4740 ns — **Best latency**: 4450 ns (**1.0652×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_and_autotune_launch_configuration`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_4_triton_poi_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3072,1,1), Waves/SM: 1.45
- **Registers/thread**: 28
- **Duration**: 4740 ns
- **SM Throughput**: 33.97 % | **Memory**: 38.52 % | **DRAM**: 10.66 %
- **Achieved Occupancy**: 70.98 %
- **IPC Active**: 2.39 inst/cycle | **No Eligible**: 37.37 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_tile_sizes_and_autotune_launch_configuration | 1.051× | 1.065× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_an... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4740 ns | 4450 ns | -290 ns (-6.1%) |
| SM Throughput (%) | 33.97 % | 15.91 % | -18.06 pp |
| Memory Throughput (%) | 38.52 % | 27.86 % | -10.66 pp |
| DRAM Throughput (%) | 10.66 % | 11.34 % | +0.68 pp |
| Achieved Occupancy (%) | 70.98 % | 48.24 % | -22.74 pp |
| Registers/Thread | 28.00 register/thread | 23.00 register/thread | -5 |
| Waves/SM | 1.45 | 0.73 | -0.72 |
| No Eligible (%) | 37.37 % | 59.87 % | +22.50 pp |
| IPC Active | 2.39 inst/cycle | 1.51 inst/cycle | -0.88 |

---

## Cumulative Progress

```
Baseline:  4740 ns  (1.000×)
Iter 1     4450 ns  (1.065×)  — fix_tile_sizes_and_autotune_launch_configuration

Final best: 4450 ns  (1.0652×)
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
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 8},   num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 24576
     xnumel = 25
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +33,8 @@

     y3 = yindex
     y0 = (yindex % 96)
     y1 = yindex // 96
-    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 96*x2 + 2400*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 96*x2 + 2400*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +42,7 @@

     ynumel = 24576
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_4[grid](arg_0, arg_1, 24576, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_4[grid](arg_0, arg_1, 24576, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

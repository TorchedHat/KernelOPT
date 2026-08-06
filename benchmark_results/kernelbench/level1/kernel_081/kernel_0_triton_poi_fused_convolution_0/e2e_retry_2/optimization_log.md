# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6020 ns — **Best latency**: 4060 ns (**1.4828×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_and_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_081/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6020 ns
- **SM Throughput**: 16.13 % | **Memory**: 41.37 % | **DRAM**: 20.50 %
- **Achieved Occupancy**: 56.08 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 71.71 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_autotune_tile_sizes | 1.263× | 1.483× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_warps_to_hide_l1tex_latency | 1.165× | 1.012× | ✓ |  |
| C | add_triton_autotune | 1.011× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4060 ns | -1960 ns (-32.6%) | 5950 ns | -70 ns (-1.2%) | 5980 ns | -40 ns (-0.7%) |
| SM Throughput (%) | 16.13 % | 17.02 % | +0.89 pp | 16.45 % | +0.32 pp | 16.30 % | +0.17 pp |
| Memory Throughput (%) | 41.37 % | 34.13 % | -7.24 pp | 42.82 % | +1.45 pp | 41.37 % | +0.00 pp |
| DRAM Throughput (%) | 20.50 % | 16.24 % | -4.26 pp | 20.70 % | +0.20 pp | 20.53 % | +0.03 pp |
| Achieved Occupancy (%) | 56.08 % | 54.33 % | -1.75 pp | 55.52 % | -0.56 pp | 55.70 % | -0.38 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.71 % | 68.84 % | -2.87 pp | 70.59 % | -1.12 pp | 70.69 % | -1.02 pp |
| IPC Active | 1.08 inst/cycle | 1.31 inst/cycle | +0.23 | 1.10 inst/cycle | +0.02 | 1.13 inst/cycle | +0.05 |

---

## Cumulative Progress

```
Baseline:  6020 ns  (1.000×)
Iter 1     4060 ns  (1.483×)  — vectorized_loads_and_autotune_tile_sizes

Final best: 4060 ns  (1.4828×)
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
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +26,8 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
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

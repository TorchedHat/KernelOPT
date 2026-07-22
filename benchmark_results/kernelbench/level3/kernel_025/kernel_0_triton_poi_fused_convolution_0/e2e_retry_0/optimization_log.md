# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5760 ns — **Best latency**: 4030 ns (**1.4293×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_025/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5760 ns
- **SM Throughput**: 16.46 % | **Memory**: 42.84 % | **DRAM**: 21.52 %
- **Achieved Occupancy**: 54.82 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.23 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.371× | 1.429× | ✓ | **✓ best** |
| B | autotune_tile_sizes_fix_uncoalesced_stores | 1.089× | 0.919× | ✓ |  |
| C | add_triton_autotune | 0.992× | 0.968× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5760 ns | 4030 ns | -1730 ns (-30.0%) | 6270 ns | +510 ns (+8.9%) | 5950 ns | +190 ns (+3.3%) |
| SM Throughput (%) | 16.46 % | 26.21 % | +9.75 pp | 18.17 % | +1.71 pp | 16.29 % | -0.17 pp |
| Memory Throughput (%) | 42.84 % | 20.82 % | -22.02 pp | 28.09 % | -14.75 pp | 41.18 % | -1.66 pp |
| DRAM Throughput (%) | 21.52 % | 4.10 % | -17.42 pp | 9.84 % | -11.68 pp | 20.67 % | -0.85 pp |
| Achieved Occupancy (%) | 54.82 % | 67.98 % | +13.16 pp | 45.70 % | -9.12 pp | 55.29 % | +0.47 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.23 % | 37.17 % | -34.06 pp | 62.23 % | -9.00 pp | 70.86 % | -0.37 pp |
| IPC Active | 1.15 inst/cycle | 2.36 inst/cycle | +1.21 | 1.42 inst/cycle | +0.27 | 1.14 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  5760 ns  (1.000×)
Iter 1     4030 ns  (1.429×)  — vectorized_loads_with_autotune

Final best: 4030 ns  (1.4293×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8}, num_warps=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +27,8 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex_1d = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xindex_1d[None, :]
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

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6340 ns — **Best latency**: 6210 ns (**1.0209×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_085/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6340 ns
- **SM Throughput**: 16.32 % | **Memory**: 39.62 % | **DRAM**: 19.49 %
- **Achieved Occupancy**: 55.31 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.276× | 1.021× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel | 1.159× | 1.101× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6340 ns | 6210 ns | -130 ns (-2.1%) | 5760 ns | -580 ns (-9.1%) |
| SM Throughput (%) | 16.32 % | 16.18 % | -0.14 pp | 23.95 % | +7.63 pp |
| Memory Throughput (%) | 39.62 % | 40.18 % | +0.56 pp | 35.45 % | -4.17 pp |
| DRAM Throughput (%) | 19.49 % | 19.87 % | +0.38 pp | 11.40 % | -8.09 pp |
| Achieved Occupancy (%) | 55.31 % | 55.98 % | +0.67 pp | 50.91 % | -4.40 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 38.00 register/thread | +12 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.99 | +0.25 |
| No Eligible (%) | 70.44 % | 70.63 % | +0.19 pp | 54.16 % | -16.28 pp |
| IPC Active | 1.15 inst/cycle | 1.11 inst/cycle | -0.04 | 1.80 inst/cycle | +0.65 |

---

## Cumulative Progress

```
Baseline:  6340 ns  (1.000×)
Iter 1     6210 ns  (1.021×)  — vectorized_loads_with_contiguous_hints

Final best: 6210 ns  (1.0209×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +24,8 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +34,13 @@

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

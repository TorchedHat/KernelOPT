# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 3970 ns (**1.4987×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_and_tile_reorientation`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_086/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 16.55 % | **Memory**: 42.50 % | **DRAM**: 20.73 %
- **Achieved Occupancy**: 55.70 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 70.68 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_tile_reorientation | 1.339× | 1.499× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_fix_uncoalesced_stores | 1.157× | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3970 ns | -1980 ns (-33.3%) | 6020 ns | +70 ns (+1.2%) |
| SM Throughput (%) | 16.55 % | 27.53 % | +10.98 pp | 16.61 % | +0.06 pp |
| Memory Throughput (%) | 42.50 % | 21.86 % | -20.64 pp | 41.67 % | -0.83 pp |
| DRAM Throughput (%) | 20.73 % | 4.14 % | -16.59 pp | 20.55 % | -0.18 pp |
| Achieved Occupancy (%) | 55.70 % | 67.60 % | +11.90 pp | 54.89 % | -0.81 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.68 % | 37.04 % | -33.64 pp | 71.51 % | +0.83 pp |
| IPC Active | 1.12 inst/cycle | 2.38 inst/cycle | +1.26 | 1.15 inst/cycle | +0.03 |

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 1     3970 ns  (1.499×)  — vectorized_loads_and_tile_reorientation

Final best: 3970 ns  (1.4987×)
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
@@ -13,7 +27,10 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints to 1D arange before broadcasting
+    xarange = tl.arange(0, XBLOCK)
+    xarange = tl.max_contiguous(tl.multiple_of(xarange, XBLOCK), XBLOCK)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +39,13 @@

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

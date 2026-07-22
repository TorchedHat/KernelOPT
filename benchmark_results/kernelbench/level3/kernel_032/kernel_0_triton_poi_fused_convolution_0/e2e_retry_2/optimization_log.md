# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5790 ns — **Best latency**: 4030 ns (**1.4367×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_032/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5790 ns
- **SM Throughput**: 16.94 % | **Memory**: 42.74 % | **DRAM**: 21.33 %
- **Achieved Occupancy**: 55.15 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.341× | 1.437× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_warps_for_coalescing | 1.269× | 1.485× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 4030 ns | -1760 ns (-30.4%) | 3900 ns | -1890 ns (-32.6%) |
| SM Throughput (%) | 16.94 % | 18.82 % | +1.88 pp | 26.52 % | +9.58 pp |
| Memory Throughput (%) | 42.74 % | 21.71 % | -21.03 pp | 21.11 % | -21.63 pp |
| DRAM Throughput (%) | 21.33 % | 4.10 % | -17.23 pp | 4.25 % | -17.08 pp |
| Achieved Occupancy (%) | 55.15 % | 48.95 % | -6.20 pp | 69.10 % | +13.95 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.34 % | 52.65 % | -18.69 pp | 37.50 % | -33.84 pp |
| IPC Active | 1.15 inst/cycle | 1.86 inst/cycle | +0.71 | 2.39 inst/cycle | +1.24 |

---

## Cumulative Progress

```
Baseline:  5790 ns  (1.000×)
Iter 1     4030 ns  (1.437×)  — vectorized_loads_with_autotune

Final best: 4030 ns  (1.4367×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +29,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints on 1D arange before broadcasting to 2D
+    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +40,13 @@

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

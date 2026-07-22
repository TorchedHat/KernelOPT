# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6530 ns — **Best latency**: 3300 ns (**1.9788×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_fix_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_065/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6530 ns
- **SM Throughput**: 16.67 % | **Memory**: 37.86 % | **DRAM**: 18.86 %
- **Achieved Occupancy**: 55.68 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 72.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune_tile_sizes | 1.238× | 1.407× | ✓ |  |
| B | autotune_tile_sizes_and_fix_coalescing | 1.421× | 1.979× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6530 ns | 4640 ns | -1890 ns (-28.9%) | 3300 ns | -3230 ns (-49.5%) |
| SM Throughput (%) | 16.67 % | 23.55 % | +6.88 pp | 9.60 % | -7.07 pp |
| Memory Throughput (%) | 37.86 % | 30.27 % | -7.59 pp | 21.16 % | -16.70 pp |
| DRAM Throughput (%) | 18.86 % | 14.28 % | -4.58 pp | 5.03 % | -13.83 pp |
| Achieved Occupancy (%) | 55.68 % | 43.92 % | -11.76 pp | 52.74 % | -2.94 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 72.12 % | 52.34 % | -19.78 pp | 69.17 % | -2.95 pp |
| IPC Active | 1.11 inst/cycle | 1.73 inst/cycle | +0.62 | 1.06 inst/cycle | -0.05 |

---

## Cumulative Progress

```
Baseline:  6530 ns  (1.000×)
Iter 1     3300 ns  (1.979×)  — autotune_tile_sizes_and_fix_coalescing

Final best: 3300 ns  (1.9788×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  1}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  2}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  4}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK':  64, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK':  64, 'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK':  32, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK':  32, 'YBLOCK': 16}, num_warps=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
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

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6370 ns — **Best latency**: 3260 ns (**1.9540×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy_and_coalescing`** beats baseline latency.

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
- **Duration**: 6370 ns
- **SM Throughput**: 16.33 % | **Memory**: 38.55 % | **DRAM**: 19.41 %
- **Achieved Occupancy**: 56.00 %
- **IPC Active**: 1.07 inst/cycle | **No Eligible**: 70.57 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints_and_autotune | 1.234× | 1.026× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.374× | 1.954× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6370 ns | 6210 ns | -160 ns (-2.5%) | 3260 ns | -3110 ns (-48.8%) |
| SM Throughput (%) | 16.33 % | 16.38 % | +0.05 pp | 9.69 % | -6.64 pp |
| Memory Throughput (%) | 38.55 % | 39.70 % | +1.15 pp | 21.08 % | -17.47 pp |
| DRAM Throughput (%) | 19.41 % | 19.77 % | +0.36 pp | 5.04 % | -14.37 pp |
| Achieved Occupancy (%) | 56.00 % | 55.32 % | -0.68 pp | 52.12 % | -3.88 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.57 % | 71.02 % | +0.45 pp | 73.26 % | +2.69 pp |
| IPC Active | 1.07 inst/cycle | 1.17 inst/cycle | +0.10 | 1.11 inst/cycle | +0.04 |

---

## Cumulative Progress

```
Baseline:  6370 ns  (1.000×)
Iter 1     3260 ns  (1.954×)  — autotune_tile_sizes_for_occupancy_and_coalescing

Final best: 3260 ns  (1.9540×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
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

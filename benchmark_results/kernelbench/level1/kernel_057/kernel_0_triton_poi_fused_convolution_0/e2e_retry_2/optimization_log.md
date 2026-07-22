# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 3230 ns (**1.8328×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_057/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5920 ns
- **SM Throughput**: 16.25 % | **Memory**: 43.24 % | **DRAM**: 20.85 %
- **Achieved Occupancy**: 55.96 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 70.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.374× | 1.746× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.444× | 1.833× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 3390 ns | -2530 ns (-42.7%) | 3230 ns | -2690 ns (-45.4%) |
| SM Throughput (%) | 16.25 % | 18.29 % | +2.04 pp | 9.94 % | -6.31 pp |
| Memory Throughput (%) | 43.24 % | 20.42 % | -22.82 pp | 10.91 % | -32.33 pp |
| DRAM Throughput (%) | 20.85 % | 4.93 % | -15.92 pp | 2.55 % | -18.30 pp |
| Achieved Occupancy (%) | 55.96 % | 44.77 % | -11.19 pp | 55.31 % | -0.65 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.65 % | 52.60 % | -18.05 pp | 73.78 % | +3.13 pp |
| IPC Active | 1.10 inst/cycle | 1.80 inst/cycle | +0.70 | 1.19 inst/cycle | +0.09 |

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)
Iter 1     3230 ns  (1.833×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3230 ns  (1.8328×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 16}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 32}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +48,13 @@

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

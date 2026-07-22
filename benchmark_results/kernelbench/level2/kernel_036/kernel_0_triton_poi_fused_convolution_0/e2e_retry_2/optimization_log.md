# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5860 ns — **Best latency**: 3170 ns (**1.8486×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_to_fix_uncoalesced_stores_and_low_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_036/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5860 ns
- **SM Throughput**: 16.32 % | **Memory**: 43.39 % | **DRAM**: 21.00 %
- **Achieved Occupancy**: 55.98 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 70.94 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_coalesced_access | 1.073× | 0.709× | ✓ |  |
| B | autotune_tile_sizes_to_fix_uncoalesced_stores_and_low_occupancy | 1.360× | 1.849× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 8260 ns | +2400 ns (+41.0%) | 3170 ns | -2690 ns (-45.9%) |
| SM Throughput (%) | 16.32 % | 42.03 % | +25.71 pp | 4.99 % | -11.33 pp |
| Memory Throughput (%) | 43.39 % | 52.75 % | +9.36 pp | 11.24 % | -32.15 pp |
| DRAM Throughput (%) | 21.00 % | 14.90 % | -6.10 pp | 2.61 % | -18.39 pp |
| Achieved Occupancy (%) | 55.98 % | 78.36 % | +22.38 pp | 25.39 % | -30.59 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.37 | -0.37 |
| No Eligible (%) | 70.94 % | 43.20 % | -27.74 pp | 78.40 % | +7.46 pp |
| IPC Active | 1.11 inst/cycle | 2.26 inst/cycle | +1.15 | 0.63 inst/cycle | -0.48 |

---

## Cumulative Progress

```
Baseline:  5860 ns  (1.000×)
Iter 1     3170 ns  (1.849×)  — autotune_tile_sizes_to_fix_uncoalesced_stores_and_low_occupancy

Final best: 3170 ns  (1.8486×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
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

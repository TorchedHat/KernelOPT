# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5820 ns — **Best latency**: 6620 ns (**0.8792×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5820 ns
- **SM Throughput**: 17.23 % | **Memory**: 42.45 % | **DRAM**: 21.18 %
- **Achieved Occupancy**: 54.37 %
- **IPC Active**: 1.18 inst/cycle | **No Eligible**: 70.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 0.442× | 0.914× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.379× | 0.879× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 6370 ns | +550 ns (+9.5%) | 6620 ns | +800 ns (+13.7%) |
| SM Throughput (%) | 17.23 % | 33.74 % | +16.51 pp | 15.42 % | -1.81 pp |
| Memory Throughput (%) | 42.45 % | 34.25 % | -8.20 pp | 41.77 % | -0.68 pp |
| DRAM Throughput (%) | 21.18 % | 23.17 % | +1.99 pp | 41.77 % | +20.59 pp |
| Achieved Occupancy (%) | 54.37 % | 67.97 % | +13.60 pp | 53.59 % | -0.78 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.34 % | 43.44 % | -26.90 pp | 74.43 % | +4.09 pp |
| IPC Active | 1.18 inst/cycle | 2.20 inst/cycle | +1.02 | 0.98 inst/cycle | -0.20 |

---

## Cumulative Progress

```
Baseline:  5820 ns  (1.000×)

Final best: 6620 ns  (0.8792×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 8}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 4}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 4}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=2, num_stages=4),
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

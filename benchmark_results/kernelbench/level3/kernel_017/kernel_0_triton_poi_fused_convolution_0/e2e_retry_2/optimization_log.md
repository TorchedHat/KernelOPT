# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5860 ns — **Best latency**: 3170 ns (**1.8486×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy_and_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_017/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5860 ns
- **SM Throughput**: 15.91 % | **Memory**: 42.37 % | **DRAM**: 21.06 %
- **Achieved Occupancy**: 55.13 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 71.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.425× | 1.849× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5860 ns | 3170 ns | -2690 ns (-45.9%) |
| SM Throughput (%) | 15.91 % | 5.32 % | -10.59 pp |
| Memory Throughput (%) | 42.37 % | 6.23 % | -36.14 pp |
| DRAM Throughput (%) | 21.06 % | 1.32 % | -19.74 pp |
| Achieved Occupancy (%) | 55.13 % | 28.67 % | -26.46 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 |
| No Eligible (%) | 71.72 % | 77.39 % | +5.67 pp |
| IPC Active | 1.13 inst/cycle | 0.59 inst/cycle | -0.54 |

---

## Cumulative Progress

```
Baseline:  5860 ns  (1.000×)
Iter 1     3170 ns  (1.849×)  — autotune_tile_sizes_for_occupancy_and_coalescing

Final best: 3170 ns  (1.8486×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 1},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 8},  num_warps=2, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +41,13 @@

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

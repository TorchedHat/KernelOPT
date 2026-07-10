# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5700 ns — **Best latency**: 4540 ns (**1.2555×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_num_stages_to_hide_memory_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_043/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5700 ns
- **SM Throughput**: 16.64 % | **Memory**: 43.25 % | **DRAM**: 21.68 %
- **Achieved Occupancy**: 55.37 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_coalesced_stores_via_tile_reorder | 1.237× | 0.990× | ✓ |  |
| B | autotune_tile_sizes_and_num_stages_to_hide_memory_latency | 1.326× | 1.256× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5700 ns | 5760 ns | +60 ns (+1.1%) | 4540 ns | -1160 ns (-20.4%) |
| SM Throughput (%) | 16.64 % | 16.87 % | +0.23 pp | 28.12 % | +11.48 pp |
| Memory Throughput (%) | 43.25 % | 43.67 % | +0.42 pp | 30.06 % | -13.19 pp |
| DRAM Throughput (%) | 21.68 % | 21.41 % | -0.27 pp | 14.54 % | -7.14 pp |
| Achieved Occupancy (%) | 55.37 % | 55.01 % | -0.36 pp | 67.62 % | +12.25 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 22.00 register/thread | -4 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.04 % | 70.63 % | -0.41 pp | 47.23 % | -23.81 pp |
| IPC Active | 1.14 inst/cycle | 1.12 inst/cycle | -0.02 | 2.05 inst/cycle | +0.91 |

---

## Cumulative Progress

```
Baseline:  5700 ns  (1.000×)
Iter 1     4540 ns  (1.256×)  — autotune_tile_sizes_and_num_stages_to_hide_memory_latency

Final best: 4540 ns  (1.2555×)
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
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 4}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=4, num_stages=3),
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

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

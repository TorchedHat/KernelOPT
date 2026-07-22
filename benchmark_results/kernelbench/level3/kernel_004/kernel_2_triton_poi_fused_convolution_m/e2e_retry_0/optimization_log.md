# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_max_pool2d_with_indices_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2340 ns — **Best latency**: 2430 ns (**0.9630×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_max_pool2d_with_indices_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_004/kernel_2_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (12,1,1), Waves/SM: 0.01
- **Registers/thread**: 30
- **Duration**: 2340 ns
- **SM Throughput**: 0.29 % | **Memory**: 0.54 % | **DRAM**: 0.12 %
- **Achieved Occupancy**: 6.19 %
- **IPC Active**: 0.43 inst/cycle | **No Eligible**: 90.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_small_grid_utilization | 1.049× | 0.963× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2340 ns | 2430 ns | +90 ns (+3.8%) |
| SM Throughput (%) | 0.29 % | 0.25 % | -0.04 pp |
| Memory Throughput (%) | 0.54 % | 0.53 % | -0.01 pp |
| DRAM Throughput (%) | 0.12 % | 0.13 % | +0.01 pp |
| Achieved Occupancy (%) | 6.19 % | 3.12 % | -3.07 pp |
| Registers/Thread | 30.00 register/thread | 32.00 register/thread | +2 |
| Waves/SM | 0.01 | 0.00 | -0.01 |
| No Eligible (%) | 90.12 % | 75.42 % | -14.70 pp |
| IPC Active | 0.43 inst/cycle | 0.50 inst/cycle | +0.07 |

---

## Cumulative Progress

```
Baseline:  2340 ns  (1.000×)

Final best: 2430 ns  (0.9630×)
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
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32}, num_warps=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 8},  num_warps=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_max_pool2d_with_indices_relu_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 96
@@ -22,15 +36,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 6*x2 + 150*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 96
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_max_pool2d_with_indices_relu_2[grid](arg_0, arg_1, 96, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_max_pool2d_with_indices_relu_2[grid](arg_0, arg_1, 96, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6140 ns — **Best latency**: 5790 ns (**1.0604×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`add_autotune_and_fix_tile_sizes_for_coalesced_access`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_069/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6140 ns
- **SM Throughput**: 16.38 % | **Memory**: 40.14 % | **DRAM**: 20.09 %
- **Achieved Occupancy**: 55.79 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 71.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.012× | 1.015× | ✓ |  |
| B | add_autotune_and_fix_tile_sizes_for_coalesced_access | 1.103× | 1.060× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | add_autotune_and_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 6050 ns | -90 ns (-1.5%) | 5790 ns | -350 ns (-5.7%) |
| SM Throughput (%) | 16.38 % | 16.47 % | +0.09 pp | 12.68 % | -3.70 pp |
| Memory Throughput (%) | 40.14 % | 41.60 % | +1.46 pp | 27.41 % | -12.73 pp |
| DRAM Throughput (%) | 20.09 % | 20.47 % | +0.38 pp | 5.34 % | -14.75 pp |
| Achieved Occupancy (%) | 55.79 % | 55.13 % | -0.66 pp | 21.77 % | -34.02 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 71.43 % | 72.17 % | +0.74 pp | 70.87 % | -0.56 pp |
| IPC Active | 1.10 inst/cycle | 1.17 inst/cycle | +0.07 | 1.03 inst/cycle | -0.07 |

---

## Cumulative Progress

```
Baseline:  6140 ns  (1.000×)
Iter 1     5790 ns  (1.060×)  — add_autotune_and_fix_tile_sizes_for_coalesced_access

Final best: 5790 ns  (1.0604×)
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
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8},  num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +36,13 @@

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

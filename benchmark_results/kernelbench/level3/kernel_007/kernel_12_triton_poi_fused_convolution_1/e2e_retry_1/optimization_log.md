# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3780 ns — **Best latency**: 2820 ns (**1.3404×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_eliminate_thread_waste`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_12_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1536,1,1), Waves/SM: 0.73
- **Registers/thread**: 28
- **Duration**: 3780 ns
- **SM Throughput**: 22.86 % | **Memory**: 21.04 % | **DRAM**: 2.43 %
- **Achieved Occupancy**: 53.90 %
- **IPC Active**: 2.09 inst/cycle | **No Eligible**: 45.91 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_small_x_dimension | 1.071× | 1.181× | ✓ |  |
| B | fix_tile_sizes_eliminate_thread_waste | 1.150× | 1.340× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | fix_tile_sizes_el... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3780 ns | 3200 ns | -580 ns (-15.3%) | 2820 ns | -960 ns (-25.4%) |
| SM Throughput (%) | 22.86 % | 14.80 % | -8.06 pp | 4.87 % | -17.99 pp |
| Memory Throughput (%) | 21.04 % | 13.47 % | -7.57 pp | 3.25 % | -17.79 pp |
| DRAM Throughput (%) | 2.43 % | 2.87 % | +0.44 pp | 3.25 % | +0.82 pp |
| Achieved Occupancy (%) | 53.90 % | 26.30 % | -27.60 pp | 16.19 % | -37.71 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 16.00 register/thread | -12 |
| Waves/SM | 0.73 | 0.36 | -0.37 | 0.36 | -0.37 |
| No Eligible (%) | 45.91 % | 49.61 % | +3.70 pp | 81.36 % | +35.45 pp |
| IPC Active | 2.09 inst/cycle | 1.78 inst/cycle | -0.31 | 0.54 inst/cycle | -1.55 |

---

## Cumulative Progress

```
Baseline:  3780 ns  (1.000×)
Iter 1     2820 ns  (1.340×)  — fix_tile_sizes_eliminate_thread_waste

Final best: 2820 ns  (1.3404×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 12288
@@ -22,14 +44,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 12288
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6020 ns — **Best latency**: 3390 ns (**1.7758×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_052/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6020 ns
- **SM Throughput**: 15.77 % | **Memory**: 41.22 % | **DRAM**: 20.56 %
- **Achieved Occupancy**: 55.83 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 72.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune_tile_sizes | 1.385× | 1.776× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.272× | 1.034× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3390 ns | -2630 ns (-43.7%) | 5820 ns | -200 ns (-3.3%) |
| SM Throughput (%) | 15.77 % | 16.85 % | +1.08 pp | 16.59 % | +0.82 pp |
| Memory Throughput (%) | 41.22 % | 20.44 % | -20.78 pp | 42.12 % | +0.90 pp |
| DRAM Throughput (%) | 20.56 % | 4.93 % | -15.63 pp | 21.05 % | +0.49 pp |
| Achieved Occupancy (%) | 55.83 % | 43.97 % | -11.86 pp | 55.62 % | -0.21 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 72.11 % | 55.29 % | -16.82 pp | 71.83 % | -0.28 pp |
| IPC Active | 1.11 inst/cycle | 1.66 inst/cycle | +0.55 | 1.10 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  6020 ns  (1.000×)
Iter 1     3390 ns  (1.776×)  — vectorized_loads_with_autotune_tile_sizes

Final best: 3390 ns  (1.7758×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -19,10 +31,9 @@

     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    x2_vec = tl.max_contiguous(tl.multiple_of(x2, [1, XBLOCK]), [1, XBLOCK])
+    tmp0 = tl.load(in_ptr0 + (x2_vec + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +41,7 @@

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

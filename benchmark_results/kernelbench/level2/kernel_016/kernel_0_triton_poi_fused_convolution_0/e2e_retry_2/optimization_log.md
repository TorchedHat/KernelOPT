# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5860 ns — **Best latency**: 6020 ns (**0.9734×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_016/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5860 ns
- **SM Throughput**: 17.06 % | **Memory**: 42.06 % | **DRAM**: 21.10 %
- **Achieved Occupancy**: 55.36 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 72.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.003× | 1.012× | ✓ |  |
| B | autotune_tile_sizes_and_fix_uncoalesced_stores | 1.056× | 0.973× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 5790 ns | -70 ns (-1.2%) | 6020 ns | +160 ns (+2.7%) |
| SM Throughput (%) | 17.06 % | 15.36 % | -1.70 pp | 16.57 % | -0.49 pp |
| Memory Throughput (%) | 42.06 % | 42.27 % | +0.21 pp | 41.06 % | -1.00 pp |
| DRAM Throughput (%) | 21.10 % | 21.28 % | +0.18 pp | 20.50 % | -0.60 pp |
| Achieved Occupancy (%) | 55.36 % | 55.40 % | +0.04 pp | 55.99 % | +0.63 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 72.89 % | 71.23 % | -1.66 pp | 70.46 % | -2.43 pp |
| IPC Active | 1.12 inst/cycle | 1.11 inst/cycle | -0.01 | 1.05 inst/cycle | -0.07 |

---

## Cumulative Progress

```
Baseline:  5860 ns  (1.000×)

Final best: 6020 ns  (0.9734×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=4),
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

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

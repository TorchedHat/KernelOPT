# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5860 ns — **Best latency**: 5860 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_005/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5860 ns
- **SM Throughput**: 15.24 % | **Memory**: 42.30 % | **DRAM**: 21.16 %
- **Achieved Occupancy**: 54.65 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 0.855× | 0.980× | ✓ |  |
| B | autotune_tile_sizes_to_fix_uncoalesced_stores_and_occupancy | 1.274× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 5980 ns | +120 ns (+2.0%) | 5860 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 15.24 % | 23.35 % | +8.11 pp | 16.72 % | +1.48 pp |
| Memory Throughput (%) | 42.30 % | 34.37 % | -7.93 pp | 41.78 % | -0.52 pp |
| DRAM Throughput (%) | 21.16 % | 11.00 % | -10.16 pp | 21.07 % | -0.09 pp |
| Achieved Occupancy (%) | 54.65 % | 51.32 % | -3.33 pp | 54.83 % | +0.18 pp |
| Registers/Thread | 26.00 register/thread | 38.00 register/thread | +12 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.99 | +0.25 | 0.74 | +0.00 |
| No Eligible (%) | 71.07 % | 54.89 % | -16.18 pp | 70.87 % | -0.20 pp |
| IPC Active | 1.14 inst/cycle | 1.82 inst/cycle | +0.68 | 1.16 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  5860 ns  (1.000×)

Final best: 5860 ns  (1.0000×)
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
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=4),
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

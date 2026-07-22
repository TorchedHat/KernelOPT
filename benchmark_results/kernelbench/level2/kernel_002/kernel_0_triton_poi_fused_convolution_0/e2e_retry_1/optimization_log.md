# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6270 ns — **Best latency**: 4260 ns (**1.4718×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalesced_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_002/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6270 ns
- **SM Throughput**: 16.66 % | **Memory**: 39.64 % | **DRAM**: 19.70 %
- **Achieved Occupancy**: 54.59 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.22 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints_and_larger_xblock | 0.990× | 1.010× | ✓ |  |
| B | autotune_tile_sizes_for_coalesced_stores | 1.221× | 1.472× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6270 ns | 6210 ns | -60 ns (-1.0%) | 4260 ns | -2010 ns (-32.1%) |
| SM Throughput (%) | 16.66 % | 16.84 % | +0.18 pp | 17.09 % | +0.43 pp |
| Memory Throughput (%) | 39.64 % | 39.52 % | -0.12 pp | 32.27 % | -7.37 pp |
| DRAM Throughput (%) | 19.70 % | 19.86 % | +0.16 pp | 15.51 % | -4.19 pp |
| Achieved Occupancy (%) | 54.59 % | 54.87 % | +0.28 pp | 53.35 % | -1.24 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 21.00 register/thread | -5 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.22 % | 70.73 % | +0.51 pp | 64.70 % | -5.52 pp |
| IPC Active | 1.14 inst/cycle | 1.15 inst/cycle | +0.01 | 1.35 inst/cycle | +0.21 |

---

## Cumulative Progress

```
Baseline:  6270 ns  (1.000×)
Iter 1     4260 ns  (1.472×)  — autotune_tile_sizes_for_coalesced_stores

Final best: 4260 ns  (1.4718×)
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
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 8},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 32}, num_warps=8, num_stages=4),
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

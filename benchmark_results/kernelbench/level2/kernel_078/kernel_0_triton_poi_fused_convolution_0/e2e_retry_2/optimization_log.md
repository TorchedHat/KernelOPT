# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5860 ns — **Best latency**: 3140 ns (**1.8662×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_078/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5860 ns
- **SM Throughput**: 16.43 % | **Memory**: 42.48 % | **DRAM**: 21.04 %
- **Achieved Occupancy**: 55.47 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 70.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.401× | 0.954× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.409× | 1.866× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.004× | 0.930× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 6140 ns | +280 ns (+4.8%) | 3140 ns | -2720 ns (-46.4%) | 6300 ns | +440 ns (+7.5%) |
| SM Throughput (%) | 16.43 % | 16.92 % | +0.49 pp | 15.01 % | -1.42 pp | 16.09 % | -0.34 pp |
| Memory Throughput (%) | 42.48 % | 40.78 % | -1.70 pp | 21.78 % | -20.70 pp | 38.90 % | -3.58 pp |
| DRAM Throughput (%) | 21.04 % | 20.14 % | -0.90 pp | 5.27 % | -15.77 pp | 19.64 % | -1.40 pp |
| Achieved Occupancy (%) | 55.47 % | 54.74 % | -0.73 pp | 46.75 % | -8.72 pp | 55.15 % | -0.32 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 16.00 register/thread | -10 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.82 % | 71.39 % | +0.57 pp | 61.45 % | -9.37 pp | 71.95 % | +1.13 pp |
| IPC Active | 1.12 inst/cycle | 1.15 inst/cycle | +0.03 | 1.46 inst/cycle | +0.34 | 1.12 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  5860 ns  (1.000×)
Iter 1     3140 ns  (1.866×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3140 ns  (1.8662×)
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
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=2),
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

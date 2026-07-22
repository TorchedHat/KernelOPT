# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5790 ns — **Best latency**: 3360 ns (**1.7232×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps_for_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_057/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5790 ns
- **SM Throughput**: 16.31 % | **Memory**: 43.18 % | **DRAM**: 21.31 %
- **Achieved Occupancy**: 56.20 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 70.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_autotune_tile_sizes | 1.281× | 1.382× | ✓ |  |
| B | autotune_tile_sizes_and_warps_for_occupancy | 1.318× | 1.723× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 4190 ns | -1600 ns (-27.6%) | 3360 ns | -2430 ns (-42.0%) |
| SM Throughput (%) | 16.31 % | 17.11 % | +0.80 pp | 16.97 % | +0.66 pp |
| Memory Throughput (%) | 43.18 % | 20.88 % | -22.30 pp | 20.51 % | -22.67 pp |
| DRAM Throughput (%) | 21.31 % | 3.95 % | -17.36 pp | 4.94 % | -16.37 pp |
| Achieved Occupancy (%) | 56.20 % | 49.61 % | -6.59 pp | 43.87 % | -12.33 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.33 % | 55.47 % | -14.86 pp | 54.06 % | -16.27 pp |
| IPC Active | 1.11 inst/cycle | 1.71 inst/cycle | +0.60 | 1.71 inst/cycle | +0.60 |

---

## Cumulative Progress

```
Baseline:  5790 ns  (1.000×)
Iter 1     3360 ns  (1.723×)  — autotune_tile_sizes_and_warps_for_occupancy

Final best: 3360 ns  (1.7232×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=4),
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

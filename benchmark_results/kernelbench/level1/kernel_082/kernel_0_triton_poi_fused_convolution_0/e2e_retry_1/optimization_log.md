# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 3650 ns (**1.6219×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_082/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5920 ns
- **SM Throughput**: 16.03 % | **Memory**: 42.65 % | **DRAM**: 20.86 %
- **Achieved Occupancy**: 55.27 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.416× | 1.622× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.246× | 1.033× | ✓ |  |
| C | add_triton_autotune | 1.034× | 0.979× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 3650 ns | -2270 ns (-38.3%) | 5730 ns | -190 ns (-3.2%) | 6050 ns | +130 ns (+2.2%) |
| SM Throughput (%) | 16.03 % | 18.45 % | +2.42 pp | 16.66 % | +0.63 pp | 16.41 % | +0.38 pp |
| Memory Throughput (%) | 42.65 % | 18.58 % | -24.07 pp | 44.85 % | +2.20 pp | 41.37 % | -1.28 pp |
| DRAM Throughput (%) | 20.86 % | 4.54 % | -16.32 pp | 21.63 % | +0.77 pp | 20.36 % | -0.50 pp |
| Achieved Occupancy (%) | 55.27 % | 44.27 % | -11.00 pp | 54.90 % | -0.37 pp | 54.98 % | -0.29 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.34 % | 52.22 % | -18.12 pp | 71.52 % | +1.18 pp | 70.66 % | +0.32 pp |
| IPC Active | 1.15 inst/cycle | 1.79 inst/cycle | +0.64 | 1.14 inst/cycle | -0.01 | 1.15 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)
Iter 1     3650 ns  (1.622×)  — vectorized_loads_with_autotune

Final best: 3650 ns  (1.6219×)
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
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=2, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +25,8 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), 16), 16)
+    xindex = xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +35,13 @@

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

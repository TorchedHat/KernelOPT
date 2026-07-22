# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 3420 ns (**1.7398×** vs baseline)
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
- **Duration**: 5950 ns
- **SM Throughput**: 16.03 % | **Memory**: 42.89 % | **DRAM**: 20.66 %
- **Achieved Occupancy**: 55.93 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 71.79 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.440× | 1.740× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_fix_uncoalesced_stores_via_evict_first | 1.359× | 1.255× | ✓ |  |
| C | add_triton_autotune | 0.970× | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3420 ns | -2530 ns (-42.5%) | 4740 ns | -1210 ns (-20.3%) | 6020 ns | +70 ns (+1.2%) |
| SM Throughput (%) | 16.03 % | 9.55 % | -6.48 pp | 21.66 % | +5.63 pp | 16.16 % | +0.13 pp |
| Memory Throughput (%) | 42.89 % | 20.21 % | -22.68 pp | 29.44 % | -13.45 pp | 41.50 % | -1.39 pp |
| DRAM Throughput (%) | 20.66 % | 4.84 % | -15.82 pp | 13.91 % | -6.75 pp | 20.60 % | -0.06 pp |
| Achieved Occupancy (%) | 55.93 % | 50.29 % | -5.64 pp | 47.43 % | -8.50 pp | 55.95 % | +0.02 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 27.00 register/thread | +1 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.79 % | 69.80 % | -1.99 pp | 54.90 % | -16.89 pp | 72.60 % | +0.81 pp |
| IPC Active | 1.10 inst/cycle | 1.04 inst/cycle | -0.06 | 1.76 inst/cycle | +0.66 | 1.06 inst/cycle | -0.04 |

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 1     3420 ns  (1.740×)  — vectorized_loads_with_autotune

Final best: 3420 ns  (1.7398×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +27,8 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), 16), 16)
+    xindex = xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +37,13 @@

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

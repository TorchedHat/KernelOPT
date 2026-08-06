# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3870 ns — **Best latency**: 3620 ns (**1.0691×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_with_autotune_for_small_xnumel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_8`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_8_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 0.97
- **Registers/thread**: 28
- **Duration**: 3870 ns
- **SM Throughput**: 27.55 % | **Memory**: 26.38 % | **DRAM**: 3.15 %
- **Achieved Occupancy**: 65.35 %
- **IPC Active**: 2.08 inst/cycle | **No Eligible**: 40.30 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_with_autotune_for_small_xnumel | — | 1.069× | ✓ | **✓ best** |
| B | optimize_tile_sizes_with_autotune | — | 1.109× | ✓ |  |
| C | add_triton_autotune | — | 0.975× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_wi... | Δ | optimize_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3620 ns | -250 ns (-6.5%) | 3490 ns | -380 ns (-9.8%) | 3970 ns | +100 ns (+2.6%) |
| SM Throughput (%) | 27.55 % | 17.74 % | -9.81 pp | 17.67 % | -9.88 pp | 27.55 % | +0.00 pp |
| Memory Throughput (%) | 26.38 % | 16.50 % | -9.88 pp | 16.47 % | -9.91 pp | 26.36 % | -0.02 pp |
| DRAM Throughput (%) | 3.15 % | 3.39 % | +0.24 pp | 3.51 % | +0.36 pp | 3.08 % | -0.07 pp |
| Achieved Occupancy (%) | 65.35 % | 36.68 % | -28.67 pp | 35.96 % | -29.39 pp | 65.68 % | +0.33 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 30.00 register/thread | +2 | 28.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.48 | -0.49 | 0.48 | -0.49 | 0.97 | +0.00 |
| No Eligible (%) | 40.30 % | 43.56 % | +3.26 pp | 44.25 % | +3.95 pp | 43.15 % | +2.85 pp |
| IPC Active | 2.08 inst/cycle | 2.00 inst/cycle | -0.08 | 2.07 inst/cycle | -0.01 | 2.26 inst/cycle | +0.18 |

---

## Cumulative Progress

```
Baseline:  3870 ns  (1.000×)
Iter 1     3620 ns  (1.069×)  — fix_tile_sizes_with_autotune_for_small_xnumel

Final best: 3620 ns  (1.0691×)
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
+        triton.Config({'YBLOCK': 32,  'XBLOCK': 16}, num_warps=2),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 16}, num_warps=4),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4),
+        triton.Config({'YBLOCK': 256, 'XBLOCK': 16}, num_warps=8),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 16}, num_warps=2),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_sub_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 16384
@@ -22,14 +34,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 16384
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_sub_8[grid](arg_0, arg_1, 16384, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_sub_8[grid](arg_0, arg_1, 16384, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5790 ns — **Best latency**: 6400 ns (**0.9047×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5790 ns
- **SM Throughput**: 16.74 % | **Memory**: 42.43 % | **DRAM**: 21.32 %
- **Achieved Occupancy**: 55.26 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 70.38 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.241× | 1.084× | ✓ |  |
| B | fix_uncoalesced_stores_via_autotune_tile_swap | 1.273× | 0.905× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.001× | 0.973× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_uncoalesced_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 5340 ns | -450 ns (-7.8%) | 6400 ns | +610 ns (+10.5%) | 5950 ns | +160 ns (+2.8%) |
| SM Throughput (%) | 16.74 % | 35.57 % | +18.83 pp | 19.73 % | +2.99 pp | 16.97 % | +0.23 pp |
| Memory Throughput (%) | 42.43 % | 36.35 % | -6.08 pp | 36.36 % | -6.07 pp | 42.57 % | +0.14 pp |
| DRAM Throughput (%) | 21.32 % | 12.26 % | -9.06 pp | 4.85 % | -16.47 pp | 20.84 % | -0.48 pp |
| Achieved Occupancy (%) | 55.26 % | 74.73 % | +19.47 pp | 56.06 % | +0.80 pp | 55.78 % | +0.52 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 32.00 register/thread | +6 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.38 % | 38.05 % | -32.33 pp | 66.09 % | -4.29 pp | 70.61 % | +0.23 pp |
| IPC Active | 1.11 inst/cycle | 2.45 inst/cycle | +1.34 | 1.51 inst/cycle | +0.40 | 1.11 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  5790 ns  (1.000×)

Final best: 6400 ns  (0.9047×)
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
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=8),
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

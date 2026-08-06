# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5760 ns — **Best latency**: 3460 ns (**1.6647×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_fix_uncoalesced_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_031/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5760 ns
- **SM Throughput**: 16.30 % | **Memory**: 42.47 % | **DRAM**: 21.31 %
- **Achieved Occupancy**: 55.94 %
- **IPC Active**: 1.09 inst/cycle | **No Eligible**: 71.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints_and_autotune | 1.158× | 1.000× | ✓ |  |
| B | autotune_tile_sizes_and_fix_uncoalesced_stores | 1.349× | 1.665× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.019× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5760 ns | 5760 ns | +0 ns (+0.0%) | 3460 ns | -2300 ns (-39.9%) | 5820 ns | +60 ns (+1.0%) |
| SM Throughput (%) | 16.30 % | 15.60 % | -0.70 pp | 10.27 % | -6.03 pp | 16.67 % | +0.37 pp |
| Memory Throughput (%) | 42.47 % | 41.62 % | -0.85 pp | 20.81 % | -21.66 pp | 41.14 % | -1.33 pp |
| DRAM Throughput (%) | 21.31 % | 21.47 % | +0.16 pp | 9.55 % | -11.76 pp | 21.15 % | -0.16 pp |
| Achieved Occupancy (%) | 55.94 % | 54.82 % | -1.12 pp | 29.15 % | -26.79 pp | 55.17 % | -0.77 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 21.00 register/thread | -5 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 0.74 | +0.00 |
| No Eligible (%) | 71.70 % | 71.53 % | -0.17 pp | 70.52 % | -1.18 pp | 70.55 % | -1.15 pp |
| IPC Active | 1.09 inst/cycle | 1.14 inst/cycle | +0.05 | 1.08 inst/cycle | -0.01 | 1.15 inst/cycle | +0.06 |

---

## Cumulative Progress

```
Baseline:  5760 ns  (1.000×)
Iter 1     3460 ns  (1.665×)  — autotune_tile_sizes_and_fix_uncoalesced_stores

Final best: 3460 ns  (1.6647×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
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

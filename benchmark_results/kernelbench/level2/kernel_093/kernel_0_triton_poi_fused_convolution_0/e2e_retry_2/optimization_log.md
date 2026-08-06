# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5980 ns — **Best latency**: 3460 ns (**1.7283×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_093/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5980 ns
- **SM Throughput**: 16.61 % | **Memory**: 42.19 % | **DRAM**: 20.62 %
- **Achieved Occupancy**: 55.40 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.005× | 0.984× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.349× | 1.728× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 6080 ns | +100 ns (+1.7%) | 3460 ns | -2520 ns (-42.1%) |
| SM Throughput (%) | 16.61 % | 16.70 % | +0.09 pp | 10.24 % | -6.37 pp |
| Memory Throughput (%) | 42.19 % | 41.60 % | -0.59 pp | 21.00 % | -21.19 pp |
| DRAM Throughput (%) | 20.62 % | 20.36 % | -0.26 pp | 9.53 % | -11.09 pp |
| Achieved Occupancy (%) | 55.40 % | 55.32 % | -0.08 pp | 29.29 % | -26.11 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 21.00 register/thread | -5 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 71.04 % | 72.27 % | +1.23 pp | 75.63 % | +4.59 pp |
| IPC Active | 1.12 inst/cycle | 1.14 inst/cycle | +0.02 | 1.06 inst/cycle | -0.06 |

---

## Cumulative Progress

```
Baseline:  5980 ns  (1.000×)
Iter 1     3460 ns  (1.728×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3460 ns  (1.7283×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=2),
+    triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4),
+    triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
+    triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 32}, num_warps=16),
+], key=['ynumel', 'xnumel'])
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

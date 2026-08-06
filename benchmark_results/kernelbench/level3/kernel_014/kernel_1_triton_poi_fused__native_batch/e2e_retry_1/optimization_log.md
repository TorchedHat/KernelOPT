# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2750 ns — **Best latency**: 2980 ns (**0.9228×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.06
- **Registers/thread**: 28
- **Duration**: 2750 ns
- **SM Throughput**: 2.68 % | **Memory**: 2.50 % | **DRAM**: 0.31 %
- **Achieved Occupancy**: 6.08 %
- **IPC Active**: 0.33 inst/cycle | **No Eligible**: 92.46 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotuned_block_sizes | 0.960× | 0.876× | ✓ |  |
| B | autotune_tile_sizes_for_small_transpose_kernel | 1.005× | 0.923× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2750 ns | 3140 ns | +390 ns (+14.2%) | 2980 ns | +230 ns (+8.4%) |
| SM Throughput (%) | 2.68 % | 0.74 % | -1.94 pp | 3.51 % | +0.83 pp |
| Memory Throughput (%) | 2.50 % | 1.01 % | -1.49 pp | 3.52 % | +1.02 pp |
| DRAM Throughput (%) | 0.31 % | 0.26 % | -0.05 pp | 2.08 % | +1.77 pp |
| Achieved Occupancy (%) | 6.08 % | 1.56 % | -4.52 pp | 6.07 % | -0.01 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 27.00 register/thread | -1 |
| Waves/SM | 0.06 | 0.03 | -0.03 | 0.06 | +0.00 |
| No Eligible (%) | 92.46 % | 92.12 % | -0.34 pp | 89.80 % | -2.66 pp |
| IPC Active | 0.33 inst/cycle | 0.08 inst/cycle | -0.25 | 0.41 inst/cycle | +0.08 |

---

## Cumulative Progress

```
Baseline:  2750 ns  (1.000×)

Final best: 2980 ns  (0.9228×)
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
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  16}, num_warps=2),
+        triton.Config({'XBLOCK':  4, 'YBLOCK': 256}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1024
@@ -19,10 +31,8 @@

     y3 = yindex
     y0 = (yindex % 32)
     y1 = yindex // 32
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=xmask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +40,7 @@

     ynumel = 1024
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1[grid](arg_0, arg_1, 1024, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1[grid](arg_0, arg_1, 1024, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

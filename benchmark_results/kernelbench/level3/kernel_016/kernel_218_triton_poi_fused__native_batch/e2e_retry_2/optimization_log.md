# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_218` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6720 ns — **Best latency**: 6690 ns (**1.0045×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_tile_size_to_eliminate_partial_wave`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_218`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_218_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6493,1,1), Waves/SM: 3.07
- **Registers/thread**: 22
- **Duration**: 6720 ns
- **SM Throughput**: 42.26 % | **Memory**: 16.16 % | **DRAM**: 10.24 %
- **Achieved Occupancy**: 62.23 %
- **IPC Active**: 2.54 inst/cycle | **No Eligible**: 35.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size_to_eliminate_partial_wave | 1.001× | 1.004× | ✓ | **✓ best** |
| B | autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.939× | 0.798× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6720 ns | 6690 ns | -30 ns (-0.4%) | 8420 ns | +1700 ns (+25.3%) |
| SM Throughput (%) | 42.26 % | 42.17 % | -0.09 pp | 63.02 % | +20.76 pp |
| Memory Throughput (%) | 16.16 % | 16.12 % | -0.04 pp | 17.20 % | +1.04 pp |
| DRAM Throughput (%) | 10.24 % | 10.31 % | +0.07 pp | 8.15 % | -2.09 pp |
| Achieved Occupancy (%) | 62.23 % | 62.43 % | +0.20 pp | 79.76 % | +17.53 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 |
| Waves/SM | 3.07 | 3.07 | +0.00 | 3.07 | +0.00 |
| No Eligible (%) | 35.20 % | 35.51 % | +0.31 pp | 16.82 % | -18.38 pp |
| IPC Active | 2.54 inst/cycle | 2.57 inst/cycle | +0.03 | 3.31 inst/cycle | +0.77 |

---

## Cumulative Progress

```
Baseline:  6720 ns  (1.000×)
Iter 1     6690 ns  (1.004×)  — autotune_xblock_tile_size_to_eliminate_partial_wave

Final best: 6690 ns  (1.0045×)
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
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_218(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 831040
@@ -33,8 +45,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -44,7 +54,7 @@

     arg_5 = inputs[5]
     xnumel = 831040
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_218[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 831040, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_218[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 831040)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

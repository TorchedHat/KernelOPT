# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3550 ns — **Best latency**: 5020 ns (**0.7072×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_30_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1715,1,1), Waves/SM: 0.81
- **Registers/thread**: 18
- **Duration**: 3550 ns
- **SM Throughput**: 12.64 % | **Memory**: 8.31 % | **DRAM**: 5.14 %
- **Achieved Occupancy**: 59.16 %
- **IPC Active**: 1.32 inst/cycle | **No Eligible**: 63.63 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_better_sm_utilization | 0.972× | 1.038× | ✓ |  |
| B | autotune_xblock_num_warps_for_underutilized_pointwise_kernel | 1.023× | 0.707× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3550 ns | 3420 ns | -130 ns (-3.7%) | 5020 ns | +1470 ns (+41.4%) |
| SM Throughput (%) | 12.64 % | 18.43 % | +5.79 pp | 44.56 % | +31.92 pp |
| Memory Throughput (%) | 8.31 % | 12.09 % | +3.78 pp | 20.62 % | +12.31 pp |
| DRAM Throughput (%) | 5.14 % | 5.32 % | +0.18 pp | 3.60 % | -1.54 pp |
| Achieved Occupancy (%) | 59.16 % | 58.06 % | -1.10 pp | 56.69 % | -2.47 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 34.00 register/thread | +16 |
| Waves/SM | 0.81 | 0.81 | +0.00 | 2.17 | +1.36 |
| No Eligible (%) | 63.63 % | 43.99 % | -19.64 pp | 29.15 % | -34.48 pp |
| IPC Active | 1.32 inst/cycle | 2.14 inst/cycle | +0.82 | 2.71 inst/cycle | +1.39 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_num_warps_for_underutilized_pointwise_kernel` (0.707×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) selected XBLOCK=256 at runtime, which nearly doubled register usage per thread (18→34 registers/thread), reducing Block Limit Registers from 21 to 6 and capping theoretical occupancy at 75% instead of 100%. Although compute throughput improved (12.6%→44.6%) and warp stall cycles dropped (31.25→12.62 cycles/instruction), the total executed instruction count exploded 5x (349K→1.74M) and elapsed cycles increased 42% (5242→7456), resulting in a 29% slowdown. The autotune selected a config that traded occupancy headroom for per-CTA work, but the register pressure increase was the dominant negative effect.

> **Applicable when**: Kernel has very low waves/SM (<1.0) and high scheduler no-eligible % (>60%) due to small grid size, AND register usage is already near the per-thread limit (close to 32 registers/thread), AND the kernel body is simple enough that larger XBLOCK would not cause register spill.

> **Avoid when**: Kernel already uses 18+ registers/thread in a simple pointwise body — increasing XBLOCK via autotune may cause the compiler to allocate significantly more registers (e.g., 18→34), reducing Block Limit Registers and capping theoretical occupancy below 100%, which can outweigh the benefit of fewer CTAs. Also avoid when the grid is already fixed and the autotune cannot change the actual launch grid (e.g., grid lambda not updated to use meta['XBLOCK']).

---

## Cumulative Progress

```
Baseline:  3550 ns  (1.000×)

Final best: 5020 ns  (0.7072×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 219520
@@ -25,7 +38,6 @@

     tmp15 = tmp11 * tmp12 + tmp14
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -34,7 +46,7 @@

     arg_4 = inputs[4]
     xnumel = 219520
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 219520, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 219520)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3070 ns — **Best latency**: 3100 ns (**0.9903×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_13`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_13_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (784,1,1), Waves/SM: 0.37
- **Registers/thread**: 18
- **Duration**: 3070 ns
- **SM Throughput**: 10.45 % | **Memory**: 4.32 % | **DRAM**: 2.74 %
- **Achieved Occupancy**: 31.29 %
- **IPC Active**: 0.99 inst/cycle | **No Eligible**: 69.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 1.028× | 0.990× | ✓ | **✓ best** |
| B | autotune_xblock_tile_size_for_occupancy | 1.027× | 0.615× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 3100 ns | +30 ns (+1.0%) | 4990 ns | +1920 ns (+62.5%) |
| SM Throughput (%) | 10.45 % | 14.07 % | +3.62 pp | 44.83 % | +34.38 pp |
| Memory Throughput (%) | 4.32 % | 4.77 % | +0.45 pp | 9.38 % | +5.06 pp |
| DRAM Throughput (%) | 2.74 % | 2.72 % | -0.02 pp | 1.69 % | -1.05 pp |
| Achieved Occupancy (%) | 31.29 % | 26.39 % | -4.90 pp | 46.31 % | +15.02 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 48.00 register/thread | +30 |
| Waves/SM | 0.37 | 0.37 | +0.00 | 1.19 | +0.82 |
| No Eligible (%) | 69.80 % | 61.27 % | -8.53 pp | 24.37 % | -45.43 pp |
| IPC Active | 0.99 inst/cycle | 1.46 inst/cycle | +0.47 | 3.03 inst/cycle | +2.04 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_tile_size_for_occupancy` (0.615×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) caused the compiler to allocate 48 registers/thread (up from 18), collapsing Block Limit Registers from 21 to 5 and capping theoretical occupancy at 62.5% instead of 100%. Although waves_per_sm improved from 0.37 to 1.19 and scheduler idle dropped from 69.8% to 24.37%, the 6.3x increase in executed instructions (269K → 1.71M) from unrolled loop bodies at larger tile sizes dominated, causing a 1.63x slowdown. When a pointwise kernel already has low register pressure at the baseline tile size, increasing XBLOCK via autotune can trigger aggressive loop unrolling that spills registers and inflates instruction count far beyond the occupancy benefit.

> **Applicable when**: Kernel has low waves_per_sm (<0.5) and low achieved occupancy due to small grid, with low register pressure at baseline (≤20 regs/thread) and low compute throughput (<15%).

> **Avoid when**: The kernel body contains enough arithmetic that the compiler will aggressively unroll at larger XBLOCK, causing register count to jump significantly (e.g., 18→48 regs/thread). Check that Block Limit Registers stays above ~10 at the new tile size before committing. Also avoid when restore_value is set in autotune on an in-place pointer, as this adds overhead. Do not apply when the kernel is already instruction-throughput bound rather than latency-bound.

---

## Cumulative Progress

```
Baseline:  3070 ns  (1.000×)

Final best: 3100 ns  (0.9903×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 100352
@@ -33,8 +43,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -43,7 +51,7 @@

     arg_4 = inputs[4]
     xnumel = 100352
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 100352, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 100352)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

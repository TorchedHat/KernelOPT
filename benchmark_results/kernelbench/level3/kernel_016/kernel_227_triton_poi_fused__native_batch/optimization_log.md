# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_227` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6980 ns — **Best latency**: 6980 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_227`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_227_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6983,1,1), Waves/SM: 3.31
- **Registers/thread**: 22
- **Duration**: 6980 ns
- **SM Throughput**: 42.16 % | **Memory**: 16.10 % | **DRAM**: 10.60 %
- **Achieved Occupancy**: 63.33 %
- **IPC Active**: 2.46 inst/cycle | **No Eligible**: 34.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size | 0.991× | 0.465× | ✓ |  |
| B | autotune_xblock_tile_size_for_occupancy_and_latency_hiding | 0.969× | 0.992× | ✓ |  |
| C | vectorized_loads_via_autotune_xblock | 0.997× | 0.470× | ✓ |  |
| D | autotune_xblock_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.907× | 0.774× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_t... | Δ | vectorized_loads_... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6980 ns | 15010 ns | +8030 ns (+115.0%) | 7040 ns | +60 ns (+0.9%) | 14850 ns | +7870 ns (+112.8%) | 9020 ns | +2040 ns (+29.2%) |
| SM Throughput (%) | 42.16 % | 72.82 % | +30.66 pp | 42.50 % | +0.34 pp | 72.00 % | +29.84 pp | 63.38 % | +21.22 pp |
| Memory Throughput (%) | 16.10 % | 16.01 % | -0.09 pp | 16.19 % | +0.09 pp | 15.80 % | -0.30 pp | 17.10 % | +1.00 pp |
| DRAM Throughput (%) | 10.60 % | 4.90 % | -5.70 pp | 10.49 % | -0.11 pp | 4.96 % | -5.64 pp | 8.16 % | -2.44 pp |
| Achieved Occupancy (%) | 63.33 % | 84.84 % | +21.51 pp | 62.43 % | -0.90 pp | 84.96 % | +21.63 pp | 80.97 % | +17.64 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 3.31 | 6.61 | +3.30 | 3.31 | +0.00 | 6.61 | +3.30 | 3.31 | +0.00 |
| No Eligible (%) | 34.98 % | 15.34 % | -19.64 pp | 37.82 % | +2.84 pp | 15.11 % | -19.87 pp | 17.80 % | -17.18 pp |
| IPC Active | 2.46 inst/cycle | 3.36 inst/cycle | +0.90 | 2.46 inst/cycle | +0.00 | 3.38 inst/cycle | +0.92 | 3.26 inst/cycle | +0.80 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_for_main_input_to_improve_channel_vector_l1_hit_rate | 0.971× | 0.983× | ✓ |  |
| B | fix_autotune_grid_lambda_and_enable_fma_fusion | 0.998× | 0.952× | ✓ |  |
| C | fuse_fp32_instructions_to_fma_in_batchnorm_relu | 0.983× | 0.747× | ✓ |  |
| D | fix_broken_autotune_grid_and_enable_fma_fusion | 0.985× | 1.034× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | fix_autotune_grid... | Δ | fuse_fp32_instruc... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6980 ns | 7100 ns | +120 ns (+1.7%) | 7330 ns | +350 ns (+5.0%) | 9340 ns | +2360 ns (+33.8%) | 6750 ns | -230 ns (-3.3%) |
| SM Throughput (%) | 42.16 % | 45.98 % | +3.82 pp | 36.74 % | -5.42 pp | 58.61 % | +16.45 pp | 33.62 % | -8.54 pp |
| Memory Throughput (%) | 16.10 % | 16.52 % | +0.42 pp | 23.81 % | +7.71 pp | 27.96 % | +11.86 pp | 16.94 % | +0.84 pp |
| DRAM Throughput (%) | 10.60 % | 10.36 % | -0.24 pp | 10.10 % | -0.50 pp | 7.90 % | -2.70 pp | 10.89 % | +0.29 pp |
| Achieved Occupancy (%) | 63.33 % | 63.21 % | -0.12 pp | 48.31 % | -15.02 pp | 79.52 % | +16.19 pp | 47.77 % | -15.56 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 20.00 register/thread | -2 |
| Waves/SM | 3.31 | 3.31 | +0.00 | 3.31 | +0.00 | 6.61 | +3.30 | 3.31 | +0.00 |
| No Eligible (%) | 34.98 % | 33.18 % | -1.80 pp | 40.47 % | +5.49 pp | 27.57 % | -7.41 pp | 58.26 % | +23.28 pp |
| IPC Active | 2.46 inst/cycle | 2.69 inst/cycle | +0.23 | 2.51 inst/cycle | +0.05 | 2.87 inst/cycle | +0.41 | 1.65 inst/cycle | -0.81 |


### Key Learnings

#### ❌ Plan C: `fuse_fp32_instructions_to_fma_in_batchnorm_relu` (0.747×)

Replacing the non-fused sqrt+int_div+mul+add chain with tl.math.rsqrt and libdevice.fma in a batch-norm+ReLU pointwise kernel caused a 25% slowdown (7.04 µs → 9.34 µs), despite the FPInstructions NCU rule estimating a 38% gain. The diff also changed the autotune config list, which caused the autotuner to select XBLOCK=256 (block_size 128→256), increasing registers-per-thread from 22 to 32 and Block Limit Registers from 21 to 8 blocks/SM — doubling waves/SM from 3.31 to 6.61 and nearly doubling executed instructions (2.37M → 4.26M). The FMA arithmetic change itself may have been neutral or mildly positive, but the concurrent autotune config change dominated and caused the regression, consistent with the pattern documented in prior negative experiments.

> **Applicable when**: Never apply this combined change (FMA rewrite + autotune config list modification) together; the autotune config change dominates and causes register pressure regression.

> **Avoid when**: Do not modify the autotune config list alongside arithmetic rewrites in short-duration (<10 µs) pointwise kernels with low baseline register count (≤22 regs/thread). The FPInstructions NCU rule may flag non-fused instructions, but if the arithmetic rewrite is bundled with a block-size change that increases registers-per-thread above ~28, the register pressure regression will dominate any FMA throughput gain. Also avoid when the kernel is already register-efficient at the baseline block size.

---

## Cumulative Progress

```
Baseline:  6980 ns  (1.000×)

Final best: 6980 ns  (1.0000×)
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
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_227(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 893760
@@ -33,8 +48,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -44,7 +57,7 @@

     arg_5 = inputs[5]
     xnumel = 893760
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_227[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 893760, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_227[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 893760)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

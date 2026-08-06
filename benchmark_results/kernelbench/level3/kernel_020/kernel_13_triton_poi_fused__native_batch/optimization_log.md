# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8290 ns — **Best latency**: 8350 ns (**0.9928×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_13_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8820,1,1), Waves/SM: 4.18
- **Registers/thread**: 18
- **Duration**: 8290 ns
- **SM Throughput**: 47.69 % | **Memory**: 18.87 % | **DRAM**: 11.14 %
- **Achieved Occupancy**: 60.61 %
- **IPC Active**: 2.63 inst/cycle | **No Eligible**: 34.02 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.805× | 0.775× | ✓ |  |
| B | autotune_xblock_num_warps_pointwise | 0.999× | 0.775× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_reduced_launch_overhead | 0.923× | 0.366× | ✓ |  |
| D | increase_xblock_to_eliminate_partial_wave_and_improve_l1_reuse | 0.808× | 0.775× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | increase_xblock_f... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8290 ns | 10690 ns | +2400 ns (+29.0%) | 10690 ns | +2400 ns (+29.0%) | 22660 ns | +14370 ns (+173.3%) | 10690 ns | +2400 ns (+29.0%) |
| SM Throughput (%) | 47.69 % | 66.41 % | +18.72 pp | 66.35 % | +18.66 pp | 77.69 % | +30.00 pp | 66.13 % | +18.44 pp |
| Memory Throughput (%) | 18.87 % | 20.01 % | +1.14 pp | 19.98 % | +1.11 pp | 22.70 % | +3.83 pp | 20.09 % | +1.22 pp |
| DRAM Throughput (%) | 11.14 % | 8.63 % | -2.51 pp | 8.65 % | -2.49 pp | 4.07 % | -7.07 pp | 8.74 % | -2.40 pp |
| Achieved Occupancy (%) | 60.61 % | 82.68 % | +22.07 pp | 82.04 % | +21.43 pp | 81.37 % | +20.76 pp | 82.23 % | +21.62 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 32.00 register/thread | +14 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 16.70 | +12.52 | 4.18 | +0.00 |
| No Eligible (%) | 34.02 % | 18.62 % | -15.40 pp | 20.69 % | -13.33 pp | 14.71 % | -19.31 pp | 18.76 % | -15.26 pp |
| IPC Active | 2.63 inst/cycle | 3.22 inst/cycle | +0.59 | 3.23 inst/cycle | +0.60 | 3.40 inst/cycle | +0.77 | 3.23 inst/cycle | +0.60 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.775×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) increased achieved occupancy from 60.6% to 82.7% and reduced scheduler idle cycles from 34% to 18.6%, but total executed instructions nearly doubled (3.03M → 5.55M) and registers per thread jumped from 18 to 32. The increased register pressure reduced the block limit from 21 to 16 blocks/SM, and the extra instruction count dominated, causing elapsed cycles to increase from 12,340 to 15,896 (a 28.8% slowdown). When a kernel is already compute-bound at small tile sizes and has low register usage, adding autotune with large XBLOCK configs can trigger register pressure increases that outweigh occupancy gains.

> **Applicable when**: Never: this specific strategy caused regression. Avoid adding autotune with large XBLOCK configs when the kernel already has low register usage (≤20 registers/thread) and the autotune search space includes configs that inflate register count.

> **Avoid when**: Kernel has low register count (≤20 registers/thread) at baseline XBLOCK; adding large XBLOCK autotune configs causes register pressure to increase significantly (e.g., 18→32 registers/thread), which inflates instruction count and negates occupancy gains. Also avoid when the kernel is already achieving reasonable compute throughput (>40% SM SOL) and the primary bottleneck is not memory bandwidth.

#### ❌ Plan B: `autotune_xblock_num_warps_pointwise` (0.775×)

Adding triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel increased register usage from 18 to 32 per thread, which reduced the block limit from registers (21→16 blocks/SM) and caused the autotuner to select a configuration that executed 83% more instructions (3.03M→5.55M) with 29% more elapsed cycles (12340→15862). Although achieved occupancy improved (60.6%→82.0%) and eligible warps per scheduler increased (1.91→6.19), the dominant effect was a massive instruction count inflation from loop unrolling at larger XBLOCK, which more than offset any latency-hiding benefit. The grid was left hardcoded at cdiv(xnumel, 128) instead of being updated to use meta['XBLOCK'], meaning the autotuner selected a larger XBLOCK but the grid did not shrink accordingly, causing each element to be processed multiple times.

> **Applicable when**: Never apply this specific pattern (autotune with mismatched static grid). If autotune is used, the grid lambda MUST use meta['XBLOCK'].

> **Avoid when**: Do not add triton.autotune to a pointwise kernel when the grid launch site is not simultaneously updated to use a dynamic lambda (lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)). The static grid causes the autotuner to select larger XBLOCK while the grid remains sized for XBLOCK=128, inflating instruction count by ~83% and increasing kernel duration by ~29%.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_redundant_arithmetic_to_enable_fma_fusion | 0.944× | 0.949× | ✓ |  |
| B | replace_int_div_sqrt_with_rsqrt_for_fma_fusion | 1.017× | 0.966× | ✓ |  |
| C | algebraic_simplification_for_fma_fusion_in_batchnorm_epilogue | 0.998× | 1.036× | ✓ |  |
| D | fix_broken_autotune_grid_and_promote_fma_instructions | 1.020× | 1.040× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.978× | 1.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_redunda... | Δ | replace_int_div_s... | Δ | algebraic_simplif... | Δ | fix_broken_autotu... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8290 ns | 8740 ns | +450 ns (+5.4%) | 8580 ns | +290 ns (+3.5%) | 8000 ns | -290 ns (-3.5%) | 7970 ns | -320 ns (-3.9%) | 8160 ns | -130 ns (-1.6%) |
| SM Throughput (%) | 47.69 % | 37.88 % | -9.81 pp | 37.97 % | -9.72 pp | 28.78 % | -18.91 pp | 28.58 % | -19.11 pp | 47.50 % | -0.19 pp |
| Memory Throughput (%) | 18.87 % | 26.74 % | +7.87 pp | 27.03 % | +8.16 pp | 19.02 % | +0.15 pp | 18.79 % | -0.08 pp | 18.87 % | +0.00 pp |
| DRAM Throughput (%) | 11.14 % | 10.57 % | -0.57 pp | 10.78 % | -0.36 pp | 11.55 % | +0.41 pp | 11.59 % | +0.45 pp | 11.35 % | +0.21 pp |
| Achieved Occupancy (%) | 60.61 % | 47.63 % | -12.98 pp | 48.72 % | -11.89 pp | 47.20 % | -13.41 pp | 49.41 % | -11.20 pp | 61.10 % | +0.49 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 34.02 % | 43.58 % | +9.56 pp | 43.51 % | +9.49 pp | 58.30 % | +24.28 pp | 59.96 % | +25.94 pp | 34.35 % | +0.33 pp |
| IPC Active | 2.63 inst/cycle | 2.21 inst/cycle | -0.42 | 2.17 inst/cycle | -0.46 | 1.64 inst/cycle | -0.99 | 1.60 inst/cycle | -1.03 | 2.62 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0402× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_size_to_256_threads_for_l1tex_latency_hiding | 1.019× | 1.049× | ✓ |  |
| B | increase_num_warps_via_larger_block_size_for_l1tex_latency_hiding | 1.021× | 0.993× | ✓ | **✓ best** |
| C | fix_autotune_grid_lambda_to_eliminate_redundant_work_and_partial_wave | 0.978× | 1.016× | ✓ |  |
| D | remove_broken_autotune_fix_grid_and_inline_hardtanh_literals_for_fma_fusion | 1.020× | 1.049× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_si... | Δ | increase_num_warp... | Δ | fix_autotune_grid... | Δ | remove_broken_aut... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8290 ns | 7900 ns | -390 ns (-4.7%) | 8350 ns | +60 ns (+0.7%) | 8160 ns | -130 ns (-1.6%) | 7900 ns | -390 ns (-4.7%) |
| SM Throughput (%) | 47.69 % | 29.22 % | -18.47 pp | 27.64 % | -20.05 pp | 47.29 % | -0.40 pp | 28.25 % | -19.44 pp |
| Memory Throughput (%) | 18.87 % | 19.11 % | +0.24 pp | 18.34 % | -0.53 pp | 18.67 % | -0.20 pp | 18.57 % | -0.30 pp |
| DRAM Throughput (%) | 11.14 % | 11.69 % | +0.55 pp | 11.05 % | -0.09 pp | 11.32 % | +0.18 pp | 11.66 % | +0.52 pp |
| Achieved Occupancy (%) | 60.61 % | 47.34 % | -13.27 pp | 47.15 % | -13.46 pp | 63.17 % | +2.56 pp | 47.02 % | -13.59 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 34.02 % | 57.42 % | +23.40 pp | 58.51 % | +24.49 pp | 34.51 % | +0.49 pp | 58.74 % | +24.72 pp |
| IPC Active | 2.63 inst/cycle | 1.69 inst/cycle | -0.94 | 1.65 inst/cycle | -0.98 | 2.63 inst/cycle | +0.00 | 1.67 inst/cycle | -0.96 |


### Key Learnings

#### ✅ Plan A: `increase_block_size_to_256_threads_for_l1tex_latency_hiding` (1.049×)

When a pointwise kernel is stalled on L1TEX scoreboard dependencies (CPIStall rule, ~55% of cycles) and has very few eligible warps per scheduler (~1.03), doubling the block size from 128 to 256 threads (XBLOCK 128→256, num_warps 4→8) with a proportionally halved grid provides more in-flight warps to hide memory latency. In this case, Warp Cycles Per Issued Instruction dropped from 18.31 to 17.65 and scheduler idle cycles fell from 59.96% to 57.42%, yielding a modest ~5% speedup. The improvement is limited when the kernel is already latency-bound and the L1TEX stall remains dominant even after doubling warps, suggesting that the stall depth exceeds what 8 warps per block can fully hide.

> **Applicable when**: Pointwise Triton kernel with CPIStall rule triggered (L1TEX stall >50% of cycles), eligible warps per scheduler near 1.0, No Eligible cycles >55%, and low register count (≤18/thread) leaving headroom to double warps without hitting register-file block limits. Grid size is large enough that halving block count does not reduce waves below 2.

> **Avoid when**: Registers per thread are already high (≥24/thread) — doubling warps per block may push the register-limited block count below the warp-count limit, reducing occupancy. Also avoid when XBLOCK≥512 has previously been shown to inflate register count (as seen in prior iterations). Do not apply when the kernel is already achieving high occupancy (>70%) or when the L1TEX stall depth is so large (>15 cycles) that even 8 warps/block cannot hide it, as the marginal gain will be small.

---

## Iteration 4

**Best so far**: 1.0402× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_fma_fusion_via_tl_math_fma_to_reduce_non_fused_fp32_instructions | 1.020× | 1.044× | ✓ |  |
| B | fuse_fp32_arithmetic_chains_via_explicit_fma_rewrite | 1.020× | 1.024× | ✓ |  |
| C | add_evict_first_to_streaming_tensor_load_store_for_l1tex_latency_reduction | 0.973× | 0.996× | ✓ |  |
| D | streaming_eviction_policy_for_main_tensor_to_improve_l1_reuse_of_channel_params | 0.995× | 1.044× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_fma_fusi... | Δ | fuse_fp32_arithme... | Δ | add_evict_first_t... | Δ | streaming_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8290 ns | 7940 ns | -350 ns (-4.2%) | 8100 ns | -190 ns (-2.3%) | 8320 ns | +30 ns (+0.4%) | 7940 ns | -350 ns (-4.2%) |
| SM Throughput (%) | 47.69 % | 29.38 % | -18.31 pp | 29.17 % | -18.52 pp | 53.20 % | +5.51 pp | 31.53 % | -16.16 pp |
| Memory Throughput (%) | 18.87 % | 19.34 % | +0.47 pp | 18.92 % | +0.05 pp | 18.67 % | -0.20 pp | 19.11 % | +0.24 pp |
| DRAM Throughput (%) | 11.14 % | 11.64 % | +0.50 pp | 11.42 % | +0.28 pp | 11.11 % | -0.03 pp | 11.68 % | +0.54 pp |
| Achieved Occupancy (%) | 60.61 % | 46.85 % | -13.76 pp | 46.57 % | -14.04 pp | 66.67 % | +6.06 pp | 47.60 % | -13.01 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 34.02 % | 57.85 % | +23.83 pp | 58.33 % | +24.31 pp | 27.15 % | -6.87 pp | 54.35 % | +20.33 pp |
| IPC Active | 2.63 inst/cycle | 1.66 inst/cycle | -0.97 | 1.65 inst/cycle | -0.98 | 2.92 inst/cycle | +0.29 | 1.79 inst/cycle | -0.84 |

---

## Iteration 5

**Best so far**: 1.0402× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_to_16_warps_per_block_via_xblock512_to_fully_hide_l1tex_stall_depth | 1.019× | 1.028× | ✓ |  |
| B | add_memory_access_hints_for_vectorized_channel_loads | 1.018× | 0.993× | ✓ |  |
| C | align_grid_to_sm_count_to_eliminate_partial_wave | 0.996× | 1.040× | ✓ |  |
| D | replace_sqrt_div_noop_with_libdevice_rsqrt_for_instruction_count_reduction | 0.996× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_to_16_wa... | Δ | add_memory_access... | Δ | align_grid_to_sm_... | Δ | replace_sqrt_div_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8290 ns | 8060 ns | -230 ns (-2.8%) | 8350 ns | +60 ns (+0.7%) | 7970 ns | -320 ns (-3.9%) | 8260 ns | -30 ns (-0.4%) |
| SM Throughput (%) | 47.69 % | 28.61 % | -19.08 pp | 31.57 % | -16.12 pp | 29.04 % | -18.65 pp | 31.43 % | -16.26 pp |
| Memory Throughput (%) | 18.87 % | 18.79 % | -0.08 pp | 19.07 % | +0.20 pp | 18.78 % | -0.09 pp | 19.07 % | +0.20 pp |
| DRAM Throughput (%) | 11.14 % | 11.46 % | +0.32 pp | 11.07 % | -0.07 pp | 11.60 % | +0.46 pp | 11.19 % | +0.05 pp |
| Achieved Occupancy (%) | 60.61 % | 47.11 % | -13.50 pp | 47.48 % | -13.13 pp | 47.71 % | -12.90 pp | 47.90 % | -12.71 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 34.02 % | 58.72 % | +24.70 pp | 54.39 % | +20.37 pp | 57.77 % | +23.75 pp | 55.25 % | +21.23 pp |
| IPC Active | 2.63 inst/cycle | 1.69 inst/cycle | -0.94 | 1.77 inst/cycle | -0.86 | 1.63 inst/cycle | -1.00 | 1.83 inst/cycle | -0.80 |

---

## Cumulative Progress

```
Baseline:  8290 ns  (1.000×)
Iter 2     7970 ns  (1.040×)  — fix_broken_autotune_grid_and_promote_fma_instructions

Final best: 8350 ns  (0.9928×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -19,23 +20,14 @@

     tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0.0, tl.float32)
-    tmp17 = tl.maximum(tmp15, tmp16)
-    tmp18 = tl.full([1], 6.0, tl.float32)
-    tmp19 = tl.minimum(tmp17, tmp18)
+    # FMA 1: fuse (tmp3 + 1e-5) into rsqrt arg to avoid separate FADD
+    inv_std = tl.math.rsqrt(tl.math.fma(tmp3, 1.0, 1e-5))
+    tmp11 = tmp2 * inv_std
+    # FMA 2: fuse (tmp11 * tmp12 + tmp14) to avoid separate FMUL + FADD
+    tmp15 = tl.math.fma(tmp11, tmp12, tmp14)
+    tmp17 = tl.maximum(tmp15, 0.0)
+    tmp19 = tl.minimum(tmp17, 6.0)
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

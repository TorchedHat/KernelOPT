# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8990 ns — **Best latency**: 8960 ns (**1.0033×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_num_warps_to_increase_block_count_and_hide_l1tex_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_32_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (10290,1,1), Waves/SM: 4.87
- **Registers/thread**: 18
- **Duration**: 8990 ns
- **SM Throughput**: 49.70 % | **Memory**: 18.72 % | **DRAM**: 12.00 %
- **Achieved Occupancy**: 61.10 %
- **IPC Active**: 2.61 inst/cycle | **No Eligible**: 33.91 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.819× | 0.749× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 0.990× | 0.986× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_better_occupancy | 0.942× | 0.248× | ✓ |  |
| D | increase_xblock_and_autotune_for_latency_hiding | 1.009× | 0.182× | ✓ |  |
| E | add_triton_autotune | 1.009× | 0.997× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | increase_xblock_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8990 ns | 12000 ns | +3010 ns (+33.5%) | 9120 ns | +130 ns (+1.4%) | 36290 ns | +27300 ns (+303.7%) | 49250 ns | +40260 ns (+447.8%) | 9020 ns | +30 ns (+0.3%) |
| SM Throughput (%) | 49.70 % | 69.07 % | +19.37 pp | 49.65 % | -0.05 pp | 81.19 % | +31.49 pp | 80.38 % | +30.68 pp | 49.06 % | -0.64 pp |
| Memory Throughput (%) | 18.72 % | 19.80 % | +1.08 pp | 18.65 % | -0.07 pp | 16.44 % | -2.28 pp | 21.99 % | +3.27 pp | 18.48 % | -0.24 pp |
| DRAM Throughput (%) | 12.00 % | 9.00 % | -3.00 pp | 11.80 % | -0.20 pp | 2.97 % | -9.03 pp | 2.18 % | -9.82 pp | 11.99 % | -0.01 pp |
| Achieved Occupancy (%) | 61.10 % | 83.32 % | +22.22 pp | 58.87 % | -2.23 pp | 55.70 % | -5.40 pp | 79.92 % | +18.82 pp | 59.16 % | -1.94 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 48.00 register/thread | +30 | 24.00 register/thread | +6 | 18.00 register/thread | +0 |
| Waves/SM | 4.87 | 4.87 | +0.00 | 4.87 | +0.00 | 15.59 | +10.72 | 38.98 | +34.11 | 4.87 | +0.00 |
| No Eligible (%) | 33.91 % | 16.53 % | -17.38 pp | 33.54 % | -0.37 pp | 13.54 % | -20.37 pp | 15.80 % | -18.11 pp | 32.59 % | -1.32 pp |
| IPC Active | 2.61 inst/cycle | 3.32 inst/cycle | +0.71 | 2.59 inst/cycle | -0.02 | 3.45 inst/cycle | +0.84 | 3.36 inst/cycle | +0.75 | 2.63 inst/cycle | +0.02 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.749×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) caused the compiler to select a configuration that nearly doubled register usage per thread (18→32), which increased executed instructions by ~83% (3.5M→6.5M) and raised warp cycles per issued instruction (14.1→16.4 cycles), resulting in a 25% longer kernel duration despite higher achieved occupancy. The autotune overhead and the instruction count explosion from unrolling larger tiles outweighed any cache reuse benefit from larger tiles. When the baseline kernel already has very low register pressure (≤18 regs/thread) and the bottleneck is L1TEX stall latency rather than pure memory bandwidth, adding autotune with large XBLOCK configs can cause the compiler to emit significantly more instructions per element, negating the tile-size benefit.

> **Applicable when**: Never applicable as described — this is a negative result. Autotune with large XBLOCK configs should be avoided when the baseline kernel already has low register pressure (≤20 regs/thread) and the bottleneck is L1TEX stall latency (CPIStall rule, ~48% stall share) but instruction count is already low. The autotune overhead and compiler unrolling of larger tiles can cause register count to jump and instruction count to nearly double.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK configs (≥512) when: (1) baseline registers/thread are already low (≤20), indicating the compiler is already near-optimal for the tile size; (2) the kernel body contains multiple broadcast loads with modulo indexing (e.g., xindex % C) that the compiler must unroll per XBLOCK, causing instruction explosion; (3) the kernel duration is short (≤10 µs) where autotune warm-up cost dominates. In these cases, the instruction count increase from unrolling larger tiles outweighs cache reuse gains.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | align_grid_to_sm_count_to_eliminate_partial_wave | 0.991× | 0.997× | ✓ |  |
| B | align_grid_to_eliminate_partial_wave | 0.991× | 1.000× | ✓ |  |
| C | fuse_fp32_multiply_add_to_fma_instructions | 0.800× | 0.755× | ✓ |  |
| D | remove_autotune_restore_small_fixed_xblock_to_eliminate_register_pressure_and_instruction_explosion | 1.011× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | align_grid_to_sm_... | Δ | align_grid_to_eli... | Δ | fuse_fp32_multipl... | Δ | remove_autotune_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8990 ns | 9020 ns | +30 ns (+0.3%) | 8990 ns | +0 ns (+0.0%) | 11900 ns | +2910 ns (+32.4%) | 8990 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 49.70 % | 49.32 % | -0.38 pp | 49.63 % | -0.07 pp | 69.04 % | +19.34 pp | 49.74 % | +0.04 pp |
| Memory Throughput (%) | 18.72 % | 18.41 % | -0.31 pp | 18.70 % | -0.02 pp | 19.78 % | +1.06 pp | 18.54 % | -0.18 pp |
| DRAM Throughput (%) | 12.00 % | 11.98 % | -0.02 pp | 11.99 % | -0.01 pp | 9.07 % | -2.93 pp | 11.99 % | -0.01 pp |
| Achieved Occupancy (%) | 61.10 % | 60.85 % | -0.25 pp | 59.03 % | -2.07 pp | 83.32 % | +22.22 pp | 60.17 % | -0.93 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 4.87 | 4.87 | +0.00 | 4.87 | +0.00 | 4.87 | +0.00 | 4.87 | +0.00 |
| No Eligible (%) | 33.91 % | 33.03 % | -0.88 pp | 33.33 % | -0.58 pp | 16.67 % | -17.24 pp | 33.14 % | -0.77 pp |
| IPC Active | 2.61 inst/cycle | 2.68 inst/cycle | +0.07 | 2.63 inst/cycle | +0.02 | 3.30 inst/cycle | +0.69 | 2.67 inst/cycle | +0.06 |


### Key Learnings

#### ❌ Plan C: `fuse_fp32_multiply_add_to_fma_instructions` (0.755×)

Applying libdevice.fma() to replace non-fused mul+add pairs and simultaneously changing the @triton.autotune configs caused a 24% regression (9.02 µs → 11.9 µs). Despite the FMA arithmetic simplification, executed instructions nearly doubled (3.5M → 6.5M) and registers per thread jumped from 18 to 32, reducing Block Limit Registers from 21 to 16 blocks/SM. The root cause is that the autotune config change (not the FMA itself) drove the compiler to select a larger XBLOCK, which caused the same instruction-explosion and register-pressure pattern seen in prior autotune regressions — the FMA optimization was masked by the autotune overhead. When the NCU FPInstructions rule fires but the kernel is already short-duration (<10 µs) with low register pressure (≤20 regs/thread), the autotune config change accompanying the FMA rewrite dominates and causes regression.

> **Applicable when**: Never apply this combined pattern. The FMA arithmetic change alone may be neutral or slightly positive, but pairing it with autotune config changes that increase XBLOCK causes the compiler to emit ~2x more instructions and spike register usage, dominating any FMA benefit.

> **Avoid when**: Do not combine libdevice.fma() rewrites with autotune config changes (especially larger XBLOCK values) when the baseline kernel already has low register pressure (≤20 regs/thread) and short duration (<10 µs). The autotune-driven XBLOCK increase causes instruction count to nearly double (3.5M → 6.5M) and registers/thread to jump (18 → 32), reducing Block Limit Registers and increasing kernel duration despite higher achieved occupancy. If FMA fusion is desired, apply it in isolation without changing autotune configs.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_fma_fusion_via_arithmetic_simplification | 1.029× | 1.003× | ✓ | **✓ best** |
| B | streaming_store_eviction_to_improve_l2_hit_rate_for_broadcast_loads | 1.007× | 0.997× | ✓ |  |
| C | replace_integer_reciprocal_with_rsqrt_to_enable_fma_fusion | 1.006× | 0.989× | ✓ |  |
| D | fix_callsite_static_grid_to_lambda_for_existing_autotune | 1.007× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_fma_fusion... | Δ | streaming_store_e... | Δ | replace_integer_r... | Δ | fix_callsite_stat... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8990 ns | 8960 ns | -30 ns (-0.3%) | 9020 ns | +30 ns (+0.3%) | 9090 ns | +100 ns (+1.1%) | 9020 ns | +30 ns (+0.3%) |
| SM Throughput (%) | 49.70 % | 29.64 % | -20.06 pp | 54.39 % | +4.69 pp | 30.81 % | -18.89 pp | 49.06 % | -0.64 pp |
| Memory Throughput (%) | 18.72 % | 18.38 % | -0.34 pp | 18.31 % | -0.41 pp | 19.07 % | +0.35 pp | 18.47 % | -0.25 pp |
| DRAM Throughput (%) | 12.00 % | 12.04 % | +0.04 pp | 11.94 % | -0.06 pp | 11.88 % | -0.12 pp | 11.94 % | -0.06 pp |
| Achieved Occupancy (%) | 61.10 % | 45.49 % | -15.61 pp | 63.89 % | +2.79 pp | 45.56 % | -15.54 pp | 58.93 % | -2.17 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.87 | 4.87 | +0.00 | 4.87 | +0.00 | 4.87 | +0.00 | 4.87 | +0.00 |
| No Eligible (%) | 33.91 % | 57.06 % | +23.15 pp | 26.08 % | -7.83 pp | 57.29 % | +23.38 pp | 32.68 % | -1.23 pp |
| IPC Active | 2.61 inst/cycle | 1.69 inst/cycle | -0.92 | 2.95 inst/cycle | +0.34 | 1.64 inst/cycle | -0.97 | 2.61 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0033× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_improve_occupancy_and_hide_l1tex_latency | 1.028× | 1.022× | ✓ |  |
| B | eliminate_integer_reciprocal_noop_multiply_to_reduce_non_fused_fp32_instructions | 1.008× | 1.022× | ✓ |  |
| C | replace_runtime_modulo_with_constexpr_channel_count_to_eliminate_integer_division_latency | 0.985× | 0.997× | ✓ |  |
| D | fix_callsite_lambda_grid_and_add_small_xblock_high_warps_configs | 1.009× | 1.003× | ✓ |  |
| E | add_triton_autotune | 1.007× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | eliminate_integer... | Δ | replace_runtime_m... | Δ | fix_callsite_lamb... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8990 ns | 8800 ns | -190 ns (-2.1%) | 8800 ns | -190 ns (-2.1%) | 9020 ns | +30 ns (+0.3%) | 8960 ns | -30 ns (-0.3%) | 8930 ns | -60 ns (-0.7%) |
| SM Throughput (%) | 49.70 % | 30.56 % | -19.14 pp | 32.97 % | -16.73 pp | 49.37 % | -0.33 pp | 30.74 % | -18.96 pp | 30.17 % | -19.53 pp |
| Memory Throughput (%) | 18.72 % | 18.96 % | +0.24 pp | 19.06 % | +0.34 pp | 18.63 % | -0.09 pp | 19.09 % | +0.37 pp | 18.72 % | +0.00 pp |
| DRAM Throughput (%) | 12.00 % | 12.26 % | +0.26 pp | 12.27 % | +0.27 pp | 12.00 % | +0.00 pp | 12.06 % | +0.06 pp | 12.09 % | +0.09 pp |
| Achieved Occupancy (%) | 61.10 % | 45.63 % | -15.47 pp | 46.46 % | -14.64 pp | 60.23 % | -0.87 pp | 45.63 % | -15.47 pp | 45.60 % | -15.50 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.87 | 4.87 | +0.00 | 4.87 | +0.00 | 4.87 | +0.00 | 4.87 | +0.00 | 4.87 | +0.00 |
| No Eligible (%) | 33.91 % | 56.86 % | +22.95 pp | 54.23 % | +20.32 pp | 33.83 % | -0.08 pp | 58.51 % | +24.60 pp | 57.87 % | +23.96 pp |
| IPC Active | 2.61 inst/cycle | 1.69 inst/cycle | -0.92 | 1.79 inst/cycle | -0.82 | 2.65 inst/cycle | +0.04 | 1.69 inst/cycle | -0.92 | 1.68 inst/cycle | -0.93 |

---

## Iteration 5

**Best so far**: 1.0033× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | precompute_rsqrt_to_break_l1tex_dependency_chain | 1.028× | 1.022× | ✓ |  |
| B | restructure_batchnorm_arithmetic_to_maximize_fma_fusion | 1.011× | 1.022× | ✓ |  |
| C | preload_broadcast_channel_params_to_eliminate_l1tex_stalls | 1.004× | 0.983× | ✓ |  |
| D | reduce_num_warps_to_increase_block_count_and_hide_l1tex_latency | 1.002× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | precompute_rsqrt_... | Δ | restructure_batch... | Δ | preload_broadcast... | Δ | reduce_num_warps_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8990 ns | 8800 ns | -190 ns (-2.1%) | 8800 ns | -190 ns (-2.1%) | 9150 ns | +160 ns (+1.8%) | 8960 ns | -30 ns (-0.3%) |
| SM Throughput (%) | 49.70 % | 30.52 % | -19.18 pp | 32.90 % | -16.80 pp | 28.91 % | -20.79 pp | 24.70 % | -25.00 pp |
| Memory Throughput (%) | 18.72 % | 18.93 % | +0.21 pp | 18.95 % | +0.23 pp | 19.22 % | +0.50 pp | 8.80 % | -9.92 pp |
| DRAM Throughput (%) | 12.00 % | 12.27 % | +0.27 pp | 12.28 % | +0.28 pp | 11.78 % | -0.22 pp | 6.04 % | -5.96 pp |
| Achieved Occupancy (%) | 61.10 % | 45.86 % | -15.24 pp | 46.25 % | -14.85 pp | 44.11 % | -16.99 pp | 36.69 % | -24.41 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.87 | 4.87 | +0.00 | 4.87 | +0.00 | 4.87 | +0.00 | 2.44 | -2.43 |
| No Eligible (%) | 33.91 % | 57.14 % | +23.23 pp | 54.16 % | +20.25 pp | 66.15 % | +32.24 pp | 58.23 % | +24.32 pp |
| IPC Active | 2.61 inst/cycle | 1.69 inst/cycle | -0.92 | 1.82 inst/cycle | -0.79 | 1.33 inst/cycle | -1.28 | 1.63 inst/cycle | -0.98 |

---

## Cumulative Progress

```
Baseline:  8990 ns  (1.000×)
Iter 3     8960 ns  (1.003×)  — enable_fma_fusion_via_arithmetic_simplification

Final best: 8960 ns  (1.0033×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -21,12 +22,8 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    tmp6 = libdevice.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0.0, tl.float32)
@@ -34,8 +31,6 @@

     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
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

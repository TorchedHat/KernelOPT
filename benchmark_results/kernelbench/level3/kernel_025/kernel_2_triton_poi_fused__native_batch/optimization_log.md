# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 144450 ns — **Best latency**: 119580 ns (**1.2080×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_y_dominant_tile_configs_for_coalesced_strided_load`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_025/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (58800,1,1), Waves/SM: 37.12
- **Registers/thread**: 40
- **Duration**: 144450 ns
- **SM Throughput**: 60.67 % | **Memory**: 70.57 % | **DRAM**: 67.29 %
- **Achieved Occupancy**: 70.91 %
- **IPC Active**: 2.44 inst/cycle | **No Eligible**: 40.30 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_better_l1_reuse_and_occupancy | 0.865× | 0.826× | ✓ |  |
| B | autotune_tile_sizes_and_fix_eviction_policy | 1.965× | 0.934× | ✓ |  |
| C | add_autotune_with_vectorized_loads_and_larger_xblock | 2.288× | 1.639× | ✓ | **✓ best** |
| D | autotune_tile_sizes_reduce_register_pressure | 1.639× | 1.456× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | add_autotune_with... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 144450 ns | 174910 ns | +30460 ns (+21.1%) | 154720 ns | +10270 ns (+7.1%) | 88160 ns | -56290 ns (-39.0%) | 99230 ns | -45220 ns (-31.3%) |
| SM Throughput (%) | 60.67 % | 73.16 % | +12.49 pp | 80.96 % | +20.29 pp | 87.62 % | +26.95 pp | 84.30 % | +23.63 pp |
| Memory Throughput (%) | 70.57 % | 59.96 % | -10.61 pp | 43.72 % | -26.85 pp | 63.09 % | -7.48 pp | 40.79 % | -29.78 pp |
| DRAM Throughput (%) | 67.29 % | 55.68 % | -11.61 pp | 31.54 % | -35.75 pp | 12.70 % | -54.59 pp | 20.45 % | -46.84 pp |
| Achieved Occupancy (%) | 70.91 % | 59.40 % | -11.51 pp | 53.23 % | -17.68 pp | 93.60 % | +22.69 pp | 69.84 % | -1.07 pp |
| Registers/Thread | 40.00 register/thread | 48.00 register/thread | +8 | 50.00 register/thread | +10 | 32.00 register/thread | -8 | 40.00 register/thread | +0 |
| Waves/SM | 37.12 | 44.55 | +7.43 | 49.49 | +12.37 | 27.84 | -9.28 | 37.12 | +0.00 |
| No Eligible (%) | 40.30 % | 23.90 % | -16.40 pp | 17.63 % | -22.67 pp | 9.93 % | -30.37 pp | 13.69 % | -26.61 pp |
| IPC Active | 2.44 inst/cycle | 2.98 inst/cycle | +0.54 | 3.30 inst/cycle | +0.86 | 3.61 inst/cycle | +1.17 | 3.45 inst/cycle | +1.01 |

---

## Iteration 2

**Best so far**: 1.6385× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | expand_autotune_configs_to_reduce_register_pressure_and_unlock_full_occupancy | 2.199× | 1.463× | ✓ |  |
| C | add_num_stages_pipelining_to_existing_autotune_to_reduce_register_pressure | 1.484× | 0.930× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | add_num_stages_pi... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 144450 ns | 98750 ns | -45700 ns (-31.6%) | 155300 ns | +10850 ns (+7.5%) |
| SM Throughput (%) | 60.67 % | 84.37 % | +23.70 pp | 81.37 % | +20.70 pp |
| Memory Throughput (%) | 70.57 % | 40.87 % | -29.70 pp | 44.25 % | -26.32 pp |
| DRAM Throughput (%) | 67.29 % | 20.58 % | -46.71 pp | 31.47 % | -35.82 pp |
| Achieved Occupancy (%) | 70.91 % | 69.87 % | -1.04 pp | 53.20 % | -17.71 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 50.00 register/thread | +10 |
| Waves/SM | 37.12 | 37.12 | +0.00 | 49.49 | +12.37 |
| No Eligible (%) | 40.30 % | 14.07 % | -26.23 pp | 17.49 % | -22.81 pp |
| IPC Active | 2.44 inst/cycle | 3.42 inst/cycle | +0.98 | 3.27 inst/cycle | +0.83 |

---

## Iteration 3

**Best so far**: 1.6385× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_non_fused_fp32_instructions_into_fma_to_reduce_alu_pressure | 2.357× | 1.724× | ✓ |  |
| B | replace_manual_reciprocal_sqrt_chain_with_rsqrt_to_reduce_fp_instructions_and_register_pressure | 2.320× | 1.752× | ✓ |  |
| C | simplify_batchnorm_arithmetic_with_rsqrt_to_reduce_register_pressure | 1.589× | 1.206× | ✓ |  |
| D | fuse_non_fused_fp32_instructions_into_fma_operations | 1.490× | 0.924× | ✓ |  |
| E | add_triton_autotune | 2.485× | 1.452× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fuse_non_fused_fp... | Δ | replace_manual_re... | Δ | simplify_batchnor... | Δ | fuse_non_fused_fp... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 144450 ns | 83810 ns | -60640 ns (-42.0%) | 82430 ns | -62020 ns (-42.9%) | 119810 ns | -24640 ns (-17.1%) | 156380 ns | +11930 ns (+8.3%) | 99520 ns | -44930 ns (-31.1%) |
| SM Throughput (%) | 60.67 % | 71.47 % | +10.80 pp | 72.09 % | +11.42 pp | 56.05 % | -4.62 pp | 81.14 % | +20.47 pp | 84.21 % | +23.54 pp |
| Memory Throughput (%) | 70.57 % | 47.63 % | -22.94 pp | 47.22 % | -23.35 pp | 57.07 % | -13.50 pp | 44.05 % | -26.52 pp | 40.75 % | -29.82 pp |
| DRAM Throughput (%) | 67.29 % | 24.07 % | -43.22 pp | 24.38 % | -42.91 pp | 40.17 % | -27.12 pp | 31.19 % | -36.10 pp | 20.42 % | -46.87 pp |
| Achieved Occupancy (%) | 70.91 % | 93.07 % | +22.16 pp | 92.67 % | +21.76 pp | 70.31 % | -0.60 pp | 53.18 % | -17.73 pp | 69.83 % | -1.08 pp |
| Registers/Thread | 40.00 register/thread | 32.00 register/thread | -8 | 32.00 register/thread | -8 | 36.00 register/thread | -4 | 50.00 register/thread | +10 | 40.00 register/thread | +0 |
| Waves/SM | 37.12 | 27.84 | -9.28 | 27.84 | -9.28 | 37.12 | +0.00 | 49.49 | +12.37 | 37.12 | +0.00 |
| No Eligible (%) | 40.30 % | 27.58 % | -12.72 pp | 26.05 % | -14.25 pp | 43.90 % | +3.60 pp | 17.67 % | -22.63 pp | 13.78 % | -26.52 pp |
| IPC Active | 2.44 inst/cycle | 2.87 inst/cycle | +0.43 | 2.97 inst/cycle | +0.53 | 2.25 inst/cycle | -0.19 | 3.30 inst/cycle | +0.86 | 3.44 inst/cycle | +1.00 |


### Key Learnings

#### ✅ Plan B: `replace_manual_reciprocal_sqrt_chain_with_rsqrt_to_reduce_fp_instructions_and_register_pressure` (1.694×)

When a batch-norm arithmetic body implements 1/sqrt(var+eps) via a verbose 11-instruction chain (sqrt_rn + int32 literal + int/float division + float literal + dead float multiply + two multiplies + add), replacing it with tl.math.rsqrt + tl.math.fma eliminates 7 dead intermediate temporaries, dropping registers_per_thread from 40 to 32. This unlocks the next occupancy tier: Block Limit Registers rises from 12 to 16 blocks/SM, theoretical occupancy jumps from 75% to 100%, and achieved occupancy from 69.8% to 93.1%. Total executed instructions drop by ~30% (66.3M to 46.4M), branch divergence is eliminated (448 avg divergent branches to 0), and kernel duration falls from 100.4µs to 85.3µs. The FPInstructions NCU rule (high non-fused FP32 count) and TheoreticalOccupancy rule (registers as binding constraint) are the key diagnostic signals.

> **Applicable when**: Kernel computes batch-norm normalization via a multi-step reciprocal-sqrt chain (sqrt_rn + integer literal + int/float division + float literal + dead multiply pattern); FPInstructions NCU rule fires with high non-fused FP32 instruction count (>50% non-fused); TheoreticalOccupancy rule fires with registers_per_thread >= 36 and Block Limit Registers is the binding occupancy constraint; the variance argument is guaranteed positive (var + epsilon > 0).

> **Avoid when**: Input to rsqrt may be zero or negative (no epsilon guard); kernel is already at 100% theoretical occupancy or registers_per_thread <= 32; the sqrt result is reused beyond the reciprocal computation; kernel is purely memory-bound (DRAM throughput > 80%) where instruction reduction has minimal wall-clock benefit.

#### ✅ Plan C: `simplify_batchnorm_arithmetic_with_rsqrt_to_reduce_register_pressure` (1.206×)

Replacing a multi-instruction reciprocal-sqrt sequence (sqrt_rn + integer-1 / float + dead float multiply) with a single tl.rsqrt() call eliminates ~13.5M executed instructions (65M→51.5M) and reduces registers per thread from 40 to 36. However, the net effect was mixed: while instruction count dropped and branch divergence was eliminated, the kernel became memory-bound (DRAM throughput rose from 20.6% to 40.2%, memory throughput from 1.01 to 1.97 TB/s) and compute throughput fell from 84% to 56%, with eligible warps per scheduler dropping from 5.08 to 2.55 and warp cycles per issued instruction worsening from 13 to 20. The 1.21x speedup came primarily from the instruction count reduction and divergence elimination, not from occupancy improvement (registers stayed above the 32/thread threshold needed to unlock higher occupancy).

> **Applicable when**: FPInstructions NCU rule fires with high non-fused FP32 instruction count; kernel body contains a reciprocal-sqrt pattern implemented as sqrt_rn + integer division + float multiply (common in batch norm / layer norm); registers_per_thread is elevated due to excess live temporaries from the multi-step sequence; branch divergence (Avg. Divergent Branches > 0) is present from masking on scalar loads.

> **Avoid when**: The argument to rsqrt may be zero or negative (no epsilon guard); kernel is already memory-bound before the change (DRAM throughput > 60%) — reducing compute without also improving memory access patterns may shift the bottleneck further toward memory and worsen eligible warps; registers are already at or below 32/thread (no register pressure benefit).

---

## Iteration 4

**Best so far**: 1.6385× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_eviction_policies_for_streaming_load_and_l2_compression | 1.731× | 1.440× | ✓ |  |
| B | add_y_dominant_tile_configs_for_coalesced_strided_load | 3.487× | 1.208× | ✓ | **✓ best** |
| C | change_eviction_policy_to_evict_first_for_streaming_loads_and_stores | 1.734× | 1.420× | ✓ |  |
| D | enable_l2_compression_and_fix_eviction_policies_for_streaming_load | 1.734× | 1.424× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_eviction_poli... | Δ | add_y_dominant_ti... | Δ | change_eviction_p... | Δ | enable_l2_compres... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 144450 ns | 100350 ns | -44100 ns (-30.5%) | 119580 ns | -24870 ns (-17.2%) | 101730 ns | -42720 ns (-29.6%) | 101410 ns | -43040 ns (-29.8%) |
| SM Throughput (%) | 60.67 % | 84.27 % | +23.60 pp | 56.08 % | -4.59 pp | 84.53 % | +23.86 pp | 83.69 % | +23.02 pp |
| Memory Throughput (%) | 70.57 % | 40.03 % | -30.54 pp | 57.09 % | -13.48 pp | 40.19 % | -30.38 pp | 39.77 % | -30.80 pp |
| DRAM Throughput (%) | 67.29 % | 20.06 % | -47.23 pp | 40.21 % | -27.08 pp | 19.84 % | -47.45 pp | 19.86 % | -47.43 pp |
| Achieved Occupancy (%) | 70.91 % | 69.81 % | -1.10 pp | 70.30 % | -0.61 pp | 69.80 % | -1.11 pp | 69.82 % | -1.09 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 36.00 register/thread | -4 | 40.00 register/thread | +0 | 40.00 register/thread | +0 |
| Waves/SM | 37.12 | 37.12 | +0.00 | 37.12 | +0.00 | 37.12 | +0.00 | 37.12 | +0.00 |
| No Eligible (%) | 40.30 % | 13.53 % | -26.77 pp | 43.41 % | +3.11 pp | 13.63 % | -26.67 pp | 13.67 % | -26.63 pp |
| IPC Active | 2.44 inst/cycle | 3.45 inst/cycle | +1.01 | 2.26 inst/cycle | -0.18 | 3.45 inst/cycle | +1.01 | 3.45 inst/cycle | +1.01 |


### Key Learnings

#### ✅ Plan C: `change_eviction_policy_to_evict_first_for_streaming_loads_and_stores` (1.420×)

Switching the primary streaming load from evict_last to evict_first and adding evict_first to the store, while simultaneously expanding the autotune config set, caused a net regression in kernel duration (82.43µs → 101.73µs, 1.42x slower wall-clock despite the reported speedup being >1 in the other direction). The autotune selected a configuration with 40 registers/thread (up from 32), collapsing theoretical occupancy from 100% to 75% and achieved occupancy from 92.7% to 69.8%. Although warp cycles per issued instruction improved (20.1→12.87) and No Eligible% dropped (26%→13.6%), the total executed instruction count ballooned from 46M to 66M (+44%) and the kernel ran longer. The eviction policy change itself is not the primary cause of the speedup — the confounding autotune config expansion selected a higher-register config that undid the prior rsqrt register reduction, making this a negative outcome masked by the reported speedup metric.

> **Applicable when**: Kernel accesses a large streaming tensor (>> L2 cache size) with non-unit stride in the tiled dimension, and the kernel is already at 100% theoretical occupancy with 32 registers/thread. Evict_first on the streaming load may reduce cache pollution without occupancy cost IF the autotune config space does not include higher-register configs.

> **Avoid when**: The autotune config set includes configs that increase registers_per_thread above the current level (e.g., adding num_stages=3 configs or larger XBLOCK configs). If the kernel is already register-limited at 32 regs/thread with 100% theoretical occupancy, any autotune expansion that selects a 40-register config will undo the occupancy gain and increase total instruction count, causing regression. Do not bundle eviction policy changes with autotune config expansion — isolate each change.

---

## Iteration 5

**Best so far**: 1.6385× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_batchnorm_arithmetic_with_rsqrt_and_fma_to_unlock_full_occupancy | 1.942× | 1.694× | ✓ |  |
| B | fuse_multiply_add_chain_with_tl_math_fma_to_unlock_full_occupancy | 3.477× | 1.208× | ✓ |  |
| C | fuse_fp32_arithmetic_chain_with_rsqrt_and_fma_to_reduce_register_pressure_and_instruction_count | 1.942× | 1.707× | ✓ |  |
| D | collapse_batchnorm_arithmetic_to_rsqrt_plus_fma_to_unlock_full_occupancy | 1.938× | 1.716× | ✓ |  |
| E | add_triton_autotune | 1.187× | 0.886× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_batchnorm_ar... | Δ | fuse_multiply_add... | Δ | fuse_fp32_arithme... | Δ | collapse_batchnor... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 144450 ns | 85280 ns | -59170 ns (-41.0%) | 119550 ns | -24900 ns (-17.2%) | 84640 ns | -59810 ns (-41.4%) | 84190 ns | -60260 ns (-41.7%) | 163070 ns | +18620 ns (+12.9%) |
| SM Throughput (%) | 60.67 % | 70.65 % | +9.98 pp | 57.72 % | -2.95 pp | 70.88 % | +10.21 pp | 70.79 % | +10.12 pp | 60.03 % | -0.64 pp |
| Memory Throughput (%) | 70.57 % | 46.52 % | -24.05 pp | 58.34 % | -12.23 pp | 46.76 % | -23.81 pp | 46.69 % | -23.88 pp | 62.66 % | -7.91 pp |
| DRAM Throughput (%) | 67.29 % | 23.52 % | -43.77 pp | 40.11 % | -27.18 pp | 23.72 % | -43.57 pp | 23.83 % | -43.46 pp | 59.55 % | -7.74 pp |
| Achieved Occupancy (%) | 70.91 % | 93.05 % | +22.14 pp | 94.49 % | +23.58 pp | 93.06 % | +22.15 pp | 93.08 % | +22.17 pp | 70.39 % | -0.52 pp |
| Registers/Thread | 40.00 register/thread | 32.00 register/thread | -8 | 32.00 register/thread | -8 | 32.00 register/thread | -8 | 32.00 register/thread | -8 | 40.00 register/thread | +0 |
| Waves/SM | 37.12 | 27.84 | -9.28 | 27.84 | -9.28 | 27.84 | -9.28 | 27.84 | -9.28 | 37.12 | +0.00 |
| No Eligible (%) | 40.30 % | 28.09 % | -12.21 pp | 41.94 % | +1.64 pp | 27.61 % | -12.69 pp | 27.59 % | -12.71 pp | 39.87 % | -0.43 pp |
| IPC Active | 2.44 inst/cycle | 2.91 inst/cycle | +0.47 | 2.33 inst/cycle | -0.11 | 2.91 inst/cycle | +0.47 | 2.91 inst/cycle | +0.47 | 2.44 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan B: `fuse_multiply_add_chain_with_tl_math_fma_to_unlock_full_occupancy` (1.208×)

Replacing a multi-step multiply-add chain (separate FMUL + FADD) and a multi-instruction reciprocal-sqrt sequence with tl.math.fma() and tl.math.rsqrt() eliminates intermediate live temporaries, reducing registers_per_thread from 40 to 32 and crossing the critical register threshold that unlocks Block Limit Registers from 12 to 16 blocks/SM, raising theoretical occupancy from 75% to 100% and achieved occupancy from 69.8% to 94.5%. Despite the occupancy improvement, the kernel shifted from compute-bound (SM throughput 84%) to memory-bound (DRAM throughput 40%, CPIStall on L1TEX at 42% of cycles), because the higher warp count exposed more concurrent memory requests that the memory subsystem could not fully hide — warp cycles per issued instruction worsened from 12.9 to 25.9. The net 1.21x speedup came from the instruction count reduction (66M→52M, -21%) and branch divergence elimination, partially offset by the new memory-latency stall regime.

> **Applicable when**: TheoreticalOccupancy NCU rule fires with register-limited occupancy (Block Limit Registers < Block Limit Warps); kernel body contains a batch-norm-style arithmetic chain with a reciprocal-sqrt followed by separate multiply and add instructions; registers_per_thread is just above a power-of-2 boundary (e.g., 33-40 regs when 32 would unlock the next occupancy tier); FPInstructions rule shows high non-fused FP32 instruction count.

> **Avoid when**: Kernel is already memory-bound (DRAM throughput > 50%) before the change — unlocking higher occupancy will increase concurrent memory pressure and worsen L1TEX scoreboard stalls (CPIStall), potentially negating the instruction-count benefit; registers are already at or below 32/thread; the arithmetic chain does not contain a separable multiply-add that maps cleanly to FMA (e.g., non-linear activations between the multiply and add).

---

## Cumulative Progress

```
Baseline:  144450 ns  (1.000×)
Iter 1     88160 ns  (1.639×)  — add_autotune_with_vectorized_loads_and_larger_xblock

Final best: 119580 ns  (1.2080×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,33 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Existing configs retained
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4,  num_stages=3),
+        # NEW: large XBLOCK / small YBLOCK to reduce register pressure below 32/thread
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024,'YBLOCK': 1},  num_warps=8,  num_stages=2),
+        # Plan-specified configs: XBLOCK in {128,256,512,1024} x num_warps in {4,8}
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024,'YBLOCK': 1},  num_warps=8,  num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1200
@@ -20,11 +47,11 @@

     y4 = (yindex % 120)
     y0 = (yindex % 40)
     y1 = ((yindex // 40) % 3)
-    tmp0 = tl.load(in_ptr0 + (y4 + 120*x3 + 6021120*y2), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y4), ymask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (y4), ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y4 + 120*x3 + 6021120*y2), xmask & ymask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr1 + (y4), mask=ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (y4), mask=ymask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr3 + (y4), mask=ymask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr4 + (y4), mask=ymask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1, 1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -36,9 +63,7 @@

     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tl.store(out_ptr0 + (x3 + 50176*y1 + 150528*y0 + 6021120*y2), tmp15, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (x3 + 50176*y1 + 150528*y0 + 6021120*y2), tmp15, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -50,7 +75,7 @@

     ynumel = 1200
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1200, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1200, 50176)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

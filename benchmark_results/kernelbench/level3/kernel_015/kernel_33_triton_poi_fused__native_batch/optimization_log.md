# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_33` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 18340 ns — **Best latency**: 18240 ns (**1.0055×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fuse_sqrt_reciprocal_to_rsqrt_in_isolation`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_33`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_33_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (25480,1,1), Waves/SM: 12.06
- **Registers/thread**: 22
- **Duration**: 18340 ns
- **SM Throughput**: 61.06 % | **Memory**: 22.94 % | **DRAM**: 14.55 %
- **Achieved Occupancy**: 61.90 %
- **IPC Active**: 2.80 inst/cycle | **No Eligible**: 31.35 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size | 1.000× | 0.386× | ✓ | **✓ best** |
| B | autotune_xblock_tile_size_for_occupancy_and_l1tex_latency_hiding | 0.991× | 0.998× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_reduced_launch_overhead | 0.864× | 0.695× | ✓ |  |
| D | autotune_xblock_increase_tile_size | 0.990× | 0.387× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18340 ns | 47460 ns | +29120 ns (+158.8%) | 18370 ns | +30 ns (+0.2%) | 26400 ns | +8060 ns (+43.9%) | 47420 ns | +29080 ns (+158.6%) |
| SM Throughput (%) | 61.06 % | 83.29 % | +22.23 pp | 61.06 % | +0.00 pp | 79.24 % | +18.18 pp | 83.59 % | +22.53 pp |
| Memory Throughput (%) | 22.94 % | 19.20 % | -3.74 pp | 22.94 % | +0.00 pp | 23.02 % | +0.08 pp | 19.25 % | -3.69 pp |
| DRAM Throughput (%) | 14.55 % | 5.61 % | -8.94 pp | 14.52 % | -0.03 pp | 10.09 % | -4.46 pp | 5.61 % | -8.94 pp |
| Achieved Occupancy (%) | 61.90 % | 87.11 % | +25.21 pp | 62.00 % | +0.10 pp | 87.39 % | +25.49 pp | 87.02 % | +25.12 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 12.06 | 24.13 | +12.07 | 12.06 | +0.00 | 12.06 | +0.00 | 24.13 | +12.07 |
| No Eligible (%) | 31.35 % | 12.59 % | -18.76 pp | 30.52 % | -0.83 pp | 13.65 % | -17.70 pp | 12.62 % | -18.73 pp |
| IPC Active | 2.80 inst/cycle | 3.48 inst/cycle | +0.68 | 2.78 inst/cycle | -0.02 | 3.43 inst/cycle | +0.63 | 3.48 inst/cycle | +0.68 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_tile_size` (0.386×)

Adding @triton.autotune to explore larger XBLOCK values (128→1024) for a pointwise BatchNorm+ReLU kernel caused a 2.6x regression (18.3 µs → 47.5 µs). Although achieved occupancy improved from 61.9% to 87.1% and eligible warps per scheduler rose from 1.9 to 8.0, the autotune mechanism selected XBLOCK=256 which doubled registers per thread (22→32) and increased total executed instructions by 3.6x (8.7M→31.1M), overwhelming any latency-hiding benefit. The critical flaw was that the grid was left hardcoded at cdiv(xnumel, 128) instead of being updated to use meta['XBLOCK'], meaning the kernel was launched with the wrong (too-small) grid while the autotuner selected a larger block size — each thread processed more elements than intended, causing redundant/incorrect computation and massively inflating instruction count.

> **Applicable when**: Pointwise kernel with low achieved occupancy (<70%), low eligible warps per scheduler (<3), and significant L1TEX scoreboard stalls (>30% of CPI stall), where registers per thread are well below 255 and shared memory usage is zero.

> **Avoid when**: When the launch grid is not updated to be a lambda consuming meta['XBLOCK'] — a static grid sized for the original XBLOCK will cause the autotuned larger block to process out-of-bounds or redundant elements, inflating instruction count by the ratio of new/old XBLOCK and causing severe regression. Also avoid when register pressure is already high (>40/thread), as larger XBLOCK will further increase register usage and reduce occupancy.

#### ❌ Plan C: `increase_xblock_for_vectorized_loads_and_reduced_launch_overhead` (0.695×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused registers per thread to jump from 22 to 32, nearly doubling the executed instruction count (8.7M → 16.3M) and increasing kernel duration from 18.3 µs to 26.4 µs. Although achieved occupancy improved (61.9% → 87.4%) and compute throughput rose (61% → 79%), the extra instructions from wider tiles overwhelmed any vectorization benefit, resulting in a 30% slowdown. When the original kernel is already compute-bound (CPIStall from L1TEX) and the instruction count doubles with wider tiles, the additional arithmetic work from processing more elements per thread negates any memory-access efficiency gains.

> **Applicable when**: Never applicable as described — this strategy caused regression. The pattern of adding autotune with large XBLOCK to a compute-bound kernel with low register count is a known anti-pattern when the wider tile causes register spill or instruction count explosion.

> **Avoid when**: Do not increase XBLOCK via autotune when: (1) the kernel is already compute-bound (Compute SM Throughput > 60%), (2) the instruction count is expected to scale proportionally with XBLOCK (no algorithmic reduction in work), (3) the grid was not updated to use meta['XBLOCK'] (fixed grid causes incorrect launch), or (4) the register count increase from wider tiles will reduce theoretical occupancy. Also avoid when the CPIStall is from L1TEX but the kernel body has no reuse opportunity — wider tiles just do more work without cache benefit.

#### ❌ Plan D: `autotune_xblock_increase_tile_size` (0.387×)

Adding @triton.autotune with larger XBLOCK configs (256–1024) caused the autotuner to select XBLOCK=256, which increased registers per thread from 22 to 32 and caused a 3.6x explosion in executed instructions (8.7M → 31.1M). Despite improving achieved occupancy (61.9% → 87%) and reducing L1TEX stalls, the massive instruction count increase dominated, raising kernel duration from 18.3 µs to 47.4 µs (2.6x slower). When a kernel already has low register pressure and moderate occupancy, increasing tile size can trigger register pressure increases that inflate instruction counts far beyond any latency-hiding benefit.

> **Applicable when**: Kernel has high L1TEX stall fraction (>40% of warp cycles), low achieved occupancy (<65%), and low register pressure (<24 regs/thread) — suggesting tile size increase could help hide latency.

> **Avoid when**: The kernel body contains enough arithmetic that larger XBLOCK causes the compiler to allocate significantly more registers (e.g., 22→32), which can trigger instruction count explosion (>3x) that overwhelms any latency-hiding benefit. Also avoid when the grid is already large (>20K blocks) and waves_per_sm is already >10, as the GPU has sufficient parallelism. Do not apply when the autotuner may select a config that increases register pressure beyond the block limit, reducing theoretical occupancy.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_sqrt_div_with_rsqrt_and_streaming_eviction | 1.003× | 1.002× | ✓ | **✓ best** |
| B | reduce_non_fused_fp32_instructions_via_rsqrt_and_fma | 0.930× | 0.957× | ✓ |  |
| C | replace_int_constants_with_float_to_enable_fma_fusion | 0.990× | 0.387× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_instruction_explosion | 0.992× | 0.387× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_sqrt_div_... | Δ | reduce_non_fused_... | Δ | replace_int_const... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18340 ns | 18300 ns | -40 ns (-0.2%) | 19170 ns | +830 ns (+4.5%) | 47420 ns | +29080 ns (+158.6%) | 47390 ns | +29050 ns (+158.4%) |
| SM Throughput (%) | 61.06 % | 45.10 % | -15.96 pp | 51.22 % | -9.84 pp | 83.05 % | +21.99 pp | 83.52 % | +22.46 pp |
| Memory Throughput (%) | 22.94 % | 22.16 % | -0.78 pp | 32.98 % | +10.04 pp | 19.14 % | -3.80 pp | 19.26 % | -3.68 pp |
| DRAM Throughput (%) | 14.55 % | 14.56 % | +0.01 pp | 13.89 % | -0.66 pp | 5.61 % | -8.94 pp | 5.62 % | -8.93 pp |
| Achieved Occupancy (%) | 61.90 % | 42.78 % | -19.12 pp | 41.25 % | -20.65 pp | 86.79 % | +24.89 pp | 86.89 % | +24.99 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 12.06 | 12.06 | +0.00 | 12.06 | +0.00 | 24.13 | +12.07 | 24.13 | +12.07 |
| No Eligible (%) | 31.35 % | 54.16 % | +22.81 pp | 39.50 % | +8.15 pp | 12.88 % | -18.47 pp | 12.80 % | -18.55 pp |
| IPC Active | 2.80 inst/cycle | 1.83 inst/cycle | -0.97 | 2.40 inst/cycle | -0.40 | 3.48 inst/cycle | +0.68 | 3.47 inst/cycle | +0.67 |


### Key Learnings

#### ❌ Plan C: `replace_int_constants_with_float_to_enable_fma_fusion` (0.387×)

Replacing integer constants (tl.full([1], 1, tl.int32), tl.full([1], 0, tl.int32)) with float literals (1.0, 0.0) to enable FMA fusion caused a 2.6x regression (18.4 µs → 47.4 µs). The executed instruction count exploded from 8.7M to 31.1M, and registers per thread jumped from 22 to 32. Critically, the diff also changed the autotune configs (removing num_stages=2 and reordering configs), which — combined with the broken static grid (not updated to use meta['XBLOCK']) — caused the autotuner to select XBLOCK=256 with doubled register pressure, identical to the regression pattern seen in iteration 1. The float constant substitution itself was not independently tested; the confounding autotune config change was the actual driver of the regression.

> **Applicable when**: NCU FPInstructions rule flags high non-fused FP32 instruction ratio (>3:1 non-fused to fused), and the change is applied in isolation without modifying autotune configs or grid expressions.

> **Avoid when**: Do not apply when the diff simultaneously modifies autotune configs (especially removing num_stages or reordering configs) and the launch grid is a static expression not updated to use meta['XBLOCK']. The confounding autotune change will dominate and cause the same register-pressure/instruction-explosion regression seen in prior iterations. Also do not apply when the kernel is already at 18 µs with 22 regs/thread — the FMA fusion opportunity may be smaller than NCU's model predicts if the compiler already partially fuses operations.

#### ❌ Plan D: `fix_autotune_grid_lambda_to_eliminate_instruction_explosion` (0.389×)

For the fourth consecutive iteration, the plan claimed to fix the autotune grid lambda but the actual diff shows the grid was NOT changed to a lambda — only autotune configs were reshuffled (num_stages removed, XBLOCK range expanded to 1024) and minor arithmetic changes were made (tl.math.rsqrt → tl.sqrt_rn + reciprocal, eviction policy removed, int constant → float in tl.maximum). The autotuner again selected XBLOCK=256, registers per thread jumped from 18 to 32, and executed instructions exploded from 5.7M to 31.1M (5.4x), yielding the same ~2.6x regression (18.3 µs → 47.1 µs). No combination of config reshuffling, arithmetic intrinsic changes, or eviction policy removal can fix a grid-sizing bug; the grid lambda at the kernel call site must be explicitly changed.

> **Applicable when**: Autotune grid lambda bug is confirmed: block_size in profiling does not match the XBLOCK value used to compute the static grid, and executed instructions are proportionally higher than expected (>3x baseline).

> **Avoid when**: Do not assume reshuffling autotune configs, removing num_stages, reordering configs, changing arithmetic intrinsics (rsqrt vs sqrt+reciprocal), removing eviction policies, or replacing int constants with float literals will fix a grid-sizing bug. The grid lambda at the call site must be explicitly changed. Do not apply autotune at all to this pointwise BN+ReLU kernel class since even a correctly-sized grid with XBLOCK=256 increases register pressure (18→32) and instruction count (5.4x), causing severe regression.

---

## Iteration 3

**Best so far**: 1.0022× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_autotune_num_warps_at_fixed_xblock | 0.996× | 1.002× | ✓ |  |
| B | fix_grid_lambda_and_streaming_eviction_for_l1tex_latency | 0.934× | 0.954× | ✓ |  |
| C | remove_broken_autotune_restore_static_xblock128 | 0.994× | 1.000× | ✓ |  |
| D | fix_autotune_grid_lambda_to_restore_baseline_performance | 0.998× | 0.386× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_grid_lambda_a... | Δ | remove_broken_aut... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18340 ns | 18300 ns | -40 ns (-0.2%) | 19230 ns | +890 ns (+4.9%) | 18340 ns | +0 ns (+0.0%) | 47460 ns | +29120 ns (+158.8%) |
| SM Throughput (%) | 61.06 % | 45.49 % | -15.57 pp | 53.87 % | -7.19 pp | 61.07 % | +0.01 pp | 83.62 % | +22.56 pp |
| Memory Throughput (%) | 22.94 % | 22.36 % | -0.58 pp | 32.70 % | +9.76 pp | 22.99 % | +0.05 pp | 19.25 % | -3.69 pp |
| DRAM Throughput (%) | 14.55 % | 14.57 % | +0.02 pp | 13.86 % | -0.69 pp | 14.54 % | -0.01 pp | 5.61 % | -8.94 pp |
| Achieved Occupancy (%) | 61.90 % | 42.14 % | -19.76 pp | 42.09 % | -19.81 pp | 61.66 % | -0.24 pp | 86.89 % | +24.99 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 32.00 register/thread | +10 |
| Waves/SM | 12.06 | 12.06 | +0.00 | 12.06 | +0.00 | 12.06 | +0.00 | 24.13 | +12.07 |
| No Eligible (%) | 31.35 % | 54.00 % | +22.65 pp | 37.12 % | +5.77 pp | 30.11 % | -1.24 pp | 12.59 % | -18.76 pp |
| IPC Active | 2.80 inst/cycle | 1.84 inst/cycle | -0.96 | 2.52 inst/cycle | -0.28 | 2.80 inst/cycle | +0.00 | 3.49 inst/cycle | +0.69 |

---

## Iteration 4

**Best so far**: 1.0022× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_and_pipeline_stages_for_l1tex_latency | 0.996× | 0.988× | ✓ |  |
| B | fuse_sqrt_reciprocal_to_rsqrt_in_isolation | 1.006× | 1.006× | ✓ | **✓ best** |
| C | fix_grid_lambda_to_unlock_existing_autotune | 1.005× | 0.998× | ✓ |  |
| D | fix_static_grid_to_lambda_for_correct_autotune_dispatch | 0.999× | 0.389× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_and_pi... | Δ | fuse_sqrt_recipro... | Δ | fix_grid_lambda_t... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18340 ns | 18560 ns | +220 ns (+1.2%) | 18240 ns | -100 ns (-0.5%) | 18370 ns | +30 ns (+0.2%) | 47140 ns | +28800 ns (+157.0%) |
| SM Throughput (%) | 61.06 % | 45.27 % | -15.79 pp | 45.59 % | -15.47 pp | 44.48 % | -16.58 pp | 83.54 % | +22.48 pp |
| Memory Throughput (%) | 22.94 % | 22.30 % | -0.64 pp | 22.52 % | -0.42 pp | 21.88 % | -1.06 pp | 19.25 % | -3.69 pp |
| DRAM Throughput (%) | 14.55 % | 14.35 % | -0.20 pp | 14.60 % | +0.05 pp | 14.51 % | -0.04 pp | 5.65 % | -8.90 pp |
| Achieved Occupancy (%) | 61.90 % | 42.13 % | -19.77 pp | 41.22 % | -20.68 pp | 42.04 % | -19.86 pp | 86.96 % | +25.06 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 32.00 register/thread | +10 |
| Waves/SM | 12.06 | 12.06 | +0.00 | 12.06 | +0.00 | 12.06 | +0.00 | 24.13 | +12.07 |
| No Eligible (%) | 31.35 % | 53.92 % | +22.57 pp | 57.97 % | +26.62 pp | 54.35 % | +23.00 pp | 12.59 % | -18.76 pp |
| IPC Active | 2.80 inst/cycle | 1.84 inst/cycle | -0.96 | 1.68 inst/cycle | -1.12 | 1.81 inst/cycle | -0.99 | 3.48 inst/cycle | +0.68 |

---

## Iteration 5

**Best so far**: 1.0055× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 0.997× | 1.002× | ✓ |  |
| B | autotune_num_warps_at_fixed_xblock128_to_hide_l1tex_latency | 1.005× | 1.002× | ✓ |  |
| C | reduce_xblock_to_64_for_higher_block_count_and_l1tex_latency_hiding | 1.003× | 1.004× | ✓ |  |
| D | restore_static_xblock128_to_eliminate_autotune_instruction_explosion | 1.001× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | autotune_num_warp... | Δ | reduce_xblock_to_... | Δ | restore_static_xb... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18340 ns | 18300 ns | -40 ns (-0.2%) | 18300 ns | -40 ns (-0.2%) | 18270 ns | -70 ns (-0.4%) | 18270 ns | -70 ns (-0.4%) |
| SM Throughput (%) | 61.06 % | 45.70 % | -15.36 pp | 45.54 % | -15.52 pp | 22.70 % | -38.36 pp | 60.95 % | -0.11 pp |
| Memory Throughput (%) | 22.94 % | 22.50 % | -0.44 pp | 22.40 % | -0.54 pp | 10.81 % | -12.13 pp | 22.90 % | -0.04 pp |
| DRAM Throughput (%) | 14.55 % | 14.56 % | +0.01 pp | 14.55 % | +0.00 pp | 7.29 % | -7.26 pp | 14.59 % | +0.04 pp |
| Achieved Occupancy (%) | 61.90 % | 41.46 % | -20.44 pp | 42.64 % | -19.26 pp | 25.60 % | -36.30 pp | 61.78 % | -0.12 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 22.00 register/thread | +0 |
| Waves/SM | 12.06 | 12.06 | +0.00 | 12.06 | +0.00 | 6.03 | -6.03 | 12.06 | +0.00 |
| No Eligible (%) | 31.35 % | 58.24 % | +26.89 pp | 53.84 % | +22.49 pp | 74.88 % | +43.53 pp | 30.81 % | -0.54 pp |
| IPC Active | 2.80 inst/cycle | 1.67 inst/cycle | -1.13 | 1.83 inst/cycle | -0.97 | 0.95 inst/cycle | -1.85 | 2.80 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  18340 ns  (1.000×)
Iter 2     18300 ns  (1.002×)  — replace_sqrt_div_with_rsqrt_and_streaming_eviction
Iter 4     18240 ns  (1.006×)  — fuse_sqrt_reciprocal_to_rsqrt_in_isolation

Final best: 18240 ns  (1.0055×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
@@ -21,19 +22,12 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = libdevice.rsqrt(tmp5)  # single SFU rsqrt replaces 5-op chain
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
+    tmp17 = tl.maximum(0.0, tmp15)  # float literal enables FP comparison fusion
     tl.store(out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

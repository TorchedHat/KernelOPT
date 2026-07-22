# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_214` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6690 ns — **Best latency**: 6500 ns (**1.0292×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_int32_to_float32_cast_in_relu_maximum`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_214`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_214_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6248,1,1), Waves/SM: 2.96
- **Registers/thread**: 22
- **Duration**: 6690 ns
- **SM Throughput**: 41.59 % | **Memory**: 15.95 % | **DRAM**: 9.92 %
- **Achieved Occupancy**: 62.33 %
- **IPC Active**: 2.55 inst/cycle | **No Eligible**: 35.81 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.948× | 0.271× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 1.000× | 1.024× | ✓ | **✓ best** |
| C | increase_xblock_with_autotune_for_vectorized_loads_and_wave_elimination | 0.944× | 0.964× | ✓ |  |
| D | increase_xblock_with_autotune_to_eliminate_partial_wave_and_improve_occupancy | 0.933× | 0.794× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_e... | Δ | increase_xblock_w... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6690 ns | 24670 ns | +17980 ns (+268.8%) | 6530 ns | -160 ns (-2.4%) | 6940 ns | +250 ns (+3.7%) | 8420 ns | +1730 ns (+25.9%) |
| SM Throughput (%) | 41.59 % | 79.60 % | +38.01 pp | 41.76 % | +0.17 pp | 51.85 % | +10.26 pp | 62.48 % | +20.89 pp |
| Memory Throughput (%) | 15.95 % | 14.93 % | -1.02 pp | 15.85 % | -0.10 pp | 19.10 % | +3.15 pp | 16.85 % | +0.90 pp |
| DRAM Throughput (%) | 9.92 % | 2.67 % | -7.25 pp | 10.14 % | +0.22 pp | 9.53 % | -0.39 pp | 7.85 % | -2.07 pp |
| Achieved Occupancy (%) | 62.33 % | 85.59 % | +23.26 pp | 62.13 % | -0.20 pp | 73.22 % | +10.89 pp | 79.68 % | +17.35 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 26.00 register/thread | +4 | 32.00 register/thread | +10 |
| Waves/SM | 2.96 | 11.83 | +8.87 | 2.96 | +0.00 | 2.96 | +0.00 | 2.96 | +0.00 |
| No Eligible (%) | 35.81 % | 12.94 % | -22.87 pp | 36.95 % | +1.14 pp | 21.02 % | -14.79 pp | 17.49 % | -18.32 pp |
| IPC Active | 2.55 inst/cycle | 3.47 inst/cycle | +0.92 | 2.53 inst/cycle | -0.02 | 3.13 inst/cycle | +0.58 | 3.27 inst/cycle | +0.72 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.271×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) while keeping the launch grid hardcoded to cdiv(xnumel, 128) instead of using a lambda with meta['XBLOCK'] caused the grid to remain at 6248 blocks regardless of the autotuned XBLOCK. This resulted in each block processing 4x more elements than the tensor contains (XBLOCK=512 with grid=6248 covers 3.2M threads vs 799K elements), causing massive redundant computation: executed instructions grew 7.25x (2.1M → 15.4M) and duration grew 3.7x (6.69 → 24.67 µs). When using @triton.autotune, the launch grid MUST be a lambda consuming meta['XBLOCK'], not a static cdiv expression; otherwise the grid does not adapt to the selected config.

> **Applicable when**: Always: this is a correctness/implementation constraint, not a profiling-signal-dependent strategy. Any time @triton.autotune is used with a tile-size config parameter.

> **Avoid when**: N/A — the static grid pattern is always wrong when using autotune with variable tile sizes.

---

## Iteration 2

**Best so far**: 1.0245× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_tune_num_warps_for_latency_hiding | 0.974× | 1.024× | ✓ |  |
| B | fix_autotune_grid_lambda_and_reduce_num_warps_for_cache_coherence | 0.965× | 0.996× | ✓ |  |
| C | fix_autotune_grid_lambda_to_enable_correct_xblock_scaling | 0.909× | 0.804× | ✓ |  |
| D | fix_autotune_grid_lambda_and_fuse_bn_arithmetic | 0.979× | 0.439× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6690 ns | 6530 ns | -160 ns (-2.4%) | 6720 ns | +30 ns (+0.4%) | 8320 ns | +1630 ns (+24.4%) | 15230 ns | +8540 ns (+127.7%) |
| SM Throughput (%) | 41.59 % | 41.50 % | -0.09 pp | 30.10 % | -11.49 pp | 62.47 % | +20.88 pp | 66.38 % | +24.79 pp |
| Memory Throughput (%) | 15.95 % | 15.93 % | -0.02 pp | 14.59 % | -1.36 pp | 16.84 % | +0.89 pp | 25.05 % | +9.10 pp |
| DRAM Throughput (%) | 9.92 % | 10.13 % | +0.21 pp | 9.83 % | -0.09 pp | 7.93 % | -1.99 pp | 4.31 % | -5.61 pp |
| Achieved Occupancy (%) | 62.33 % | 61.69 % | -0.64 pp | 51.48 % | -10.85 pp | 79.87 % | +17.54 pp | 79.39 % | +17.06 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 26.00 register/thread | +4 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 2.96 | 2.96 | +0.00 | 1.48 | -1.48 | 2.96 | +0.00 | 11.83 | +8.87 |
| No Eligible (%) | 35.81 % | 35.32 % | -0.49 pp | 40.22 % | +4.41 pp | 17.43 % | -18.38 pp | 23.19 % | -12.62 pp |
| IPC Active | 2.55 inst/cycle | 2.52 inst/cycle | -0.03 | 2.21 inst/cycle | -0.34 | 3.22 inst/cycle | +0.67 | 3.07 inst/cycle | +0.52 |


### Key Learnings

#### ❌ Plan D: `fix_autotune_grid_lambda_and_fuse_bn_arithmetic` (0.439×)

Fixing the autotune grid lambda (static cdiv → lambda meta: cdiv(xnumel, meta['XBLOCK'])) and replacing the non-fused reciprocal-sqrt sequence with tl.math.rsqrt() still caused a 2.27x slowdown. Despite the grid lambda being correct, autotune selected XBLOCK=512 which caused registers per thread to jump from 22 to 32, reducing Block Limit Registers from 21 to 4 blocks/SM and inflating executed instructions from 2.1M to 8.0M (3.8x). The waves_per_sm grew from 2.96 to 11.83, meaning the larger tile size caused far more total work per SM rather than reducing the partial-wave tail. The dominant causal chain is: larger XBLOCK → more register pressure → fewer concurrent blocks → more waves → higher total instruction count → slower kernel, which overwhelms any benefit from the grid fix or rsqrt fusion.

> **Applicable when**: Never for this kernel profile — do not apply when the baseline kernel uses 22 registers/thread with XBLOCK=128 and any larger XBLOCK causes register count to jump to 32+, as the register-pressure cliff dominates all other optimizations.

> **Avoid when**: Kernel has low register count (≤24 registers/thread) at the baseline tile size that is near a register-occupancy cliff; pointwise BN+ReLU kernels where the compiler allocates significantly more registers for larger tiles; cases where waves_per_sm is already near 3 (adequate latency hiding) and the partial-wave tail is not the true bottleneck. Do not combine grid lambda fix with autotune larger XBLOCK configs when register pressure is the limiting factor.

---

## Iteration 3

**Best so far**: 1.0245× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_nonfused_fp32_instructions_via_rsqrt_at_fixed_xblock | 1.005× | 1.062× | ✓ |  |
| B | fuse_reciprocal_sqrt_to_rsqrt_instruction | 1.014× | 1.055× | ✓ |  |
| C | fuse_sqrt_reciprocal_to_rsqrt_at_fixed_xblock128 | 1.012× | 1.024× | ✓ |  |
| D | revert_to_small_xblock_and_cap_register_pressure | 1.015× | 1.029× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.970× | 1.024× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_nonfused_f... | Δ | fuse_reciprocal_s... | Δ | fuse_sqrt_recipro... | Δ | revert_to_small_x... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6690 ns | 6300 ns | -390 ns (-5.8%) | 6340 ns | -350 ns (-5.2%) | 6530 ns | -160 ns (-2.4%) | 6500 ns | -190 ns (-2.8%) | 6530 ns | -160 ns (-2.4%) |
| SM Throughput (%) | 41.59 % | 31.64 % | -9.95 pp | 16.13 % | -25.46 pp | 32.48 % | -9.11 pp | 32.45 % | -9.14 pp | 39.36 % | -2.23 pp |
| Memory Throughput (%) | 15.95 % | 15.94 % | -0.01 pp | 15.67 % | -0.28 pp | 16.40 % | +0.45 pp | 16.38 % | +0.43 pp | 15.44 % | -0.51 pp |
| DRAM Throughput (%) | 9.92 % | 10.51 % | +0.59 pp | 10.42 % | +0.50 pp | 10.12 % | +0.20 pp | 10.20 % | +0.28 pp | 10.13 % | +0.21 pp |
| Achieved Occupancy (%) | 62.33 % | 49.67 % | -12.66 pp | 41.67 % | -20.66 pp | 50.76 % | -11.57 pp | 50.64 % | -11.69 pp | 62.41 % | +0.08 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 24.00 register/thread | +2 | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 22.00 register/thread | +0 |
| Waves/SM | 2.96 | 2.96 | +0.00 | 1.48 | -1.48 | 2.96 | +0.00 | 2.96 | +0.00 | 2.96 | +0.00 |
| No Eligible (%) | 35.81 % | 58.57 % | +22.76 pp | 65.71 % | +29.90 pp | 55.68 % | +19.87 pp | 51.87 % | +16.06 pp | 36.71 % | +0.90 pp |
| IPC Active | 2.55 inst/cycle | 1.72 inst/cycle | -0.83 | 1.25 inst/cycle | -1.30 | 1.73 inst/cycle | -0.82 | 1.88 inst/cycle | -0.67 | 2.54 inst/cycle | -0.01 |


### Key Learnings

#### ✅ Plan A: `reduce_nonfused_fp32_instructions_via_rsqrt_at_fixed_xblock` (1.062×)

Replacing a verbose 5-instruction reciprocal-sqrt sequence (tl.sqrt_rn + int32(1)/float + multiply by 1.0) with a single tl.math.rsqrt() call reduced executed instructions by 37.6% (2.12M → 1.32M) and duration by 3.5% (6.53 → 6.30 µs). Critically, this was done at a fixed XBLOCK=128 with no autotune, preserving register count (22→20 regs/thread) and waves_per_sm (2.96). Removing the @triton.autotune decorator also eliminated branch divergence (Avg. Divergent Branches: 47.33 → 0.0) and reduced active threads per warp variance (28.38 → 32.0), indicating the autotune's config-selection branches were themselves causing warp divergence. The FPInstructions NCU rule is the reliable signal: when non-fused FP32 instructions dominate (>70% of total FP32 ops), replacing multi-step arithmetic with hardware intrinsics (rsqrt, rcp) at the same tile size yields modest but reliable speedups.

> **Applicable when**: FPInstructions NCU rule fires with non-fused FP32 fraction >70%; kernel contains multi-step reciprocal-sqrt patterns (sqrt + int_div + float_mul); XBLOCK is fixed (no autotune) so register count is stable; the kernel is not already memory-bandwidth bound (DRAM throughput <30%).

> **Avoid when**: Kernel is already at peak memory bandwidth (DRAM throughput >80%); the arithmetic simplification is combined with autotune or larger XBLOCK configs that push registers past a register-occupancy cliff (e.g., 22→32 regs causing Block Limit Registers to drop from 21 to 4); inputs to rsqrt may be negative or zero (rsqrt is only valid for positive inputs).

---

## Iteration 4

**Best so far**: 1.0292× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_with_proportional_xblock_for_l1tex_latency_hiding | 1.006× | 1.029× | ✓ |  |
| B | increase_num_warps_at_fixed_xblock128_for_latency_hiding | 1.011× | 1.024× | ✓ |  |
| C | increase_num_warps_for_l1tex_latency_hiding | 0.989× | 1.055× | ✓ |  |
| D | increase_num_warps_for_l1tex_latency_hiding_at_fixed_xblock | 1.007× | 1.062× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6690 ns | 6500 ns | -190 ns (-2.8%) | 6530 ns | -160 ns (-2.4%) | 6340 ns | -350 ns (-5.2%) | 6300 ns | -390 ns (-5.8%) |
| SM Throughput (%) | 41.59 % | 32.21 % | -9.38 pp | 31.92 % | -9.67 pp | 32.49 % | -9.10 pp | 32.51 % | -9.08 pp |
| Memory Throughput (%) | 15.95 % | 16.24 % | +0.29 pp | 16.09 % | +0.14 pp | 16.36 % | +0.41 pp | 16.41 % | +0.46 pp |
| DRAM Throughput (%) | 9.92 % | 10.17 % | +0.25 pp | 10.14 % | +0.22 pp | 10.45 % | +0.53 pp | 10.48 % | +0.56 pp |
| Achieved Occupancy (%) | 62.33 % | 50.82 % | -11.51 pp | 50.33 % | -12.00 pp | 50.94 % | -11.39 pp | 50.98 % | -11.35 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 20.00 register/thread | -2 |
| Waves/SM | 2.96 | 2.96 | +0.00 | 2.96 | +0.00 | 2.96 | +0.00 | 2.96 | +0.00 |
| No Eligible (%) | 35.81 % | 56.04 % | +20.23 pp | 57.02 % | +21.21 pp | 54.03 % | +18.22 pp | 56.87 % | +21.06 pp |
| IPC Active | 2.55 inst/cycle | 1.73 inst/cycle | -0.82 | 1.73 inst/cycle | -0.82 | 1.88 inst/cycle | -0.67 | 1.74 inst/cycle | -0.81 |


### Key Learnings

#### ✅ Plan D: `increase_num_warps_for_l1tex_latency_hiding_at_fixed_xblock` (1.062×)

When CPIStall on L1TEX scoreboard dependency accounts for ~50% of warp stall cycles and the scheduler 'No Eligible' rate exceeds 55%, increasing num_warps at a fixed XBLOCK can provide more warps to interleave during L1TEX stalls. In this case, the actual profiling metrics showed only marginal improvement: achieved occupancy rose from 49.67% to 50.98%, active warps per scheduler from 7.68 to 8.61, and No Eligible dropped from 58.57% to 56.87% — all small changes that produced a modest 1.06x speedup. The kernel duration remained at 6.3 µs and instruction count was unchanged, indicating the num_warps hint had limited effect, likely because the 128-thread block already constrains the hardware warp count to 4 and the compiler may not fully honor num_warps > block_size/32 in all cases.

> **Applicable when**: CPIStall on L1TEX scoreboard accounts for >45% of warp stall cycles; scheduler No Eligible rate >50%; active warps per scheduler <10; registers per thread are well below the register-occupancy cliff (e.g., 20 regs with Block Limit Registers=21); XBLOCK is fixed and small (128 threads = 4 hardware warps).

> **Avoid when**: Kernel is already memory-bandwidth bound (DRAM throughput >60%); block size is large enough that hardware warp count already saturates the scheduler (e.g., 512+ threads = 16+ warps); register pressure is near the occupancy cliff (registers per thread near Block Limit Registers threshold); the speedup signal is weak (<1.05x) and other bottlenecks (partial-wave tail, L2 imbalance) dominate.

---

## Iteration 5

**Best so far**: 1.0292× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_size_to_256_threads_with_xblock256_for_latency_hiding_and_partial_wave_reduction | 1.013× | 1.062× | ✓ |  |
| B | eliminate_int32_to_float32_cast_in_relu_maximum | 1.009× | 1.029× | ✓ |  |
| C | replace_int32_relu_with_float_fmax_to_enable_fma_fusion | 1.011× | 1.055× | ✓ |  |
| D | enable_l2_compression_via_streaming_store_cache_modifier | 0.984× | 1.062× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_si... | Δ | eliminate_int32_t... | Δ | replace_int32_rel... | Δ | enable_l2_compres... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6690 ns | 6300 ns | -390 ns (-5.8%) | 6500 ns | -190 ns (-2.8%) | 6340 ns | -350 ns (-5.2%) | 6300 ns | -390 ns (-5.8%) |
| SM Throughput (%) | 41.59 % | 31.71 % | -9.88 pp | 32.45 % | -9.14 pp | 32.50 % | -9.09 pp | 31.85 % | -9.74 pp |
| Memory Throughput (%) | 15.95 % | 15.99 % | +0.04 pp | 16.37 % | +0.42 pp | 16.37 % | +0.42 pp | 16.08 % | +0.13 pp |
| DRAM Throughput (%) | 9.92 % | 10.47 % | +0.55 pp | 10.14 % | +0.22 pp | 10.44 % | +0.52 pp | 10.46 % | +0.54 pp |
| Achieved Occupancy (%) | 62.33 % | 50.70 % | -11.63 pp | 51.07 % | -11.26 pp | 50.72 % | -11.61 pp | 50.84 % | -11.49 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 20.00 register/thread | -2 |
| Waves/SM | 2.96 | 2.96 | +0.00 | 2.96 | +0.00 | 2.96 | +0.00 | 2.96 | +0.00 |
| No Eligible (%) | 35.81 % | 56.44 % | +20.63 pp | 53.47 % | +17.66 pp | 56.16 % | +20.35 pp | 56.34 % | +20.53 pp |
| IPC Active | 2.55 inst/cycle | 1.72 inst/cycle | -0.83 | 1.91 inst/cycle | -0.64 | 1.74 inst/cycle | -0.81 | 1.71 inst/cycle | -0.84 |

---

## Cumulative Progress

```
Baseline:  6690 ns  (1.000×)
Iter 1     6530 ns  (1.024×)  — autotune_xblock_eliminate_partial_wave
Iter 3     6500 ns  (1.029×)  — revert_to_small_xblock_and_cap_register_pressure

Final best: 6500 ns  (1.0292×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_214(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
@@ -21,19 +22,13 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = tl.math.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
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

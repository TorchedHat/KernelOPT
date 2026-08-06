# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_212` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6460 ns — **Best latency**: 6460 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_212`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_212_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6125,1,1), Waves/SM: 2.90
- **Registers/thread**: 22
- **Duration**: 6460 ns
- **SM Throughput**: 40.60 % | **Memory**: 15.42 % | **DRAM**: 10.01 %
- **Achieved Occupancy**: 61.49 %
- **IPC Active**: 2.54 inst/cycle | **No Eligible**: 36.30 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.958× | 0.804× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave_and_improve_cache_reuse | 1.000× | 1.000× | ✓ |  |
| C | vectorized_loads_stores_via_larger_xblock | 0.938× | 0.287× | ✓ |  |
| D | autotune_xblock_eliminate_partial_wave | 0.963× | 0.818× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | vectorized_loads_... | Δ | autotune_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6460 ns | 8030 ns | +1570 ns (+24.3%) | 6460 ns | +0 ns (+0.0%) | 22500 ns | +16040 ns (+248.3%) | 7900 ns | +1440 ns (+22.3%) |
| SM Throughput (%) | 40.60 % | 62.23 % | +21.63 pp | 40.85 % | +0.25 pp | 77.94 % | +37.34 pp | 60.96 % | +20.36 pp |
| Memory Throughput (%) | 15.42 % | 16.96 % | +1.54 pp | 15.50 % | +0.08 pp | 16.00 % | +0.58 pp | 16.62 % | +1.20 pp |
| DRAM Throughput (%) | 10.01 % | 8.04 % | -1.97 pp | 10.03 % | +0.02 pp | 2.87 % | -7.14 pp | 8.18 % | -1.83 pp |
| Achieved Occupancy (%) | 61.49 % | 79.48 % | +17.99 pp | 61.17 % | -0.32 pp | 54.82 % | -6.67 pp | 78.95 % | +17.46 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 48.00 register/thread | +26 | 32.00 register/thread | +10 |
| Waves/SM | 2.90 | 2.90 | +0.00 | 2.90 | +0.00 | 9.28 | +6.38 | 2.90 | +0.00 |
| No Eligible (%) | 36.30 % | 16.87 % | -19.43 pp | 35.63 % | -0.67 pp | 13.97 % | -22.33 pp | 16.91 % | -19.39 pp |
| IPC Active | 2.54 inst/cycle | 3.20 inst/cycle | +0.66 | 2.50 inst/cycle | -0.04 | 3.43 inst/cycle | +0.89 | 3.29 inst/cycle | +0.75 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.804×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) increased registers per thread from 22 to 32 and nearly doubled executed instructions (2.08M → 3.93M), causing a 24% wall-clock regression despite improving achieved occupancy (61.5% → 79.5%) and compute throughput (40.6% → 62.2%). The autotune selected a larger XBLOCK that unrolled more work per block, inflating the instruction count and register pressure without a commensurate reduction in memory latency — warp cycles per issued instruction remained essentially unchanged (~15.6 → ~15.3). When the kernel is already latency-bound (CPIStall on L1TEX) and the grid is small enough that the partial-wave tail is modest, increasing tile size amplifies instruction overhead without meaningfully improving cache reuse or reducing stall cycles.

> **Applicable when**: Kernel is memory-bandwidth-bound with low instruction count and low register pressure; increasing tile size meaningfully reduces grid overhead (e.g., >50% partial-wave waste) AND does not increase register spill or instruction count significantly.

> **Avoid when**: Kernel is already latency-bound (CPIStall on L1TEX) with warp cycles per issued instruction ~15+; the kernel body is instruction-heavy (BN normalization with multiple FP32 ops per element); register pressure is near the block limit (registers per thread close to 32); the grid is static and not updated to use a lambda — autotune will select a large XBLOCK but the grid remains sized for the original small XBLOCK, causing each block to process more elements than intended or wasting threads.

#### ❌ Plan D: `autotune_xblock_eliminate_partial_wave` (0.818×)

Adding a @triton.autotune decorator with larger XBLOCK configs (512–2048) on a pointwise kernel caused a 22% regression despite improving occupancy (61% → 79%) and compute throughput (40% → 61%). The root cause is that the grid lambda was not updated to use meta['XBLOCK'], so the kernel was still launched with the fixed grid=(cdiv(xnumel,128),) while the autotuner selected a larger XBLOCK — this caused each thread to process more elements, nearly doubling executed instructions (2.08M → 3.93M) and increasing register pressure (22 → 32 registers/thread), which more than offset any partial-wave or occupancy gains. When applying autotune to change XBLOCK, the launch grid must be updated to lambda meta: (cdiv(xnumel, meta['XBLOCK']),) or the instruction count will scale with XBLOCK while block count stays fixed.

> **Applicable when**: Never apply this specific pattern (autotune XBLOCK with fixed grid). The corrected pattern (autotune XBLOCK with dynamic grid lambda) is applicable when NCU LaunchConfiguration rule fires with >20% estimated speedup and the kernel is purely pointwise with no shared memory.

> **Avoid when**: Do not add @triton.autotune for XBLOCK without simultaneously converting the launch grid to a lambda that reads meta['XBLOCK']. If the grid remains fixed at cdiv(xnumel, old_XBLOCK), each block will process more elements than intended, inflating instruction count and register pressure without reducing block count.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_increase_num_warps_for_latency_hiding | 1.018× | 1.000× | ✓ |  |
| B | fix_grid_lambda_and_fuse_fp32_instructions_to_fma | 1.033× | 1.000× | ✓ |  |
| C | reduce_register_pressure_via_rsqrt_arithmetic_simplification | 1.039× | 1.000× | ✓ | **✓ best** |
| D | fix_autotune_grid_lambda_to_restore_baseline_performance | 1.021× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_grid_lambda_a... | Δ | reduce_register_p... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6460 ns | 6460 ns | +0 ns (+0.0%) | 6460 ns | +0 ns (+0.0%) | 6460 ns | +0 ns (+0.0%) | 6460 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 40.60 % | 40.47 % | -0.13 pp | 36.05 % | -4.55 pp | 31.00 % | -9.60 pp | 40.08 % | -0.52 pp |
| Memory Throughput (%) | 15.42 % | 15.37 % | -0.05 pp | 23.45 % | +8.03 pp | 15.75 % | +0.33 pp | 15.22 % | -0.20 pp |
| DRAM Throughput (%) | 10.01 % | 10.05 % | +0.04 pp | 10.02 % | +0.01 pp | 10.00 % | -0.01 pp | 10.02 % | +0.01 pp |
| Achieved Occupancy (%) | 61.49 % | 61.90 % | +0.41 pp | 48.90 % | -12.59 pp | 49.16 % | -12.33 pp | 60.20 % | -1.29 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 18.00 register/thread | -4 | 22.00 register/thread | +0 |
| Waves/SM | 2.90 | 2.90 | +0.00 | 2.90 | +0.00 | 2.90 | +0.00 | 2.90 | +0.00 |
| No Eligible (%) | 36.30 % | 35.68 % | -0.62 pp | 37.23 % | +0.93 pp | 60.24 % | +23.94 pp | 35.80 % | -0.50 pp |
| IPC Active | 2.54 inst/cycle | 2.54 inst/cycle | +0.00 | 2.55 inst/cycle | +0.01 | 1.58 inst/cycle | -0.96 | 2.47 inst/cycle | -0.07 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_instructions_via_rsqrt_and_eliminate_int_cast_chain | 1.032× | 1.000× | ✓ |  |
| B | streaming_eviction_policy_for_non_reused_tensors | 1.034× | 0.970× | ✓ |  |
| C | increase_num_warps_for_l1tex_latency_hiding | 1.009× | 1.000× | ✓ |  |
| D | increase_num_warps_for_l1tex_latency_hiding | 1.002× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_instruc... | Δ | streaming_evictio... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6460 ns | 6460 ns | +0 ns (+0.0%) | 6660 ns | +200 ns (+3.1%) | 6460 ns | +0 ns (+0.0%) | 6430 ns | -30 ns (-0.5%) |
| SM Throughput (%) | 40.60 % | 30.56 % | -10.04 pp | 37.35 % | -3.25 pp | 30.94 % | -9.66 pp | 39.52 % | -1.08 pp |
| Memory Throughput (%) | 15.42 % | 15.31 % | -0.11 pp | 23.05 % | +7.63 pp | 15.70 % | +0.28 pp | 14.94 % | -0.48 pp |
| DRAM Throughput (%) | 10.01 % | 10.03 % | +0.02 pp | 9.77 % | -0.24 pp | 10.04 % | +0.03 pp | 10.04 % | +0.03 pp |
| Achieved Occupancy (%) | 61.49 % | 48.20 % | -13.29 pp | 49.59 % | -11.90 pp | 48.16 % | -13.33 pp | 61.53 % | +0.04 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 32.00 register/thread | +10 | 18.00 register/thread | -4 | 22.00 register/thread | +0 |
| Waves/SM | 2.90 | 2.90 | +0.00 | 2.90 | +0.00 | 2.90 | +0.00 | 2.90 | +0.00 |
| No Eligible (%) | 36.30 % | 60.51 % | +24.21 pp | 33.14 % | -3.16 pp | 60.41 % | +24.11 pp | 38.29 % | +1.99 pp |
| IPC Active | 2.54 inst/cycle | 1.54 inst/cycle | -1.00 | 2.64 inst/cycle | +0.10 | 1.59 inst/cycle | -0.95 | 2.53 inst/cycle | -0.01 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_int_div_sqrt_with_rsqrt_for_fma_fusion | 1.038× | 1.000× | ✓ |  |
| B | software_pipeline_num_stages_to_hide_l1tex_latency | 1.037× | 1.005× | ✓ |  |
| C | increase_xblock_256_with_correct_grid_and_8_warps | 1.036× | 0.994× | ✓ |  |
| D | fuse_bn_arithmetic_to_fma_via_precomputed_scale_bias | 0.999× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_int_div_s... | Δ | software_pipeline... | Δ | increase_xblock_2... | Δ | fuse_bn_arithmeti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6460 ns | 6460 ns | +0 ns (+0.0%) | 6430 ns | -30 ns (-0.5%) | 6500 ns | +40 ns (+0.6%) | 6530 ns | +70 ns (+1.1%) |
| SM Throughput (%) | 40.60 % | 30.77 % | -9.83 pp | 31.07 % | -9.53 pp | 31.08 % | -9.52 pp | 37.75 % | -2.85 pp |
| Memory Throughput (%) | 15.42 % | 15.68 % | +0.26 pp | 15.81 % | +0.39 pp | 15.71 % | +0.29 pp | 23.16 % | +7.74 pp |
| DRAM Throughput (%) | 10.01 % | 10.04 % | +0.03 pp | 10.03 % | +0.02 pp | 9.96 % | -0.05 pp | 9.93 % | -0.08 pp |
| Achieved Occupancy (%) | 61.49 % | 49.56 % | -11.93 pp | 48.65 % | -12.84 pp | 50.84 % | -10.65 pp | 49.06 % | -12.43 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 32.00 register/thread | +10 |
| Waves/SM | 2.90 | 2.90 | +0.00 | 2.90 | +0.00 | 2.90 | +0.00 | 2.90 | +0.00 |
| No Eligible (%) | 36.30 % | 60.94 % | +24.64 pp | 59.93 % | +23.63 pp | 59.96 % | +23.66 pp | 33.60 % | -2.70 pp |
| IPC Active | 2.54 inst/cycle | 1.55 inst/cycle | -0.99 | 1.58 inst/cycle | -0.96 | 1.51 inst/cycle | -1.03 | 2.64 inst/cycle | +0.10 |

---

## Cumulative Progress

```
Baseline:  6460 ns  (1.000×)

Final best: 6460 ns  (1.0000×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_212(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
@@ -32,8 +33,6 @@

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

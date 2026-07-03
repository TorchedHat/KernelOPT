# Kernel Optimization Log

## Summary

- **Kernel**: `depthwise_conv2d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1580000 ns — **Best latency**: 1790000 ns (**0.8827×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `depthwise_conv2d_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_085/synthesis/kernel_0_fused_fused_convolution.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (504064,1,1), Waves/SM: 318.22
- **Registers/thread**: 40
- **Duration**: 1580000 ns
- **SM Throughput**: 74.55 % | **Memory**: 50.66 % | **DRAM**: 13.41 %
- **Achieved Occupancy**: 70.55 %
- **IPC Active**: 2.99 inst/cycle | **No Eligible**: 25.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.007× | 0.007× | ✓ |  |
| B | expand_autotune_configs_with_larger_block_m_and_channel_tiling | 0.999× | 1.000× | ✓ |  |
| C | add_channel_blocking_and_weight_preload_to_reduce_uncoalesced_access | 0.867× | 0.868× | ✓ |  |
| D | vectorized_loads_stores_with_aligned_2d_tiling | 0.754× | 0.760× | ✓ |  |
| E | add_triton_autotune | 1.000× | 0.883× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | expand_autotune_c... | Δ | add_channel_block... | Δ | vectorized_loads_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1580000 ns | 217600000 ns | +216020000 ns (+13672.2%) | 1580000 ns | +0 ns (+0.0%) | 1820000 ns | +240000 ns (+15.2%) | 2080000 ns | +500000 ns (+31.6%) | 1790000 ns | +210000 ns (+13.3%) |
| SM Throughput (%) | 74.55 % | 75.37 % | +0.82 pp | 74.57 % | +0.02 pp | 79.41 % | +4.86 pp | 82.11 % | +7.56 pp | 75.75 % | +1.20 pp |
| Memory Throughput (%) | 50.66 % | 46.93 % | -3.73 pp | 50.64 % | -0.02 pp | 43.79 % | -6.87 pp | 38.01 % | -12.65 pp | 56.78 % | +6.12 pp |
| DRAM Throughput (%) | 13.41 % | 12.78 % | -0.63 pp | 13.42 % | +0.01 pp | 11.64 % | -1.77 pp | 12.38 % | -1.03 pp | 11.82 % | -1.59 pp |
| Achieved Occupancy (%) | 70.55 % | 74.87 % | +4.32 pp | 70.56 % | +0.01 pp | 70.86 % | +0.31 pp | 58.90 % | -11.65 pp | 92.87 % | +22.32 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 40.00 register/thread | +0 | 40.00 register/thread | +0 | 42.00 register/thread | +2 | 32.00 register/thread | -8 |
| Waves/SM | 318.22 | 318.22 | +0.00 | 318.22 | +0.00 | 318.22 | +0.00 | 381.87 | +63.65 | 477.27 | +159.05 |
| No Eligible (%) | 25.31 % | 24.62 % | -0.69 pp | 25.31 % | +0.00 pp | 20.48 % | -4.83 pp | 17.77 % | -7.54 pp | 24.13 % | -1.18 pp |
| IPC Active | 2.99 inst/cycle | 3.02 inst/cycle | +0.03 | 2.99 inst/cycle | +0.00 | 3.18 inst/cycle | +0.19 | 3.29 inst/cycle | +0.30 | 3.04 inst/cycle | +0.05 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.007×)

Replacing a large 2D grid (504,064 CTAs) with a small persistent 1D grid that loops over tiles inside the kernel caused executed instructions to increase by 140x (from 929M to 129.9B), inflating kernel duration from 1.58ms to 217.6ms. The root cause is that computing `tl.cdiv(M, BLOCK_M)` and `num_tiles_total` inside the kernel body using runtime values (not constexpr) forces Triton to emit scalar integer division and modulo for every iteration of the persistent loop, and the loop itself is not unrolled — creating enormous per-iteration overhead. The grid size did not actually shrink (still 504,064) because the persistent grid cap was not correctly applied in the launch configuration, meaning the kernel ran the same number of CTAs but each executed the full tile loop, multiplying work by the number of tiles.

> **Applicable when**: Never applicable in this form. Persistent scheduling in Triton requires the grid to actually be reduced (e.g., grid = NUM_SMS * TILES_PER_CTA, not the original full grid size). The tile count arithmetic must use constexpr values or be passed as kernel arguments, not computed via runtime tl.cdiv inside the kernel.

> **Avoid when**: Do not use tl.cdiv(M, BLOCK_M) inside the kernel body to compute loop bounds when BLOCK_M is a constexpr but M is a runtime value — this generates expensive scalar integer ops per loop iteration. Do not apply persistent scheduling when the grid size is not actually reduced (the original grid size was preserved at 504,064 in this case, negating any benefit). Do not apply when the inner loop body contains many memory accesses that would be replicated across all iterations.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_weight_preload_and_maxnreg | 0.693× | 0.664× | ✓ |  |
| B | reduce_num_warps_increase_num_stages_for_cpi_stall_reduction | 0.999× | 1.000× | ✓ |  |
| C | hoist_loop_invariant_index_computation_to_reduce_register_pressure | 0.999× | 1.000× | ✓ |  |
| D | remove_eviction_hints_and_reduce_register_pressure_via_num_warps_tuning | 1.000× | 1.013× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_num_warps_... | Δ | hoist_loop_invari... | Δ | remove_eviction_h... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1580000 ns | 2380000 ns | +800000 ns (+50.6%) | 1580000 ns | +0 ns (+0.0%) | 1580000 ns | +0 ns (+0.0%) | 1560000 ns | -20000 ns (-1.3%) |
| SM Throughput (%) | 74.55 % | 78.89 % | +4.34 pp | 74.51 % | -0.04 pp | 74.51 % | -0.04 pp | 73.90 % | -0.65 pp |
| Memory Throughput (%) | 50.66 % | 84.45 % | +33.79 pp | 51.83 % | +1.17 pp | 50.66 % | +0.00 pp | 51.42 % | +0.76 pp |
| DRAM Throughput (%) | 13.41 % | 8.95 % | -4.46 pp | 13.41 % | +0.00 pp | 13.41 % | +0.00 pp | 13.54 % | +0.13 pp |
| Achieved Occupancy (%) | 70.55 % | 95.42 % | +24.87 pp | 69.06 % | -1.49 pp | 70.43 % | -0.12 pp | 46.85 % | -23.70 pp |
| Registers/Thread | 40.00 register/thread | 32.00 register/thread | -8 | 40.00 register/thread | +0 | 40.00 register/thread | +0 | 64.00 register/thread | +24 |
| Waves/SM | 318.22 | 238.67 | -79.55 | 318.18 | -0.04 | 318.22 | +0.00 | 477.27 | +159.05 |
| No Eligible (%) | 25.31 % | 21.01 % | -4.30 pp | 25.36 % | +0.05 pp | 25.36 % | +0.05 pp | 25.97 % | +0.66 pp |
| IPC Active | 2.99 inst/cycle | 3.16 inst/cycle | +0.17 | 2.98 inst/cycle | -0.01 | 2.98 inst/cycle | -0.01 | 2.96 inst/cycle | -0.03 |


### Key Learnings

#### ❌ Plan A: `reduce_register_pressure_via_weight_preload_and_maxnreg` (0.693×)

Replacing 21 scalar global weight loads with a single vectorized preload followed by `tl.sum(tl.where(w_offs == (kh*KW+kw), w_vals, 0.0))` to extract each scalar caused a 31% regression (0.69x). Despite the plan explicitly warning against this anti-pattern, the diff shows it was applied anyway: executed instructions ballooned from 913M to 1403M (+54%), Warp Cycles Per Issued Instruction rose from 10.09 to 14.56, and kernel duration increased from 1.56ms to 2.28ms. The tl.sum(tl.where(...)) idiom generates a full vector comparison and reduction per weight access rather than a register read, negating any benefit from reducing global load count. This is the same anti-pattern documented in exp_triton_maxnreg_weight_preload_regression_002 and must not be used for scalar extraction from preloaded vectors.

> **Applicable when**: Never. The tl.sum(tl.where(vector == const_idx, vector, 0.0)) pattern for scalar extraction is always an anti-pattern in Triton regardless of whether the index is a compile-time constant.

> **Avoid when**: Do not use tl.sum(tl.where(offs == idx, buf, 0.0)) to extract scalar values from a preloaded vector register in any Triton kernel. This pattern generates a full vector comparison and reduction per extraction, inflating instruction count by 50%+ and increasing Warp Cycles Per Issued Instruction. The compiler cannot optimize this to a register read even when idx is a compile-time constant derived from tl.static_range induction variables.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | amortize_spatial_index_decode_by_looping_channels_inside_cta | 0.009× | 0.009× | ✓ |  |
| C | weight_vector_preload_with_constexpr_index_access | 0.720× | 0.693× | ✓ |  |
| D | eliminate_register_spill_and_fuse_fp32_multiply_add | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | amortize_spatial_... | Δ | weight_vector_pre... | Δ | eliminate_registe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1580000 ns | 167380000 ns | +165800000 ns (+10493.7%) | 2280000 ns | +700000 ns (+44.3%) | 1580000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 74.55 % | 60.25 % | -14.30 pp | 77.70 % | +3.15 pp | 74.56 % | +0.01 pp |
| Memory Throughput (%) | 50.66 % | 57.81 % | +7.15 pp | 72.50 % | +21.84 pp | 50.65 % | -0.01 pp |
| DRAM Throughput (%) | 13.41 % | 16.38 % | +2.97 pp | 9.26 % | -4.15 pp | 13.42 % | +0.01 pp |
| Achieved Occupancy (%) | 70.55 % | 18.72 % | -51.83 pp | 70.95 % | +0.40 pp | 70.56 % | +0.01 pp |
| Registers/Thread | 40.00 register/thread | 168.00 register/thread | +128 | 40.00 register/thread | +0 | 40.00 register/thread | +0 |
| Waves/SM | 318.22 | 1272.73 | +954.51 | 318.22 | +0.00 | 318.22 | +0.00 |
| No Eligible (%) | 25.31 % | 39.74 % | +14.43 pp | 22.21 % | -3.10 pp | 25.31 % | +0.00 pp |
| IPC Active | 2.99 inst/cycle | 2.41 inst/cycle | -0.58 | 3.11 inst/cycle | +0.12 | 2.99 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan A: `amortize_spatial_index_decode_by_looping_channels_inside_cta` (0.009×)

Moving the channel dimension from the grid (2D grid with pid_c = tl.program_id(1)) into a runtime tl.range(C) loop inside the kernel body caused a catastrophic 106x regression (1.56ms → 167ms). The root cause is that the runtime channel loop prevented Triton's compiler from unrolling or statically scheduling the loop, causing registers per thread to explode from 64 to 168 — far exceeding the hardware register file budget per warp. This crushed theoretical occupancy from 50% to 18.75% (Block Limit Registers dropped from 32 to 12 blocks/SM), and total executed instructions ballooned from 913M to 79.85B (~87x increase), indicating massive register spill traffic and loop overhead. The grid size did not actually shrink (remained 2,016,000) because the grid lambda was not correctly updated to remove the channel dimension, meaning each CTA now executed the full C=128 channel loop on top of the original per-channel work.

> **Applicable when**: Never in this form. A runtime tl.range(C) loop over a large dimension (C=128) inside a Triton kernel body causes register pressure to scale with C, not remain constant, because the compiler cannot statically schedule or pipeline the loop body.

> **Avoid when**: Do not move a grid dimension into a runtime tl.range() loop inside the kernel when: (1) the loop trip count is large (e.g., C=128), (2) the loop body contains many live values (accumulators, pointer arithmetic), or (3) the grid lambda is not simultaneously updated to remove that dimension. The combination of a large runtime loop and unchanged grid size means each CTA does C times more work than intended, multiplying total work by C. Register pressure scales with the number of live values across loop iterations, and Triton cannot spill-free compile a 128-iteration loop with 21 FMAs per iteration and a BLOCK_M-sized accumulator.

---

## Cumulative Progress

```
Baseline:  1580000 ns  (1.000×)

Final best: 1790000 ns  (0.8827×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,17 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
+        triton.Config({'BLOCK_M': 64},  num_warps=1, num_stages=4),
+        triton.Config({'BLOCK_M': 128}, num_warps=1, num_stages=3),
+        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 64},  num_warps=2, num_stages=4),
         triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=3),
         triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=3),
-        triton.Config({'BLOCK_M': 512}, num_warps=8, num_stages=3),
-        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=4),
-        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_M': 512}, num_warps=4, num_stages=3),
     ],
     key=['M', 'C'],
 )
@@ -68,7 +71,6 @@

     # Y shape: [N, C, OUT_H, OUT_W], strides: [C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1]
     y_ptrs = Y_ptr + batch * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + out_h * OUT_W + out_w
     tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m)
-
 
 def kernel_function(weight, x):
     # weight: [128, 1, 3, 7], x: [32, 128, 128, 256]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

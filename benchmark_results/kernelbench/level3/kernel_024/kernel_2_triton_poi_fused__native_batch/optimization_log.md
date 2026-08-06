# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 24900 ns — **Best latency**: 24580 ns (**1.0130×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_broken_autotune_grid_and_constrain_xblock_to_restore_register_efficiency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (36000,1,1), Waves/SM: 17.05
- **Registers/thread**: 18
- **Duration**: 24900 ns
- **SM Throughput**: 63.10 % | **Memory**: 23.90 % | **DRAM**: 15.14 %
- **Achieved Occupancy**: 58.10 %
- **IPC Active**: 2.81 inst/cycle | **No Eligible**: 29.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_better_occupancy_and_ilp | 1.002× | 0.384× | ✓ |  |
| B | autotune_xblock_num_warps_for_occupancy_and_ipc | 0.722× | 0.705× | ✓ |  |
| C | increase_xblock_with_autotune_for_vectorized_loads_and_occupancy | 0.999× | 0.296× | ✓ |  |
| D | increase_xblock_with_autotune_for_occupancy_and_latency_hiding | 1.003× | 0.709× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | increase_xblock_w... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24900 ns | 64800 ns | +39900 ns (+160.2%) | 35300 ns | +10400 ns (+41.8%) | 83970 ns | +59070 ns (+237.2%) | 35140 ns | +10240 ns (+41.1%) |
| SM Throughput (%) | 63.10 % | 86.14 % | +23.04 pp | 82.93 % | +19.83 pp | 85.82 % | +22.72 pp | 83.34 % | +20.24 pp |
| Memory Throughput (%) | 23.90 % | 20.21 % | -3.69 pp | 24.78 % | +0.88 pp | 24.76 % | +0.86 pp | 24.90 % | +1.00 pp |
| DRAM Throughput (%) | 15.14 % | 5.80 % | -9.34 pp | 10.65 % | -4.49 pp | 4.48 % | -10.66 pp | 10.70 % | -4.44 pp |
| Achieved Occupancy (%) | 58.10 % | 85.42 % | +27.32 pp | 87.28 % | +29.18 pp | 82.79 % | +24.69 pp | 87.01 % | +28.91 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 32.00 register/thread | +14 |
| Waves/SM | 17.05 | 34.09 | +17.04 | 17.05 | +0.00 | 68.18 | +51.13 | 17.05 | +0.00 |
| No Eligible (%) | 29.74 % | 10.90 % | -18.84 pp | 11.45 % | -18.29 pp | 11.91 % | -17.83 pp | 11.50 % | -18.24 pp |
| IPC Active | 2.81 inst/cycle | 3.55 inst/cycle | +0.74 | 3.54 inst/cycle | +0.73 | 3.51 inst/cycle | +0.70 | 3.53 inst/cycle | +0.72 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_better_occupancy_and_ilp` (0.384×)

Increasing XBLOCK via @triton.autotune caused a 2.6x regression despite improving achieved occupancy (58.1% → 85.4%) and eligible warps (1.6 → 7.72). The autotuner selected XBLOCK=256 with 32 registers/thread (up from 18), nearly doubling register pressure and reducing Block Limit Registers from 21 to 8 blocks/SM. This forced 2x more waves (17 → 34) and nearly tripled total executed instructions (12.4M → 43.9M), increasing kernel duration from 24.9 µs to 64.8 µs. The grid was not updated to use a lambda, so the fixed grid=(cdiv(xnumel,128),) launched 36,000 blocks regardless of the autotuned XBLOCK, causing each block to process redundant or out-of-bounds work — the instruction count explosion (3.5x) is the primary evidence of this implementation flaw.

> **Applicable when**: Kernel has CPIStall from L1TEX latency >40% of CPI, low eligible warps per scheduler (<2.0), achieved occupancy well below theoretical, and low register pressure (<30 registers/thread) leaving headroom for larger tiles.

> **Avoid when**: The grid is not updated to a lambda when adding @triton.autotune — this is a critical implementation requirement. Also avoid when register count is already near spill threshold (>48 registers/thread), as larger XBLOCK will increase register pressure and reduce occupancy. Do not apply when the kernel is already memory-bandwidth bound (DRAM throughput >70%).

#### ❌ Plan B: `autotune_xblock_num_warps_for_occupancy_and_ipc` (0.705×)

Adding triton.autotune with larger XBLOCK configs (512–2048) and more warps increased achieved occupancy from 58% to 87% and eligible warps per scheduler from 1.6 to 7.35, but the kernel regressed by ~42% in wall-clock time. The root cause of the regression is that larger XBLOCK forced the compiler to allocate more registers per thread (18→32), which reduced the block limit from 21 to 16 blocks/SM and increased total executed instructions by ~86% (12.4M→23.1M). Additionally, the fixed grid in kernel_function was not updated to use a lambda with META['XBLOCK'], so the grid remained at cdiv(xnumel, 128) regardless of the winning autotune config, causing each CTA to process far more elements than its XBLOCK tile size, leading to redundant work and incorrect tiling. The L1 hit rate also dropped from 83% to 66%, indicating that larger tiles broke the cache reuse pattern for the 32-element channel broadcast tensors.

> **Applicable when**: Kernel has low achieved occupancy (<65%), low eligible warps per scheduler (<2.0), and L1TEX scoreboard stalls dominate (>40% of warp cycles), suggesting insufficient warp-level parallelism to hide memory latency.

> **Avoid when**: Do not apply autotune with larger XBLOCK when: (1) the kernel has high L1 hit rate (>80%) from small broadcast/channel tensors that fit in cache at the current tile size — larger tiles will break this reuse; (2) the current register count is already near the register-limit boundary (e.g., 18 regs/thread with block limit of 21), as larger tiles may push register usage up and reduce occupancy via a different bottleneck; (3) the grid launch is not updated to use a META lambda — a fixed grid with variable XBLOCK causes incorrect tiling and redundant computation.

#### ⚠️ Plan C: `increase_xblock_with_autotune_for_vectorized_loads_and_occupancy` (0.296×)

Increasing XBLOCK from 128 to 512 via @triton.autotune improved occupancy (58% → 83%) and eligible warps per scheduler (1.6 → 7.56), but caused a 4.6x increase in total executed instructions (12.4M → 57M) and a 3.4x increase in kernel duration (24.9µs → 84µs). The grid was not updated to use `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` — instead the fixed grid `(triton.cdiv(xnumel, 128),)` was kept, so the kernel launched 4x too many CTAs (36,000 instead of 9,000 for XBLOCK=512), each doing redundant work. When using @triton.autotune, the launch grid MUST be a lambda that reads `meta['XBLOCK']`; a hardcoded grid defeats the purpose and causes massive over-computation.

> **Applicable when**: Any Triton kernel using @triton.autotune with a tile-size parameter (XBLOCK, BLOCK_M, etc.) where the grid size depends on that parameter.

> **Avoid when**: Grid size is independent of the autotuned parameter (e.g., autotuning only num_warps/num_stages without changing tile size).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_eliminate_nonfused_fp32_via_algebraic_simplification | 1.012× | 0.949× | ✓ | **✓ best** |
| B | eliminate_nonfused_fp32_instructions_via_rsqrt_and_float_literals | 0.918× | 0.948× | ✓ |  |
| C | simplify_arithmetic_to_enable_fma_fusion_and_fix_grid_lambda | 1.002× | 0.402× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_redundant_cta_launches | 0.999× | 0.296× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | eliminate_nonfuse... | Δ | simplify_arithmet... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24900 ns | 26240 ns | +1340 ns (+5.4%) | 26270 ns | +1370 ns (+5.5%) | 61950 ns | +37050 ns (+148.8%) | 84000 ns | +59100 ns (+237.3%) |
| SM Throughput (%) | 63.10 % | 53.04 % | -10.06 pp | 53.08 % | -10.02 pp | 86.23 % | +23.13 pp | 85.90 % | +22.80 pp |
| Memory Throughput (%) | 23.90 % | 34.88 % | +10.98 pp | 34.75 % | +10.85 pp | 21.07 % | -2.83 pp | 24.72 % | +0.82 pp |
| DRAM Throughput (%) | 15.14 % | 14.33 % | -0.81 pp | 14.33 % | -0.81 pp | 6.07 % | -9.07 pp | 4.47 % | -10.67 pp |
| Achieved Occupancy (%) | 58.10 % | 35.81 % | -22.29 pp | 36.09 % | -22.01 pp | 86.00 % | +27.90 pp | 82.76 % | +24.66 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 24.00 register/thread | +6 |
| Waves/SM | 17.05 | 17.05 | +0.00 | 17.05 | +0.00 | 34.09 | +17.04 | 68.18 | +51.13 |
| No Eligible (%) | 29.74 % | 40.13 % | +10.39 pp | 40.57 % | +10.83 pp | 10.67 % | -19.07 pp | 11.93 % | -17.81 pp |
| IPC Active | 2.81 inst/cycle | 2.39 inst/cycle | -0.42 | 2.39 inst/cycle | -0.42 | 3.57 inst/cycle | +0.76 | 3.52 inst/cycle | +0.71 |


### Key Learnings

#### ❌ Plan C: `simplify_arithmetic_to_enable_fma_fusion_and_fix_grid_lambda` (0.402×)

Simplifying the batch-norm arithmetic (removing an int32 constant and a redundant multiply-by-1.0) to enable FMA fusion was mathematically correct but caused a 1.76x regression in kernel duration (35.1 µs → 62.0 µs). The profiling data shows total executed instructions nearly doubled (23.1M → 42.2M) while SM throughput barely changed (83.3% → 86.2%), indicating the autotune configs changed alongside the arithmetic rewrite — the autotuner selected XBLOCK=256 (block_size 128→256, waves 17→34) which doubled the instruction count independently of the FMA change. The FMA simplification itself did not reduce non-fused instruction count in a measurable way; the dominant effect was the autotune config selection doubling the work per launch. When the grid lambda fix is applied simultaneously with autotune config changes, the autotuner may select a larger XBLOCK that inflates instruction count and masks any FMA benefit.

> **Applicable when**: NCU FPInstructions rule flags >50% non-fused FP32 instructions AND the autotune config set is held constant (same XBLOCK/num_warps) so that only the arithmetic change is isolated.

> **Avoid when**: When simultaneously changing autotune configs (XBLOCK, num_warps, num_stages) alongside arithmetic rewrites — the autotune config selection dominates the instruction count and masks any FMA benefit. Do not apply when the kernel is already compute-bound at >80% SM SOL with high waves_per_sm, as a larger autotuned XBLOCK will double instruction count and duration. Do not combine grid-lambda fix with autotune config changes in the same iteration.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_tune_num_warps_num_stages_at_fixed_xblock | 1.008× | 1.017× | ✓ |  |
| B | fix_broken_autotune_grid_and_constrain_xblock_to_restore_register_efficiency | 1.013× | 1.013× | ✓ | **✓ best** |
| C | restrict_autotune_to_xblock128_with_num_warps_sweep_to_prevent_instruction_bloat | 0.999× | 0.999× | ✓ |  |
| D | fix_grid_lambda_with_locked_xblock_to_eliminate_redundant_cta_launches | 0.579× | 0.554× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_broken_autotu... | Δ | restrict_autotune... | Δ | fix_grid_lambda_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24900 ns | 24480 ns | -420 ns (-1.7%) | 24580 ns | -320 ns (-1.3%) | 24930 ns | +30 ns (+0.1%) | 44930 ns | +20030 ns (+80.4%) |
| SM Throughput (%) | 63.10 % | 38.50 % | -24.60 pp | 22.03 % | -41.07 pp | 44.66 % | -18.44 pp | 83.90 % | +20.80 pp |
| Memory Throughput (%) | 23.90 % | 23.61 % | -0.29 pp | 21.19 % | -2.71 pp | 21.43 % | -2.47 pp | 27.19 % | +3.29 pp |
| DRAM Throughput (%) | 15.14 % | 15.38 % | +0.24 pp | 15.31 % | +0.17 pp | 15.10 % | -0.04 pp | 8.37 % | -6.77 pp |
| Achieved Occupancy (%) | 58.10 % | 41.24 % | -16.86 pp | 27.86 % | -30.24 pp | 36.48 % | -21.62 pp | 83.01 % | +24.91 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 24.00 register/thread | +6 | 24.00 register/thread | +6 |
| Waves/SM | 17.05 | 17.05 | +0.00 | 8.52 | -8.53 | 8.52 | -8.53 | 34.09 | +17.04 |
| No Eligible (%) | 29.74 % | 57.13 % | +27.39 pp | 73.64 % | +43.90 pp | 47.39 % | +17.65 pp | 11.96 % | -17.78 pp |
| IPC Active | 2.81 inst/cycle | 1.70 inst/cycle | -1.11 | 1.03 inst/cycle | -1.78 | 2.09 inst/cycle | -0.72 | 3.52 inst/cycle | +0.71 |


### Key Learnings

#### ❌ Plan D: `fix_grid_lambda_with_locked_xblock_to_eliminate_redundant_cta_launches` (0.554×)

Applying the grid lambda fix (changing hardcoded grid to `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)`) alongside an arithmetic rewrite (replacing `tl.math.rsqrt` with `tl.sqrt_rn` + integer division) caused a 1.8x regression in kernel duration (26.2 µs → 44.9 µs). The profiling data shows total executed instructions nearly tripled (10.9M → 29.7M) and waves_per_sm doubled (17 → 34), indicating the autotune selected XBLOCK=256 (block_size 128→256) rather than the intended XBLOCK=512. Additionally, the arithmetic rewrite introduced 272 avg divergent branches per warp (up from 0), suggesting the integer division path (`tl.full([1], 1, tl.int32) / tmp6`) created type-mismatch or predication overhead. The grid lambda fix is correct in principle but the simultaneous arithmetic change and autotune config narrowing to a single suboptimal config (XBLOCK=512, num_warps=8, num_stages=2) caused the autotuner to land on XBLOCK=256 with doubled instruction count.

> **Applicable when**: Never combine grid lambda fix with arithmetic rewrites in the same iteration; isolate each change to identify the causal effect independently.

> **Avoid when**: Do not replace `tl.math.rsqrt` with `tl.sqrt_rn` + integer division (`tl.full([1], 1, tl.int32) / float_value`) — this introduces type mismatches that create divergent branches (0 → 272 avg divergent branches) and inflates instruction count. Do not combine autotune config changes with arithmetic rewrites in the same iteration.

---

## Iteration 4

**Best so far**: 1.013× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_evict_first_on_main_load_to_reduce_l1tex_stalls | 1.011× | 1.009× | ✓ |  |
| B | fix_grid_lambda_and_sweep_num_warps_at_xblock128_to_restore_occupancy | 1.005× | 1.011× | ✓ |  |
| C | eliminate_xmask_predication_overhead_via_mask_free_loads_stores | 1.004× | 1.009× | ✓ |  |
| D | fix_grid_lambda_lock_xblock128_and_isolate_fma_arithmetic | 0.999× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evict_f... | Δ | fix_grid_lambda_a... | Δ | eliminate_xmask_p... | Δ | fix_grid_lambda_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24900 ns | 24670 ns | -230 ns (-0.9%) | 24640 ns | -260 ns (-1.0%) | 24670 ns | -230 ns (-0.9%) | 24700 ns | -200 ns (-0.8%) |
| SM Throughput (%) | 63.10 % | 42.57 % | -20.53 pp | 22.46 % | -40.64 pp | 52.46 % | -10.64 pp | 63.93 % | +0.83 pp |
| Memory Throughput (%) | 23.90 % | 23.80 % | -0.10 pp | 21.34 % | -2.56 pp | 23.93 % | +0.03 pp | 24.21 % | +0.31 pp |
| DRAM Throughput (%) | 15.14 % | 15.26 % | +0.12 pp | 15.29 % | +0.15 pp | 15.25 % | +0.11 pp | 15.21 % | +0.07 pp |
| Achieved Occupancy (%) | 58.10 % | 41.70 % | -16.40 pp | 28.00 % | -30.10 pp | 53.70 % | -4.40 pp | 58.04 % | -0.06 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 17.00 register/thread | -1 | 18.00 register/thread | +0 |
| Waves/SM | 17.05 | 17.05 | +0.00 | 8.52 | -8.53 | 17.05 | +0.00 | 17.05 | +0.00 |
| No Eligible (%) | 29.74 % | 52.83 % | +23.09 pp | 73.24 % | +43.50 pp | 41.81 % | +12.07 pp | 30.02 % | +0.28 pp |
| IPC Active | 2.81 inst/cycle | 1.88 inst/cycle | -0.93 | 1.06 inst/cycle | -1.75 | 2.33 inst/cycle | -0.48 | 2.81 inst/cycle | +0.00 |

---

## Iteration 5

**Best so far**: 1.013× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_stalls | 1.006× | 1.015× | ✓ |  |
| B | fuse_nonfused_fp32_via_inline_epsilon_constant_for_fma_fusion | 1.011× | 1.006× | ✓ |  |
| C | sweep_num_warps_downward_at_fixed_xblock128_to_improve_block_level_occupancy | 1.004× | 0.978× | ✓ |  |
| D | isolate_fma_fusion_via_arithmetic_simplification_at_fixed_xblock128 | 1.004× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fuse_nonfused_fp3... | Δ | sweep_num_warps_d... | Δ | isolate_fma_fusio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24900 ns | 24540 ns | -360 ns (-1.4%) | 24740 ns | -160 ns (-0.6%) | 25470 ns | +570 ns (+2.3%) | 24740 ns | -160 ns (-0.6%) |
| SM Throughput (%) | 63.10 % | 39.21 % | -23.89 pp | 42.26 % | -20.84 pp | 33.64 % | -29.46 pp | 62.08 % | -1.02 pp |
| Memory Throughput (%) | 23.90 % | 23.55 % | -0.35 pp | 23.61 % | -0.29 pp | 20.07 % | -3.83 pp | 24.04 % | +0.14 pp |
| DRAM Throughput (%) | 15.14 % | 15.33 % | +0.19 pp | 15.22 % | +0.08 pp | 14.78 % | -0.36 pp | 15.21 % | +0.07 pp |
| Achieved Occupancy (%) | 58.10 % | 40.95 % | -17.15 pp | 41.73 % | -16.37 pp | 23.22 % | -34.88 pp | 57.53 % | -0.57 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 28.00 register/thread | +10 | 18.00 register/thread | +0 |
| Waves/SM | 17.05 | 17.05 | +0.00 | 17.05 | +0.00 | 8.52 | -8.53 | 17.05 | +0.00 |
| No Eligible (%) | 29.74 % | 56.45 % | +26.71 pp | 53.74 % | +24.00 pp | 60.69 % | +30.95 pp | 31.08 % | +1.34 pp |
| IPC Active | 2.81 inst/cycle | 1.75 inst/cycle | -1.06 | 1.88 inst/cycle | -0.93 | 1.54 inst/cycle | -1.27 | 2.72 inst/cycle | -0.09 |

---

## Cumulative Progress

```
Baseline:  24900 ns  (1.000×)
Iter 3     24580 ns  (1.013×)  — fix_broken_autotune_grid_and_constrain_xblock_to_restore_register_efficiency

Final best: 24580 ns  (1.0130×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,9 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 4608000
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
@@ -21,18 +29,13 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    tmp6 = tl.math.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -42,7 +45,7 @@

     arg_4 = inputs[4]
     xnumel = 4608000
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4608000, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4608000)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__native_batch_norm_legit_convolution_div_view_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4590000 ns — **Best latency**: 5860000 ns (**0.7833×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__native_batch_norm_legit_convolution_div_view_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_017/kernel_2_triton_red_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16384,1,1), Waves/SM: 15.52
- **Registers/thread**: 64
- **Duration**: 4590000 ns
- **SM Throughput**: 10.98 % | **Memory**: 98.18 % | **DRAM**: 44.18 %
- **Achieved Occupancy**: 49.34 %
- **IPC Active**: 0.44 inst/cycle | **No Eligible**: 89.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_xblock_coalescing | 0.932× | 0.937× | ✓ |  |
| C | autotune_xblock_and_software_pipelining | 1.482× | 1.022× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4590000 ns | 4900000 ns | +310000 ns (+6.8%) | 4490000 ns | -100000 ns (-2.2%) |
| SM Throughput (%) | 10.98 % | 60.54 % | +49.56 pp | 12.76 % | +1.78 pp |
| Memory Throughput (%) | 98.18 % | 27.76 % | -70.42 pp | 98.10 % | -0.08 pp |
| DRAM Throughput (%) | 44.18 % | 13.62 % | -30.56 pp | 40.82 % | -3.36 pp |
| Achieved Occupancy (%) | 49.34 % | 12.44 % | -36.90 pp | 49.35 % | +0.01 pp |
| Registers/Thread | 64.00 register/thread | 255.00 register/thread | +191 | 62.00 register/thread | -2 |
| Waves/SM | 15.52 | 62.06 | +46.54 | 15.52 | +0.00 |
| No Eligible (%) | 89.01 % | 39.13 % | -49.88 pp | 87.25 % | -1.76 pp |
| IPC Active | 0.44 inst/cycle | 2.43 inst/cycle | +1.99 | 0.51 inst/cycle | +0.07 |

---

## Iteration 2

**Best so far**: 1.0223× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.485× | 1.022× | ✓ |  |
| B | memory_coalescing_reorder_xindex_decomposition | 0.089× | 0.082× | ✓ |  |
| D | reduce_register_pressure_via_smaller_r0_block_configs | 1.603× | 1.408× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4590000 ns | 4490000 ns | -100000 ns (-2.2%) | 55910000 ns | +51320000 ns (+1118.1%) | 3260000 ns | -1330000 ns (-29.0%) |
| SM Throughput (%) | 10.98 % | 12.76 % | +1.78 pp | 77.22 % | +66.24 pp | 40.41 % | +29.43 pp |
| Memory Throughput (%) | 98.18 % | 98.00 % | -0.18 pp | 26.95 % | -71.23 pp | 39.45 % | -58.73 pp |
| DRAM Throughput (%) | 44.18 % | 40.85 % | -3.33 pp | 1.13 % | -43.05 pp | 32.14 % | -12.04 pp |
| Achieved Occupancy (%) | 49.34 % | 49.34 % | +0.00 pp | 24.76 % | -24.58 pp | 93.25 % | +43.91 pp |
| Registers/Thread | 64.00 register/thread | 62.00 register/thread | -2 | 122.00 register/thread | +58 | 32.00 register/thread | -32 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 31.03 | +15.51 | 3.88 | -11.64 |
| No Eligible (%) | 89.01 % | 87.19 % | -1.82 pp | 22.25 % | -66.76 pp | 59.07 % | -29.94 pp |
| IPC Active | 0.44 inst/cycle | 0.51 inst/cycle | +0.07 | 3.11 inst/cycle | +2.67 | 1.64 inst/cycle | +1.20 |


### Key Learnings

#### ❌ Plan B: `memory_coalescing_reorder_xindex_decomposition` (0.082×)

Restructuring a reduction kernel so that XBLOCK=128 covers the channel dimension as the fast-varying axis (to coalesce memory) while transposing the accumulator shape from [XBLOCK, R0_BLOCK] to [R0_BLOCK, XBLOCK] caused registers per thread to nearly double (62→122), halving theoretical occupancy (50%→25%) and increasing waves per SM from 15.5 to 31. Although L2 throughput dropped dramatically (98.1%→1.58%) and memory stalls were eliminated, the instruction count exploded 76× (450M→34B instructions) because the larger tile size (XBLOCK=128 × R0_BLOCK) forced far more compute work per CTA with severely reduced parallelism. The net effect was a 12× slowdown despite fixing the coalescing problem.

> **Applicable when**: Never: this specific transformation (transposing accumulator shape and welford axis to enable coalescing) caused a 76× instruction count explosion due to register pressure doubling and occupancy halving, resulting in 12× slowdown despite fixing L2 throughput.

> **Avoid when**: When the kernel already has high register pressure (≥60 registers/thread) and moderate occupancy (≥50%). Transposing the tile shape from [XBLOCK, R0_BLOCK] to [R0_BLOCK, XBLOCK] with large XBLOCK (128) will double register usage, halve occupancy, and cause instruction count to explode even if memory coalescing improves. Do not apply when the register budget is already near the occupancy-limiting threshold.

---

## Iteration 3

**Best so far**: 1.408× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_loop_invariants_and_increase_pipeline_stages | 1.602× | 1.408× | ✓ |  |
| B | shrink_reduction_tile_configs_to_cut_register_pressure_and_boost_occupancy | 1.558× | 1.572× | ✓ |  |
| C | shrink_accumulator_tile_to_cut_register_pressure_and_restore_occupancy | 1.601× | 0.785× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_loop_invari... | Δ | shrink_reduction_... | Δ | shrink_accumulato... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4590000 ns | 3260000 ns | -1330000 ns (-29.0%) | 2920000 ns | -1670000 ns (-36.4%) | 5850000 ns | +1260000 ns (+27.5%) |
| SM Throughput (%) | 10.98 % | 40.50 % | +29.52 pp | 44.49 % | +33.51 pp | 46.52 % | +35.54 pp |
| Memory Throughput (%) | 98.18 % | 39.62 % | -58.56 pp | 42.73 % | -55.45 pp | 93.46 % | -4.72 pp |
| DRAM Throughput (%) | 44.18 % | 32.10 % | -12.08 pp | 32.39 % | -11.79 pp | 20.90 % | -23.28 pp |
| Achieved Occupancy (%) | 49.34 % | 93.20 % | +43.86 pp | 71.86 % | +22.52 pp | 95.09 % | +45.75 pp |
| Registers/Thread | 64.00 register/thread | 32.00 register/thread | -32 | 40.00 register/thread | -24 | 23.00 register/thread | -41 |
| Waves/SM | 15.52 | 3.88 | -11.64 | 5.17 | -10.35 | 3.88 | -11.64 |
| No Eligible (%) | 89.01 % | 59.01 % | -30.00 pp | 55.15 % | -33.86 pp | 59.53 % | -29.48 pp |
| IPC Active | 0.44 inst/cycle | 1.64 inst/cycle | +1.20 | 1.80 inst/cycle | +1.36 | 1.62 inst/cycle | +1.18 |


### Key Learnings

#### ✅ Plan A: `hoist_loop_invariants_and_increase_pipeline_stages` (1.567×)

Moving loop-invariant scalar computations (constants, rsqrt, and scale factor) out of a Triton tl.range loop body eliminates redundant FP32 ops per iteration and enables FMA fusion on the remaining multiply. Simultaneously increasing num_stages from 2 to 3 in autotune configs deepens the software pipeline to better hide L1TEX scoreboard stalls (CPIStall rule: ~12.9 cycles/warp stalled on L1TEX, ~50% of total stall budget). The NCU profiling metrics showed nearly identical before/after values for the measured kernel instance, suggesting the speedup was captured at the autotuner selection level rather than in the profiled kernel instance itself.

> **Applicable when**: Triton kernel with a secondary tl.range loop that recomputes scalar constants, rsqrt, or other expressions depending only on values computed before the loop (e.g., post-Welford statistics). CPIStall rule triggered with L1TEX stall contributing >40% of total warp stall cycles. FPInstructions rule triggered with large ratio of non-fused to fused FP32 instructions (e.g., 364M non-fused vs 66M fused). num_stages=2 in autotune configs. Registers per thread moderate (≤40) so increasing num_stages does not cause register spill.

> **Avoid when**: When registers per thread are already high (≥60) and increasing num_stages would push register usage over the occupancy-limiting threshold. When the 'invariant' expressions actually depend on loop-varying values (verify dependency chain carefully). When the loop body is already minimal and the bottleneck is purely memory bandwidth rather than instruction latency.

---

## Iteration 4

**Best so far**: 1.408× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_loop_invariants_and_fuse_scale_multiply_for_fma | 1.560× | 1.567× | ✓ |  |
| B | fuse_fp32_instructions_via_explicit_fma_in_reduction_loops | 1.607× | 1.412× | ✓ | **✓ best** |
| C | hoist_normalization_scalars_and_increase_pipeline_stages | 1.603× | 0.787× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_loop_invari... | Δ | fuse_fp32_instruc... | Δ | hoist_normalizati... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4590000 ns | 2930000 ns | -1660000 ns (-36.2%) | 3250000 ns | -1340000 ns (-29.2%) | 5830000 ns | +1240000 ns (+27.0%) |
| SM Throughput (%) | 10.98 % | 44.59 % | +33.61 pp | 39.75 % | +28.77 pp | 46.59 % | +35.61 pp |
| Memory Throughput (%) | 98.18 % | 42.81 % | -55.37 pp | 40.32 % | -57.86 pp | 93.60 % | -4.58 pp |
| DRAM Throughput (%) | 44.18 % | 32.19 % | -11.99 pp | 32.47 % | -11.71 pp | 20.84 % | -23.34 pp |
| Achieved Occupancy (%) | 49.34 % | 71.83 % | +22.49 pp | 93.29 % | +43.95 pp | 95.10 % | +45.76 pp |
| Registers/Thread | 64.00 register/thread | 40.00 register/thread | -24 | 32.00 register/thread | -32 | 23.00 register/thread | -41 |
| Waves/SM | 15.52 | 5.17 | -10.35 | 3.88 | -11.64 | 3.88 | -11.64 |
| No Eligible (%) | 89.01 % | 55.57 % | -33.44 pp | 59.86 % | -29.15 pp | 59.46 % | -29.55 pp |
| IPC Active | 0.44 inst/cycle | 1.79 inst/cycle | +1.35 | 1.59 inst/cycle | +1.15 | 1.62 inst/cycle | +1.18 |


### Key Learnings

#### ❌ Plan C: `hoist_normalization_scalars_and_increase_pipeline_stages` (0.787×)

Hoisting loop-invariant scalar computations (constants, rsqrt, divisions) out of a Triton reduction loop, combined with replacing larger autotune configs (XBLOCK=4–8, R0_BLOCK=64–128) with smaller ones (XBLOCK=1–8, R0_BLOCK=16–64), reduced registers per thread (40→23) and raised theoretical occupancy (75%→100%), but caused L1/TEX throughput to saturate at 93.6% Mem Busy (up from 42.73%) and eligible warps per scheduler to collapse from 2.58 to 0.51. The instruction count nearly doubled (1.03B→1.83B) and Warp Cycles Per Issued Instruction worsened (25.66→37.62 cycles), resulting in a 1.27× slowdown. The root cause is that the smaller tile configs (especially XBLOCK=1–2) generate far more narrow memory transactions per element, overwhelming L1/TEX bandwidth even though register pressure decreased and cache hit rates improved. This is the same L1 saturation anti-pattern observed in iteration 3 (item c3e1a7f2): the autotune selected a small-XBLOCK config that traded register pressure for L1 bandwidth exhaustion.

> **Applicable when**: Never in this kernel class: hoisting loop-invariant scalars while simultaneously shrinking autotune tile configs to XBLOCK=1–4 causes L1/TEX saturation and instruction count explosion, negating any register pressure benefit.

> **Avoid when**: When the kernel is already running at moderate memory utilization (Mem Busy ~40%) with reasonable occupancy (>70% achieved) and the autotune configs use XBLOCK=4–8 with R0_BLOCK=64–128. Hoisting loop-invariant scalars alone is a correct transformation but does not help if the autotune simultaneously selects smaller tile configs that saturate L1/TEX. Do not combine scalar hoisting with a wholesale replacement of autotune configs toward smaller XBLOCK (1–4) values — the resulting L1 saturation (Mem Busy 93.6%, eligible warps 0.51) is worse than the original CPIStall bottleneck. Also do not apply when the baseline Mem Busy is below 50% and the bottleneck is scheduler stalls rather than memory bandwidth.

---

## Iteration 5

**Best so far**: 1.4123× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_with_higher_warp_counts_and_larger_r0_blocks | 1.556× | 1.577× | ✓ |  |
| B | eliminate_partial_wave_via_larger_xblock_and_num_warps8_configs | 1.608× | 1.421× | ✓ |  |
| C | bypass_l1_for_strided_stores_and_fuse_fp32_multiply_chain | 1.662× | 0.783× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | expand_autotune_w... | Δ | eliminate_partial... | Δ | bypass_l1_for_str... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4590000 ns | 2910000 ns | -1680000 ns (-36.6%) | 3230000 ns | -1360000 ns (-29.6%) | 5860000 ns | +1270000 ns (+27.7%) |
| SM Throughput (%) | 10.98 % | 44.46 % | +33.48 pp | 39.50 % | +28.52 pp | 47.99 % | +37.01 pp |
| Memory Throughput (%) | 98.18 % | 42.68 % | -55.50 pp | 40.36 % | -57.82 pp | 93.61 % | -4.57 pp |
| DRAM Throughput (%) | 44.18 % | 32.13 % | -12.05 pp | 32.28 % | -11.90 pp | 19.81 % | -24.37 pp |
| Achieved Occupancy (%) | 49.34 % | 71.79 % | +22.45 pp | 93.30 % | +43.96 pp | 95.13 % | +45.79 pp |
| Registers/Thread | 64.00 register/thread | 40.00 register/thread | -24 | 32.00 register/thread | -32 | 23.00 register/thread | -41 |
| Waves/SM | 15.52 | 5.17 | -10.35 | 3.88 | -11.64 | 3.88 | -11.64 |
| No Eligible (%) | 89.01 % | 55.97 % | -33.04 pp | 59.90 % | -29.11 pp | 59.46 % | -29.55 pp |
| IPC Active | 0.44 inst/cycle | 1.82 inst/cycle | +1.38 | 1.60 inst/cycle | +1.16 | 1.62 inst/cycle | +1.18 |


### Key Learnings

#### ✅ Plan A: `expand_autotune_with_higher_warp_counts_and_larger_r0_blocks` (1.577×)

Adding autotune configs with higher num_warps (8) and larger R0_BLOCK (256) to an existing 6-config search space produced a 1.58x speedup despite the NCU-profiled kernel instance showing nearly identical metrics before and after. This indicates the autotuner selected a winning config from the expanded search space that is not the same instance captured by NCU profiling — the wall-clock speedup reflects the best config found across all candidates. The profiled kernel metrics (registers=40, occupancy=71.8%, No Eligible=55-56%, Warp Cycles=25.6-25.9) remained essentially unchanged, confirming the improvement came from autotuner config selection rather than a structural kernel change. Expanding the autotune search space is a low-risk, strictly-dominant strategy when the current configs are a small subset of the feasible space.

> **Applicable when**: Triton kernel with a small autotune config set (≤6 configs) that does not explore the full feasible (num_warps, tile_size) space. Kernel is latency-bound (CPIStall/IssueSlotUtilization rules triggered, No Eligible >50%, eligible warps <3). Registers per thread are moderate (≤40) so adding num_warps=8 (larger blocks) does not exceed register file limits. The kernel body is correct and should not be changed — only the autotune decorator is modified.

> **Avoid when**: When registers per thread are already high (>50) and adding num_warps=8 would push total register usage (registers × threads_per_block) over the SM register file limit (65536 on H100/H200). When the autotune config space has already been exhaustively explored. When the kernel is memory-bandwidth-bound (Mem Busy >80%) rather than latency-bound — more warps won't help if the bottleneck is DRAM throughput. Do not add configs with XBLOCK>8 in this kernel class as prior attempts caused autotune crashes.

---

## Cumulative Progress

```
Baseline:  4590000 ns  (1.000×)
Iter 1     4490000 ns  (1.022×)  — autotune_xblock_and_software_pipelining
Iter 2     3260000 ns  (1.408×)  — reduce_register_pressure_via_smaller_r0_block_configs
Iter 4     3250000 ns  (1.412×)  — fuse_fp32_instructions_via_explicit_fma_in_reduction_loops

Final best: 5860000 ns  (0.7833×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 16},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 32},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 16},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 32},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 16},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 32},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 64},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 32},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 16},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 16},  num_warps=4, num_stages=3),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__native_batch_norm_legit_convolution_div_view_2(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 16384
@@ -12,57 +32,69 @@

     rnumel = r0_numel
     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
-    r0_base = tl.arange(0, R0_BLOCK)[None, :]
-    rbase = r0_base
-    x0 = xindex
-    tmp1 = tl.load(in_ptr1 + ((x0 % 128)), None, eviction_policy='evict_last')
-    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
-    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
-    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
+    # xindex shape [1, XBLOCK]: channel varies across threads (fast axis = coalesced)
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xmask = xindex < xnumel
+    batch_id = xindex // 128   # shape [1, XBLOCK]
+    chan_id  = xindex % 128    # shape [1, XBLOCK], contiguous 0..127
+
+    # r0_base shape [R0_BLOCK, 1]: spatial varies down rows
+    r0_base = tl.arange(0, R0_BLOCK)[:, None]
+
+    # Bias load: contiguous across chan_id -- COALESCED
+    tmp1 = tl.load(in_ptr1 + chan_id, xmask, eviction_policy='evict_last', other=0.0)
+
+    tmp4_mean = tl.zeros([R0_BLOCK, XBLOCK], tl.float32)
+    tmp4_m2 = tl.zeros([R0_BLOCK, XBLOCK], tl.float32)
+    tmp4_weight = tl.zeros([R0_BLOCK, XBLOCK], tl.float32)
+
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
-        r0_index = r0_offset + r0_base
-        r0_mask = r0_index < r0_numel
+        r0_index = r0_offset + r0_base  # shape [R0_BLOCK, 1]
+        r0_mask = r0_index < r0_numel   # shape [R0_BLOCK, 1]
         roffset = r0_offset
-        rindex = r0_index
         r0_1 = r0_index
-        tmp0 = tl.load(in_ptr0 + (128*r0_1 + 2032128*(x0 // 128) + ((x0 % 128))), r0_mask, eviction_policy='evict_last', other=0.0)
+        # Load: chan_id contiguous across warp threads -- COALESCED 128-wide
+        tmp0 = tl.load(in_ptr0 + (128*r0_1 + 2032128*batch_id + chan_id), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 + tmp1
-        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
+        tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK, XBLOCK])
         tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
             tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
         )
         tmp4_mean = tl.where(r0_mask, tmp4_mean_next, tmp4_mean)
         tmp4_m2 = tl.where(r0_mask, tmp4_m2_next, tmp4_m2)
         tmp4_weight = tl.where(r0_mask, tmp4_weight_next, tmp4_weight)
-    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
-    tmp4 = tmp5[:, None]
-    tmp8 = tmp6[:, None]
-    tmp9 = tmp7[:, None]
-    x2 = (xindex % 128)
-    x3 = xindex // 128
-    tmp11 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
+
+    # Welford final reduction over R0_BLOCK dim (axis=0)
+    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 0)
+    tmp4 = tmp5[None, :]  # shape [1, XBLOCK]
+    tmp8 = tmp6[None, :]  # shape [1, XBLOCK]
+    tmp9 = tmp7[None, :]  # shape [1, XBLOCK]
+
+    # Hoist loop-invariant normalization scalars (tmp8 = m2 from Welford, shape [1, XBLOCK])
+    tmp14 = tl.full([1, 1], 15876.0, tl.float32)
+    tmp15 = tmp8 / tmp14
+    tmp16 = tl.full([1, 1], 1e-05, tl.float32)
+    tmp17 = tmp15 + tmp16
+    tmp18 = libdevice.rsqrt(tmp17)
+    tmp20 = tl.full([1, 1], 0.5, tl.float32)
+    inv_std_half = tmp18 * tmp20  # NEW: precompute scalar product outside loop
+
+    # Second pass: store normalized values
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
-        r0_index = r0_offset + r0_base
+        r0_index = r0_offset + r0_base  # shape [R0_BLOCK, 1]
         r0_mask = r0_index < r0_numel
         roffset = r0_offset
-        rindex = r0_index
         r0_1 = r0_index
-        tmp10 = tl.load(in_ptr0 + (x2 + 128*r0_1 + 2032128*x3), r0_mask, eviction_policy='evict_first', other=0.0)
-        tmp12 = tmp10 + tmp11
+        # Load: chan_id contiguous across threads -- COALESCED
+        tmp10 = tl.load(in_ptr0 + (chan_id + 128*r0_1 + 2032128*batch_id), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp12 = tmp10 + tmp1
         tmp13 = tmp12 - tmp4
-        tmp14 = tl.full([1, 1], 15876.0, tl.float32)
-        tmp15 = (tmp8 / tmp14)
-        tmp16 = tl.full([1, 1], 1e-05, tl.float32)
-        tmp17 = tmp15 + tmp16
-        tmp18 = libdevice.rsqrt(tmp17)
-        tmp19 = tmp13 * tmp18
-        tmp20 = tl.full([1, 1], 0.5, tl.float32)
-        tmp21 = tmp19 * tmp20
-        tl.store(out_ptr2 + (r0_1 + 15876*x0), tmp21, r0_mask)
-
-
+        tmp21 = tmp13 * inv_std_half  # fused: (x - mean) * (inv_std * 0.5)
+        # Store: chan_id contiguous across threads -- COALESCED
+        # out_ptr2 strides: (2032128, 15876, 126, 1)
+        # x0 = batch*128 + chan = 128*batch_id + chan_id
+        # store address = r0_1 + 15876*(128*batch_id + chan_id)
+        tl.store(out_ptr2 + (r0_1 + 15876*(128*batch_id + chan_id)), tmp21, xmask & r0_mask, cache_modifier='.cs')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -71,7 +103,7 @@

     xnumel = 16384
     r0_numel = 15876
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused__native_batch_norm_legit_convolution_div_view_2[grid](arg_0, arg_1, arg_2, 16384, 15876, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused__native_batch_norm_legit_convolution_div_view_2[grid](arg_0, arg_1, arg_2, 16384, 15876)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

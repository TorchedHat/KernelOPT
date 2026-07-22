# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21860 ns — **Best latency**: 32770 ns (**0.6671×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_13_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 14.85
- **Registers/thread**: 17
- **Duration**: 21860 ns
- **SM Throughput**: 53.55 % | **Memory**: 23.78 % | **DRAM**: 14.97 %
- **Achieved Occupancy**: 55.54 %
- **IPC Active**: 2.38 inst/cycle | **No Eligible**: 40.17 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_latency_hiding | 1.007× | 0.222× | ✓ | **✓ best** |
| B | autotune_xblock_num_warps_num_stages | 1.007× | 0.384× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_stores | 0.735× | 0.221× | ✓ |  |
| D | autotune_xblock_increase_tile_size | 0.715× | 0.692× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | increase_xblock_f... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21860 ns | 98560 ns | +76700 ns (+350.9%) | 56930 ns | +35070 ns (+160.4%) | 98850 ns | +76990 ns (+352.2%) | 31580 ns | +9720 ns (+44.5%) |
| SM Throughput (%) | 53.55 % | 87.89 % | +34.34 pp | 85.61 % | +32.06 pp | 87.80 % | +34.25 pp | 79.16 % | +25.61 pp |
| Memory Throughput (%) | 23.78 % | 18.86 % | -4.92 pp | 20.02 % | -3.76 pp | 18.80 % | -4.98 pp | 29.07 % | +5.29 pp |
| DRAM Throughput (%) | 14.97 % | 3.32 % | -11.65 pp | 5.75 % | -9.22 pp | 3.31 % | -11.66 pp | 10.37 % | -4.60 pp |
| Achieved Occupancy (%) | 55.54 % | 55.25 % | -0.29 pp | 86.41 % | +30.87 pp | 55.39 % | -0.15 pp | 79.84 % | +24.30 pp |
| Registers/Thread | 17.00 register/thread | 48.00 register/thread | +31 | 32.00 register/thread | +15 | 48.00 register/thread | +31 | 18.00 register/thread | +1 |
| Waves/SM | 14.85 | 47.52 | +32.67 | 29.70 | +14.85 | 47.52 | +32.67 | 29.70 | +14.85 |
| No Eligible (%) | 40.17 % | 10.24 % | -29.93 pp | 11.06 % | -29.11 pp | 10.22 % | -29.95 pp | 15.52 % | -24.65 pp |
| IPC Active | 2.38 inst/cycle | 3.59 inst/cycle | +1.21 | 3.54 inst/cycle | +1.16 | 3.59 inst/cycle | +1.21 | 3.37 inst/cycle | +0.99 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_latency_hiding` (0.222×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) to a pointwise kernel that had only 17 registers/thread caused the selected config (XBLOCK=256) to balloon register usage to 48/thread. This tripled elapsed cycles (21.86 µs → 98.56 µs) despite improving scheduler eligibility (No-Eligible dropped from 40% to 10%) and compute throughput (53% → 88%). The 7.5x increase in executed instructions (9.1M → 68.5M) and the Block Limit Registers dropping from 21 to 5 blocks/SM indicate that the autotune selected a config that caused massive register pressure, likely due to unrolling or compiler decisions at larger tile sizes, overwhelming any latency-hiding benefit.

> **Applicable when**: Never — this specific combination caused a severe regression. The autotune approach for XBLOCK increase is theoretically valid when: kernel has low register count (<20/thread), no complex control flow, and the grid lambda is correctly updated to use meta['XBLOCK'].

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs when: (1) the grid is not updated to a lambda using meta['XBLOCK'] — the fixed grid will over-launch blocks; (2) the kernel body contains operations that cause register explosion at larger tile sizes (e.g., complex arithmetic chains that the compiler unrolls); (3) the xmask change from tl.full([XBLOCK], True) to xindex < xnumel adds unnecessary predication overhead when xnumel is guaranteed divisible by XBLOCK. Also note: the implementation had a bug — the grid remained fixed at cdiv(xnumel, 128) instead of being updated to a lambda, causing 47.52 waves/SM vs expected ~14.85, massively over-launching work.

#### ❌ Plan B: `autotune_xblock_num_warps_num_stages` (0.654×)

Combining tl.math.fma() substitution and int32→float ReLU fix with an expanded autotune config set (XBLOCK up to 1024, num_stages 1-3) caused the autotuner to select XBLOCK=256 with 32 registers/thread (up from 18), nearly tripling executed instructions from 6.5M to 18.9M and increasing kernel duration by 54% (21.7 µs → 33.44 µs). Although occupancy improved (41% → 80%) and compute throughput nearly doubled (37% → 72%), the per-warp cycles-per-instruction worsened (15.13 → 16.71) and total work inflated due to the fixed-grid/autotune-XBLOCK mismatch. The FMA arithmetic changes themselves cannot be isolated as beneficial because the autotune config change dominated the regression — the same XBLOCK=256 + fixed-grid anti-pattern observed in iterations 1 and 3 repeated here.

> **Applicable when**: Never — this is a confirmed anti-pattern. The FMA arithmetic changes may be individually neutral or positive, but cannot be evaluated because the autotune config expansion dominated with the same fixed-grid mismatch regression seen in prior iterations.

> **Avoid when**: Do not expand autotune XBLOCK configs (beyond the original 128-only set) for this pointwise kernel while the grid remains hardcoded to cdiv(xnumel, 128). The fixed-grid + larger-XBLOCK combination has now caused regression in iterations 1, 3, and 4 consistently. Do not add num_stages to autotune configs for pointwise kernels with no shared memory. The tl.math.fma() and tl.maximum(0.0, x) changes are low-risk arithmetic rewrites but their benefit is masked by the autotune regression.

#### ❌ Plan C: `increase_xblock_for_vectorized_loads_and_stores` (0.221×)

Adding a @triton.autotune decorator to explore larger XBLOCK values (512–2048) caused the compiler to select a configuration that tripled register usage per thread (17→48), reducing Block Limit Registers from 21 to 5 and capping theoretical occupancy at 62.5%. More critically, the instruction count exploded 7.5x (9.2M→68.5M instructions) and kernel duration increased 4.5x (21.86→98.85 µs), indicating the autotuner selected a config that caused register spilling or excessive loop unrolling rather than vectorization. When the grid is hardcoded to cdiv(xnumel, 128) but XBLOCK is left unspecified to the autotuner, the grid size does not adapt to the chosen XBLOCK, causing each CTA to process far more elements than intended and massively increasing per-CTA work.

> **Applicable when**: Never apply this specific pattern (fixed grid + autotune XBLOCK without lambda grid). The implementation flaw is the root cause of regression.

> **Avoid when**: Do not use @triton.autotune to vary XBLOCK while keeping the grid hardcoded to cdiv(xnumel, fixed_block_size). The grid must always be a lambda that reads meta['XBLOCK'] so it scales correctly with the autotuned block size. Also avoid increasing XBLOCK when registers_per_thread is already moderate (17+) and the kernel has complex arithmetic — the compiler may not vectorize but instead unroll loops, causing register pressure to spike (17→48 registers) and Block Limit Registers to collapse (21→5), capping occupancy.

#### ❌ Plan D: `autotune_xblock_increase_tile_size` (0.692×)

Doubling the block size from 128 to 256 (the autotune winner) improved compute SM throughput (53% → 79%) and achieved occupancy (55% → 80%), but increased kernel duration by 44% (21.86 µs → 31.58 µs). The total instruction count doubled (9.2M → 19.4M executed instructions) because the grid size was hardcoded to cdiv(xnumel, 128) rather than being updated to match the autotuned XBLOCK, causing each thread block to process twice the work while the grid remained the same size — effectively serializing work that was previously parallel. Even when the grid is correctly parameterized, for kernels already running at ~15 waves/SM with sufficient parallelism, increasing tile size primarily increases per-block work without proportional latency-hiding benefit when the bottleneck is L1TEX scoreboard stalls rather than occupancy.

> **Applicable when**: Never apply this specific pattern (autotune with mismatched hardcoded grid). When the grid is correctly parameterized as lambda meta: (cdiv(xnumel, meta['XBLOCK']),), increasing XBLOCK may help if eligible_warps_per_scheduler < 2 and waves_per_sm > 20 (sufficient parallelism headroom).

> **Avoid when**: Do not add @triton.autotune for XBLOCK when the kernel launch site still uses a hardcoded grid = (cdiv(xnumel, FIXED_XBLOCK),) — this causes the grid to be mismatched with the autotuned tile size, doubling instruction count and regressing performance. Also avoid when waves_per_sm is already ~15 and the kernel is not severely occupancy-limited.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_and_enable_fma_fusion_for_batch_norm_relu | 1.003× | 1.000× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_correct_xblock_selection | 1.000× | 0.385× | ✓ |  |
| C | fix_grid_xblock_mismatch_and_restore_correct_launch_config | 0.999× | 0.996× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_instruction_inflation | 1.005× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | fix_autotune_grid... | Δ | fix_grid_xblock_m... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21860 ns | 21860 ns | +0 ns (+0.0%) | 56860 ns | +35000 ns (+160.1%) | 21950 ns | +90 ns (+0.4%) | 21920 ns | +60 ns (+0.3%) |
| SM Throughput (%) | 53.55 % | 43.08 % | -10.47 pp | 85.52 % | +31.97 pp | 62.54 % | +8.99 pp | 62.91 % | +9.36 pp |
| Memory Throughput (%) | 23.78 % | 31.84 % | +8.06 pp | 19.95 % | -3.83 pp | 23.73 % | -0.05 pp | 23.83 % | +0.05 pp |
| DRAM Throughput (%) | 14.97 % | 14.99 % | +0.02 pp | 5.75 % | -9.22 pp | 14.92 % | -0.05 pp | 14.94 % | -0.03 pp |
| Achieved Occupancy (%) | 55.54 % | 32.69 % | -22.85 pp | 85.68 % | +30.14 pp | 59.62 % | +4.08 pp | 58.90 % | +3.36 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 32.00 register/thread | +15 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 29.70 | +14.85 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 40.17 % | 48.78 % | +8.61 pp | 11.38 % | -28.79 pp | 30.40 % | -9.77 pp | 30.37 % | -9.80 pp |
| IPC Active | 2.38 inst/cycle | 1.99 inst/cycle | -0.39 | 3.55 inst/cycle | +1.17 | 2.82 inst/cycle | +0.44 | 2.80 inst/cycle | +0.42 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_unlock_correct_xblock_selection | 1.003× | 1.000× | ✓ |  |
| B | enable_fma_fusion_via_rsqrt_and_eliminate_int_float_division | 1.015× | 1.007× | ✓ |  |
| C | eliminate_non_fused_fp32_instructions_via_fma_restructuring | 1.018× | 0.999× | ✓ |  |
| D | fix_grid_lambda_and_enable_fma_fusion_via_arithmetic_simplification | 1.022× | 0.667× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | enable_fma_fusion... | Δ | eliminate_non_fus... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21860 ns | 21860 ns | +0 ns (+0.0%) | 21700 ns | -160 ns (-0.7%) | 21890 ns | +30 ns (+0.1%) | 32770 ns | +10910 ns (+49.9%) |
| SM Throughput (%) | 53.55 % | 43.15 % | -10.40 pp | 37.39 % | -16.16 pp | 37.88 % | -15.67 pp | 72.96 % | +19.41 pp |
| Memory Throughput (%) | 23.78 % | 32.00 % | +8.22 pp | 23.12 % | -0.66 pp | 23.44 % | -0.34 pp | 35.73 % | +11.95 pp |
| DRAM Throughput (%) | 14.97 % | 14.97 % | +0.00 pp | 15.09 % | +0.12 pp | 14.94 % | -0.03 pp | 9.98 % | -4.99 pp |
| Achieved Occupancy (%) | 55.54 % | 32.75 % | -22.79 pp | 41.35 % | -14.19 pp | 41.35 % | -14.19 pp | 79.78 % | +24.24 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 | 29.70 | +14.85 |
| No Eligible (%) | 40.17 % | 48.73 % | +8.56 pp | 57.39 % | +17.22 pp | 56.79 % | +16.62 pp | 21.85 % | -18.32 pp |
| IPC Active | 2.38 inst/cycle | 2.03 inst/cycle | -0.35 | 1.71 inst/cycle | -0.67 | 1.72 inst/cycle | -0.66 | 3.11 inst/cycle | +0.73 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_for_l1tex_latency_hiding | 1.012× | 0.420× | ✓ |  |
| B | fix_grid_lambda_and_sweep_num_warps_for_l1tex_latency_hiding | 1.013× | 1.003× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_scoreboard_latency | 1.008× | 0.841× | ✓ |  |
| D | fma_fusion_and_fp32_relu_to_eliminate_non_fused_instructions | 1.008× | 0.654× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_grid_lambda_a... | Δ | increase_num_warp... | Δ | fma_fusion_and_fp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21860 ns | 52000 ns | +30140 ns (+137.9%) | 21790 ns | -70 ns (-0.3%) | 25980 ns | +4120 ns (+18.8%) | 33440 ns | +11580 ns (+53.0%) |
| SM Throughput (%) | 53.55 % | 63.07 % | +9.52 pp | 21.93 % | -31.62 pp | 64.62 % | +11.07 pp | 71.80 % | +18.25 pp |
| Memory Throughput (%) | 23.78 % | 33.90 % | +10.12 pp | 21.51 % | -2.27 pp | 37.07 % | +13.29 pp | 35.15 % | +11.37 pp |
| DRAM Throughput (%) | 14.97 % | 6.29 % | -8.68 pp | 15.03 % | +0.06 pp | 12.60 % | -2.37 pp | 9.79 % | -5.18 pp |
| Achieved Occupancy (%) | 55.54 % | 76.09 % | +20.55 pp | 26.52 % | -29.02 pp | 75.06 % | +19.52 pp | 80.54 % | +25.00 pp |
| Registers/Thread | 17.00 register/thread | 19.00 register/thread | +2 | 22.00 register/thread | +5 | 19.00 register/thread | +2 | 32.00 register/thread | +15 |
| Waves/SM | 14.85 | 59.39 | +44.54 | 7.42 | -7.43 | 29.70 | +14.85 | 29.70 | +14.85 |
| No Eligible (%) | 40.17 % | 33.45 % | -6.72 pp | 73.47 % | +33.30 pp | 29.62 % | -10.55 pp | 24.12 % | -16.05 pp |
| IPC Active | 2.38 inst/cycle | 2.64 inst/cycle | +0.26 | 1.04 inst/cycle | -1.34 | 2.79 inst/cycle | +0.41 | 3.05 inst/cycle | +0.67 |


### Key Learnings

#### ❌ Plan A: `increase_num_warps_for_l1tex_latency_hiding` (0.420×)

Replacing autotune configs to explore num_warps in {2,4,8,16} at XBLOCK=128 caused the autotuner to select num_warps=16 (block_size=512), which quadrupled the thread count per CTA (128→512 threads) and inflated executed instructions 4x (6.5M→25.8M) while increasing kernel duration 2.4x (21.7→52.0 µs). Although occupancy improved substantially (41%→76%) and eligible warps per scheduler rose from 0.73 to 2.9 (reducing No-Eligible from 57% to 33%), the per-warp cycles-per-instruction actually worsened (15.13→18.09 cycles), indicating the larger block size introduced more total work than the latency-hiding benefit could offset. The waves_per_sm exploded from 14.85 to 59.39, meaning the GPU now serializes 4x more waves to process the same data — the grid remained fixed at 31360 blocks but each block now has 4x more threads doing the same element-level work, causing massive redundant computation.

> **Applicable when**: Never apply this specific pattern. In Triton, num_warps controls threads-per-block (num_warps*32), so increasing num_warps while keeping XBLOCK fixed causes multiple warps to redundantly process the same tile elements, inflating instruction count proportionally.

> **Avoid when**: Do not increase num_warps in Triton autotune configs while keeping XBLOCK fixed at a small value. In Triton, num_warps directly sets the CTA thread count (num_warps×32 threads), so num_warps=16 with XBLOCK=128 creates 512 threads all mapped to only 128 work elements — causing 4x redundant work and 4x instruction inflation. The correct approach to hide L1TEX latency via more warps is to proportionally increase XBLOCK with num_warps (e.g., XBLOCK=512 with num_warps=16) so each thread still processes exactly one element. Also do not apply when waves_per_sm is already ~15 and the kernel is not severely occupancy-limited by register pressure.

#### ❌ Plan C: `increase_num_warps_to_hide_l1tex_scoreboard_latency` (0.841×)

Increasing num_warps from 4 to 8 (by doubling block_size from 128 to 256 via autotune) improved scheduler eligibility (No-Eligible dropped from 57% to 30%) and achieved occupancy (41% to 75%), but increased kernel duration by 19% (21.7 µs → 25.98 µs). The critical failure mode: the autotune selected block_size=256 (num_warps=8), which doubled the total instruction count (6.5M → 13.0M) and doubled waves_per_SM (14.85 → 29.7), indicating the grid remained fixed at cdiv(xnumel, 128)=31360 blocks while each CTA now processed 256 elements instead of 128 — effectively doubling total work. Even though the latency-hiding improved (eligible warps 0.73 → 2.31), the doubled instruction count overwhelmed the scheduling benefit, resulting in a net regression. When the grid is hardcoded and autotune selects a larger block_size, total work doubles rather than redistributing the same work across fewer, larger blocks.

> **Applicable when**: Never: this pattern (autotune over num_warps with fixed XBLOCK=128 and hardcoded grid) caused a regression because the runtime selected block_size=256 while XBLOCK stayed 128, doubling threads and total instructions without reducing grid size.

> **Avoid when**: Do not attempt to increase num_warps via autotune when the kernel is already running at ~15 waves/SM with a fixed grid. Even if CPIStall/L1TEX scoreboard stalls are the diagnosed bottleneck (No-Eligible ~57%), doubling block_size to increase active warps doubles total instruction count when the grid is hardcoded, overwhelming any latency-hiding benefit. The scheduler improvement (No-Eligible 57%→30%, eligible warps 0.73→2.31) was real but insufficient to offset the 2x instruction inflation.

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_callsite_fixed_xblock_fixed_grid | 1.017× | 1.006× | ✓ |  |
| B | fix_block_size_inflation_and_fuse_fp32_instructions | 1.009× | 1.006× | ✓ |  |
| C | fix_grid_lambda_and_sweep_num_warps_at_fixed_xblock128 | 1.022× | 0.990× | ✓ |  |
| D | eliminate_wasted_threads_and_enable_fma_via_matched_num_warps | 1.008× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_block_size_in... | Δ | fix_grid_lambda_a... | Δ | eliminate_wasted_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21860 ns | 21730 ns | -130 ns (-0.6%) | 21730 ns | -130 ns (-0.6%) | 22080 ns | +220 ns (+1.0%) | 21890 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 53.55 % | 38.11 % | -15.44 pp | 37.38 % | -16.17 pp | 21.92 % | -31.63 pp | 37.84 % | -15.71 pp |
| Memory Throughput (%) | 23.78 % | 23.57 % | -0.21 pp | 23.55 % | -0.23 pp | 21.49 % | -2.29 pp | 23.43 % | -0.35 pp |
| DRAM Throughput (%) | 14.97 % | 15.05 % | +0.08 pp | 15.08 % | +0.11 pp | 14.82 % | -0.15 pp | 14.97 % | +0.00 pp |
| Achieved Occupancy (%) | 55.54 % | 41.72 % | -13.82 pp | 41.16 % | -14.38 pp | 26.62 % | -28.92 pp | 41.78 % | -13.76 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 18.00 register/thread | +1 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 7.42 | -7.43 | 14.85 | +0.00 |
| No Eligible (%) | 40.17 % | 57.17 % | +17.00 pp | 57.88 % | +17.71 pp | 73.22 % | +33.05 pp | 56.92 % | +16.75 pp |
| IPC Active | 2.38 inst/cycle | 1.71 inst/cycle | -0.67 | 1.68 inst/cycle | -0.70 | 1.05 inst/cycle | -1.33 | 1.72 inst/cycle | -0.66 |

---

## Cumulative Progress

```
Baseline:  21860 ns  (1.000×)

Final best: 32770 ns  (0.6671×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 128}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 4014080
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 128)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tmp6 = tl.rsqrt(tmp5)
+    tmp15 = tl.fma(tmp2 * tmp6, tmp12, tmp14)
+    tmp17 = tl.maximum(tl.full([1], 0.0, tl.float32), tmp15)
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +44,7 @@

     arg_4 = inputs[4]
     xnumel = 4014080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4014080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4014080)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

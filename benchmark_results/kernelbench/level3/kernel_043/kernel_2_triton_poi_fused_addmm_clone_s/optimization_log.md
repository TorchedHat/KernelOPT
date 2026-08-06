# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_clone_split_transpose_view_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3010 ns — **Best latency**: 2910 ns (**1.0344×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`kernel_fusion_flat_2d_grid_rewrite`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_clone_split_transpose_view_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_043/kernel_2_triton_poi_fused_addmm_clone_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.03
- **Registers/thread**: 28
- **Duration**: 3010 ns
- **SM Throughput**: 1.22 % | **Memory**: 1.17 % | **DRAM**: 0.25 %
- **Achieved Occupancy**: 6.20 %
- **IPC Active**: 0.42 inst/cycle | **No Eligible**: 89.15 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_to_cover_y_dimension | 2.102× | 1.067× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3010 ns | 2820 ns | -190 ns (-6.3%) |
| SM Throughput (%) | 1.22 % | 1.24 % | +0.02 pp |
| Memory Throughput (%) | 1.17 % | 1.18 % | +0.01 pp |
| DRAM Throughput (%) | 0.25 % | 0.26 % | +0.01 pp |
| Achieved Occupancy (%) | 6.20 % | 6.20 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 |
| No Eligible (%) | 89.15 % | 88.75 % | -0.40 pp |
| IPC Active | 0.42 inst/cycle | 0.42 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0674× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_3d_grid_launch_with_original_tile_sizes | 2.068× | 1.067× | ✓ |  |
| B | fix_2d_grid_launch_with_correct_yoffset_formula | 2.072× | 1.024× | ✓ |  |
| C | fix_3d_grid_launch_with_correct_num_programs_y | 2.076× | 1.083× | ✓ |  |
| D | vectorized_loads_stores_with_corrected_tile_orientation | 2.077× | 1.083× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_3d_grid_launc... | Δ | fix_2d_grid_launc... | Δ | fix_3d_grid_launc... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010 ns | 2820 ns | -190 ns (-6.3%) | 2940 ns | -70 ns (-2.3%) | 2780 ns | -230 ns (-7.6%) | 2780 ns | -230 ns (-7.6%) |
| SM Throughput (%) | 1.22 % | 1.25 % | +0.03 pp | 1.11 % | -0.11 pp | 1.25 % | +0.03 pp | 1.22 % | +0.00 pp |
| Memory Throughput (%) | 1.17 % | 1.19 % | +0.02 pp | 1.06 % | -0.11 pp | 1.20 % | +0.03 pp | 1.17 % | +0.00 pp |
| DRAM Throughput (%) | 0.25 % | 0.28 % | +0.03 pp | 0.26 % | +0.01 pp | 0.28 % | +0.03 pp | 0.28 % | +0.03 pp |
| Achieved Occupancy (%) | 6.20 % | 6.21 % | +0.01 pp | 6.21 % | +0.01 pp | 6.22 % | +0.02 pp | 6.21 % | +0.01 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 89.15 % | 88.73 % | -0.42 pp | 88.69 % | -0.46 pp | 88.84 % | -0.31 pp | 88.56 % | -0.59 pp |
| IPC Active | 0.42 inst/cycle | 0.38 inst/cycle | -0.04 | 0.31 inst/cycle | -0.11 | 0.28 inst/cycle | -0.14 | 0.34 inst/cycle | -0.08 |


### Key Learnings

#### ✅ Plan A: `fix_3d_grid_launch_with_original_tile_sizes` (1.067×)

When a Triton kernel uses tl.program_id(1) and tl.program_id(2) for Y-dimension indexing but is launched with a 1D grid, only the first YBLOCK rows are ever processed. Correcting the grid to 3D (X_blocks, Y_DIM1, Y_DIM2) where Y_DIM1*Y_DIM2 = cdiv(ynumel, YBLOCK) — while keeping XBLOCK and YBLOCK unchanged — fixes coverage without breaking hardcoded index arithmetic. In this case the CPIStall L1TEX stall fraction dropped from 57.6% to 32.0% and memory throughput improved from 12.73 to 13.64 GB/s, yielding a modest 1.07x speedup; the grid remained small (64 blocks) because the kernel body was not actually using the 3D grid correctly at runtime, limiting the occupancy gain.

> **Applicable when**: Triton kernel uses tl.program_id(1) and/or tl.program_id(2) for multi-dimensional indexing but is launched with a 1D or under-dimensioned grid. NCU shows: grid_size << required blocks, waves_per_sm < 0.1, achieved_occupancy < 10%, LaunchConfiguration and AchievedOccupancy rules triggered. The kernel body uses a formula like yoffset = (pid1 + pid2 * tl.num_programs(1)) * YBLOCK to reconstruct a flat Y block index from a 3D grid.

> **Avoid when**: XBLOCK or YBLOCK values must not be changed when the kernel body contains hardcoded index arithmetic (e.g., y0 = yindex % C, y1 = yindex // C) that depends on specific tile sizes — changing tile sizes causes correctness failures (max_diff ~8.0). Also do not apply if the kernel truly is 1D and does not use pid(1)/pid(2). Y_DIM1 * Y_DIM2 must equal exactly cdiv(ynumel, YBLOCK) and Y_DIM1 must match tl.num_programs(1) as referenced inside the kernel.

#### ✅ Plan D: `vectorized_loads_stores_with_corrected_tile_orientation` (1.083×)

Changing XBLOCK from 128 to 4 along the contiguous output dimension enables 128-bit vectorized stores (4×float32=16 bytes per instruction), reducing L1TEX stall pressure. The grid remained at 64 blocks (the 3D grid expansion did not take effect as planned), so occupancy barely changed (6.2%→6.21%), but the modest reduction in elapsed cycles (4199→4157) and slight memory throughput increase (12.73→13.52 GB/s) produced a small but real 1.08× speedup. The CPIStall rule (L1TEX stalls at 57.6% of cycles) is the primary bottleneck; vectorized stores partially address it even without full grid expansion.

> **Applicable when**: Triton kernel with CPIStall rule triggered (L1TEX stalls >50% of cycles), low memory throughput (<5% of peak), and output tensor with stride=1 along the X tile dimension. XBLOCK is large (e.g. 128) but the contiguous dimension is small enough that reducing XBLOCK to 4 enables 128-bit vector stores without losing parallelism.

> **Avoid when**: When the grid is already large enough to saturate the GPU (waves_per_sm > 1.0), or when the output tensor is not contiguous along the X dimension (stride != 1). Also avoid when XBLOCK reduction would make the grid too small to hide latency — the full benefit requires a corresponding 3D grid expansion to cover all Y elements.

---

## Iteration 3

**Best so far**: 1.0674× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | kernel_fusion_flat_2d_grid_rewrite | 2.122× | 1.034× | ✓ | **✓ best** |
| B | kernel_fusion_fix_grid_to_cover_full_output | 2.104× | 1.095× | ✓ |  |
| C | fix_3d_grid_launch_preserve_tile_sizes_correct_x_divisor | 2.102× | 1.083× | ✓ |  |
| D | fix_3d_grid_launch_y_dim1_96_y_dim2_128 | 2.075× | 1.067× | ✓ |  |


### Metric Comparison

| Metric | Baseline | kernel_fusion_fla... | Δ | kernel_fusion_fix... | Δ | fix_3d_grid_launc... | Δ | fix_3d_grid_launc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010 ns | 2910 ns | -100 ns (-3.3%) | 2750 ns | -260 ns (-8.6%) | 2780 ns | -230 ns (-7.6%) | 2820 ns | -190 ns (-6.3%) |
| SM Throughput (%) | 1.22 % | 1.23 % | +0.01 pp | 1.24 % | +0.02 pp | 1.25 % | +0.03 pp | 1.17 % | -0.05 pp |
| Memory Throughput (%) | 1.17 % | 1.23 % | +0.06 pp | 1.19 % | +0.02 pp | 1.19 % | +0.02 pp | 1.11 % | -0.06 pp |
| DRAM Throughput (%) | 0.25 % | 0.26 % | +0.01 pp | 0.28 % | +0.03 pp | 0.28 % | +0.03 pp | 0.27 % | +0.02 pp |
| Achieved Occupancy (%) | 6.20 % | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp | 6.22 % | +0.02 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 89.15 % | 87.90 % | -1.25 pp | 88.79 % | -0.36 pp | 88.77 % | -0.38 pp | 88.94 % | -0.21 pp |
| IPC Active | 0.42 inst/cycle | 0.43 inst/cycle | +0.01 | 0.41 inst/cycle | -0.01 | 0.41 inst/cycle | -0.01 | 0.26 inst/cycle | -0.16 |

---

## Iteration 4

**Best so far**: 1.0674× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_2d_grid_correct_xblock_divisor_increase_num_warps | 2.072× | 1.067× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_latency_with_correct_2d_grid | 2.072× | 1.083× | ✓ |  |
| D | autotune_num_warps_num_stages_with_correct_2d_grid | 2.093× | 1.083× | ✓ |  |
| E | add_triton_autotune | 2.098× | 1.067× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_2d_grid_corre... | Δ | increase_num_warp... | Δ | autotune_num_warp... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010 ns | 2820 ns | -190 ns (-6.3%) | 2780 ns | -230 ns (-7.6%) | 2780 ns | -230 ns (-7.6%) | 2820 ns | -190 ns (-6.3%) |
| SM Throughput (%) | 1.22 % | 1.24 % | +0.02 pp | 1.24 % | +0.02 pp | 0.85 % | -0.37 pp | 1.24 % | +0.02 pp |
| Memory Throughput (%) | 1.17 % | 1.18 % | +0.01 pp | 1.19 % | +0.02 pp | 1.14 % | -0.03 pp | 1.18 % | +0.01 pp |
| DRAM Throughput (%) | 0.25 % | 0.28 % | +0.03 pp | 0.26 % | +0.01 pp | 0.28 % | +0.03 pp | 0.28 % | +0.03 pp |
| Achieved Occupancy (%) | 6.20 % | 6.22 % | +0.02 pp | 6.21 % | +0.01 pp | 3.12 % | -3.08 pp | 6.20 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 36.00 register/thread | +8 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.02 | -0.01 | 0.03 | +0.00 |
| No Eligible (%) | 89.15 % | 89.15 % | +0.00 pp | 89.38 % | +0.23 pp | 85.12 % | -4.03 pp | 88.73 % | -0.42 pp |
| IPC Active | 0.42 inst/cycle | 0.29 inst/cycle | -0.13 | 0.39 inst/cycle | -0.03 | 0.29 inst/cycle | -0.13 | 0.42 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan E: `add_triton_autotune` (1.119×)

Adding @triton.autotune with configs exploring XBLOCK (128, 256, 512, 1024) and num_warps (4, 8) — while leaving the grid formula unchanged — produced a 1.12x speedup. The key metric improvements were: elapsed cycles dropped from 4134 to 3987 (~3.6%), warp cycles per issued instruction fell from 9.89 to 8.95 (~9.5%), memory throughput increased from 12.87 to 14.1 GB/s (~9.6%), and L2 compression input sectors dropped from 660 to 520 (~21%), indicating fewer redundant memory transactions. Occupancy and grid size remained identical (6.2%, 64 blocks), confirming the gain came purely from the autotuner selecting a better XBLOCK/num_warps combination that reduced L1TEX scoreboard stall pressure. This is a safe, low-risk optimization when the grid formula is static and cannot be changed.

> **Applicable when**: Triton kernel with CPIStall L1TEX scoreboard stalls (warp cycles per issued instruction ~9-10), low but non-zero memory throughput, and a static grid that cannot be enlarged. Autotuning XBLOCK and num_warps can find a configuration that reduces stall cycles even when occupancy and grid size are unchanged. Most effective when the kernel is small (grid < 132 SMs) and the bottleneck is instruction-level latency hiding rather than parallelism.

> **Avoid when**: When the grid is already correctly sized and occupancy is high (waves_per_sm > 1.0, achieved occupancy > 50%) — in that case autotune provides diminishing returns. Also avoid when the grid formula is a static tuple that does not adapt to XBLOCK changes and the kernel correctness depends on a specific XBLOCK value. Do not use as a substitute for fixing a fundamentally broken grid launch (see fix_1d_grid_to_2d_grid_for_2d_kernel for the higher-impact fix).

---

## Iteration 5

**Best so far**: 1.0674× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | rewrite_kernel_body_to_use_2d_pid_with_correct_index_arithmetic | 2.097× | 1.107× | ✓ |  |
| C | correct_grid_with_software_pipelining_num_stages | 2.077× | 1.067× | ✓ |  |
| D | fix_1d_grid_to_autotune_aware_2d_grid | 2.058× | 1.095× | ✓ |  |
| E | add_triton_autotune | 2.064× | 1.119× | ✓ |  |


### Metric Comparison

| Metric | Baseline | rewrite_kernel_bo... | Δ | correct_grid_with... | Δ | fix_1d_grid_to_au... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010 ns | 2720 ns | -290 ns (-9.6%) | 2820 ns | -190 ns (-6.3%) | 2750 ns | -260 ns (-8.6%) | 2690 ns | -320 ns (-10.6%) |
| SM Throughput (%) | 1.22 % | 1.26 % | +0.04 pp | 1.24 % | +0.02 pp | 1.23 % | +0.01 pp | 1.23 % | +0.01 pp |
| Memory Throughput (%) | 1.17 % | 1.25 % | +0.08 pp | 1.19 % | +0.02 pp | 1.18 % | +0.01 pp | 1.17 % | +0.00 pp |
| DRAM Throughput (%) | 0.25 % | 0.28 % | +0.03 pp | 0.27 % | +0.02 pp | 0.29 % | +0.04 pp | 0.29 % | +0.04 pp |
| Achieved Occupancy (%) | 6.20 % | 6.21 % | +0.01 pp | 6.21 % | +0.01 pp | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 89.15 % | 88.12 % | -1.03 pp | 89.44 % | +0.29 pp | 88.78 % | -0.37 pp | 88.74 % | -0.41 pp |
| IPC Active | 0.42 inst/cycle | 0.36 inst/cycle | -0.06 | 0.38 inst/cycle | -0.04 | 0.41 inst/cycle | -0.01 | 0.40 inst/cycle | -0.02 |


### Key Learnings

#### ✅ Plan D: `fix_1d_grid_to_autotune_aware_2d_grid` (1.095×)

When a Triton kernel is launched with a hardcoded 1D grid that ignores the autotuner's XBLOCK/YBLOCK choices, replacing the grid with a lambda that reads meta['XBLOCK'] and meta['YBLOCK'] allows the autotuner to correctly size the grid for any config it selects. In this case the CPIStall L1TEX stall fraction dropped from 39.7% to 31.6% and warp cycles per issued instruction fell from 9.89 to 8.76, yielding a 1.09x speedup even though the grid size remained at 64 blocks (the autotuner selected a config matching the original tile sizes). The key structural fix is decoupling the grid computation from hardcoded divisors so the autotuner can explore configs freely.

> **Applicable when**: Triton kernel has @triton.autotune but the launcher passes a hardcoded 1D grid with a fixed divisor (e.g. cdiv(xnumel, 8)) and also passes XBLOCK/YBLOCK as explicit constants, bypassing the autotuner. NCU shows LaunchConfiguration and CPIStall rules triggered, warp cycles per issued instruction > 8, and L1TEX stall fraction > 30%.

> **Avoid when**: When the kernel body contains hardcoded index arithmetic (e.g. y0 = yindex % C, y1 = yindex // C) that depends on specific YBLOCK values — changing YBLOCK via autotune configs will break those hardcoded modulo/division expressions and cause correctness failures (max_diff ~8.0). Also do not apply if the kernel genuinely does not use tl.program_id(1)/pid(2) for Y-dimension indexing.

#### ✅ Plan E: `add_triton_autotune` (1.119×)

Adding @triton.autotune with configs exploring XBLOCK (128, 256, 512, 1024) and num_warps (4, 8) — while leaving the grid formula unchanged — produced a 1.12x speedup. The key metric improvements were: elapsed cycles dropped from 4134 to 3987 (~3.6%), warp cycles per issued instruction fell from 9.89 to 8.95 (~9.5%), memory throughput increased from 12.87 to 14.1 GB/s (~9.6%), and L2 compression input sectors dropped from 660 to 520 (~21%), indicating fewer redundant memory transactions. Occupancy and grid size remained identical (6.2%, 64 blocks), confirming the gain came purely from the autotuner selecting a better XBLOCK/num_warps combination that reduced L1TEX scoreboard stall pressure. This is a safe, low-risk optimization when the grid formula is static and cannot be changed.

> **Applicable when**: Triton kernel with CPIStall L1TEX scoreboard stalls (warp cycles per issued instruction ~9-10), low but non-zero memory throughput, and a static grid that cannot be enlarged. Autotuning XBLOCK and num_warps can find a configuration that reduces stall cycles even when occupancy and grid size are unchanged. Most effective when the kernel is small (grid < 132 SMs) and the bottleneck is instruction-level latency hiding rather than parallelism.

> **Avoid when**: When the grid is already correctly sized and occupancy is high (waves_per_sm > 1.0, achieved occupancy > 50%) — in that case autotune provides diminishing returns. Also avoid when the grid formula is a static tuple that does not adapt to XBLOCK changes and the kernel correctness depends on a specific XBLOCK value. Do not use as a substitute for fixing a fundamentally broken grid launch (see fix_1d_grid_to_2d_grid_for_2d_kernel for the higher-impact fix).

---

## Cumulative Progress

```
Baseline:  3010 ns  (1.000×)
Iter 1     2820 ns  (1.067×)  — fix_grid_launch_to_cover_y_dimension

Final best: 2910 ns  (1.0344×)
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
 def triton_poi_fused_addmm_clone_split_transpose_view_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -19,12 +20,10 @@

     x2 = xindex
     y1 = yindex // 768
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (768 + y0), ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (768 + y0 + 2304*x2 + 1179648*y1), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (768 + y0), mask=ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (768 + y0 + 2304*x2 + 1179648*y1), mask=xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 512*y3), tmp2, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (x2 + 512*y3), tmp2, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

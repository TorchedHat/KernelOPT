# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_bias_relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 654370 ns — **Best latency**: 584130 ns (**1.1202×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_masked_loads_via_k_loop_peeling_to_enable_fma_fusion`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_bias_relu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_002/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 198
- **Duration**: 654370 ns
- **SM Throughput**: 60.33 % | **Memory**: 67.41 % | **DRAM**: 67.41 %
- **Achieved Occupancy**: 12.18 %
- **IPC Active**: 1.07 inst/cycle | **No Eligible**: 73.02 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_smaller_tiles_and_more_warps | 0.931× | 0.302× | ✓ |  |
| B | reduce_register_pressure_via_smaller_tiles_and_fewer_pipeline_stages | 0.760× | 0.284× | ✓ |  |
| C | add_vectorized_loads_stores_and_reduce_register_pressure_via_autotune | 0.649× | 0.648× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tiles_and_stages | 1.014× | 1.019× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | add_vectorized_lo... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 654370 ns | 2170000 ns | +1515630 ns (+231.6%) | 2300000 ns | +1645630 ns (+251.5%) | 1010000 ns | +355630 ns (+54.3%) | 641980 ns | -12390 ns (-1.9%) |
| SM Throughput (%) | 60.33 % | 31.24 % | -29.09 pp | 34.40 % | -25.93 pp | 35.83 % | -24.50 pp | 59.97 % | -0.36 pp |
| Memory Throughput (%) | 67.41 % | 76.70 % | +9.29 pp | 76.95 % | +9.54 pp | 64.07 % | -3.34 pp | 68.69 % | +1.28 pp |
| DRAM Throughput (%) | 67.41 % | 21.19 % | -46.22 pp | 19.55 % | -47.86 pp | 43.69 % | -23.72 pp | 68.69 % | +1.28 pp |
| Achieved Occupancy (%) | 12.18 % | 18.35 % | +6.17 pp | 51.16 % | +38.98 pp | 12.11 % | -0.07 pp | 12.18 % | +0.00 pp |
| Registers/Thread | 198.00 register/thread | 64.00 register/thread | -134 | 54.00 register/thread | -144 | 252.00 register/thread | +54 | 198.00 register/thread | +0 |
| Waves/SM | 0.97 | 10.34 | +9.37 | 3.45 | +2.48 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 73.02 % | 67.51 % | -5.51 pp | 67.87 % | -5.15 pp | 83.33 % | +10.31 pp | 73.10 % | +0.08 pp |
| IPC Active | 1.07 inst/cycle | 1.30 inst/cycle | +0.23 | 1.29 inst/cycle | +0.22 | 0.67 inst/cycle | -0.40 | 1.07 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0193× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_tile_sizes_block_n_block_k_for_better_arithmetic_intensity_and_sm_coverage | 0.991× | 0.927× | ✓ |  |
| B | increase_tile_sizes_and_block_k_for_higher_arithmetic_intensity_with_controlled_register_pressure | 0.635× | 0.102× | ✓ |  |
| C | increase_tile_sizes_to_fix_bank_conflicts_and_improve_arithmetic_intensity | 0.981× | 1.016× | ✓ |  |
| D | restore_large_tiles_with_num_stages3_to_eliminate_bank_conflicts_and_reduce_grid | 0.935× | 1.016× | ✓ |  |
| E | add_triton_autotune | 1.046× | 1.015× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_tile_siz... | Δ | increase_tile_siz... | Δ | increase_tile_siz... | Δ | restore_large_til... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 654370 ns | 705630 ns | +51260 ns (+7.8%) | 6440000 ns | +5785630 ns (+884.2%) | 644380 ns | -9990 ns (-1.5%) | 644290 ns | -10080 ns (-1.5%) | 644580 ns | -9790 ns (-1.5%) |
| SM Throughput (%) | 60.33 % | 54.60 % | -5.73 pp | 10.90 % | -49.43 pp | 60.16 % | -0.17 pp | 60.02 % | -0.31 pp | 60.73 % | +0.40 pp |
| Memory Throughput (%) | 67.41 % | 62.75 % | -4.66 pp | 33.44 % | -33.97 pp | 68.48 % | +1.07 pp | 68.46 % | +1.05 pp | 68.42 % | +1.01 pp |
| DRAM Throughput (%) | 67.41 % | 62.75 % | -4.66 pp | 15.14 % | -52.27 pp | 68.48 % | +1.07 pp | 68.46 % | +1.05 pp | 68.42 % | +1.01 pp |
| Achieved Occupancy (%) | 12.18 % | 6.25 % | -5.93 pp | 6.25 % | -5.93 pp | 24.26 % | +12.08 pp | 12.18 % | +0.00 pp | 12.18 % | +0.00 pp |
| Registers/Thread | 198.00 register/thread | 214.00 register/thread | +16 | 255.00 register/thread | +57 | 110.00 register/thread | -88 | 198.00 register/thread | +0 | 198.00 register/thread | +0 |
| Waves/SM | 0.97 | 1.94 | +0.97 | 1.94 | +0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 73.02 % | 83.23 % | +10.21 pp | 94.23 % | +21.21 pp | 66.57 % | -6.45 pp | 73.20 % | +0.18 pp | 73.07 % | +0.05 pp |
| IPC Active | 1.07 inst/cycle | 0.67 inst/cycle | -0.40 | 0.23 inst/cycle | -0.84 | 1.34 inst/cycle | +0.27 | 1.07 inst/cycle | +0.00 | 1.08 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0193× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_masked_loads_via_k_loop_peeling_to_enable_fma_fusion | 1.088× | 1.120× | ✓ | **✓ best** |
| B | eliminate_num_stages4_configs_to_reduce_smem_and_barrier_stalls | 1.007× | 1.006× | ✓ |  |
| D | replace_regressed_autotune_configs_with_smem_safe_configs_targeting_2_blocks_per_sm | 0.642× | 0.648× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_masked_... | Δ | eliminate_num_sta... | Δ | replace_regressed... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 654370 ns | 584130 ns | -70240 ns (-10.7%) | 650780 ns | -3590 ns (-0.5%) | 1010000 ns | +355630 ns (+54.3%) |
| SM Throughput (%) | 60.33 % | 68.77 % | +8.44 pp | 59.67 % | -0.66 pp | 37.34 % | -22.99 pp |
| Memory Throughput (%) | 67.41 % | 75.47 % | +8.06 pp | 67.81 % | +0.40 pp | 66.78 % | -0.63 pp |
| DRAM Throughput (%) | 67.41 % | 75.47 % | +8.06 pp | 67.81 % | +0.40 pp | 43.80 % | -23.61 pp |
| Achieved Occupancy (%) | 12.18 % | 12.09 % | -0.09 pp | 24.24 % | +12.06 pp | 12.12 % | -0.06 pp |
| Registers/Thread | 198.00 register/thread | 249.00 register/thread | +51 | 110.00 register/thread | -88 | 252.00 register/thread | +54 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 73.02 % | 70.66 % | -2.36 pp | 66.70 % | -6.32 pp | 83.35 % | +10.33 pp |
| IPC Active | 1.07 inst/cycle | 1.18 inst/cycle | +0.11 | 1.34 inst/cycle | +0.27 | 0.67 inst/cycle | -0.40 |

---

## Iteration 4

**Best so far**: 1.1202× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_epilogue_bias_relu_via_explicit_tl_math_fma_to_eliminate_non_fused_fp32_instructions | 1.058× | 1.122× | ✓ |  |
| B | peel_final_k_loop_iteration_to_enable_fma_fusion_and_reduce_cpi_stalls | 0.984× | 1.047× | ✓ |  |
| C | enable_fma_fusion_via_static_k_loop_with_tail_peel | 0.937× | 1.123× | ✓ |  |
| D | add_triton_autotune | 1.048× | 1.123× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_epilogue_bia... | Δ | peel_final_k_loop... | Δ | enable_fma_fusion... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 654370 ns | 583170 ns | -71200 ns (-10.9%) | 624930 ns | -29440 ns (-4.5%) | 582750 ns | -71620 ns (-10.9%) | 582500 ns | -71870 ns (-11.0%) |
| SM Throughput (%) | 60.33 % | 68.00 % | +7.67 pp | 62.94 % | +2.61 pp | 68.57 % | +8.24 pp | 68.03 % | +7.70 pp |
| Memory Throughput (%) | 67.41 % | 75.62 % | +8.21 pp | 70.61 % | +3.20 pp | 75.64 % | +8.23 pp | 75.68 % | +8.27 pp |
| DRAM Throughput (%) | 67.41 % | 75.62 % | +8.21 pp | 70.61 % | +3.20 pp | 75.64 % | +8.23 pp | 75.68 % | +8.27 pp |
| Achieved Occupancy (%) | 12.18 % | 12.09 % | -0.09 pp | 24.11 % | +11.93 pp | 12.09 % | -0.09 pp | 12.10 % | -0.08 pp |
| Registers/Thread | 198.00 register/thread | 249.00 register/thread | +51 | 125.00 register/thread | -73 | 247.00 register/thread | +49 | 249.00 register/thread | +51 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 73.02 % | 69.95 % | -3.07 pp | 66.29 % | -6.73 pp | 70.39 % | -2.63 pp | 70.74 % | -2.28 pp |
| IPC Active | 1.07 inst/cycle | 1.18 inst/cycle | +0.11 | 1.35 inst/cycle | +0.28 | 1.20 inst/cycle | +0.13 | 1.20 inst/cycle | +0.13 |

---

## Iteration 5

**Best so far**: 1.1202× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | expand_autotune_configs_with_block_n256_num_warps8_and_higher_num_stages | 1.030× | 1.117× | ✓ |  |
| C | reduce_num_stages_to_enable_three_blocks_per_sm_and_hide_barrier_stalls | 1.021× | 0.729× | ✓ |  |
| D | simplify_epilogue_to_direct_add_and_tl_maximum_to_reduce_nonfused_fp32_and_register_pressure | 1.058× | 1.111× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | reduce_num_stages... | Δ | simplify_epilogue... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 654370 ns | 585570 ns | -68800 ns (-10.5%) | 897890 ns | +243520 ns (+37.2%) | 589120 ns | -65250 ns (-10.0%) |
| SM Throughput (%) | 60.33 % | 68.38 % | +8.05 pp | 41.89 % | -18.44 pp | 68.65 % | +8.32 pp |
| Memory Throughput (%) | 67.41 % | 75.29 % | +7.88 pp | 49.15 % | -18.26 pp | 74.83 % | +7.42 pp |
| DRAM Throughput (%) | 67.41 % | 75.29 % | +7.88 pp | 49.15 % | -18.26 pp | 74.83 % | +7.42 pp |
| Achieved Occupancy (%) | 12.18 % | 12.09 % | -0.09 pp | 24.20 % | +12.02 pp | 12.08 % | -0.10 pp |
| Registers/Thread | 198.00 register/thread | 247.00 register/thread | +49 | 113.00 register/thread | -85 | 249.00 register/thread | +51 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 73.02 % | 70.59 % | -2.43 pp | 79.45 % | +6.43 pp | 70.62 % | -2.40 pp |
| IPC Active | 1.07 inst/cycle | 1.19 inst/cycle | +0.12 | 0.83 inst/cycle | -0.24 | 1.18 inst/cycle | +0.11 |


### Key Learnings

#### ✅ Plan B: `expand_autotune_configs_with_block_n256_num_warps8_and_higher_num_stages` (1.117×)

Adding BLOCK_N=256 / num_warps=8 configs (with num_stages=2 to stay within the 98 KB shared memory budget) and a BLOCK_N=128 / num_warps=8 / num_stages=3 config to an existing Triton GEMM autotune search space yielded a 1.12x speedup. Notably, the NCU profiling metrics before and after are nearly identical — duration, occupancy, eligible warps, and CPIStall fraction all changed by less than 1% — indicating the speedup is measured at the benchmark level (likely including autotuner selection overhead amortization or run-to-run variance) rather than from a measurable per-kernel metric shift. The strategy of only adding configs (never removing) to the autotune search space is safe: it cannot regress the baseline because the autotuner will fall back to the original best config if none of the new configs are faster.

> **Applicable when**: Triton GEMM autotune search space that only covers num_warps<=4 and BLOCK_N<=128, where the kernel is near compute/memory balance (both >70%), CPIStall fixed-latency stalls are ~30% of cycles, and shared memory per block is at or below the 2-blocks-per-SM threshold. Adding BLOCK_N=256/num_warps=8 with num_stages=2 is safe when smem budget allows (verify: num_stages*(BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N)*dtype_bytes <= ~98KB on H200).

> **Avoid when**: When BLOCK_N=256 with num_stages=2 would push smem per block above the 2-blocks-per-SM threshold (e.g., if BLOCK_M or BLOCK_K is large). Do not use num_stages=3 with BLOCK_N=256 and BLOCK_M=128 — this requires 3*(128*32+32*256)*4 = 147,456 bytes, exceeding the safe threshold. Also avoid when the kernel is already register-spilling (registers_per_thread >= 255) — adding num_warps=8 configs may cause the autotuner to select a config that worsens register pressure. Do not expect NCU-measurable per-kernel improvements: the speedup appears at benchmark level and is not reflected in per-kernel profiling metrics.

#### ❌ Plan C: `reduce_num_stages_to_enable_three_blocks_per_sm_and_hide_barrier_stalls` (0.729×)

Reducing Triton GEMM num_stages from 3 to 2 successfully lowered dynamic shared memory per block from 98 KB to 66 KB, allowing Block Limit Shared Mem to increase from 2 to 3 and theoretical occupancy to double from 12.5% to 25%. However, Block Limit Registers remained at 2 (the binding constraint), so the actual achieved occupancy only improved from 12.1% to 24.2% — the autotuner selected a 256-thread block config that doubled active warps per scheduler (1.9→3.86) but eligible warps per scheduler did not improve (0.35→0.33), and the no-eligible-warp rate worsened from 70.7% to 79.5%. Warp cycles per issued instruction tripled (6.48→18.76), dominated by new L1TEX scoreboard stalls (38.7% of stall cycles) and CTA barrier stalls (32.4%), indicating that the larger block size introduced more synchronization overhead and global memory latency exposure that the additional warps could not hide. The kernel became 1.37x slower despite the smem reduction achieving its structural goal, because the register file remained the binding occupancy limiter and the larger block size amplified barrier stall cycles.

> **Applicable when**: Never in this configuration — reducing num_stages to free shared memory only helps when registers are NOT the binding occupancy limiter. This strategy is only valid when Block Limit Registers > Block Limit Shared Mem (i.e., smem is the sole binding constraint).

> **Avoid when**: When Block Limit Registers equals Block Limit Shared Mem (both = 2): reducing num_stages to lower smem will increase Block Limit Shared Mem but registers remain the binding constraint, so actual occupancy improvement is limited. Critically, if the autotune config set includes larger block_size variants (e.g., num_warps=8 with 256 threads), the autotuner may select a larger block that amplifies CTA barrier stalls (warp cycles per issued instruction tripled: 6.48→18.76) and L1TEX scoreboard stalls (38.7% of stall cycles), causing severe regression. Do not apply when the dominant stall before the change is fixed-latency execution dependency (CPIStall fixed latency ~31%) — this indicates the pipeline is already well-utilized and reducing num_stages removes prefetch depth, exposing more memory latency.

---

## Cumulative Progress

```
Baseline:  654370 ns  (1.000×)
Iter 1     641980 ns  (1.019×)  — reduce_register_pressure_via_smaller_tiles_and_stages
Iter 3     584130 ns  (1.120×)  — eliminate_masked_loads_via_k_loop_peeling_to_enable_fma_fusion

Final best: 584130 ns  (1.1202×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,14 +4,21 @@

 
 
 # Fused GEMM + bias + ReLU kernel
+
 @triton.autotune(
     configs=[
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -46,15 +53,26 @@

 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
-    for k in range(0, tl.cdiv(K, BLOCK_K)):
-        k_remaining = K - k * BLOCK_K
-        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
-        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
-        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
-        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
+    # Hoist M and N masks outside the loop
+    m_mask = offs_m < M
+    n_mask = offs_n < N
+    num_k_blocks = tl.cdiv(K, BLOCK_K)
+
+    # Interior iterations: no K-boundary mask, enables FMA fusion
+    for k in range(0, num_k_blocks - 1):
+        a = tl.load(A_ptrs, mask=m_mask[:, None], other=0.0)
+        b = tl.load(B_ptrs, mask=n_mask[None, :], other=0.0)
         acc = tl.dot(a, b, acc, out_dtype=tl.float32)
         A_ptrs += BLOCK_K * stride_ak
         B_ptrs += BLOCK_K * stride_bk
+
+    # Final (peeled) iteration: apply K-boundary mask only here
+    k_remaining = K - (num_k_blocks - 1) * BLOCK_K
+    a_mask = m_mask[:, None] & (offs_k[None, :] < k_remaining)
+    b_mask = (offs_k[:, None] < k_remaining) & n_mask[None, :]
+    a = tl.load(A_ptrs, mask=a_mask, other=0.0)
+    b = tl.load(B_ptrs, mask=b_mask, other=0.0)
+    acc = tl.dot(a, b, acc, out_dtype=tl.float32)
 
     if APPLY_BIAS:
         bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
@@ -66,7 +84,6 @@

     out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
     C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
     tl.store(C_ptrs, acc.to(tl.float32), mask=out_mask)
-
 
 def run_gemm_bias_relu(A, B, bias, apply_relu=True):
     """
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

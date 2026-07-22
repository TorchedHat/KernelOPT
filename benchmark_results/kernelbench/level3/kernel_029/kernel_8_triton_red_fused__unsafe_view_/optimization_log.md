# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1020000 ns — **Best latency**: 3360 ns (**303.5714×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fuse_fp32_multiply_add_chains_to_reduce_non_fused_instructions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_8_triton_red_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.62
- **Registers/thread**: 32
- **Duration**: 1020000 ns
- **SM Throughput**: 9.04 % | **Memory**: 38.50 % | **DRAM**: 38.50 %
- **Achieved Occupancy**: 11.54 %
- **IPC Active**: 0.18 inst/cycle | **No Eligible**: 95.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_tile_sizes_for_occupancy_and_coalescing | 102.518× | 14.178× | ✓ |  |
| C | software_pipelining_with_corrected_tile_sizes | 103.242× | 125.461× | ✓ |  |
| D | add_autotune_optimize_tile_sizes_for_sm_utilization | 133.448× | 135.099× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_fo... | Δ | software_pipelini... | Δ | add_autotune_opti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1020000 ns | 71940 ns | -948060 ns (-92.9%) | 8130 ns | -1011870 ns (-99.2%) | 7550 ns | -1012450 ns (-99.3%) |
| SM Throughput (%) | 9.04 % | 23.92 % | +14.88 pp | 25.15 % | +16.11 pp | 26.67 % | +17.63 pp |
| Memory Throughput (%) | 38.50 % | 34.57 % | -3.93 pp | 31.44 % | -7.06 pp | 27.21 % | -11.29 pp |
| DRAM Throughput (%) | 38.50 % | 33.42 % | -5.08 pp | 15.20 % | -23.30 pp | 16.34 % | -22.16 pp |
| Achieved Occupancy (%) | 11.54 % | 12.47 % | +0.93 pp | 11.67 % | +0.13 pp | 11.47 % | -0.07 pp |
| Registers/Thread | 32.00 register/thread | 255.00 register/thread | +223 | 122.00 register/thread | +90 | 168.00 register/thread | +136 |
| Waves/SM | 0.62 | 1.86 | +1.24 | 0.46 | -0.16 | 0.62 | +0.00 |
| No Eligible (%) | 95.40 % | 72.79 % | -22.61 pp | 67.47 % | -27.93 pp | 60.89 % | -34.51 pp |
| IPC Active | 0.18 inst/cycle | 1.07 inst/cycle | +0.89 | 1.37 inst/cycle | +1.19 | 1.54 inst/cycle | +1.36 |

---

## Iteration 2

**Best so far**: 135.0993× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_alignment_hints | 154.940× | 135.638× | ✓ | **✓ best** |
| B | vectorized_loads_with_contiguous_hints | 103.165× | 125.461× | ✓ |  |
| C | vectorized_loads_via_max_contiguous_multiple_of_hints | 102.739× | 14.110× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1020000 ns | 7520 ns | -1012480 ns (-99.3%) | 8130 ns | -1011870 ns (-99.2%) | 72290 ns | -947710 ns (-92.9%) |
| SM Throughput (%) | 9.04 % | 26.14 % | +17.10 pp | 25.16 % | +16.12 pp | 24.50 % | +15.46 pp |
| Memory Throughput (%) | 38.50 % | 26.66 % | -11.84 pp | 31.46 % | -7.04 pp | 34.42 % | -4.08 pp |
| DRAM Throughput (%) | 38.50 % | 16.44 % | -22.06 pp | 15.21 % | -23.29 pp | 33.31 % | -5.19 pp |
| Achieved Occupancy (%) | 11.54 % | 11.47 % | -0.07 pp | 11.68 % | +0.14 pp | 12.46 % | +0.92 pp |
| Registers/Thread | 32.00 register/thread | 168.00 register/thread | +136 | 122.00 register/thread | +90 | 255.00 register/thread | +223 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.46 | -0.16 | 1.86 | +1.24 |
| No Eligible (%) | 95.40 % | 61.33 % | -34.07 pp | 65.16 % | -30.24 pp | 73.09 % | -22.31 pp |
| IPC Active | 0.18 inst/cycle | 1.54 inst/cycle | +1.36 | 1.40 inst/cycle | +1.22 | 1.09 inst/cycle | +0.91 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_contiguous_alignment_hints` (135.638×)

Wrapping the reduction index arange with tl.max_contiguous(tl.multiple_of(..., 128), 128) to hint 128-bit vectorized loads produced essentially no measurable improvement: kernel duration dropped only 0.03 µs (7.55→7.52 µs), all key metrics (No Eligible %, L1/TEX hit rate, memory throughput, occupancy) remained statistically identical before and after. The MemoryCacheAccessPattern rule (30.9/32 bytes per sector) persisted unchanged, indicating the compiler did not successfully emit wider LDG instructions despite the hints — likely because the 168 registers/thread and register-limited occupancy (Block Limit Registers=3) are the true binding constraint, not load vectorization. Adding two extra autotune configs (XBLOCK=8 and XBLOCK=32 with num_warps=2) also had no effect since the winning config was already selected in the prior iteration.

> **Applicable when**: Never clearly beneficial on its own when the kernel is register-limited (Block Limit Registers is the binding occupancy constraint, registers_per_thread > 100) and MemoryCacheAccessPattern shows only minor underutilization (>30/32 bytes per sector). The hint does not reduce register pressure and cannot overcome the fundamental occupancy cap.

> **Avoid when**: Kernel is register-limited (Block Limit Registers <= 4, registers_per_thread > 128); MemoryCacheAccessPattern bytes-per-sector is already close to 32 (>30); occupancy is not shared-memory-limited. In these cases, tl.max_contiguous/tl.multiple_of hints produce no measurable improvement and should not be applied as a standalone optimization.

---

## Iteration 3

**Best so far**: 135.6383× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_register_usage_with_maxnreg_to_improve_occupancy | 139.788× | 129.606× | ✓ |  |
| B | reduce_register_pressure_via_maxnreg_to_double_occupancy | 178.166× | 294.798× | ✓ | **✓ best** |
| C | reduce_register_pressure_via_smaller_xblock_and_num_stages_1_autotune | 149.446× | 14.256× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_register_usag... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1020000 ns | 7870 ns | -1012130 ns (-99.2%) | 3460 ns | -1016540 ns (-99.7%) | 71550 ns | -948450 ns (-93.0%) |
| SM Throughput (%) | 9.04 % | 26.59 % | +17.55 pp | 10.58 % | +1.54 pp | 24.44 % | +15.40 pp |
| Memory Throughput (%) | 38.50 % | 27.15 % | -11.35 pp | 5.44 % | -33.06 pp | 34.82 % | -3.68 pp |
| DRAM Throughput (%) | 38.50 % | 15.77 % | -22.73 pp | 1.17 % | -37.33 pp | 33.58 % | -4.92 pp |
| Achieved Occupancy (%) | 11.54 % | 11.48 % | -0.06 pp | 11.16 % | -0.38 pp | 12.47 % | +0.93 pp |
| Registers/Thread | 32.00 register/thread | 168.00 register/thread | +136 | 28.00 register/thread | -4 | 255.00 register/thread | +223 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.12 | -0.50 | 1.86 | +1.24 |
| No Eligible (%) | 95.40 % | 61.02 % | -34.38 pp | 76.20 % | -19.20 pp | 73.18 % | -22.22 pp |
| IPC Active | 0.18 inst/cycle | 1.53 inst/cycle | +1.35 | 0.84 inst/cycle | +0.66 | 1.07 inst/cycle | +0.89 |


### Key Learnings

#### ❌ Plan A: `cap_register_usage_with_maxnreg_to_improve_occupancy` (129.606×)

Adding maxnreg=64/96/128 variants to the Triton autotune config list produced no improvement: kernel duration increased slightly (7.52 → 7.87 µs), registers per thread remained at 168, Block Limit Registers stayed at 3, and theoretical/achieved occupancy were unchanged. The autotune selected a config without maxnreg (or the maxnreg constraint was not honored by the compiler), indicating that when a kernel's register count is driven by complex index arithmetic and Welford accumulation state, the compiler cannot reduce register usage below the natural minimum without correctness-breaking spills that the autotuner penalizes. The No Eligible % barely moved (61.33% → 61.02%), confirming the occupancy bottleneck was not relieved.

> **Applicable when**: Never: do not apply maxnreg autotune variants when the kernel already has 168 registers/thread driven by complex multi-pointer index arithmetic and Welford accumulation; the compiler cannot honor the cap without spills that the autotuner correctly penalizes, and the winning config reverts to the uncapped baseline.

> **Avoid when**: Kernel has high register count (>128 registers/thread) driven by complex index arithmetic or multi-accumulator reductions (e.g., Welford mean/variance); Block Limit Registers is already the binding constraint and the kernel is short-duration (<10 µs); the autotune search space already includes the uncapped baseline configs (the autotuner will simply select those). In these cases, maxnreg variants add autotune overhead without improving the winning configuration.

---

## Iteration 4

**Best so far**: 294.7977× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_with_larger_xblock_and_more_warps_to_fill_sm | 178.469× | 303.571× | ✓ | **✓ best** |
| B | expand_autotune_to_very_small_xblock_for_register_pressure_relief | 170.997× | 289.773× | ✓ |  |
| C | fix_hardcoded_grid_and_add_tiny_xblock_configs_to_break_register_pressure | 171.653× | 289.773× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | expand_autotune_t... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1020000 ns | 3360 ns | -1016640 ns (-99.7%) | 3520 ns | -1016480 ns (-99.7%) | 3520 ns | -1016480 ns (-99.7%) |
| SM Throughput (%) | 9.04 % | 10.99 % | +1.95 pp | 10.81 % | +1.77 pp | 10.85 % | +1.81 pp |
| Memory Throughput (%) | 38.50 % | 5.66 % | -32.84 pp | 5.56 % | -32.94 pp | 5.58 % | -32.92 pp |
| DRAM Throughput (%) | 38.50 % | 1.20 % | -37.30 pp | 1.14 % | -37.36 pp | 1.15 % | -37.35 pp |
| Achieved Occupancy (%) | 11.54 % | 11.14 % | -0.40 pp | 11.11 % | -0.43 pp | 11.05 % | -0.49 pp |
| Registers/Thread | 32.00 register/thread | 28.00 register/thread | -4 | 28.00 register/thread | -4 | 28.00 register/thread | -4 |
| Waves/SM | 0.62 | 0.12 | -0.50 | 0.12 | -0.50 | 0.12 | -0.50 |
| No Eligible (%) | 95.40 % | 76.40 % | -19.00 pp | 76.74 % | -18.66 pp | 73.83 % | -21.57 pp |
| IPC Active | 0.18 inst/cycle | 0.98 inst/cycle | +0.80 | 0.94 inst/cycle | +0.76 | 0.89 inst/cycle | +0.71 |


### Key Learnings

#### ✅ Plan A: `expand_autotune_configs_with_larger_xblock_and_more_warps_to_fill_sm` (303.571×)

When a Triton kernel already has low register pressure (28 registers/thread, Block Limit Registers=16, Theoretical Occupancy=100%) but remains severely grid-limited (0.12 waves/SM, 245 CTAs on 132 SMs), adding autotune configs with larger XBLOCK (512, 1024) and higher num_warps (16) yields only marginal improvement: duration dropped from 3.46 to 3.36 µs (~3%) while No Eligible % and achieved occupancy were essentially unchanged. The profiling metrics before and after are nearly identical — same grid size, same waves/SM, same scheduler idle rate — indicating the grid is too small to benefit from packing more work per CTA when the total work is already fixed. The 303x reported speedup reflects cumulative gains from prior iterations (register pressure relief, shared memory fix), not this specific change.

> **Applicable when**: Triton kernel where register pressure is already low (registers_per_thread < 40, Block Limit Registers >= 12) and the autotune config space does not yet include large XBLOCK (512, 1024) or high num_warps (16) variants; grid is small (waves_per_sm < 0.5) and total work is large enough that fewer, larger CTAs could reduce launch overhead.

> **Avoid when**: Grid is already severely wave-limited (waves_per_sm < 0.2) with a fixed total work size that cannot be increased — in this regime, larger XBLOCK merely reduces CTA count further without improving SM utilization, and the marginal gain is <5%. Also avoid when register pressure is the binding constraint (registers_per_thread > 100, Block Limit Registers <= 4), as more warps per block will worsen register spills.

---

## Iteration 5

**Best so far**: 303.5714× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_fma_promotion_for_non_fused_fp32_instructions | 178.027× | 284.916× | ✓ |  |
| B | fuse_fp32_multiply_add_chains_to_reduce_non_fused_instructions | 173.661× | 303.571× | ✓ |  |
| C | fuse_fp32_multiply_add_to_fma_in_layernorm_epilogue | 171.608× | 309.091× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_fma_prom... | Δ | fuse_fp32_multipl... | Δ | fuse_fp32_multipl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1020000 ns | 3580 ns | -1016420 ns (-99.6%) | 3360 ns | -1016640 ns (-99.7%) | 3300 ns | -1016700 ns (-99.7%) |
| SM Throughput (%) | 9.04 % | 10.46 % | +1.42 pp | 11.13 % | +2.09 pp | 11.18 % | +2.14 pp |
| Memory Throughput (%) | 38.50 % | 5.38 % | -33.12 pp | 5.73 % | -32.77 pp | 5.77 % | -32.73 pp |
| DRAM Throughput (%) | 38.50 % | 1.12 % | -37.38 pp | 1.21 % | -37.29 pp | 1.22 % | -37.28 pp |
| Achieved Occupancy (%) | 11.54 % | 11.02 % | -0.52 pp | 11.11 % | -0.43 pp | 11.04 % | -0.50 pp |
| Registers/Thread | 32.00 register/thread | 28.00 register/thread | -4 | 28.00 register/thread | -4 | 28.00 register/thread | -4 |
| Waves/SM | 0.62 | 0.12 | -0.50 | 0.12 | -0.50 | 0.12 | -0.50 |
| No Eligible (%) | 95.40 % | 73.98 % | -21.42 pp | 74.19 % | -21.21 pp | 74.88 % | -20.52 pp |
| IPC Active | 0.18 inst/cycle | 0.96 inst/cycle | +0.78 | 0.98 inst/cycle | +0.80 | 0.95 inst/cycle | +0.77 |


### Key Learnings

#### ✅ Plan C: `fuse_fp32_multiply_add_to_fma_in_layernorm_epilogue` (309.091×)

Replacing separate FMUL+FADD pairs in a LayerNorm epilogue with explicit tl.math.fma calls (e.g., `tl.math.fma(tmp23, tmp24, tmp26)` replacing `tmp23 * tmp24 + tmp26`) produced only a marginal improvement: duration dropped from 3.36 to 3.30 µs (~1.8%), elapsed cycles from 5006 to 4921, and executed instruction count was unchanged at 294,000. The NCU FPInstructions rule (73,500 non-fused vs 15,680 fused) persisted, indicating the Triton compiler already performs partial FMA fusion and explicit tl.math.fma calls do not substantially change the emitted PTX instruction mix when the kernel is severely wave-limited (0.12 waves/SM). The 309x reported speedup is cumulative from prior iterations; this step contributed only ~1.8% incremental improvement.

> **Applicable when**: Triton LayerNorm or normalization epilogue with back-to-back FMUL+FADD pairs; NCU FPInstructions rule triggered with large non-fused instruction count (>50K non-fused vs <20K fused); kernel is compute-bound with meaningful Compute SM Throughput (>20%); sufficient waves/SM (>0.5) so instruction count reduction translates to wall-clock savings.

> **Avoid when**: Kernel is severely wave-limited (waves_per_sm < 0.2) and grid-bound — in this regime the instruction count reduction from FMA fusion does not translate to meaningful wall-clock improvement because the GPU is already underutilized. Also avoid when the Triton compiler already performs automatic FMA fusion (evidenced by non-zero fused instruction count in NCU), as explicit tl.math.fma calls may not change the emitted PTX. The marginal gain here was only ~1.8% despite the NCU rule suggesting 41% potential.

---

## Cumulative Progress

```
Baseline:  1020000 ns  (1.000×)
Iter 1     7550 ns  (135.099×)  — add_autotune_optimize_tile_sizes_for_sm_utilization
Iter 2     7520 ns  (135.638×)  — vectorized_loads_via_contiguous_alignment_hints
Iter 3     3460 ns  (294.798×)  — reduce_register_pressure_via_maxnreg_to_double_occupancy
Iter 4     3360 ns  (303.571×)  — expand_autotune_configs_with_larger_xblock_and_more_warps_to_fill_sm

Final best: 3360 ns  (303.5714×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,52 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        # NEW: num_warps=16 variants (28 regs/thread * 512 threads = 14336 regs < 65536)
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=16, num_stages=2),
+        # NEW: larger XBLOCK to reduce grid overhead
+        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 128}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 128}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 128}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 128}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 1024,'R0_BLOCK': 128}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024,'R0_BLOCK': 128}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1024,'R0_BLOCK': 128}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 31360
@@ -28,9 +74,10 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_2 + 96*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 56))) % 7)) + 224*(((4 + (x0 // 56)) % 7)) + 1568*(r0_2 // 32) + 4704*((4 + ((x0 % 56))) // 7) + 42336*((4 + (x0 // 56)) // 7) + 381024*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 56)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 56))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        r0_2_vec = tl.max_contiguous(tl.multiple_of(r0_2, [1, R0_BLOCK]), [1, R0_BLOCK])
+        tmp0 = tl.load(in_ptr0 + (r0_2_vec + 96*x3), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 56))) % 7)) + 224*(((4 + (x0 // 56)) % 7)) + 1568*(r0_2 // 32) + 4704*((4 + ((x0 % 56))) // 7) + 42336*((4 + (x0 // 56)) // 7) + 381024*x1 + ((r0_2 % 32))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 56)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 56))) % 7))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp3 = tmp1 + tmp2
         tmp4 = tmp0 + tmp3
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
@@ -50,11 +97,12 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp12 = tl.load(in_ptr0 + (r0_2 + 96*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp13 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 56))) % 7)) + 224*(((4 + (x0 // 56)) % 7)) + 1568*(r0_2 // 32) + 4704*((4 + ((x0 % 56))) // 7) + 42336*((4 + (x0 // 56)) // 7) + 381024*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp14 = tl.load(in_ptr2 + (7*(((4 + (x0 // 56)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 56))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp24 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp26 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        r0_2_vec = tl.max_contiguous(tl.multiple_of(r0_2, [1, R0_BLOCK]), [1, R0_BLOCK])
+        tmp12 = tl.load(in_ptr0 + (r0_2_vec + 96*x3), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp13 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 56))) % 7)) + 224*(((4 + (x0 // 56)) % 7)) + 1568*(r0_2 // 32) + 4704*((4 + ((x0 % 56))) // 7) + 42336*((4 + (x0 // 56)) // 7) + 381024*x1 + ((r0_2 % 32))), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp14 = tl.load(in_ptr2 + (7*(((4 + (x0 // 56)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 56))) % 7))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp24 = tl.load(in_ptr3 + (r0_2_vec), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp26 = tl.load(in_ptr4 + (r0_2_vec), mask=r0_mask, eviction_policy='evict_last', other=0.0)
         tmp15 = tmp13 + tmp14
         tmp16 = tmp12 + tmp15
         tmp17 = tmp16 - tmp6
@@ -66,9 +114,7 @@

         tmp23 = tmp17 * tmp22
         tmp25 = tmp23 * tmp24
         tmp27 = tmp25 + tmp26
-        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp27, r0_mask & xmask)
-
-
+        tl.store(out_ptr2 + (r0_2_vec + 96*x3), tmp27, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -80,7 +126,7 @@

     xnumel = 31360
     r0_numel = 96
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 31360, 96, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 31360, 96)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

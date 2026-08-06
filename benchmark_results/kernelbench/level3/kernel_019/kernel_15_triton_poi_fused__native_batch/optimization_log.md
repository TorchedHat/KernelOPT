# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7490 ns — **Best latency**: 7460 ns (**1.0040×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_small_xblock_configs_with_adaptive_grid_lambda`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_15_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 17
- **Duration**: 7490 ns
- **SM Throughput**: 38.06 % | **Memory**: 16.65 % | **DRAM**: 10.98 %
- **Achieved Occupancy**: 58.57 %
- **IPC Active**: 2.16 inst/cycle | **No Eligible**: 43.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.845× | 0.767× | ✓ |  |
| B | autotune_xblock_to_hide_l1tex_latency_and_eliminate_partial_wave | 0.837× | 0.770× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.826× | 0.369× | ✓ |  |
| D | autotune_xblock_with_register_tiling | 0.831× | 0.757× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ | autotune_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 9760 ns | +2270 ns (+30.3%) | 9730 ns | +2240 ns (+29.9%) | 20290 ns | +12800 ns (+170.9%) | 9890 ns | +2400 ns (+32.0%) |
| SM Throughput (%) | 38.06 % | 66.51 % | +28.45 pp | 65.22 % | +27.16 pp | 77.29 % | +39.23 pp | 66.40 % | +28.34 pp |
| Memory Throughput (%) | 16.65 % | 17.97 % | +1.32 pp | 17.70 % | +1.05 pp | 21.93 % | +5.28 pp | 17.88 % | +1.23 pp |
| DRAM Throughput (%) | 10.98 % | 8.43 % | -2.55 pp | 8.44 % | -2.54 pp | 4.04 % | -6.94 pp | 8.33 % | -2.65 pp |
| Achieved Occupancy (%) | 58.57 % | 81.84 % | +23.27 pp | 81.55 % | +22.98 pp | 81.90 % | +23.33 pp | 81.93 % | +23.36 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 24.00 register/thread | +7 | 32.00 register/thread | +15 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 14.85 | +11.14 | 3.71 | +0.00 |
| No Eligible (%) | 43.44 % | 16.54 % | -26.90 pp | 16.15 % | -27.29 pp | 13.33 % | -30.11 pp | 16.55 % | -26.89 pp |
| IPC Active | 2.16 inst/cycle | 3.29 inst/cycle | +1.13 | 3.32 inst/cycle | +1.16 | 3.45 inst/cycle | +1.29 | 3.32 inst/cycle | +1.16 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.767×)

Adding @triton.autotune with larger XBLOCK values (512–2048) improved occupancy (58.6%→81.8%) and eligible warps/scheduler (1.37→6.45), but caused a 30% regression in wall time. The autotune overhead doubled register usage (17→32 registers/thread), which reduced Block Limit Registers from 21 to 16 blocks/SM, and the total executed instruction count more than doubled (2.29M→5.06M), indicating the larger tiles introduced significant extra work (likely from the xmask predication overhead on non-power-of-two boundaries and autotune compilation cost amortized into measured time). Despite better scheduler utilization, the net effect was slower execution because the instruction count increase outweighed the latency hiding benefit for this short, memory-light kernel.

> **Applicable when**: Kernel has very low eligible warps/scheduler (<2) and high L1TEX stall fraction (>40%), with low register count (<20/thread) and large grid sizes causing tail-wave penalties.

> **Avoid when**: The kernel is already short-running (<10 µs) and the total instruction count is low; in such cases, autotune compilation overhead and increased register pressure from larger tiles can dominate, causing regression even when occupancy and scheduler metrics improve. Also avoid when the autotune itself doubles register usage, as this reduces block-level parallelism and negates occupancy gains.

#### ❌ Plan C: `increase_xblock_for_vectorized_loads_and_latency_hiding` (0.542×)

Changing autotune configs to larger XBLOCK values (512/1024) with more num_warps (8/16) and adding rsqrt fusion caused a 46% regression (9.73→13.82 µs). Despite registers/thread dropping from 32 to 22, the autotuner selected XBLOCK=512 with block_size=512, which caused waves/SM to jump from 3.71 to 14.85 — indicating the grid was not properly adapting to the selected XBLOCK, leading to 7.3M executed instructions vs 5.0M before. Achieved occupancy also dropped (81.6%→72.8%) because Block Limit Warps fell to 4 blocks/SM with the larger block size. For this kernel, the existing autotune configs already produced a reasonable operating point; adding more aggressive configs with larger num_warps caused the scheduler to select a worse configuration.

> **Applicable when**: Never: this specific combination of replacing autotune configs with higher num_warps variants and adding rsqrt fusion caused regression for this kernel profile.

> **Avoid when**: Do not replace existing autotune configs that already produce reasonable waves/SM (~3-4) with configs that force larger block sizes (512+) and higher num_warps (8-16) when the kernel is already compute-bound (Compute SM Throughput >60%). Doing so causes the autotuner to select a configuration with excessive waves/SM (14+), inflating total instruction count and reducing achieved occupancy. Also avoid when Block Limit Warps would drop to ≤4 blocks/SM with the new block size.

#### ❌ Plan D: `autotune_xblock_with_register_tiling` (0.757×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) for a pointwise kernel improved occupancy (58.6%→81.9%) and scheduler eligibility (1.37→6.48 eligible warps/scheduler), but caused a 32% regression in wall-clock time (7.49→9.89 µs). The root cause is that the autotuner selected a configuration that doubled register usage per thread (17→32 registers/thread), which reduced Block Limit Registers from 21 to 16 blocks/SM, and the 2.2x increase in executed instructions (2.29M→5.06M) from processing more elements per thread outweighed the latency-hiding benefit. For short-duration memory-bound pointwise kernels where the original register count is already low and the kernel is not deeply latency-bound, increasing XBLOCK via autotune adds instruction overhead faster than it hides memory latency.

> **Applicable when**: Kernel has very high L1TEX scoreboard stall fraction (>50% of warp cycles), low ILP (1 element/thread), and register count is well below the register-limit threshold such that doubling registers would not reduce Block Limit Registers.

> **Avoid when**: Kernel duration is short (<10 µs), register count per thread is already moderate (≥17) and near the boundary where doubling would reduce Block Limit Registers, or the kernel is not deeply memory-latency-bound (L1TEX stall <30% of cycles). Also avoid when the total instruction count increase from larger tiles will dominate over latency-hiding gains.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_div_sqrt_with_rsqrt_for_fused_fp32_instructions | 0.974× | 0.968× | ✓ |  |
| B | eliminate_partial_wave_and_fuse_fp32_instructions | 1.011× | 1.026× | ✓ | **✓ best** |
| C | replace_reciprocal_sqrt_with_rsqrt_to_enable_fma_fusion | 0.992× | 0.959× | ✓ |  |
| D | fix_grid_lambda_and_fuse_bn_rsqrt_for_fma | 0.961× | 0.542× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_div_sqrt_... | Δ | eliminate_partial... | Δ | replace_reciproca... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7740 ns | +250 ns (+3.3%) | 7300 ns | -190 ns (-2.5%) | 7810 ns | +320 ns (+4.3%) | 13820 ns | +6330 ns (+84.5%) |
| SM Throughput (%) | 38.06 % | 38.62 % | +0.56 pp | 28.85 % | -9.21 pp | 38.03 % | -0.03 pp | 66.35 % | +28.29 pp |
| Memory Throughput (%) | 16.65 % | 24.47 % | +7.82 pp | 17.64 % | +0.99 pp | 24.10 % | +7.45 pp | 32.94 % | +16.29 pp |
| DRAM Throughput (%) | 10.98 % | 10.59 % | -0.39 pp | 11.25 % | +0.27 pp | 10.52 % | -0.46 pp | 5.92 % | -5.06 pp |
| Achieved Occupancy (%) | 58.57 % | 44.92 % | -13.65 pp | 48.31 % | -10.26 pp | 45.08 % | -13.49 pp | 72.84 % | +14.27 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 22.00 register/thread | +5 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 14.85 | +11.14 |
| No Eligible (%) | 43.44 % | 39.22 % | -4.22 pp | 56.46 % | +13.02 pp | 40.76 % | -2.68 pp | 22.20 % | -21.24 pp |
| IPC Active | 2.16 inst/cycle | 2.40 inst/cycle | +0.24 | 1.71 inst/cycle | -0.45 | 2.40 inst/cycle | +0.24 | 3.09 inst/cycle | +0.93 |

---

## Iteration 3

**Best so far**: 1.026× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | pipeline_loads_with_num_stages_and_streaming_eviction | 1.011× | 1.000× | ✓ |  |
| B | fix_grid_lambda_and_add_small_xblock_configs_for_partial_wave_elimination | 1.026× | 1.026× | ✓ |  |
| C | fix_grid_lambda_to_enable_correct_autotune_exploration | 1.012× | 1.022× | ✓ |  |
| D | fix_grid_lambda_and_fuse_bn_ops_with_fma | 1.034× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | pipeline_loads_wi... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7490 ns | +0 ns (+0.0%) | 7300 ns | -190 ns (-2.5%) | 7330 ns | -160 ns (-2.1%) | 7490 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 38.06 % | 31.54 % | -6.52 pp | 28.40 % | -9.66 pp | 28.08 % | -9.98 pp | 28.02 % | -10.04 pp |
| Memory Throughput (%) | 16.65 % | 17.65 % | +1.00 pp | 17.69 % | +1.04 pp | 17.11 % | +0.46 pp | 17.41 % | +0.76 pp |
| DRAM Throughput (%) | 10.98 % | 10.96 % | -0.02 pp | 11.28 % | +0.30 pp | 11.24 % | +0.26 pp | 10.95 % | -0.03 pp |
| Achieved Occupancy (%) | 58.57 % | 48.35 % | -10.22 pp | 47.56 % | -11.01 pp | 47.91 % | -10.66 pp | 47.11 % | -11.46 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 43.44 % | 52.67 % | +9.23 pp | 56.93 % | +13.49 pp | 56.33 % | +12.89 pp | 57.77 % | +14.33 pp |
| IPC Active | 2.16 inst/cycle | 1.86 inst/cycle | -0.30 | 1.69 inst/cycle | -0.47 | 1.72 inst/cycle | -0.44 | 1.70 inst/cycle | -0.46 |

---

## Iteration 4

**Best so far**: 1.026× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_bn_multiply_add_chains_into_explicit_fma_instructions | 1.015× | 1.036× | ✓ |  |
| B | fix_broken_grid_lambda_and_add_small_warp_configs_for_occupancy | 1.015× | 1.014× | ✓ |  |
| C | reduce_num_warps_to_increase_blocks_per_sm_and_fill_occupancy_gap | 1.049× | 1.026× | ✓ |  |
| D | add_small_xblock_configs_with_adaptive_grid_lambda | 1.057× | 1.004× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fuse_bn_multiply_... | Δ | fix_broken_grid_l... | Δ | reduce_num_warps_... | Δ | add_small_xblock_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7230 ns | -260 ns (-3.5%) | 7390 ns | -100 ns (-1.3%) | 7300 ns | -190 ns (-2.5%) | 7460 ns | -30 ns (-0.4%) |
| SM Throughput (%) | 38.06 % | 28.42 % | -9.64 pp | 10.50 % | -27.56 pp | 15.91 % | -22.15 pp | 14.26 % | -23.80 pp |
| Memory Throughput (%) | 16.65 % | 17.67 % | +1.02 pp | 15.58 % | -1.07 pp | 8.63 % | -8.02 pp | 8.60 % | -8.05 pp |
| DRAM Throughput (%) | 10.98 % | 11.32 % | +0.34 pp | 11.14 % | +0.16 pp | 5.63 % | -5.35 pp | 5.51 % | -5.47 pp |
| Achieved Occupancy (%) | 58.57 % | 48.98 % | -9.59 pp | 16.81 % | -41.76 pp | 39.44 % | -19.13 pp | 31.89 % | -26.68 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 1.86 | -1.85 | 1.86 | -1.85 | 1.86 | -1.85 |
| No Eligible (%) | 43.44 % | 57.36 % | +13.92 pp | 77.47 % | +34.03 pp | 69.40 % | +25.96 pp | 71.80 % | +28.36 pp |
| IPC Active | 2.16 inst/cycle | 1.63 inst/cycle | -0.53 | 0.72 inst/cycle | -1.44 | 1.14 inst/cycle | -1.02 | 1.00 inst/cycle | -1.16 |

---

## Iteration 5

**Best so far**: 1.026× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_autotune_small_xblock_with_fewer_warps | 1.056× | 1.045× | ✓ |  |
| B | fix_grid_lambda_and_add_medium_xblock_config_for_occupancy | 1.049× | 1.036× | ✓ |  |
| C | fix_num_warps_to_match_xblock_for_intra_block_latency_hiding | 1.050× | 1.000× | ✓ |  |
| D | fix_grid_lambda_and_target_num_warps4_xblock128_for_occupancy | 1.056× | 1.032× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | fix_num_warps_to_... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7170 ns | -320 ns (-4.3%) | 7230 ns | -260 ns (-3.5%) | 7490 ns | +0 ns (+0.0%) | 7260 ns | -230 ns (-3.1%) |
| SM Throughput (%) | 38.06 % | 7.24 % | -30.82 pp | 15.68 % | -22.38 pp | 28.79 % | -9.27 pp | 28.41 % | -9.65 pp |
| Memory Throughput (%) | 16.65 % | 4.42 % | -12.23 pp | 8.39 % | -8.26 pp | 17.63 % | +0.98 pp | 17.65 % | +1.00 pp |
| DRAM Throughput (%) | 10.98 % | 2.89 % | -8.09 pp | 5.68 % | -5.30 pp | 10.97 % | -0.01 pp | 11.31 % | +0.33 pp |
| Achieved Occupancy (%) | 58.57 % | 15.98 % | -42.59 pp | 38.71 % | -19.86 pp | 47.57 % | -11.00 pp | 48.77 % | -9.80 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 1.86 | -1.85 | 1.86 | -1.85 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 43.44 % | 83.26 % | +39.82 pp | 69.51 % | +26.07 pp | 56.58 % | +13.14 pp | 57.19 % | +13.75 pp |
| IPC Active | 2.16 inst/cycle | 0.52 inst/cycle | -1.64 | 1.12 inst/cycle | -1.04 | 1.71 inst/cycle | -0.45 | 1.62 inst/cycle | -0.54 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_and_autotune_small_xblock_with_fewer_warps` (1.045×)

When a Triton pointwise kernel has a hardcoded grid size (e.g., cdiv(xnumel, 128)) that ignores the autotuned XBLOCK, the autotune configs for larger/smaller XBLOCK are dead code and the kernel always runs at the hardcoded tile size. Fixing the grid to a lambda (lambda meta: (cdiv(xnumel, meta['XBLOCK']),)) and adding smaller XBLOCK configs (32/64) with proportionally fewer warps (1-2) keeps register pressure constant (18/thread) while reducing waves/SM from 3.71 to 1.86 and total executed instructions from 1.63M to 0.41M. The speedup is modest (4.5%) because the smaller block size also reduces active warps/scheduler (7.87→2.80) and increases the no-eligible fraction (57%→83%), meaning the latency-hiding benefit of more concurrent blocks is partially offset by fewer warps per scheduler to hide stalls.

> **Applicable when**: Triton pointwise kernel where the grid is hardcoded to a fixed tile size (e.g., cdiv(xnumel, 128)) while autotune configs vary XBLOCK — making all non-matching configs dead code. Kernel has moderate L1TEX stall fraction (CPIStall rule), low achieved occupancy relative to theoretical, and registers/thread are already low (≤18) so smaller tiles won't increase register pressure.

> **Avoid when**: Kernel already has a correct lambda grid. Do not use very small XBLOCK (32/64) with few warps (1-2) as the primary strategy when the kernel is deeply latency-bound and needs many active warps per scheduler to hide stalls — the reduced active warps/scheduler (2.80 vs 7.87) can worsen the no-eligible fraction (83% vs 57%), limiting the benefit. Also avoid when the kernel duration is already very short (<5 µs) and scheduling overhead dominates.

---

## Cumulative Progress

```
Baseline:  7490 ns  (1.000×)
Iter 2     7300 ns  (1.026×)  — eliminate_partial_wave_and_fuse_fp32_instructions

Final best: 7460 ns  (1.0040×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,38 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1003520
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
+    tmp10 = tl.math.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    tmp15 = tl.math.fma(tmp11, tmp12, tmp14)
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +46,7 @@

     arg_4 = inputs[4]
     xnumel = 1003520
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1003520, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1003520)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

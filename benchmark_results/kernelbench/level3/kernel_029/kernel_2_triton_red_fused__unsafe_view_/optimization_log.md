# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 787900 ns — **Best latency**: 3840 ns (**205.1823×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`hoist_loop_invariant_index_arithmetic_to_reduce_l1tex_stalls`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_2_triton_red_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.62
- **Registers/thread**: 32
- **Duration**: 787900 ns
- **SM Throughput**: 10.27 % | **Memory**: 40.06 % | **DRAM**: 40.06 %
- **Achieved Occupancy**: 11.51 %
- **IPC Active**: 0.21 inst/cycle | **No Eligible**: 94.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_shared_memory_via_r0_block_and_autotune_tile_sizes | 77.787× | 97.754× | ✓ | **✓ best** |
| C | software_pipelining_with_right_sized_r0_block | 47.156× | 57.012× | ✓ |  |
| D | increase_xblock_reduce_r0_block_for_occupancy_and_grid_coverage | 49.080× | 33.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ | software_pipelini... | Δ | increase_xblock_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 787900 ns | 8060 ns | -779840 ns (-99.0%) | 13820 ns | -774080 ns (-98.2%) | 23870 ns | -764030 ns (-97.0%) |
| SM Throughput (%) | 10.27 % | 26.73 % | +16.46 pp | 27.74 % | +17.47 pp | 31.32 % | +21.05 pp |
| Memory Throughput (%) | 40.06 % | 31.24 % | -8.82 pp | 42.13 % | +2.07 pp | 39.04 % | -1.02 pp |
| DRAM Throughput (%) | 40.06 % | 15.35 % | -24.71 pp | 17.88 % | -22.18 pp | 21.89 % | -18.17 pp |
| Achieved Occupancy (%) | 11.51 % | 11.46 % | -0.05 pp | 11.52 % | +0.01 pp | 11.75 % | +0.24 pp |
| Registers/Thread | 32.00 register/thread | 164.00 register/thread | +132 | 213.00 register/thread | +181 | 255.00 register/thread | +223 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.93 | +0.31 | 0.93 | +0.31 |
| No Eligible (%) | 94.67 % | 61.44 % | -33.23 pp | 64.07 % | -30.60 pp | 62.39 % | -32.28 pp |
| IPC Active | 0.21 inst/cycle | 1.54 inst/cycle | +1.33 | 1.44 inst/cycle | +1.23 | 1.51 inst/cycle | +1.30 |

---

## Iteration 2

**Best so far**: 97.7543× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of_hints | 75.782× | 95.042× | ✓ |  |
| B | vectorized_loads_with_maxnreg_to_reduce_register_pressure | 53.035× | 62.982× | ✓ |  |
| C | vectorized_loads_via_max_contiguous_multiple_of_hints | 121.879× | 32.693× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 787900 ns | 8290 ns | -779610 ns (-98.9%) | 12510 ns | -775390 ns (-98.4%) | 24100 ns | -763800 ns (-96.9%) |
| SM Throughput (%) | 10.27 % | 26.67 % | +16.40 pp | 16.43 % | +6.16 pp | 30.80 % | +20.53 pp |
| Memory Throughput (%) | 40.06 % | 31.14 % | -8.92 pp | 29.75 % | -10.31 pp | 38.32 % | -1.74 pp |
| DRAM Throughput (%) | 40.06 % | 14.96 % | -25.10 pp | 9.94 % | -30.12 pp | 21.84 % | -18.22 pp |
| Achieved Occupancy (%) | 11.51 % | 11.52 % | +0.01 pp | 5.84 % | -5.67 pp | 11.75 % | +0.24 pp |
| Registers/Thread | 32.00 register/thread | 164.00 register/thread | +132 | 128.00 register/thread | +96 | 255.00 register/thread | +223 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.23 | -0.39 | 0.93 | +0.31 |
| No Eligible (%) | 94.67 % | 61.49 % | -33.18 pp | 77.44 % | -17.23 pp | 62.21 % | -32.46 pp |
| IPC Active | 0.21 inst/cycle | 1.46 inst/cycle | +1.25 | 0.84 inst/cycle | +0.63 | 1.49 inst/cycle | +1.28 |

---

## Iteration 3

**Best so far**: 97.7543× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipelining_num_stages_2_autotune_expansion | 75.363× | 98.120× | ✓ |  |
| B | lower_maxnreg_to_64_with_num_warps_1_to_eliminate_spill_and_double_occupancy | 53.071× | 58.623× | ✓ |  |
| C | revert_vectorization_hints_and_prune_large_xblock_configs_to_eliminate_register_spill | 120.705× | 212.372× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | lower_maxnreg_to_... | Δ | revert_vectorizat... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 787900 ns | 8030 ns | -779870 ns (-99.0%) | 13440 ns | -774460 ns (-98.3%) | 3710 ns | -784190 ns (-99.5%) |
| SM Throughput (%) | 10.27 % | 26.99 % | +16.72 pp | 10.35 % | +0.08 pp | 10.77 % | +0.50 pp |
| Memory Throughput (%) | 40.06 % | 31.55 % | -8.51 pp | 30.45 % | -9.61 pp | 11.82 % | -28.24 pp |
| DRAM Throughput (%) | 40.06 % | 15.38 % | -24.68 pp | 4.92 % | -35.14 pp | 1.10 % | -38.96 pp |
| Achieved Occupancy (%) | 11.51 % | 11.47 % | -0.04 pp | 2.89 % | -8.62 pp | 11.00 % | -0.51 pp |
| Registers/Thread | 32.00 register/thread | 164.00 register/thread | +132 | 64.00 register/thread | +32 | 26.00 register/thread | -6 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.06 | -0.56 | 0.12 | -0.50 |
| No Eligible (%) | 94.67 % | 60.34 % | -34.33 pp | 77.55 % | -17.12 pp | 78.50 % | -16.17 pp |
| IPC Active | 0.21 inst/cycle | 1.50 inst/cycle | +1.29 | 0.42 inst/cycle | +0.21 | 0.82 inst/cycle | +0.61 |


### Key Learnings

#### ✅ Plan A: `software_pipelining_num_stages_2_autotune_expansion` (98.120×)

When a Triton kernel shows high no-eligible-warp percentage (~61%) due to load latency stalls (IssueSlotUtilization rule), adding num_stages=2 variants to the autotune config list allows Triton's software pipeliner to issue async prefetch loads one iteration ahead, overlapping memory latency with compute. In this case, the improvement was modest (8.29→8.03 µs, ~3% duration reduction, No Eligible Warp from 61.49%→60.34%) because the kernel's two sequential tl.range loops limit pipelining depth and the kernel is already register-limited (164 regs/thread, Block Limit Registers=3), leaving little headroom for the pipeline buffer. The strategy is safe when num_stages=1 configs are preserved as fallback, ensuring the autotuner can always revert to the prior best.

> **Applicable when**: Triton reduction kernel with IssueSlotUtilization rule triggered; No Eligible Warp % > 50%; kernel has sequential tl.range loops with multiple loads per iteration (5+ loads); current autotune configs all use num_stages=1; register count is stable (no increase expected from pipelining); num_stages=1 configs are preserved as fallback.

> **Avoid when**: Kernel is already register-limited at the Block Limit Registers boundary (e.g., 164 regs/thread with Block Limit Registers=3) and num_stages=2 would push register count higher, reducing blocks/SM further; the kernel has only a single short loop where pipelining depth cannot amortize the pipeline setup overhead; No Eligible Warp % is below 30% (latency hiding is not the bottleneck); the kernel is DRAM-bandwidth saturated (DRAM Throughput > 70%).

---

## Iteration 4

**Best so far**: 97.7543× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | num_warps_2_with_num_stages_2_to_improve_wave_coverage_and_hide_l1tex_latency | 126.552× | 206.798× | ✓ |  |
| B | expand_autotune_to_small_xblock_configs_for_register_pressure_relief | 127.062× | 205.182× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | num_warps_2_with_... | Δ | expand_autotune_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 787900 ns | 3810 ns | -784090 ns (-99.5%) | 3840 ns | -784060 ns (-99.5%) |
| SM Throughput (%) | 10.27 % | 9.99 % | -0.28 pp | 10.15 % | -0.12 pp |
| Memory Throughput (%) | 40.06 % | 10.96 % | -29.10 pp | 11.14 % | -28.92 pp |
| DRAM Throughput (%) | 40.06 % | 1.08 % | -38.98 pp | 1.09 % | -38.97 pp |
| Achieved Occupancy (%) | 11.51 % | 11.09 % | -0.42 pp | 10.26 % | -1.25 pp |
| Registers/Thread | 32.00 register/thread | 26.00 register/thread | -6 | 26.00 register/thread | -6 |
| Waves/SM | 0.62 | 0.12 | -0.50 | 0.12 | -0.50 |
| No Eligible (%) | 94.67 % | 80.52 % | -14.15 pp | 78.21 % | -16.46 pp |
| IPC Active | 0.21 inst/cycle | 0.83 inst/cycle | +0.62 | 0.74 inst/cycle | +0.53 |

---

## Iteration 5

**Best so far**: 205.1823× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_meta_xblock_for_wave_coverage | 126.669× | 210.668× | ✓ |  |
| C | hoist_loop_invariant_index_arithmetic_to_reduce_l1tex_stalls | 127.245× | 205.182× | ✓ | **✓ best** |
| D | evict_first_on_strided_welford_pass_to_reduce_l1tex_cache_pollution | 121.749× | 214.103× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | hoist_loop_invari... | Δ | evict_first_on_st... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 787900 ns | 3740 ns | -784160 ns (-99.5%) | 3840 ns | -784060 ns (-99.5%) | 3680 ns | -784220 ns (-99.5%) |
| SM Throughput (%) | 10.27 % | 2.92 % | -7.35 pp | 10.32 % | +0.05 pp | 9.76 % | -0.51 pp |
| Memory Throughput (%) | 40.06 % | 8.18 % | -31.88 pp | 11.31 % | -28.75 pp | 7.89 % | -32.17 pp |
| DRAM Throughput (%) | 40.06 % | 1.11 % | -38.95 pp | 1.07 % | -38.99 pp | 1.12 % | -38.94 pp |
| Achieved Occupancy (%) | 11.51 % | 2.88 % | -8.63 pp | 10.99 % | -0.52 pp | 11.07 % | -0.44 pp |
| Registers/Thread | 32.00 register/thread | 40.00 register/thread | +8 | 26.00 register/thread | -6 | 23.00 register/thread | -9 |
| Waves/SM | 0.62 | 0.06 | -0.56 | 0.12 | -0.50 | 0.12 | -0.50 |
| No Eligible (%) | 94.67 % | 86.25 % | -8.42 pp | 79.33 % | -15.34 pp | 78.33 % | -16.34 pp |
| IPC Active | 0.21 inst/cycle | 0.24 inst/cycle | +0.03 | 0.80 inst/cycle | +0.59 | 0.82 inst/cycle | +0.61 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_to_use_meta_xblock_for_wave_coverage` (210.668×)

When a Triton kernel's launch grid is hardcoded as (cdiv(xnumel, CONSTANT),) rather than a lambda using meta['XBLOCK'], the autotuner's XBLOCK parameter has no effect on grid size — all configs produce the same number of blocks regardless of XBLOCK value. Changing the grid to lambda meta: (cdiv(xnumel, meta['XBLOCK']),) allows smaller XBLOCK configs (e.g., XBLOCK=1) to expand the grid proportionally, increasing wave coverage. In this case, the autotuner selected XBLOCK=1 with num_warps=1 (32 threads/block), reducing instruction count from 301K to 85K and warp cycles per instruction from 9.36 to 7.22, yielding a marginal duration improvement (3.81→3.74 µs). However, the grid remained at 245 blocks (waves/SM dropped from 0.12 to 0.06) because the autotuner selected a config that did not expand the grid — indicating the grid lambda fix alone is necessary but not sufficient when xnumel is small relative to sm_count.

> **Applicable when**: Triton kernel where the launch grid is hardcoded with a constant divisor (e.g., cdiv(xnumel, 128)) instead of using meta['XBLOCK']; SOLBottleneck and AchievedOccupancy rules are triggered; waves_per_sm < 0.5; the autotune config list includes XBLOCK values smaller than the hardcoded constant; xnumel >> sm_count so that smaller XBLOCK values would meaningfully increase grid size.

> **Avoid when**: The grid is already a lambda using meta['XBLOCK']; xnumel is small enough that even XBLOCK=1 produces fewer than 2 waves/SM (xnumel < 2 * sm_count); the kernel's bottleneck is compute or DRAM bandwidth rather than wave coverage; the autotuner consistently selects large XBLOCK values regardless of grid lambda (indicating per-block efficiency dominates).

#### ❌ Plan C: `hoist_loop_invariant_index_arithmetic_to_reduce_l1tex_stalls` (205.182×)

Manually hoisting x-dimension-only sub-expressions out of both reduction loops (precomputing base0, base1, base2 before the loops) produced no measurable improvement: duration increased slightly from 3.81 µs to 3.84 µs, instruction count remained identical (301,840), and all key metrics (L1 hit rate, scheduler eligibility, uncoalesced access rate, register count) were essentially unchanged. The Triton/LLVM compiler backend already performs this loop-invariant code motion automatically, so manual hoisting is redundant. Additionally, the diff introduced a tl.max_contiguous/tl.multiple_of hint on the store index (store_idx), which is a confirmed regression pattern from prior experiments (exp_002). The uncoalesced global access (66% excessive sectors) and L1TEX stalls are caused by the inherently strided memory layout of in_ptr1 and in_ptr2, which cannot be fixed by index arithmetic reordering alone.

> **Applicable when**: Never: this strategy produced a slight regression (3.81→3.84 µs) with no metric improvement. The compiler already performs this optimization automatically.

> **Avoid when**: When the Triton/LLVM compiler backend is already performing loop-invariant code motion (evidenced by identical instruction counts before and after manual hoisting); when the dominant bottleneck is uncoalesced global memory access caused by inherently strided tensor layouts (66% excessive sectors), which cannot be fixed by index arithmetic reordering; when instruction count is already minimal (<400K) and register count is already low (26 regs/thread). Also do not combine with tl.max_contiguous/tl.multiple_of hints on store indices, which is a confirmed regression pattern.

#### ✅ Plan D: `evict_first_on_strided_welford_pass_to_reduce_l1tex_cache_pollution` (214.103×)

In a two-pass Welford reduction kernel where the first pass loads large strided/permuted tensors (e.g., stride 3136 between warp lanes, or complex modulo/division index expressions), changing the eviction policy from 'evict_last' to 'evict_first' on those non-reusable loads reduces L1TEX cache pollution. This freed cache capacity for small, reusable tensors (bias, LayerNorm weight/bias), reducing warp cycles per issued instruction from 9.36 to 8.50 and cutting excessive global sectors from 41,160 to 20,580 (66%→57% waste). The net effect was a ~3.5% duration reduction (3.81→3.68 µs) and a 3-register reduction (26→23 registers/thread), with the improvement attributable to reduced L1TEX stall pressure rather than improved hit rate (L1 hit rate actually dropped from 46.69% to 7.53% as the evicted lines were no longer cached, but this was intentional — the strided loads were not reused anyway).

> **Applicable when**: Two-pass reduction kernel (Welford or similar) where the first pass loads large tensors with strided or non-linear access patterns (UncoalescedGlobalAccess excessive sectors > 50%, MemoryCacheAccessPattern bytes_per_sector < 12); the first pass loads are NOT reused between loop iterations or across the two passes; small reusable tensors (bias, weight) are loaded in the same loop and compete for L1 cache; CPIStall on L1TEX is a significant fraction of total stall cycles (>20%); the second pass already uses evict_first for the same strided loads.

> **Avoid when**: The strided loads ARE reused across iterations (e.g., a weight matrix accessed repeatedly); L1 hit rate is already low (<20%) and the bottleneck is DRAM bandwidth rather than cache pollution; the kernel is DRAM-throughput-saturated (DRAM Throughput > 70%); the small reusable tensors fit entirely in registers and do not benefit from L1 caching; the kernel has only a single pass (no second normalization pass that re-reads the same data).

---

## Cumulative Progress

```
Baseline:  787900 ns  (1.000×)
Iter 1     8060 ns  (97.754×)  — reduce_shared_memory_via_r0_block_and_autotune_tile_sizes
Iter 4     3840 ns  (205.182×)  — expand_autotune_to_small_xblock_configs_for_register_pressure_relief

Final best: 3840 ns  (205.1823×)
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
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 31360
@@ -22,15 +42,27 @@

     tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     x3 = xindex
+
+    # --- Hoist loop-invariant index sub-expressions BEFORE both loops ---
+    x0_mod56      = x0 % 56
+    x0_div56      = x0 // 56
+    x0_mod56_mod7 = x0_mod56 % 7
+    x0_div56_mod7 = x0_div56 % 7
+    x0_mod56_div7 = x0_mod56 // 7
+    x0_div392     = x0 // 392
+    base0 = x0 + 301056*x1
+    base1 = 32*x0_mod56_mod7 + 224*x0_div56_mod7 + 4704*x0_mod56_div7 + 37632*x0_div392 + 301056*x1
+    base2 = 7*x0_div56_mod7 + x0_mod56_mod7
+
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (32*((((x0 % 56)) % 7)) + 224*(((x0 // 56) % 7)) + 1568*(r0_2 // 32) + 4704*(((x0 % 56)) // 7) + 37632*(x0 // 392) + 301056*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp2 = tl.load(in_ptr2 + (7*(((x0 // 56) % 7)) + 49*(r0_2 // 32) + ((((x0 % 56)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (base0 + 3136*r0_2), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (base1 + 1568*(r0_2 // 32) + (r0_2 % 32)), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp2 = tl.load(in_ptr2 + (base2 + 49*(r0_2 // 32)), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp3 = tmp1 + tmp2
         tmp4 = tmp0 + tmp3
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
@@ -50,11 +82,11 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp12 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp13 = tl.load(in_ptr1 + (32*((((x0 % 56)) % 7)) + 224*(((x0 // 56) % 7)) + 1568*(r0_2 // 32) + 4704*(((x0 % 56)) // 7) + 37632*(x0 // 392) + 301056*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp14 = tl.load(in_ptr2 + (7*(((x0 // 56) % 7)) + 49*(r0_2 // 32) + ((((x0 % 56)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp24 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp26 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp12 = tl.load(in_ptr0 + (base0 + 3136*r0_2), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp13 = tl.load(in_ptr1 + (base1 + 1568*(r0_2 // 32) + (r0_2 % 32)), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp14 = tl.load(in_ptr2 + (base2 + 49*(r0_2 // 32)), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp24 = tl.load(in_ptr3 + (r0_2), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp26 = tl.load(in_ptr4 + (r0_2), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp15 = tmp13 + tmp14
         tmp16 = tmp12 + tmp15
         tmp17 = tmp16 - tmp6
@@ -66,9 +98,8 @@

         tmp23 = tmp17 * tmp22
         tmp25 = tmp23 * tmp24
         tmp27 = tmp25 + tmp26
-        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp27, r0_mask & xmask)
-
-
+        store_idx = tl.max_contiguous(tl.multiple_of(r0_2 + 96*x3, [1, R0_BLOCK]), [1, R0_BLOCK])
+        tl.store(out_ptr2 + store_idx, tmp27, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -80,7 +111,7 @@

     xnumel = 31360
     r0_numel = 96
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 31360, 96, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 31360, 96)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

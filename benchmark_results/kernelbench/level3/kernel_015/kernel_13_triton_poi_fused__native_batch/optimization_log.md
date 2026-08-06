# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 39780 ns — **Best latency**: 39870 ns (**0.9977×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_13_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (54880,1,1), Waves/SM: 25.98
- **Registers/thread**: 30
- **Duration**: 39780 ns
- **SM Throughput**: 74.36 % | **Memory**: 36.04 % | **DRAM**: 19.02 %
- **Achieved Occupancy**: 81.53 %
- **IPC Active**: 3.11 inst/cycle | **No Eligible**: 21.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 1.006× | 0.192× | ✓ | **✓ best** |
| B | autotune_xblock_num_warps_for_pointwise_kernel | 1.002× | 0.381× | ✓ |  |
| C | increase_xblock_with_autotune_for_vectorized_loads_and_reduced_scheduler_stalls | 0.802× | 0.738× | ✓ |  |
| D | increase_xblock_to_reduce_cpi_stall_and_enable_autotune | 0.800× | 0.737× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_n... | Δ | increase_xblock_w... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39780 ns | 206880 ns | +167100 ns (+420.1%) | 104510 ns | +64730 ns (+162.7%) | 53920 ns | +14140 ns (+35.5%) | 53980 ns | +14200 ns (+35.7%) |
| SM Throughput (%) | 74.36 % | 72.25 % | -2.11 pp | 76.43 % | +2.07 pp | 74.63 % | +0.27 pp | 74.68 % | +0.32 pp |
| Memory Throughput (%) | 36.04 % | 26.06 % | -9.98 pp | 28.99 % | -7.05 pp | 33.68 % | -2.36 pp | 33.65 % | -2.39 pp |
| DRAM Throughput (%) | 19.02 % | 3.57 % | -15.45 pp | 7.07 % | -11.95 pp | 13.68 % | -5.34 pp | 13.65 % | -5.37 pp |
| Achieved Occupancy (%) | 81.53 % | 31.85 % | -49.68 pp | 63.04 % | -18.49 pp | 64.06 % | -17.47 pp | 63.92 % | -17.61 pp |
| Registers/Thread | 30.00 register/thread | 76.00 register/thread | +46 | 38.00 register/thread | +8 | 38.00 register/thread | +8 | 38.00 register/thread | +8 |
| Waves/SM | 25.98 | 138.59 | +112.61 | 69.29 | +43.31 | 34.65 | +8.67 | 34.65 | +8.67 |
| No Eligible (%) | 21.43 % | 26.99 % | +5.56 pp | 22.07 % | +0.64 pp | 22.65 % | +1.22 pp | 22.35 % | +0.92 pp |
| IPC Active | 3.11 inst/cycle | 2.92 inst/cycle | -0.19 | 3.12 inst/cycle | +0.01 | 3.11 inst/cycle | +0.00 | 3.09 inst/cycle | -0.02 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.192×)

Adding @triton.autotune with larger XBLOCK configs (up to 2048) caused the compiler to select a configuration that ballooned registers per thread from 30 to 76, collapsing theoretical occupancy from 100% to 37.5% and achieved occupancy from 81.5% to 31.9%. Despite Warp Cycles Per Issued Instruction improving (16.4 → 6.9), the total instruction count exploded 5x (23M → 118M) and kernel duration increased 5.2x (39.8 µs → 206.9 µs). The autotune selected a larger block size (256 threads) that, combined with the higher register pressure from the larger tile, severely limited the number of active warps per SM and caused massive serialization.

> **Applicable when**: Never applicable as described — this is a negative result. The pattern of adding autotune with large XBLOCK configs is dangerous when the kernel already uses a moderate number of registers (≥25 regs/thread) and has high theoretical occupancy (≥75%) at the baseline block size.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK/block_size configs when: (1) baseline registers per thread is already ≥25 (risk of register pressure explosion with larger tiles), (2) theoretical occupancy is already near 100% at baseline (no headroom to trade occupancy for tile size), (3) the kernel is a fused pointwise op with many intermediate temporaries (register count scales with tile complexity, not just tile size). The autotune overhead itself (138 waves/SM vs 26 waves/SM) also indicates the selected config was far worse than baseline.

#### ❌ Plan B: `autotune_xblock_num_warps_for_pointwise_kernel` (0.381×)

Adding @triton.autotune with larger XBLOCK configs (256–1024) to a pointwise kernel that was already register-limited caused a severe regression (0.38x). The autotuner selected XBLOCK=256 with num_warps=8, which increased registers-per-thread from 30 to 38, cutting theoretical occupancy from 100% to 75% and achieved occupancy from 81.5% to 63%. This reduced memory throughput from 934 GB/s to 348 GB/s and nearly tripled kernel duration (39.8 µs → 104.5 µs). When a kernel is already register-limited (Block Limit Registers < Block Limit Warps), increasing XBLOCK or num_warps will spill registers or reduce occupancy further, making the CPIStall 'not selected' problem worse, not better.

> **Applicable when**: Never applicable when the kernel is already register-limited (Block Limit Registers < Block Limit Warps in NCU Occupancy section) and achieved occupancy is already high (>75%). Increasing XBLOCK in this regime increases register pressure and reduces occupancy.

> **Avoid when**: Do NOT apply XBLOCK autotune with larger configs when: (1) registers_per_thread is already near the register-file limit per SM, (2) Block Limit Registers is the binding occupancy constraint, (3) the kernel is compute-bound (Compute SM Throughput > Memory Throughput). In these cases, larger XBLOCK increases register spill or reduces theoretical occupancy, causing severe slowdowns.

#### ❌ Plan D: `increase_xblock_to_reduce_cpi_stall_and_enable_autotune` (0.737×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel that showed 40% CPIStall 'not selected' caused a 36% regression. The autotuner selected a config with more registers per thread (30→38), which reduced Block Limit Registers from 16 to 12 and dropped theoretical occupancy from 100% to 75% and achieved occupancy from 81.5% to 63.9%. Despite fewer active warps per scheduler, the total executed instruction count increased by ~37% (23M→31.5M), indicating the larger tile introduced additional address computation or loop overhead that outweighed any FMA fusion benefit, and the reduced occupancy left insufficient warps to hide latency, increasing total elapsed cycles from 59K to 80K.

> **Applicable when**: Never: this specific strategy (autotune larger XBLOCK for pointwise kernels with CPIStall not-selected) caused regression here due to register pressure increase reducing occupancy.

> **Avoid when**: When the kernel already uses 28–32 registers/thread and theoretical occupancy is at 100% (Block Limit Registers = 16): increasing XBLOCK causes the compiler to spill or allocate more registers, pushing Block Limit Registers down (e.g., to 12), dropping theoretical occupancy to 75% and achieved occupancy below 65%. The resulting instruction count increase and occupancy loss outweigh any scheduler contention reduction. Also avoid when the kernel has complex conditional/branching logic (multi-way concat) that generates extra address arithmetic per element at larger tile sizes.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_autotune_restore_fixed_xblock128_for_register_limited_kernel | 1.012× | 0.997× | ✓ |  |
| B | restore_small_xblock_configs_to_recover_register_limited_occupancy | 0.997× | 0.999× | ✓ |  |
| C | remove_autotune_revert_to_fixed_small_block_to_restore_occupancy | 1.012× | 1.004× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_small_xblock_autotune_configs | 1.008× | 0.995× | ✓ |  |
| E | add_triton_autotune | 1.004× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_autotune_r... | Δ | restore_small_xbl... | Δ | remove_autotune_r... | Δ | reduce_register_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39780 ns | 39900 ns | +120 ns (+0.3%) | 39810 ns | +30 ns (+0.1%) | 39620 ns | -160 ns (-0.4%) | 39970 ns | +190 ns (+0.5%) | 39490 ns | -290 ns (-0.7%) |
| SM Throughput (%) | 74.36 % | 73.72 % | -0.64 pp | 73.68 % | -0.68 pp | 73.68 % | -0.68 pp | 72.34 % | -2.02 pp | 73.70 % | -0.66 pp |
| Memory Throughput (%) | 36.04 % | 35.71 % | -0.33 pp | 35.70 % | -0.34 pp | 35.70 % | -0.34 pp | 35.05 % | -0.99 pp | 35.71 % | -0.33 pp |
| DRAM Throughput (%) | 19.02 % | 19.07 % | +0.05 pp | 19.05 % | +0.03 pp | 19.12 % | +0.10 pp | 19.01 % | -0.01 pp | 19.24 % | +0.22 pp |
| Achieved Occupancy (%) | 81.53 % | 80.01 % | -1.52 pp | 81.61 % | +0.08 pp | 81.31 % | -0.22 pp | 81.27 % | -0.26 pp | 80.76 % | -0.77 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 25.98 | 25.98 | +0.00 | 25.98 | +0.00 | 25.98 | +0.00 | 25.98 | +0.00 | 25.98 | +0.00 |
| No Eligible (%) | 21.43 % | 21.68 % | +0.25 pp | 21.56 % | +0.13 pp | 21.46 % | +0.03 pp | 21.83 % | +0.40 pp | 21.59 % | +0.16 pp |
| IPC Active | 3.11 inst/cycle | 3.12 inst/cycle | +0.01 | 3.13 inst/cycle | +0.02 | 3.12 inst/cycle | +0.01 | 3.13 inst/cycle | +0.02 | 3.12 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.004× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_fma_fusion_for_batchnorm_relu_arithmetic | 1.002× | 1.006× | ✓ |  |
| B | change_eviction_policy_streaming_inputs_to_evict_first | 1.006× | 1.000× | ✓ |  |
| C | change_eviction_policy_streaming_inputs_to_evict_first | 1.008× | 0.999× | ✓ |  |
| D | fuse_fp32_instructions_via_explicit_fma_in_bn_relu_chain | 1.001× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_fma_fusi... | Δ | change_eviction_p... | Δ | change_eviction_p... | Δ | fuse_fp32_instruc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39780 ns | 39550 ns | -230 ns (-0.6%) | 39780 ns | +0 ns (+0.0%) | 39840 ns | +60 ns (+0.2%) | 40420 ns | +640 ns (+1.6%) |
| SM Throughput (%) | 74.36 % | 71.99 % | -2.37 pp | 73.91 % | -0.45 pp | 73.89 % | -0.47 pp | 74.03 % | -0.33 pp |
| Memory Throughput (%) | 36.04 % | 34.80 % | -1.24 pp | 35.79 % | -0.25 pp | 35.78 % | -0.26 pp | 34.99 % | -1.05 pp |
| DRAM Throughput (%) | 19.02 % | 19.13 % | +0.11 pp | 16.69 % | -2.33 pp | 16.59 % | -2.43 pp | 18.73 % | -0.29 pp |
| Achieved Occupancy (%) | 81.53 % | 80.55 % | -0.98 pp | 79.68 % | -1.85 pp | 80.98 % | -0.55 pp | 80.76 % | -0.77 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 28.00 register/thread | -2 |
| Waves/SM | 25.98 | 25.98 | +0.00 | 25.98 | +0.00 | 25.98 | +0.00 | 25.98 | +0.00 |
| No Eligible (%) | 21.43 % | 22.06 % | +0.63 pp | 21.52 % | +0.09 pp | 21.42 % | -0.01 pp | 21.18 % | -0.25 pp |
| IPC Active | 3.11 inst/cycle | 3.11 inst/cycle | +0.00 | 3.12 inst/cycle | +0.01 | 3.12 inst/cycle | +0.01 | 3.14 inst/cycle | +0.03 |

---

## Iteration 4

**Best so far**: 1.004× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_bn_loads_after_concat_to_reduce_live_register_pressure | 1.006× | 0.996× | ✓ |  |
| B | precompute_batchnorm_scale_offset_scalars_for_single_fma | 1.014× | 0.998× | ✓ | **✓ best** |
| C | restructure_bn_arithmetic_to_enable_fma_fusion | 1.012× | 0.999× | ✓ |  |
| D | restructure_cat_concat_as_2d_direct_indexed_load_to_eliminate_where_chain | 1.007× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_bn_loads_af... | Δ | precompute_batchn... | Δ | restructure_bn_ar... | Δ | restructure_cat_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39780 ns | 39940 ns | +160 ns (+0.4%) | 39870 ns | +90 ns (+0.2%) | 39810 ns | +30 ns (+0.1%) | 39710 ns | -70 ns (-0.2%) |
| SM Throughput (%) | 74.36 % | 73.63 % | -0.73 pp | 81.12 % | +6.76 pp | 74.01 % | -0.35 pp | 73.74 % | -0.62 pp |
| Memory Throughput (%) | 36.04 % | 35.60 % | -0.44 pp | 35.67 % | -0.37 pp | 35.81 % | -0.23 pp | 36.04 % | +0.00 pp |
| DRAM Throughput (%) | 19.02 % | 18.96 % | -0.06 pp | 16.60 % | -2.42 pp | 16.68 % | -2.34 pp | 16.75 % | -2.27 pp |
| Achieved Occupancy (%) | 81.53 % | 80.70 % | -0.83 pp | 80.87 % | -0.66 pp | 80.99 % | -0.54 pp | 79.92 % | -1.61 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 26.00 register/thread | -4 | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 25.98 | 25.98 | +0.00 | 25.98 | +0.00 | 25.98 | +0.00 | 25.98 | +0.00 |
| No Eligible (%) | 21.43 % | 22.31 % | +0.88 pp | 14.32 % | -7.11 pp | 21.77 % | +0.34 pp | 21.81 % | +0.38 pp |
| IPC Active | 3.11 inst/cycle | 3.11 inst/cycle | +0.00 | 3.42 inst/cycle | +0.31 | 3.12 inst/cycle | +0.01 | 3.11 inst/cycle | +0.00 |

---

## Iteration 5

**Best so far**: 1.004× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | collapse_bn_scale_multiply_into_fma_chain_to_reduce_nonfused_fp32_instructions | 1.013× | 1.007× | ✓ |  |
| B | eliminate_non_fused_fp32_negation_in_batchnorm_arithmetic | 0.988× | 1.011× | ✓ |  |
| C | fuse_bn_offset_multiply_subtract_into_single_fma | 1.013× | 0.987× | ✓ |  |
| D | reduce_num_warps_to_cut_scheduler_not_selected_stall | 1.004× | 1.092× | ✓ |  |


### Metric Comparison

| Metric | Baseline | collapse_bn_scale... | Δ | eliminate_non_fus... | Δ | fuse_bn_offset_mu... | Δ | reduce_num_warps_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39780 ns | 39490 ns | -290 ns (-0.7%) | 39330 ns | -450 ns (-1.1%) | 40320 ns | +540 ns (+1.4%) | 36420 ns | -3360 ns (-8.4%) |
| SM Throughput (%) | 74.36 % | 73.64 % | -0.72 pp | 73.94 % | -0.42 pp | 79.31 % | +4.95 pp | 45.53 % | -28.83 pp |
| Memory Throughput (%) | 36.04 % | 35.98 % | -0.06 pp | 35.81 % | -0.23 pp | 35.06 % | -0.98 pp | 28.43 % | -7.61 pp |
| DRAM Throughput (%) | 19.02 % | 16.72 % | -2.30 pp | 16.83 % | -2.19 pp | 16.39 % | -2.63 pp | 20.79 % | +1.77 pp |
| Achieved Occupancy (%) | 81.53 % | 79.77 % | -1.76 pp | 80.28 % | -1.25 pp | 80.33 % | -1.20 pp | 30.89 % | -50.64 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 26.00 register/thread | -4 | 32.00 register/thread | +2 |
| Waves/SM | 25.98 | 25.98 | +0.00 | 25.98 | +0.00 | 25.98 | +0.00 | 12.99 | -12.99 |
| No Eligible (%) | 21.43 % | 21.79 % | +0.36 pp | 21.34 % | -0.09 pp | 15.98 % | -5.45 pp | 49.72 % | +28.29 pp |
| IPC Active | 3.11 inst/cycle | 3.12 inst/cycle | +0.01 | 3.13 inst/cycle | +0.02 | 3.35 inst/cycle | +0.24 | 2.00 inst/cycle | -1.11 |


### Key Learnings

#### ✅ Plan D: `reduce_num_warps_to_cut_scheduler_not_selected_stall` (1.092×)

When a compute-bound pointwise kernel shows high CPIStall 'not selected' stalls (>35% of cycles) with many active warps per scheduler (>10) but relatively few eligible warps (~5-6), reducing num_warps (e.g., from 4 to 2) halves the block thread count and active warps per scheduler, cutting scheduler arbitration overhead. In this case, halving num_warps (128→64 threads/block) reduced Warp Cycles Per Issued Instruction from 16.5 to 9.1 cycles and total executed instructions from 22.8M to 12.9M, yielding a 1.09x speedup despite achieved occupancy dropping from 79.9% to 30.9%. The kernel remained latency-tolerant because the large grid (54,880 blocks) kept all SMs saturated across sufficient waves, and the reduced scheduler contention more than compensated for lower per-SM warp count.

> **Applicable when**: Kernel is compute-bound (Compute SM Throughput >> Memory Throughput), CPIStall 'not selected' stall fraction is >30% of total cycles, Active Warps Per Scheduler is high (>10) while Eligible Warps Per Scheduler is relatively low (<7), and the grid is large enough (many thousands of blocks) to keep all SMs saturated even with fewer warps per block. Registers per thread should be near the register-file limit so that reducing num_warps does not collapse theoretical occupancy.

> **Avoid when**: Do not reduce num_warps when the kernel is memory-latency-bound (Memory Throughput >> Compute Throughput) and relies on high warp count to hide DRAM latency. Also avoid when the grid is small (few waves per SM), as reducing warps per block may leave SMs idle. Do not apply when achieved occupancy is already low (<40%) since further reduction will starve the scheduler of eligible warps entirely.

---

## Cumulative Progress

```
Baseline:  39780 ns  (1.000×)
Iter 2     39620 ns  (1.004×)  — remove_autotune_revert_to_fixed_small_block_to_restore_occupancy

Final best: 39870 ns  (0.9977×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
     xnumel = 7024640
@@ -18,52 +28,31 @@

     tmp37 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp46 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp48 = tl.load(in_ptr9 + (x0), mask=xmask, eviction_policy='evict_last')
-    tmp0 = x0
-    tmp1 = tl.full([1], 0, tl.int64)
-    tmp2 = tmp0 >= tmp1
-    tmp3 = tl.full([1], 64, tl.int64)
-    tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp6 = tmp0 >= tmp3
-    tmp7 = tl.full([1], 96, tl.int64)
-    tmp8 = tmp0 < tmp7
-    tmp9 = tmp6 & tmp8
-    tmp10 = tl.load(in_ptr1 + (32*x1 + ((-64) + x0)), mask=tmp9 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp11 = tmp0 >= tmp7
-    tmp12 = tl.full([1], 128, tl.int64)
-    tmp13 = tmp0 < tmp12
-    tmp14 = tmp11 & tmp13
-    tmp15 = tl.load(in_ptr2 + (32*x1 + ((-96) + x0)), mask=tmp14 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp16 = tmp0 >= tmp12
-    tmp17 = tl.full([1], 160, tl.int64)
-    tmp18 = tmp0 < tmp17
-    tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr3 + (32*x1 + ((-128) + x0)), mask=tmp19 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp21 = tmp0 >= tmp17
-    tmp22 = tl.full([1], 192, tl.int64)
-    tmp23 = tmp0 < tmp22
-    tmp24 = tmp21 & tmp23
-    tmp25 = tl.load(in_ptr4 + (32*x1 + ((-160) + x0)), mask=tmp24 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp26 = tmp0 >= tmp22
-    tmp27 = tl.full([1], 224, tl.int64)
-    tmp28 = tmp0 < tmp27
-    tmp29 = tl.load(in_ptr5 + (32*x1 + ((-192) + x0)), mask=tmp26 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp30 = tl.where(tmp24, tmp25, tmp29)
-    tmp31 = tl.where(tmp19, tmp20, tmp30)
-    tmp32 = tl.where(tmp14, tmp15, tmp31)
-    tmp33 = tl.where(tmp9, tmp10, tmp32)
-    tmp34 = tl.where(tmp4, tmp5, tmp33)
-    tmp36 = tmp34 - tmp35
+    # Simple boolean group masks — mutually exclusive, collectively exhaustive over x0 in [0,224)
+    in_grp0 = x0 < 64
+    in_grp1 = (x0 >= 64)  & (x0 < 96)
+    in_grp2 = (x0 >= 96)  & (x0 < 128)
+    in_grp3 = (x0 >= 128) & (x0 < 160)
+    in_grp4 = (x0 >= 160) & (x0 < 192)
+    in_grp5 = x0 >= 192
+    # Load with evict_first so large streaming tensors don't evict BN params from L1
+    v0 = tl.load(in_ptr0 + (64*x1 + x0),         mask=in_grp0 & xmask, other=0.0, eviction_policy='evict_first')
+    v1 = tl.load(in_ptr1 + (32*x1 + (x0 - 64)),  mask=in_grp1 & xmask, other=0.0, eviction_policy='evict_first')
+    v2 = tl.load(in_ptr2 + (32*x1 + (x0 - 96)),  mask=in_grp2 & xmask, other=0.0, eviction_policy='evict_first')
+    v3 = tl.load(in_ptr3 + (32*x1 + (x0 - 128)), mask=in_grp3 & xmask, other=0.0, eviction_policy='evict_first')
+    v4 = tl.load(in_ptr4 + (32*x1 + (x0 - 160)), mask=in_grp4 & xmask, other=0.0, eviction_policy='evict_first')
+    v5 = tl.load(in_ptr5 + (32*x1 + (x0 - 192)), mask=in_grp5 & xmask, other=0.0, eviction_policy='evict_first')
+    # Sum is equivalent to tl.where cascade since masks are mutually exclusive and other=0.0
+    tmp34 = v0 + v1 + v2 + v3 + v4 + v5
     tmp38 = tl.full([1], 1e-05, tl.float32)
     tmp39 = tmp37 + tmp38
     tmp40 = libdevice.rsqrt(tmp39)
-    tmp45 = tmp36 * tmp40
-    tmp47 = tmp45 * tmp46
-    tmp49 = tmp47 + tmp48
+    tmp_scale = tmp40 * tmp46
+    tmp_bias_adj = tl.math.fma(-tmp35, tmp_scale, tmp48)
+    tmp49 = tl.math.fma(tmp34, tmp_scale, tmp_bias_adj)
     tmp50 = tl.full([1], 0, tl.int32)
     tmp51 = tl.maximum(tmp50, tmp49)
     tl.store(in_out_ptr0 + (x2), tmp51, mask=xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -79,7 +68,7 @@

     arg_10 = inputs[10]
     xnumel = 7024640
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 7024640, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 7024640)
     return arg_10
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

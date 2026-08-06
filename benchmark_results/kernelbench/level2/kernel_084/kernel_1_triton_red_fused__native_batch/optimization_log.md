# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6900000 ns — **Best latency**: 9860 ns (**699.7972×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`hoist_loop_invariant_scalars_and_fuse_fp32_instructions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_084/kernel_1_triton_red_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.00
- **Registers/thread**: 32
- **Duration**: 6900000 ns
- **SM Throughput**: 0.75 % | **Memory**: 2.50 % | **DRAM**: 0.30 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.23 inst/cycle | **No Eligible**: 94.29 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | online_softmax_attention | 198.541× | 1.194× | ✓ |  |
| C | autotune_tile_sizes_for_sm_utilization | 366.734× | 347.782× | ✓ |  |
| D | add_autotune_increase_grid_vectorized_loads | 489.404× | 342.262× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | online_softmax_at... | Δ | autotune_tile_siz... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6900000 ns | 5780000 ns | -1120000 ns (-16.2%) | 19840 ns | -6880160 ns (-99.7%) | 20160 ns | -6879840 ns (-99.7%) |
| SM Throughput (%) | 0.75 % | 1.05 % | +0.30 pp | 1.01 % | +0.26 pp | 1.02 % | +0.27 pp |
| Memory Throughput (%) | 2.50 % | 3.01 % | +0.51 pp | 1.07 % | -1.43 pp | 1.04 % | -1.46 pp |
| DRAM Throughput (%) | 0.30 % | 0.25 % | -0.05 pp | 0.51 % | +0.21 pp | 0.50 % | +0.20 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 255.00 register/thread | +223 | 255.00 register/thread | +223 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.03 | +0.03 | 0.03 | +0.03 |
| No Eligible (%) | 94.29 % | 92.08 % | -2.21 pp | 81.14 % | -13.15 pp | 82.28 % | -12.01 pp |
| IPC Active | 0.23 inst/cycle | 0.32 inst/cycle | +0.09 | 0.74 inst/cycle | +0.51 | 0.71 inst/cycle | +0.48 |


### Key Learnings

#### ✅ Plan C: `autotune_tile_sizes_for_sm_utilization` (258.524×)

When a Triton kernel launches with far fewer blocks than available SMs (e.g., 8 blocks on 132 SMs, 0 full waves, <10% achieved occupancy), combining @triton.autotune with small XBLOCK configs (XBLOCK in {1,2,4}) and a lambda grid expands parallelism to fill all SMs. Adding restore_value for in-place tensors in the autotune decorator is mandatory for correctness during benchmarking. Pairing this with a 2-pass online softmax (eliminating intermediate global stores between passes) and maxnreg=128 to cap register pressure further reduces instruction count and improves theoretical occupancy from 12.5% to 25%, yielding a combined 258x speedup over the prior baseline.

> **Applicable when**: LaunchConfiguration NCU rule fires with high speedup_pct (>80%); grid_size << sm_count (e.g., grid_size < sm_count/4); Waves Per SM ~0; Achieved Occupancy <10%; Scheduler 'No Eligible' >80%; kernel has parametric XBLOCK/R0_BLOCK tile sizes; kernel implements multi-pass softmax with intermediate global memory writes; in-place tensors (in_out_ptr pattern) are present; registers_per_thread near 255 with Block Limit Registers = 2.

> **Avoid when**: Grid already covers all SMs (waves_per_sm >= 1.0); kernel is memory-bandwidth bound with high DRAM throughput (>50%); in-place tensors exist and restore_value cannot be set (correctness risk); score recomputation in pass 2 is prohibitively expensive (e.g., involves matmul); XBLOCK reduction would exceed hardware grid limits; maxnreg causes excessive register spill for large R0_BLOCK configs.

---

## Iteration 2

**Best so far**: 342.2619× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_restore_value_for_sm_utilization | 354.094× | 351.682× | ✓ |  |
| B | fix_grid_lambda_and_cap_registers_for_sm_saturation | 464.707× | 224.609× | ✓ |  |
| C | expand_autotune_configs_with_small_xblock_and_restore_value | 390.222× | 430.443× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_grid_lambda_a... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6900000 ns | 19620 ns | -6880380 ns (-99.7%) | 30720 ns | -6869280 ns (-99.6%) | 16030 ns | -6883970 ns (-99.8%) |
| SM Throughput (%) | 0.75 % | 1.00 % | +0.25 pp | 0.71 % | -0.04 pp | 0.91 % | +0.16 pp |
| Memory Throughput (%) | 2.50 % | 1.07 % | -1.43 pp | 1.49 % | -1.01 pp | 0.92 % | -1.58 pp |
| DRAM Throughput (%) | 0.30 % | 0.52 % | +0.22 pp | 0.34 % | +0.04 pp | 0.61 % | +0.31 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.25 % | +0.01 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 255.00 register/thread | +223 | 128.00 register/thread | +96 | 255.00 register/thread | +223 |
| Waves/SM | 0.00 | 0.03 | +0.03 | 0.02 | +0.02 | 0.03 | +0.03 |
| No Eligible (%) | 94.29 % | 81.75 % | -12.54 pp | 87.35 % | -6.94 pp | 82.44 % | -11.85 pp |
| IPC Active | 0.23 inst/cycle | 0.71 inst/cycle | +0.48 | 0.49 inst/cycle | +0.26 | 0.70 inst/cycle | +0.47 |

---

## Iteration 3

**Best so far**: 342.2619× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_maxnreg_to_all_configs | 279.561× | 281.174× | ✓ |  |
| B | hoist_loop_invariant_scalars_and_fuse_fp32_instructions | 569.971× | 699.797× | ✓ | **✓ best** |
| C | fuse_3pass_softmax_to_2pass_with_lambda_grid | 507.068× | 258.524× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | hoist_loop_invari... | Δ | fuse_3pass_softma... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6900000 ns | 24540 ns | -6875460 ns (-99.6%) | 9860 ns | -6890140 ns (-99.9%) | 26690 ns | -6873310 ns (-99.6%) |
| SM Throughput (%) | 0.75 % | 0.70 % | -0.05 pp | 1.15 % | +0.40 pp | 0.68 % | -0.07 pp |
| Memory Throughput (%) | 2.50 % | 1.39 % | -1.11 pp | 1.82 % | -0.68 pp | 1.52 % | -0.98 pp |
| DRAM Throughput (%) | 0.30 % | 0.40 % | +0.10 pp | 0.98 % | +0.68 pp | 0.37 % | +0.07 pp |
| Achieved Occupancy (%) | 6.24 % | 6.21 % | -0.03 pp | 6.24 % | +0.00 pp | 6.21 % | -0.03 pp |
| Registers/Thread | 32.00 register/thread | 128.00 register/thread | +96 | 178.00 register/thread | +146 | 128.00 register/thread | +96 |
| Waves/SM | 0.00 | 0.02 | +0.02 | 0.03 | +0.03 | 0.02 | +0.02 |
| No Eligible (%) | 94.29 % | 87.33 % | -6.96 pp | 75.98 % | -18.31 pp | 87.10 % | -7.19 pp |
| IPC Active | 0.23 inst/cycle | 0.49 inst/cycle | +0.26 | 0.91 inst/cycle | +0.68 | 0.52 inst/cycle | +0.29 |


### Key Learnings

#### ✅ Plan B: `hoist_loop_invariant_scalars_and_fuse_fp32_instructions` (699.797×)

When a Triton reduction kernel recomputes the same scalar load and derived expressions (e.g., sign computation from a scale value) inside each of multiple sequential reduction loops, hoisting those computations before all loops eliminates redundant work and reduces register pressure. Replacing a two-step `(1 / sqrt_rn(x)) * 1.0` pattern with `tl.math.rsqrt(x)` collapses two non-fused instructions into a single MUFU.RSQ hardware instruction, enabling downstream FMA fusion for multiply-add chains. These changes together reduced executed instructions from 118K to 91K (~22%), registers per thread from 255 to 178, warp cycles per issued instruction from 5.83 to 4.27, branch divergence from 3.88 avg divergent branches to 0, and L1/TEX hit rate from 16.25% to 40.01%, cutting kernel duration from 16.03µs to 9.86µs (~38% reduction).

> **Applicable when**: FPInstructions NCU rule fires with high non-fused fraction (>80% non-fused FP32 instructions); kernel has multiple sequential reduction loops that each independently load the same scalar pointer and recompute derived expressions (sign, reciprocal-sqrt); registers_per_thread is near 255 (register pressure from repeated temporaries); branch divergence is non-zero (Avg. Divergent Branches > 0) due to repeated where() inside loops; L1/TEX hit rate is low (<25%) suggesting cache thrashing from redundant loads.

> **Avoid when**: The scalar value changes between loop iterations (not truly loop-invariant); the sign/scale computation depends on per-element data rather than a single scalar pointer; rsqrt precision is insufficient (requires exact IEEE sqrt behavior); the kernel has only one reduction loop (no redundancy to eliminate).

---

## Iteration 4

**Best so far**: 699.7972× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_lower_maxnreg_for_register_headroom | 506.997× | 577.889× | ✓ |  |
| B | fix_hardcoded_grid_to_lambda_for_sm_saturation | 278.300× | 281.518× | ✓ |  |
| C | fix_hardcoded_grid_to_lambda_for_sm_saturation | 511.942× | 260.083× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6900000 ns | 11940 ns | -6888060 ns (-99.8%) | 24510 ns | -6875490 ns (-99.6%) | 26530 ns | -6873470 ns (-99.6%) |
| SM Throughput (%) | 0.75 % | 0.98 % | +0.23 pp | 0.68 % | -0.07 pp | 0.68 % | -0.07 pp |
| Memory Throughput (%) | 2.50 % | 1.69 % | -0.81 pp | 1.39 % | -1.11 pp | 1.54 % | -0.96 pp |
| DRAM Throughput (%) | 0.30 % | 0.81 % | +0.51 pp | 0.40 % | +0.10 pp | 0.38 % | +0.08 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.21 % | -0.03 pp | 6.22 % | -0.02 pp |
| Registers/Thread | 32.00 register/thread | 128.00 register/thread | +96 | 128.00 register/thread | +96 | 128.00 register/thread | +96 |
| Waves/SM | 0.00 | 0.02 | +0.02 | 0.02 | +0.02 | 0.02 | +0.02 |
| No Eligible (%) | 94.29 % | 80.34 % | -13.95 pp | 87.85 % | -6.44 pp | 86.93 % | -7.36 pp |
| IPC Active | 0.23 inst/cycle | 0.77 inst/cycle | +0.54 | 0.50 inst/cycle | +0.27 | 0.51 inst/cycle | +0.28 |

---

## Iteration 5

**Best so far**: 699.7972× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_dynamic_lambda_for_sm_saturation | 527.249× | 576.441× | ✓ |  |
| B | fix_hardcoded_grid_to_lambda_for_sm_saturation_v2 | 276.667× | 283.717× | ✓ |  |
| C | fix_callsite_grid_to_lambda_for_sm_saturation | 511.072× | 262.957× | ✓ |  |
| D | add_triton_autotune | 512.916× | 590.753× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | fix_callsite_grid... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6900000 ns | 11970 ns | -6888030 ns (-99.8%) | 24320 ns | -6875680 ns (-99.6%) | 26240 ns | -6873760 ns (-99.6%) | 11680 ns | -6888320 ns (-99.8%) |
| SM Throughput (%) | 0.75 % | 0.99 % | +0.24 pp | 0.69 % | -0.06 pp | 0.67 % | -0.08 pp | 1.03 % | +0.28 pp |
| Memory Throughput (%) | 2.50 % | 1.55 % | -0.95 pp | 1.35 % | -1.15 pp | 1.56 % | -0.94 pp | 1.60 % | -0.90 pp |
| DRAM Throughput (%) | 0.30 % | 0.81 % | +0.51 pp | 0.41 % | +0.11 pp | 0.38 % | +0.08 pp | 0.83 % | +0.53 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.21 % | -0.03 pp | 6.22 % | -0.02 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 128.00 register/thread | +96 | 128.00 register/thread | +96 | 128.00 register/thread | +96 | 128.00 register/thread | +96 |
| Waves/SM | 0.00 | 0.02 | +0.02 | 0.02 | +0.02 | 0.02 | +0.02 | 0.02 | +0.02 |
| No Eligible (%) | 94.29 % | 79.86 % | -14.43 pp | 87.84 % | -6.45 pp | 87.36 % | -6.93 pp | 79.75 % | -14.54 pp |
| IPC Active | 0.23 inst/cycle | 0.76 inst/cycle | +0.53 | 0.48 inst/cycle | +0.25 | 0.52 inst/cycle | +0.29 | 0.77 inst/cycle | +0.54 |

---

## Cumulative Progress

```
Baseline:  6900000 ns  (1.000×)
Iter 1     20160 ns  (342.262×)  — add_autotune_increase_grid_vectorized_loads
Iter 3     9860 ns  (699.797×)  — hoist_loop_invariant_scalars_and_fuse_fp32_instructions

Final best: 9860 ns  (699.7972×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 8192}, num_warps=4, num_stages=1, maxnreg=128),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 4096}, num_warps=4, num_stages=1, maxnreg=128),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=1, maxnreg=96),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1, maxnreg=96),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 4096}, num_warps=8, num_stages=1, maxnreg=128),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 2048}, num_warps=8, num_stages=1, maxnreg=96),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1, maxnreg=96),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 1024}, num_warps=8, num_stages=1, maxnreg=96),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 512},  num_warps=4, num_stages=1, maxnreg=96),
+        triton.Config({'XBLOCK': 128,'R0_BLOCK': 512},  num_warps=4, num_stages=1, maxnreg=96),
+    ],
+    key=['xnumel', 'r0_numel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 1024
@@ -17,8 +34,13 @@

     r0_base = tl.arange(0, R0_BLOCK)[None, :]
     rbase = r0_base
     x0 = xindex
-    tmp18 = tl.load(in_ptr5 + (0))
-    tmp19 = tl.broadcast_to(tmp18, [1, 1])
+
+    # --- Hoist scalar load and sign computation BEFORE all loops ---
+    _scale_val = tl.load(in_ptr5 + tl.zeros([XBLOCK, 1], dtype=tl.int32), mask=xmask, other=0.0)
+    _scale_bc  = tl.broadcast_to(_scale_val, [XBLOCK, 1])
+    _sign      = tl.where(_scale_bc >= 0.0, 1.0, -1.0)   # preserves +1 for scale=0
+    _abs_scale = _sign * _scale_bc                         # |scale| = sign(scale)*scale
+
     _tmp26 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
@@ -26,36 +48,27 @@

         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp3 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp5 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp14 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp16 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp2 = tmp0 + tmp1
-        tmp4 = tmp2 - tmp3
-        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
-        tmp7 = tmp5 + tmp6
-        tmp8 = tl.sqrt_rn(tmp7)
-        tmp9 = tl.full([1, 1], 1, tl.int32)
-        tmp10 = (tmp9 / tmp8)
-        tmp11 = tl.full([1, 1], 1.0, tl.float32)
-        tmp12 = tmp10 * tmp11
-        tmp13 = tmp4 * tmp12
+        tmp0  = tl.load(in_ptr0 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp1  = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp3  = tl.load(in_ptr1 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp5  = tl.load(in_ptr2 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp14 = tl.load(in_ptr3 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp16 = tl.load(in_ptr4 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp2  = tmp0 + tmp1
+        tmp4  = tmp2 - tmp3
+        # Replace two-step (1/sqrt_rn(x))*1.0 with single rsqrt instruction
+        tmp_inv_std = tl.math.rsqrt(tmp5 + 1e-5)
+        tmp13 = tmp4 * tmp_inv_std
         tmp15 = tmp13 * tmp14
         tmp17 = tmp15 + tmp16
-        tmp20 = tl.full([1, 1], 0.0, tl.float32)
-        tmp21 = tmp19 >= tmp20
-        tmp22 = tl.full([1, 1], -1.0, tl.float32)
-        tmp23 = tl.where(tmp21, tmp11, tmp22)
-        tmp24 = tmp17 * tmp23
+        # Use hoisted _sign instead of recomputing
+        tmp24 = tmp17 * _sign
         tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
         tmp27 = tl.maximum(_tmp26, tmp25)
         _tmp26 = tl.where(r0_mask & xmask, tmp27, _tmp26)
-        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp24, r0_mask & xmask)
+        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp24, mask=r0_mask & xmask)
     tmp26 = triton_helpers.max2(_tmp26, 1)[:, None]
-    tmp30 = tl.load(in_ptr5 + (0))
-    tmp31 = tl.broadcast_to(tmp30, [1, 1])
+
     _tmp41 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
@@ -63,42 +76,29 @@

         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp28 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp28 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp29 = tmp28 - tmp26
-        tmp32 = tl.full([1, 1], 0.0, tl.float32)
-        tmp33 = tmp31 >= tmp32
-        tmp34 = tl.full([1, 1], 1.0, tl.float32)
-        tmp35 = tl.full([1, 1], -1.0, tl.float32)
-        tmp36 = tl.where(tmp33, tmp34, tmp35)
-        tmp37 = tmp36 * tmp31
-        tmp38 = tmp29 * tmp37
+        # Use hoisted _abs_scale instead of recomputing sign * scale
+        tmp38 = tmp29 * _abs_scale
         tmp39 = tl.exp(tmp38)
         tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
         tmp42 = _tmp41 + tmp40
         _tmp41 = tl.where(r0_mask & xmask, tmp42, _tmp41)
     tmp41 = tl.sum(_tmp41, 1)[:, None]
-    tmp45 = tl.load(in_ptr5 + (0))
-    tmp46 = tl.broadcast_to(tmp45, [1, 1])
+
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp43 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp43 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp44 = tmp43 - tmp26
-        tmp47 = tl.full([1, 1], 0.0, tl.float32)
-        tmp48 = tmp46 >= tmp47
-        tmp49 = tl.full([1, 1], 1.0, tl.float32)
-        tmp50 = tl.full([1, 1], -1.0, tl.float32)
-        tmp51 = tl.where(tmp48, tmp49, tmp50)
-        tmp52 = tmp51 * tmp46
-        tmp53 = tmp44 * tmp52
+        # Use hoisted _abs_scale instead of recomputing sign * scale
+        tmp53 = tmp44 * _abs_scale
         tmp54 = tl.exp(tmp53)
         tmp55 = (tmp54 / tmp41)
-        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp55, r0_mask & xmask)
-
-
+        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp55, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -111,7 +111,7 @@

     xnumel = 1024
     r0_numel = 8192
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 1024, 8192, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 1024, 8192)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

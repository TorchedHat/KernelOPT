# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 742050 ns — **Best latency**: 192610 ns (**3.8526×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`persistent_kernel_grid_stride_loop_over_rows`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_38_triton_per_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.04
- **Registers/thread**: 32
- **Duration**: 742050 ns
- **SM Throughput**: 1.00 % | **Memory**: 3.23 % | **DRAM**: 1.00 %
- **Achieved Occupancy**: 6.21 %
- **IPC Active**: 0.15 inst/cycle | **No Eligible**: 96.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency | 0.999× | 0.996× | ✓ |  |
| C | vectorized_loads_and_increase_num_warps | 1.118× | 1.113× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 742050 ns | 745250 ns | +3200 ns (+0.4%) | 666910 ns | -75140 ns (-10.1%) |
| SM Throughput (%) | 1.00 % | 1.01 % | +0.01 pp | 1.07 % | +0.07 pp |
| Memory Throughput (%) | 3.23 % | 3.21 % | -0.02 pp | 3.36 % | +0.13 pp |
| DRAM Throughput (%) | 1.00 % | 0.98 % | -0.02 pp | 0.93 % | -0.07 pp |
| Achieved Occupancy (%) | 6.21 % | 6.21 % | +0.00 pp | 6.25 % | +0.04 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.04 | 0.04 | +0.00 | 0.04 | +0.00 |
| No Eligible (%) | 96.24 % | 96.23 % | -0.01 pp | 95.80 % | -0.44 pp |
| IPC Active | 0.15 inst/cycle | 0.15 inst/cycle | +0.00 | 0.17 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.1127× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_r0block_to_eliminate_padding_waste_and_reduce_smem_pressure | 1.119× | 1.111× | ✓ | **✓ best** |
| C | fix_xblock1_correct_masked_reduction_to_saturate_sms | 0.998× | 0.998× | ✓ |  |
| D | simplify_load_masks_to_enable_vectorized_memory_access | 1.093× | 1.100× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_r0block_to_el... | Δ | fix_xblock1_corre... | Δ | simplify_load_mas... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 742050 ns | 667900 ns | -74150 ns (-10.0%) | 743740 ns | +1690 ns (+0.2%) | 674560 ns | -67490 ns (-9.1%) |
| SM Throughput (%) | 1.00 % | 1.08 % | +0.08 pp | 1.02 % | +0.02 pp | 1.04 % | +0.04 pp |
| Memory Throughput (%) | 3.23 % | 3.35 % | +0.12 pp | 3.21 % | -0.02 pp | 3.16 % | -0.07 pp |
| DRAM Throughput (%) | 1.00 % | 0.93 % | -0.07 pp | 0.99 % | -0.01 pp | 0.83 % | -0.17 pp |
| Achieved Occupancy (%) | 6.21 % | 6.25 % | +0.04 pp | 6.21 % | +0.00 pp | 6.24 % | +0.03 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.04 | 0.04 | +0.00 | 0.04 | +0.00 | 0.04 | +0.00 |
| No Eligible (%) | 96.24 % | 95.79 % | -0.45 pp | 96.23 % | -0.01 pp | 95.97 % | -0.27 pp |
| IPC Active | 0.15 inst/cycle | 0.17 inst/cycle | +0.02 | 0.15 inst/cycle | +0.00 | 0.16 inst/cycle | +0.01 |


### Key Learnings

#### ❌ Plan B: `fix_r0block_to_eliminate_padding_waste_and_reduce_smem_pressure` (1.111×)

Changing R0_BLOCK from 512 to 384 (to match r0_numel exactly) was expected to reduce shared memory pressure and increase Block Limit Shared Mem from 3 to 4 blocks/SM, raising theoretical occupancy from 18.75% to 25%. However, all occupancy metrics remained completely unchanged (Block Limit Shared Mem=3, Theoretical Occupancy=18.75%, Achieved Occupancy=6.25%), and Dynamic Shared Memory Per Block stayed at 65.54 KB. The elapsed cycles were essentially identical (1,000,315 vs 1,001,816), indicating the change had no measurable effect. Triton's shared memory allocation for reduction scratch buffers is apparently not proportional to R0_BLOCK in the way predicted, or the dynamic smem is dominated by other allocations unrelated to R0_BLOCK.

> **Applicable when**: Never — this strategy produced no improvement even when R0_BLOCK exactly matched r0_numel.

> **Avoid when**: When the kernel's dynamic shared memory is dominated by allocations other than the reduction scratch buffer (e.g., Triton's per-block communication buffers for persistent reductions). Reducing R0_BLOCK does not reduce dynamic smem if smem is allocated based on XBLOCK or other fixed parameters. Also avoid when Block Limit Shared Mem and Theoretical Occupancy remain unchanged after the modification — this confirms smem is not the binding constraint being addressed.

---

## Iteration 4

**Best so far**: 1.1127× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | rewrite_reduction_to_eliminate_scratch_smem_and_expand_grid | 1.119× | 1.117× | ✓ |  |
| B | restructure_to_one_row_per_block_for_full_sm_coverage | 0.998× | 1.007× | ✓ |  |
| C | persistent_kernel_one_block_per_sm_loop_over_x_rows | 3.061× | 3.153× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | rewrite_reduction... | Δ | restructure_to_on... | Δ | persistent_kernel... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 742050 ns | 664190 ns | -77860 ns (-10.5%) | 736960 ns | -5090 ns (-0.7%) | 235330 ns | -506720 ns (-68.3%) |
| SM Throughput (%) | 1.00 % | 1.08 % | +0.08 pp | 1.02 % | +0.02 pp | 0.83 % | -0.17 pp |
| Memory Throughput (%) | 3.23 % | 3.37 % | +0.14 pp | 3.24 % | +0.01 pp | 1.22 % | -2.01 pp |
| DRAM Throughput (%) | 1.00 % | 0.93 % | -0.07 pp | 1.00 % | +0.00 pp | 0.78 % | -0.22 pp |
| Achieved Occupancy (%) | 6.21 % | 6.25 % | +0.04 pp | 6.22 % | +0.01 pp | 6.25 % | +0.04 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 40.00 register/thread | +8 |
| Waves/SM | 0.04 | 0.04 | +0.00 | 0.04 | +0.00 | 0.01 | -0.03 |
| No Eligible (%) | 96.24 % | 95.77 % | -0.47 pp | 96.23 % | -0.01 pp | 92.95 % | -3.29 pp |
| IPC Active | 0.15 inst/cycle | 0.17 inst/cycle | +0.02 | 0.15 inst/cycle | +0.00 | 0.28 inst/cycle | +0.13 |

---

## Iteration 5

**Best so far**: 3.1532× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launch_grid_to_saturate_all_132_sms_and_increase_num_warps | 1.000× | 1.002× | ✓ |  |
| B | persistent_kernel_grid_stride_loop_over_rows | 3.060× | 3.142× | ✓ |  |
| C | persistent_kernel_grid_stride_loop_over_rows | 3.785× | 3.853× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_launch_grid_t... | Δ | persistent_kernel... | Δ | persistent_kernel... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 742050 ns | 740480 ns | -1570 ns (-0.2%) | 236160 ns | -505890 ns (-68.2%) | 192610 ns | -549440 ns (-74.0%) |
| SM Throughput (%) | 1.00 % | 1.02 % | +0.02 pp | 0.82 % | -0.18 pp | 1.00 % | +0.00 pp |
| Memory Throughput (%) | 3.23 % | 3.22 % | -0.01 pp | 1.21 % | -2.02 pp | 0.98 % | -2.25 pp |
| DRAM Throughput (%) | 1.00 % | 0.99 % | -0.01 pp | 0.78 % | -0.22 pp | 0.96 % | -0.04 pp |
| Achieved Occupancy (%) | 6.21 % | 6.22 % | +0.01 pp | 6.25 % | +0.04 pp | 6.25 % | +0.04 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 40.00 register/thread | +8 | 32.00 register/thread | +0 |
| Waves/SM | 0.04 | 0.04 | +0.00 | 0.01 | -0.03 | 0.01 | -0.03 |
| No Eligible (%) | 96.24 % | 96.22 % | -0.02 pp | 92.94 % | -3.30 pp | 91.48 % | -4.76 pp |
| IPC Active | 0.15 inst/cycle | 0.15 inst/cycle | +0.00 | 0.28 inst/cycle | +0.13 | 0.34 inst/cycle | +0.19 |


### Key Learnings

#### ✅ Plan B: `persistent_kernel_grid_stride_loop_over_rows` (3.853×)

After converting a 2D tiled kernel (XBLOCK=128) to a persistent grid-stride kernel (XBLOCK=1) with a [1, R0_BLOCK] shaped r0_index, a further refinement of using a flat 1D r0_index (shape [R0_BLOCK] instead of [1, R0_BLOCK]) eliminates the extra broadcast dimension, reducing registers per thread from 40 to 32 and dynamic shared memory from 2050 bytes to 16 bytes. This raised theoretical occupancy from 75% to 100%, cut warp cycles per issued instruction from 14.18 to 11.74, and improved L1/TEX hit rate from 42.26% to 49.65%, yielding an additional ~18% speedup on top of the already-optimized persistent kernel. The scalar stores (tl.store(out_ptr + x3, scalar_val)) also become cleaner without the tl.full([1,1], x3, tl.int32) index workaround.

> **Applicable when**: A persistent grid-stride kernel (XBLOCK=1) still uses a [1, R0_BLOCK] shaped r0_index (with [None, :] indexing) inherited from the original 2D tiled kernel. Registers per thread are elevated (>32) and dynamic shared memory is non-trivially large (>1 KB) despite XBLOCK=1. Theoretical occupancy is below 100% and Block Limit Registers is the binding constraint. The reduction axis is 1D and independent per row.

> **Avoid when**: When XBLOCK > 1 is still in use (the 2D [XBLOCK, R0_BLOCK] shape is needed for the outer x-dimension). When the [1, R0_BLOCK] shape is required for broadcasting against other [XBLOCK, R0_BLOCK] tensors in the computation. When registers per thread are already at or below 32.

#### ✅ Plan C: `persistent_kernel_grid_stride_loop_over_rows` (3.853×)

After converting a 2D tiled kernel (XBLOCK=128) to a persistent grid-stride kernel (XBLOCK=1) with a [1, R0_BLOCK] shaped r0_index, a further refinement of using a flat 1D r0_index (shape [R0_BLOCK] instead of [1, R0_BLOCK]) eliminates the extra broadcast dimension, reducing registers per thread from 40 to 32 and dynamic shared memory from 2050 bytes to 16 bytes. This raised theoretical occupancy from 75% to 100%, cut warp cycles per issued instruction from 14.18 to 11.74, and improved L1/TEX hit rate from 42.26% to 49.65%, yielding an additional ~18% speedup on top of the already-optimized persistent kernel. The scalar stores (tl.store(out_ptr + x3, scalar_val)) also become cleaner without the tl.full([1,1], x3, tl.int32) index workaround.

> **Applicable when**: A persistent grid-stride kernel (XBLOCK=1) still uses a [1, R0_BLOCK] shaped r0_index (with [None, :] indexing) inherited from the original 2D tiled kernel. Registers per thread are elevated (>32) and dynamic shared memory is non-trivially large (>1 KB) despite XBLOCK=1. Theoretical occupancy is below 100% and Block Limit Registers is the binding constraint. The reduction axis is 1D and independent per row.

> **Avoid when**: When XBLOCK > 1 is still in use (the 2D [XBLOCK, R0_BLOCK] shape is needed for the outer x-dimension). When the [1, R0_BLOCK] shape is required for broadcasting against other [XBLOCK, R0_BLOCK] tensors in the computation. When registers per thread are already at or below 32.

---

## Cumulative Progress

```
Baseline:  742050 ns  (1.000×)
Iter 2     666910 ns  (1.113×)  — vectorized_loads_and_increase_num_warps
Iter 4     235330 ns  (3.153×)  — persistent_kernel_one_block_per_sm_loop_over_x_rows
Iter 5     192610 ns  (3.853×)  — persistent_kernel_grid_stride_loop_over_rows

Final best: 192610 ns  (3.8526×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,52 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 1960
     r0_numel = 384
     R0_BLOCK: tl.constexpr = 512
-    rnumel = r0_numel
-    RBLOCK: tl.constexpr = R0_BLOCK
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = xindex < xnumel
-    r0_index = tl.arange(0, R0_BLOCK)[None, :]
-    r0_offset = 0
+    pid = tl.program_id(0)
+    num_programs = tl.num_programs(0)
+    r0_index = tl.arange(0, R0_BLOCK)
     r0_mask = r0_index < r0_numel
-    roffset = r0_offset
-    rindex = r0_index
     r0_2 = r0_index
-    x3 = xindex
-    x0 = (xindex % 196)
-    x1 = xindex // 196
-    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
-    tmp1 = tl.load(in_ptr0 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
-    tmp2 = tl.load(in_ptr1 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-    tmp5 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-    tmp6 = tl.load(in_ptr3 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
-    tmp3 = tmp1 + tmp2
-    tmp4 = tmp0 + tmp3
-    tmp7 = tmp5 + tmp6
-    tmp8 = tmp4 + tmp7
-    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
-    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
-    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
-    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
-    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
-    tmp16 = tl.full([1, 1], 384, tl.int32)
-    tmp17 = tmp16.to(tl.float32)
-    tmp18 = (tmp15 / tmp17)
-    tmp19 = tmp9 - tmp18
-    tmp20 = tmp19 * tmp19
-    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
-    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
-    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
-    tl.store(in_out_ptr0 + (r0_2 + 384*x3), tmp8, r0_mask & xmask)
-    tl.store(out_ptr0 + (x3), tmp18, xmask)
-    tl.store(out_ptr1 + (x3), tmp24, xmask)
-
-
+    for x3 in tl.range(pid, xnumel, num_programs):
+        x0 = x3 % 196
+        x1 = x3 // 196
+        tmp0 = tl.load(in_out_ptr0 + (r0_2 + 384*x3), mask=r0_mask, other=0.0)
+        tmp1 = tl.load(in_ptr0 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), mask=r0_mask, other=0.0)
+        tmp2 = tl.load(in_ptr1 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp5 = tl.load(in_ptr2 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp6 = tl.load(in_ptr3 + (r0_2 + 384*x3), mask=r0_mask, other=0.0)
+        tmp3 = tmp1 + tmp2
+        tmp4 = tmp0 + tmp3
+        tmp7 = tmp5 + tmp6
+        tmp8 = tmp4 + tmp7
+        tmp14 = tl.where(r0_mask, tmp8, 0.0)
+        tmp15 = tl.sum(tmp14, 0).to(tl.float32)
+        tmp16 = tl.full([], 384, tl.int32)
+        tmp17 = tmp16.to(tl.float32)
+        tmp18 = (tmp15 / tmp17)
+        tmp19 = tmp8 - tmp18
+        tmp20 = tmp19 * tmp19
+        tmp23 = tl.where(r0_mask, tmp20, 0.0)
+        tmp24 = tl.sum(tmp23, 0).to(tl.float32)
+        tl.store(in_out_ptr0 + (r0_2 + 384*x3), tmp8, mask=r0_mask)
+        tl.store(out_ptr0 + x3, tmp18)
+        tl.store(out_ptr1 + x3, tmp24)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

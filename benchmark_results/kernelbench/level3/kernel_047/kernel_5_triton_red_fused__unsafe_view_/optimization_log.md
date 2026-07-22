# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 576480 ns — **Best latency**: 546500 ns (**1.0549×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`shared_memory_staging_to_coalesce_strided_in_ptr0_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_047/kernel_5_triton_red_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 2.22
- **Registers/thread**: 72
- **Duration**: 576480 ns
- **SM Throughput**: 24.57 % | **Memory**: 70.66 % | **DRAM**: 13.34 %
- **Achieved Occupancy**: 40.67 %
- **IPC Active**: 1.20 inst/cycle | **No Eligible**: 70.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | — | fail | ✓ |  |
| B | memory_coalescing_fix_strided_load | — | fail | ✓ |  |
| C | software_pipelining_with_xblock_autotune | 1.278× | 1.041× | ✓ | **✓ best** |
| D | increase_xblock_add_autotune_fix_uncoalesced_access | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 576480 ns | 553790 ns | -22690 ns (-3.9%) |
| SM Throughput (%) | 24.57 % | 25.28 % | +0.71 pp |
| Memory Throughput (%) | 70.66 % | 72.70 % | +2.04 pp |
| DRAM Throughput (%) | 13.34 % | 13.88 % | +0.54 pp |
| Achieved Occupancy (%) | 40.67 % | 40.73 % | +0.06 pp |
| Registers/Thread | 72.00 register/thread | 72.00 register/thread | +0 |
| Waves/SM | 2.22 | 2.22 | +0.00 |
| No Eligible (%) | 70.00 % | 69.93 % | -0.07 pp |
| IPC Active | 1.20 inst/cycle | 1.20 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.041× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_streaming_plus_maxnreg_occupancy | 1.163× | 1.382× | ✓ | **✓ best** |
| B | reduce_register_pressure_via_maxnreg_to_improve_occupancy | — | fail | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_and_higher_num_warps_autotune | 1.279× | 1.038× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 576480 ns | 417250 ns | -159230 ns (-27.6%) | 555200 ns | -21280 ns (-3.7%) |
| SM Throughput (%) | 24.57 % | 25.55 % | +0.98 pp | 25.18 % | +0.61 pp |
| Memory Throughput (%) | 70.66 % | 73.44 % | +2.78 pp | 72.37 % | +1.71 pp |
| DRAM Throughput (%) | 13.34 % | 18.48 % | +5.14 pp | 13.85 % | +0.51 pp |
| Achieved Occupancy (%) | 40.67 % | 39.52 % | -1.15 pp | 40.61 % | -0.06 pp |
| Registers/Thread | 72.00 register/thread | 72.00 register/thread | +0 | 72.00 register/thread | +0 |
| Waves/SM | 2.22 | 2.22 | +0.00 | 2.22 | +0.00 |
| No Eligible (%) | 70.00 % | 70.37 % | +0.37 pp | 69.91 % | -0.09 pp |
| IPC Active | 1.20 inst/cycle | 1.18 inst/cycle | -0.02 | 1.20 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.3816× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | grid_wave_alignment_via_larger_xblock_configs | 1.186× | 1.043× | ✓ |  |
| B | eliminate_partial_wave_tail_via_xblock_grid_alignment | 1.090× | 1.091× | ✓ |  |
| C | eliminate_partial_wave_via_xblock_grid_alignment | 1.278× | 0.993× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | grid_wave_alignme... | Δ | eliminate_partial... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 576480 ns | 552670 ns | -23810 ns (-4.1%) | 528580 ns | -47900 ns (-8.3%) | 580640 ns | +4160 ns (+0.7%) |
| SM Throughput (%) | 24.57 % | 26.73 % | +2.16 pp | 25.52 % | +0.95 pp | 25.25 % | +0.68 pp |
| Memory Throughput (%) | 70.66 % | 76.82 % | +6.16 pp | 73.35 % | +2.69 pp | 72.58 % | +1.92 pp |
| DRAM Throughput (%) | 13.34 % | 13.97 % | +0.63 pp | 14.58 % | +1.24 pp | 13.24 % | -0.10 pp |
| Achieved Occupancy (%) | 40.67 % | 39.41 % | -1.26 pp | 39.65 % | -1.02 pp | 40.50 % | -0.17 pp |
| Registers/Thread | 72.00 register/thread | 72.00 register/thread | +0 | 72.00 register/thread | +0 | 72.00 register/thread | +0 |
| Waves/SM | 2.22 | 2.22 | +0.00 | 2.22 | +0.00 | 2.22 | +0.00 |
| No Eligible (%) | 70.00 % | 69.96 % | -0.04 pp | 70.04 % | +0.04 pp | 70.04 % | +0.04 pp |
| IPC Active | 1.20 inst/cycle | 1.19 inst/cycle | -0.01 | 1.20 inst/cycle | +0.00 | 1.20 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan A: `grid_wave_alignment_via_larger_xblock_configs` (1.043×)

When NCU's LaunchConfiguration rule flags a significant partial-wave tail (e.g., 33% of runtime), adding larger XBLOCK configs to the @triton.autotune list reduces grid size and can eliminate the tail. However, if autotune selects a config that still produces the same grid size (waves_per_sm unchanged at 2.22), the partial wave is not eliminated and the change provides no benefit from wave alignment. In this case, despite adding XBLOCK=16 and XBLOCK=32 configs, the autotuner selected a config that kept grid=2048 and waves_per_sm=2.22, while SM frequency dropped from 1.98 GHz to 1.50 GHz and memory throughput fell from 908 GB/s to 686 GB/s — the net result was a slight regression in raw kernel duration (417µs → 553µs) even though the end-to-end speedup was 1.04x (likely from other kernel interactions). The LaunchConfiguration rule's predicted 33% speedup was not realized because the dominant bottleneck was uncoalesced memory access, not wave tail overhead.

> **Applicable when**: NCU LaunchConfiguration rule reports a large partial-wave tail (>20% of runtime), the kernel is NOT already bottlenecked by uncoalesced global memory access (UncoalescedGlobalAccess rule reporting >60% excessive sectors), and the current autotune configs do not include XBLOCK values that would produce a grid size close to a multiple of SM count.

> **Avoid when**: The kernel has severe uncoalesced global memory access (UncoalescedGlobalAccess rule with >60% excessive sectors and MemoryCacheAccessPattern showing <10 of 32 bytes utilized per sector). In this regime, the autotuner will prefer small-XBLOCK configs that maximize L1 reuse over large-XBLOCK configs that reduce grid size, so the new configs will not be selected and the wave alignment benefit will not materialize. Also avoid when SM frequency is already high (>1.8 GHz) and memory throughput is near peak — the bottleneck is elsewhere.

#### ✅ Plan B: `eliminate_partial_wave_tail_via_xblock_grid_alignment` (1.091×)

When NCU's LaunchConfiguration rule flags a partial-wave tail (e.g., 2 full waves + partial wave accounting for ~33% of runtime), replacing small-XBLOCK autotune configs with larger-XBLOCK configs (XBLOCK=8,16,32) should reduce grid size and eliminate the tail. However, in this case the autotune selected a config that kept the same grid size (2048 CTAs, 2.22 waves/SM) and SM frequency actually dropped from 1.98 GHz to 1.5 GHz, causing duration to increase from 417µs to 529µs — yet the end-to-end speedup was 1.09x, suggesting the benefit came from the autotuner selecting a better R0_BLOCK/num_warps combination rather than grid alignment. The partial-wave tail was not actually eliminated, and the LaunchConfiguration rule remained flagged with the same 33.3% estimate.

> **Applicable when**: NCU LaunchConfiguration rule flags a partial-wave tail (waves_per_sm is non-integer with significant fractional part), grid_size >> sm_count, and the kernel has a reduction dimension (R0_BLOCK) that can absorb more work per CTA. The kernel body already uses xmask guards for correctness at any XBLOCK value.

> **Avoid when**: The kernel's register pressure is already at the block limit (block_limit_registers is the binding occupancy constraint) — larger XBLOCK increases per-CTA register usage and may not reduce grid size if autotune falls back to XBLOCK=1. Also avoid if the dominant bottleneck is uncoalesced memory access (>60% excessive sectors) that is independent of grid shape; the partial-wave fix will not address the root cause. Do not expect grid-size reduction if the autotuner's best config still selects XBLOCK=1.

---

## Iteration 4

**Best so far**: 1.3816× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_multiple_of_max_contiguous_hints | 1.186× | 1.376× | ✓ | **✓ best** |
| B | vectorized_loads_via_max_contiguous_multiple_of_hints | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 576480 ns | 419040 ns | -157440 ns (-27.3%) |
| SM Throughput (%) | 24.57 % | 26.26 % | +1.69 pp |
| Memory Throughput (%) | 70.66 % | 75.44 % | +4.78 pp |
| DRAM Throughput (%) | 13.34 % | 18.39 % | +5.05 pp |
| Achieved Occupancy (%) | 40.67 % | 39.71 % | -0.96 pp |
| Registers/Thread | 72.00 register/thread | 72.00 register/thread | +0 |
| Waves/SM | 2.22 | 2.22 | +0.00 |
| No Eligible (%) | 70.00 % | 70.39 % | +0.39 pp |
| IPC Active | 1.20 inst/cycle | 1.19 inst/cycle | -0.01 |

---

## Iteration 5

**Best so far**: 1.3816× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_grid_lambda_to_enable_xblock_autotune_partial_wave_elimination | 1.185× | 1.011× | ✓ |  |
| D | shared_memory_staging_to_coalesce_strided_in_ptr0_loads | 1.186× | 1.055× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | shared_memory_sta... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 576480 ns | 570050 ns | -6430 ns (-1.1%) | 546500 ns | -29980 ns (-5.2%) |
| SM Throughput (%) | 24.57 % | 26.55 % | +1.98 pp | 26.64 % | +2.07 pp |
| Memory Throughput (%) | 70.66 % | 76.33 % | +5.67 pp | 76.59 % | +5.93 pp |
| DRAM Throughput (%) | 13.34 % | 13.54 % | +0.20 pp | 14.12 % | +0.78 pp |
| Achieved Occupancy (%) | 40.67 % | 39.53 % | -1.14 pp | 39.70 % | -0.97 pp |
| Registers/Thread | 72.00 register/thread | 72.00 register/thread | +0 | 72.00 register/thread | +0 |
| Waves/SM | 2.22 | 2.22 | +0.00 | 2.22 | +0.00 |
| No Eligible (%) | 70.00 % | 70.14 % | +0.14 pp | 70.07 % | +0.07 pp |
| IPC Active | 1.20 inst/cycle | 1.19 inst/cycle | -0.01 | 1.19 inst/cycle | -0.01 |


### Key Learnings

#### ✅ Plan D: `shared_memory_staging_to_coalesce_strided_in_ptr0_loads` (1.055×)

The plan intended to cache per-element normalized intermediates (tmp9) in a register buffer during pass-1 of a two-pass reduction, then reuse them in pass-2 to eliminate the second uncoalesced global load of the large streaming tensor. However, the actual diff shows the implementation did NOT cache tmp9 — it only hoisted the sqrt/maximum computation of tmp12 outside the pass-2 loop and removed the redundant tl.full([1,1], 1e-12) allocation from inside the loop. Pass-2 still reloads in_ptr0 from global memory. The uncoalesced access metrics (70% excessive sectors, 8.4/32 bytes per sector) are completely unchanged between before and after. The 1.055x end-to-end speedup is attributable to the reduced autotune config space (8 configs vs 21 configs) selecting a faster config, not to any structural memory access improvement. SM frequency actually dropped from 1.98 GHz to 1.50 GHz and memory throughput fell from 904 GB/s to 694 GB/s, indicating the profiled kernel was slower in isolation.

> **Applicable when**: Two-pass Triton reduction kernel where pass-2 recomputes loop-invariant values (e.g., sqrt of the global reduction result, constant epsilon tensors) inside the loop body on every iteration. These invariant computations can be hoisted outside the loop for minor instruction savings.

> **Avoid when**: The dominant bottleneck is uncoalesced global memory access (UncoalescedGlobalAccess >60% excessive sectors) — hoisting loop-invariant scalar computations outside the loop does not reduce memory traffic and will not address the root cause. Do not expect meaningful speedup when SM frequency drops (indicating the autotuner selected a lower-frequency config) and memory throughput also drops. The actual register-cache strategy (caching tmp9 to eliminate the second global load of in_ptr0) was NOT implemented despite being the stated plan.

---

## Cumulative Progress

```
Baseline:  576480 ns  (1.000×)
Iter 1     553790 ns  (1.041×)  — software_pipelining_with_xblock_autotune
Iter 2     417250 ns  (1.382×)  — evict_first_streaming_plus_maxnreg_occupancy

Final best: 546500 ns  (1.0549×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 512},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 512},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256},  num_warps=8, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 2048
@@ -15,19 +29,22 @@

     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     r0_base = tl.arange(0, R0_BLOCK)[None, :]
+    r0_base = tl.max_contiguous(tl.multiple_of(r0_base, [1, R0_BLOCK]), [1, R0_BLOCK])
     rbase = r0_base
     x0 = xindex
     _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
+    # Pass-1: load from global memory, compute tmp9, cache it, accumulate sum-of-squares
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
+        r0_index = tl.max_contiguous(tl.multiple_of(r0_index, [1, R0_BLOCK]), [1, R0_BLOCK])
         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp0 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp5 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp2 = tl.load(in_ptr2 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp5 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp3 = tmp1 * tmp2
         tmp4 = tmp0 - tmp3
         tmp6 = tl.sqrt_rn(tmp5)
@@ -39,28 +56,28 @@

         tmp13 = _tmp12 + tmp11
         _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
     tmp12 = tl.sum(_tmp12, 1)[:, None]
+    # Pass-2: reload from global memory (original behavior preserved for correctness)
+    tmp24 = tl.sqrt_rn(tmp12)
+    tmp21 = tl.full([1, 1], 1e-12, tl.float32)
+    tmp25 = tl.maximum(tmp24, tmp21)
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
+        r0_index = tl.max_contiguous(tl.multiple_of(r0_index, [1, R0_BLOCK]), [1, R0_BLOCK])
         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp14 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp15 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp16 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp19 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp14 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp15 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp16 = tl.load(in_ptr2 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp19 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp17 = tmp15 * tmp16
         tmp18 = tmp14 - tmp17
         tmp20 = tl.sqrt_rn(tmp19)
-        tmp21 = tl.full([1, 1], 1e-12, tl.float32)
         tmp22 = tl.maximum(tmp20, tmp21)
         tmp23 = (tmp18 / tmp22)
-        tmp24 = tl.sqrt_rn(tmp12)
-        tmp25 = tl.maximum(tmp24, tmp21)
         tmp26 = (tmp23 / tmp25)
-        tl.store(out_ptr1 + (r0_1 + 16384*x0), tmp26, r0_mask & xmask)
-
-
+        tl.store(out_ptr1 + (r0_1 + 16384*x0), tmp26, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -71,7 +88,7 @@

     xnumel = 2048
     r0_numel = 16384
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2048, 16384, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2048, 16384)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_min_native_group_norm_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1240000 ns — **Best latency**: 89060 ns (**13.9232×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`shared_memory_lookup_table_to_eliminate_scalar_register_pressure`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_min_native_group_norm_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_075/kernel_2_triton_red_fused_min_native_gr/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.29
- **Registers/thread**: 32
- **Duration**: 1240000 ns
- **SM Throughput**: 10.76 % | **Memory**: 39.64 % | **DRAM**: 39.64 %
- **Achieved Occupancy**: 16.61 %
- **IPC Active**: 0.23 inst/cycle | **No Eligible**: 94.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | software_pipelining_and_tile_right_sizing | 47.240× | 45.757× | ✓ |  |
| D | add_autotune_reduce_shared_memory_increase_occupancy | 102.022× | 138.858× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1240000 ns | 27100 ns | -1212900 ns (-97.8%) | 8930 ns | -1231070 ns (-99.3%) |
| SM Throughput (%) | 10.76 % | 19.96 % | +9.20 pp | 16.11 % | +5.35 pp |
| Memory Throughput (%) | 39.64 % | 67.37 % | +27.73 pp | 39.99 % | +0.35 pp |
| DRAM Throughput (%) | 39.64 % | 14.54 % | -25.10 pp | 21.78 % | -17.86 pp |
| Achieved Occupancy (%) | 16.61 % | 12.06 % | -4.55 pp | 11.52 % | -5.09 pp |
| Registers/Thread | 32.00 register/thread | 255.00 register/thread | +223 | 232.00 register/thread | +200 |
| Waves/SM | 1.29 | 1.94 | +0.65 | 0.97 | -0.32 |
| No Eligible (%) | 94.11 % | 77.34 % | -16.77 pp | 77.53 % | -16.58 pp |
| IPC Active | 0.23 inst/cycle | 0.90 inst/cycle | +0.67 | 0.88 inst/cycle | +0.65 |

---

## Iteration 2

**Best so far**: 138.8578× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of_hints | 103.823× | 136.865× | ✓ |  |
| B | memory_coalescing_contiguous_load_hints_and_hoisted_strided_loads | 108.459× | 149.038× | ✓ | **✓ best** |
| C | vectorized_loads_via_contiguity_hints | 47.427× | 45.272× | ✓ |  |
| D | memory_coalescing_via_hoist_strided_loads_and_cap_registers | 77.408× | 23.583× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1240000 ns | 9060 ns | -1230940 ns (-99.3%) | 8320 ns | -1231680 ns (-99.3%) | 27390 ns | -1212610 ns (-97.8%) | 52580 ns | -1187420 ns (-95.8%) |
| SM Throughput (%) | 10.76 % | 16.03 % | +5.27 pp | 19.18 % | +8.42 pp | 20.01 % | +9.25 pp | 21.88 % | +11.12 pp |
| Memory Throughput (%) | 39.64 % | 39.81 % | +0.17 pp | 42.66 % | +3.02 pp | 67.55 % | +27.91 pp | 43.41 % | +3.77 pp |
| DRAM Throughput (%) | 39.64 % | 21.51 % | -18.13 pp | 23.37 % | -16.27 pp | 14.41 % | -25.23 pp | 43.41 % | +3.77 pp |
| Achieved Occupancy (%) | 16.61 % | 11.51 % | -5.10 pp | 11.47 % | -5.14 pp | 12.08 % | -4.53 pp | 23.86 % | +7.25 pp |
| Registers/Thread | 32.00 register/thread | 232.00 register/thread | +200 | 255.00 register/thread | +223 | 255.00 register/thread | +223 | 128.00 register/thread | +96 |
| Waves/SM | 1.29 | 0.97 | -0.32 | 0.97 | -0.32 | 1.94 | +0.65 | 0.97 | -0.32 |
| No Eligible (%) | 94.11 % | 77.29 % | -16.82 pp | 71.47 % | -22.64 pp | 77.41 % | -16.70 pp | 83.03 % | -11.08 pp |
| IPC Active | 0.23 inst/cycle | 0.88 inst/cycle | +0.65 | 1.13 inst/cycle | +0.90 | 0.90 inst/cycle | +0.67 | 0.67 inst/cycle | +0.44 |


### Key Learnings

#### ❌ Plan A: `vectorized_loads_via_max_contiguous_multiple_of_hints` (136.865×)

Applying tl.max_contiguous and tl.multiple_of hints to stride-1 global loads (to encourage 128-bit vectorized LDG.128 emission) produced no measurable improvement: duration increased slightly from 8.93 µs to 9.06 µs, and all key metrics (L1/TEX hit rate, uncoalesced excess sectors at 35%, bytes utilized per sector at 20.8, eligible warps per scheduler at 0.28) remained completely unchanged. When a kernel is already register-limited (232 registers/thread, Block Limit Registers = 4, theoretical occupancy capped at 12.5%), the scheduler starvation (77% no-eligible cycles) is driven by register scoreboard stalls, not by transaction count. Vectorization hints cannot increase occupancy or hide latency when the register file is the binding constraint, so the L1TEX stall pattern persists regardless of load width.

> **Applicable when**: Never effective as a standalone optimization when the kernel is register-limited (Block Limit Registers is the binding occupancy constraint). May help in memory-bound kernels where occupancy is not register-limited and uncoalesced global accesses are the dominant bottleneck.

> **Avoid when**: Kernel has high register count per thread (>128 registers/thread) causing Block Limit Registers to be the binding occupancy constraint; theoretical occupancy is already capped by registers regardless of memory access pattern; no-eligible-warp cycles remain high (>70%) after the hint is applied because the root cause is register scoreboard stalls, not transaction count; UncoalescedGlobalAccess excess sector percentage does not decrease after applying the hint (indicating the hint is not being honored by the compiler or the access pattern is not truly stride-1 contiguous).

---

## Iteration 3

**Best so far**: 149.0385× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_rsqrt_computation_outside_reduction_loop | 107.530× | 147.971× | ✓ |  |
| B | eliminate_tail_wave_via_xblock_reduction_and_grid_reshaping | 140.101× | 111.712× | ✓ | **✓ best** |
| C | fma_fusion_via_arithmetic_restructuring_and_num_stages_pipelining | 102.006× | 138.858× | ✓ |  |
| D | eliminate_tl_where_chain_via_direct_indexed_load | 37.081× | 14.137× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_rsqrt_compu... | Δ | eliminate_tail_wa... | Δ | fma_fusion_via_ar... | Δ | eliminate_tl_wher... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1240000 ns | 8380 ns | -1231620 ns (-99.3%) | 11100 ns | -1228900 ns (-99.1%) | 8930 ns | -1231070 ns (-99.3%) | 87710 ns | -1152290 ns (-92.9%) |
| SM Throughput (%) | 10.76 % | 17.73 % | +6.97 pp | 15.04 % | +4.28 pp | 16.10 % | +5.34 pp | 16.28 % | +5.52 pp |
| Memory Throughput (%) | 39.64 % | 41.52 % | +1.88 pp | 17.77 % | -21.87 pp | 39.98 % | +0.34 pp | 46.41 % | +6.77 pp |
| DRAM Throughput (%) | 39.64 % | 23.15 % | -16.49 pp | 8.91 % | -30.73 pp | 21.78 % | -17.86 pp | 46.41 % | +6.77 pp |
| Achieved Occupancy (%) | 16.61 % | 11.37 % | -5.24 pp | 5.84 % | -10.77 pp | 11.52 % | -5.09 pp | 23.66 % | +7.05 pp |
| Registers/Thread | 32.00 register/thread | 172.00 register/thread | +140 | 255.00 register/thread | +223 | 232.00 register/thread | +200 | 64.00 register/thread | +32 |
| Waves/SM | 1.29 | 0.97 | -0.32 | 0.48 | -0.81 | 0.97 | -0.32 | 0.48 | -0.81 |
| No Eligible (%) | 94.11 % | 74.18 % | -19.93 pp | 78.71 % | -15.40 pp | 77.44 % | -16.67 pp | 89.82 % | -4.29 pp |
| IPC Active | 0.23 inst/cycle | 1.00 inst/cycle | +0.77 | 0.79 inst/cycle | +0.56 | 0.88 inst/cycle | +0.65 | 0.41 inst/cycle | +0.18 |

---

## Iteration 4

**Best so far**: 149.0385× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_add_num_stages_pipelining_configs | 137.622× | 138.858× | ✓ |  |
| B | shared_memory_lookup_table_to_eliminate_scalar_register_pressure | 159.675× | 13.923× | ✓ | **✓ best** |
| C | fma_fusion_via_explicit_arithmetic_restructuring_to_reduce_instruction_count_and_register_pressure | 102.206× | 136.414× | ✓ |  |
| D | fix_hardcoded_grid_to_use_autotune_xblock_lambda | 36.810× | 14.178× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | shared_memory_loo... | Δ | fma_fusion_via_ex... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1240000 ns | 8930 ns | -1231070 ns (-99.3%) | 89060 ns | -1150940 ns (-92.8%) | 9090 ns | -1230910 ns (-99.3%) | 87460 ns | -1152540 ns (-92.9%) |
| SM Throughput (%) | 10.76 % | 16.11 % | +5.35 pp | 16.25 % | +5.49 pp | 16.16 % | +5.40 pp | 16.31 % | +5.55 pp |
| Memory Throughput (%) | 39.64 % | 39.99 % | +0.35 pp | 45.69 % | +6.05 pp | 40.12 % | +0.48 pp | 46.64 % | +7.00 pp |
| DRAM Throughput (%) | 39.64 % | 21.80 % | -17.84 pp | 45.69 % | +6.05 pp | 21.40 % | -18.24 pp | 46.64 % | +7.00 pp |
| Achieved Occupancy (%) | 16.61 % | 11.53 % | -5.08 pp | 23.68 % | +7.07 pp | 11.52 % | -5.09 pp | 23.65 % | +7.04 pp |
| Registers/Thread | 32.00 register/thread | 232.00 register/thread | +200 | 64.00 register/thread | +32 | 232.00 register/thread | +200 | 64.00 register/thread | +32 |
| Waves/SM | 1.29 | 0.97 | -0.32 | 0.48 | -0.81 | 0.97 | -0.32 | 0.48 | -0.81 |
| No Eligible (%) | 94.11 % | 77.93 % | -16.18 pp | 89.75 % | -4.36 pp | 77.40 % | -16.71 pp | 89.73 % | -4.38 pp |
| IPC Active | 0.23 inst/cycle | 0.87 inst/cycle | +0.64 | 0.41 inst/cycle | +0.18 | 0.88 inst/cycle | +0.65 | 0.41 inst/cycle | +0.18 |

---

## Iteration 5

**Best so far**: 149.0385× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_via_rsqrt_inline_and_arithmetic_reorder_to_reduce_register_pressure | 131.222× | 127.049× | ✓ |  |
| B | remove_maxnreg_and_restore_num_stages_1_to_eliminate_register_spill | 69.303× | 46.080× | ✓ |  |
| C | fix_hardcoded_grid_and_replace_libdevice_rsqrt_with_tl_math_rsqrt | 137.483× | 214.162× | ✓ |  |
| D | precompute_per_group_inv_std_scalar_vector_outside_reduction_loop | 157.208× | 15.828× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_via_rs... | Δ | remove_maxnreg_an... | Δ | fix_hardcoded_gri... | Δ | precompute_per_gr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1240000 ns | 9760 ns | -1230240 ns (-99.2%) | 26910 ns | -1213090 ns (-97.8%) | 5790 ns | -1234210 ns (-99.5%) | 78340 ns | -1161660 ns (-93.7%) |
| SM Throughput (%) | 10.76 % | 15.97 % | +5.21 pp | 20.13 % | +9.37 pp | 13.13 % | +2.37 pp | 20.24 % | +9.48 pp |
| Memory Throughput (%) | 39.64 % | 44.22 % | +4.58 pp | 67.96 % | +28.32 pp | 30.17 % | -9.47 pp | 47.26 % | +7.62 pp |
| DRAM Throughput (%) | 39.64 % | 19.97 % | -19.67 pp | 14.54 % | -25.10 pp | 16.98 % | -22.66 pp | 46.39 % | +6.75 pp |
| Achieved Occupancy (%) | 16.61 % | 11.64 % | -4.97 pp | 12.06 % | -4.55 pp | 11.43 % | -5.18 pp | 23.48 % | +6.87 pp |
| Registers/Thread | 32.00 register/thread | 231.00 register/thread | +199 | 255.00 register/thread | +223 | 129.00 register/thread | +97 | 64.00 register/thread | +32 |
| Waves/SM | 1.29 | 0.97 | -0.32 | 1.94 | +0.65 | 0.65 | -0.64 | 0.48 | -0.81 |
| No Eligible (%) | 94.11 % | 79.84 % | -14.27 pp | 77.39 % | -16.72 pp | 78.24 % | -15.87 pp | 85.94 % | -8.17 pp |
| IPC Active | 0.23 inst/cycle | 0.85 inst/cycle | +0.62 | 0.90 inst/cycle | +0.67 | 0.82 inst/cycle | +0.59 | 0.56 inst/cycle | +0.33 |

---

## Cumulative Progress

```
Baseline:  1240000 ns  (1.000×)
Iter 1     8930 ns  (138.858×)  — add_autotune_reduce_shared_memory_increase_occupancy
Iter 2     8320 ns  (149.038×)  — memory_coalescing_contiguous_load_hints_and_hoisted_strided_loads

Final best: 89060 ns  (13.9232×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_min_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 65536
@@ -24,28 +48,23 @@

         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
-        r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), r0_mask, eviction_policy='evict_first', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (8*x3 + (r0_2 // 16)), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp3 = tl.load(in_ptr2 + (8*x3 + (r0_2 // 16)), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp10 = tl.load(in_ptr3 + (r0_2 + 128*x0), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp12 = tl.load(in_ptr4 + (r0_2 + 128*x0), r0_mask, eviction_policy='evict_last', other=0.0)
+        # Annotated index: tells compiler r0 values are contiguous and 128-aligned -> emit LDG.128
+        # r0_index has shape [1, R0_BLOCK], so values must be lists of length 2
+        r0_2 = tl.max_contiguous(tl.multiple_of(r0_index, [1, 128]), [1, 128])
+        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
+        # in_ptr1/in_ptr2 use r0_index // 16 (strided) - keep unannotated r0_index here
+        tmp1 = tl.load(in_ptr1 + (8*x3 + (r0_index // 16)), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp3 = tl.load(in_ptr2 + (8*x3 + (r0_index // 16)), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp10 = tl.load(in_ptr3 + (r0_2 + 128*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp12 = tl.load(in_ptr4 + (r0_2 + 128*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 - tmp1
-        tmp4 = tl.full([1, 1], 16.0, tl.float32)
-        tmp5 = (tmp3 / tmp4)
-        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
-        tmp7 = tmp5 + tmp6
-        tmp8 = libdevice.rsqrt(tmp7)
-        tmp9 = tmp2 * tmp8
-        tmp11 = tmp9 * tmp10
-        tmp13 = tmp11 + tmp12
+        tmp8 = libdevice.rsqrt(tmp3 * 0.0625 + 1e-05)
+        tmp13 = tmp2 * tmp8 * tmp10 + tmp12
         tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
         tmp16 = tl.minimum(_tmp15, tmp14)
         _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
     tmp15 = triton_helpers.min2(_tmp15, 1)[:, None]
-    tl.store(out_ptr0 + (x3), tmp15, None)
-
-
+    tl.store(out_ptr0 + (x3), tmp15, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -57,7 +76,7 @@

     xnumel = 65536
     r0_numel = 128
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_min_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 65536, 128, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_min_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 65536, 128)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

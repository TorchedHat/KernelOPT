# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_native_group_norm_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 388640 ns — **Best latency**: 387200 ns (**1.0037×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`revert_to_small_xblock_with_targeted_autotune_over_small_tiles`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_native_group_norm_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_061/kernel_2_triton_poi_fused_native_group_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (628864,1,1), Waves/SM: 297.76
- **Registers/thread**: 18
- **Duration**: 388640 ns
- **SM Throughput**: 52.31 % | **Memory**: 32.76 % | **DRAM**: 32.76 %
- **Achieved Occupancy**: 44.72 %
- **IPC Active**: 1.48 inst/cycle | **No Eligible**: 62.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.780× | 0.786× | ✓ |  |
| B | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.782× | 0.776× | ✓ |  |
| C | vectorized_loads_stores_via_larger_xblock_and_autotune | 0.784× | 0.787× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_latency_hiding | 1.000× | 0.204× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_w... | Δ | vectorized_loads_... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 388640 ns | 494300 ns | +105660 ns (+27.2%) | 500930 ns | +112290 ns (+28.9%) | 494080 ns | +105440 ns (+27.1%) | 1900000 ns | +1511360 ns (+388.9%) |
| SM Throughput (%) | 52.31 % | 80.62 % | +28.31 pp | 81.09 % | +28.78 pp | 80.65 % | +28.34 pp | 82.48 % | +30.17 pp |
| Memory Throughput (%) | 32.76 % | 53.49 % | +20.73 pp | 53.81 % | +21.05 pp | 53.52 % | +20.76 pp | 49.84 % | +17.08 pp |
| DRAM Throughput (%) | 32.76 % | 25.58 % | -7.18 pp | 25.25 % | -7.51 pp | 25.60 % | -7.16 pp | 6.66 % | -26.10 pp |
| Achieved Occupancy (%) | 44.72 % | 85.41 % | +40.69 pp | 85.42 % | +40.70 pp | 85.31 % | +40.59 pp | 78.89 % | +34.17 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 297.76 | 297.76 | +0.00 | 297.76 | +0.00 | 297.76 | +0.00 | 1191.03 | +893.27 |
| No Eligible (%) | 62.88 % | 18.44 % | -44.44 pp | 18.64 % | -44.24 pp | 18.76 % | -44.12 pp | 17.50 % | -45.38 pp |
| IPC Active | 1.48 inst/cycle | 3.25 inst/cycle | +1.77 | 3.25 inst/cycle | +1.77 | 3.25 inst/cycle | +1.77 | 3.30 inst/cycle | +1.82 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.786×)

Adding a @triton.autotune decorator with larger XBLOCK configs (512–2048) caused registers_per_thread to jump from 18 to 32 and executed instructions to increase nearly 3x (113M → 311M), resulting in a 27% wall-clock regression despite occupancy improving from 44.7% to 85.4%. The autotune selected a config that dramatically increased register pressure and instruction count, overwhelming any latency-hiding benefit from more warps per CTA. When the original kernel is already register-light and the bottleneck is L1TEX stall latency, blindly enlarging XBLOCK via autotune can cause the compiler to emit far more instructions per element (e.g., due to predication overhead from xmask, loop unrolling, or register spilling), negating the scheduling benefit.

> **Applicable when**: Never apply this specific pattern (autotune XBLOCK increase + xmask addition) when the kernel is already register-light (<20 registers/thread) and the instruction count is low; the autotune overhead and mask predication cost can dominate.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK configs when: (1) the kernel has very low register usage that may spike with larger tiles, (2) xnumel is exactly divisible by the current XBLOCK so xmask=True is a no-op and adding a real mask adds predication overhead, (3) the kernel body contains index arithmetic (e.g., x6 // 16, xindex // 39304) that gets replicated per unrolled iteration, causing instruction count to multiply with XBLOCK.

#### ❌ Plan B: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.786×)

Eliminating a duplicate floor-division in index arithmetic and adding num_stages=2 autotune configs did not reduce register pressure or instruction count: registers/thread stayed at 32 and executed instructions exploded 8.8x (35M → 310M), causing a 27% wall-clock regression (388 µs → 495 µs). The autotune selected a larger XBLOCK config (128 vs original 32), and the instruction explosion from wider tiling overwhelmed any benefit from the index simplification or software pipelining. This confirms that for this kernel, the original small block size (32) with low instruction count is superior to any larger XBLOCK configuration, regardless of index arithmetic cleanup or num_stages tuning.

> **Applicable when**: Never: this pattern (autotune XBLOCK increase for this group-norm pointwise kernel, even with index simplification and num_stages=2) consistently regresses across all iterations.

> **Avoid when**: Do not apply autotune XBLOCK increase (even with correct lambda grid, index arithmetic simplification, or num_stages=2) when: (1) the original kernel uses a small block size (≤32) with very low instruction count; (2) the instruction count grows super-linearly with XBLOCK (e.g., 8.8x for 4x XBLOCK increase), indicating the compiler emits more instructions per element for larger tiles; (3) source-level index simplification (eliminating duplicate floor-divs) does not reduce the compiler's register allocation below the block_limit_registers threshold. The original 35M-instruction, 20-register kernel at XBLOCK=32 is the performance optimum for this kernel shape.

#### ❌ Plan D: `autotune_xblock_and_num_warps_for_latency_hiding` (0.204×)

Adding @triton.autotune with larger XBLOCK values (512 selected) on a pointwise kernel where xnumel is exactly divisible by all tile sizes caused a ~11x increase in executed instructions (113M → 1238M) and a 4.9x wall-time regression (388 µs → 1900 µs). The root cause is that the autotune decorator's grid lambda was not updated: the call site still used a fixed grid=(cdiv(xnumel, 128),) instead of a dynamic lambda grid, so with XBLOCK=512 selected by autotune, each CTA processed 512 elements but the grid still launched 628,864 CTAs — causing each element to be processed ~4x redundantly. Although occupancy improved (44.7% → 78.9%) and compute throughput rose (52% → 82%), the redundant work overwhelmed any latency-hiding benefit. When adding autotune to a Triton kernel, the call-site grid must be changed to a lambda: `grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` — a static grid is incompatible with autotuned XBLOCK.

> **Applicable when**: Never apply this pattern as implemented. The autotune + static grid combination is always wrong.

> **Avoid when**: Do not use a static grid expression when @triton.autotune is tuning XBLOCK. The grid must be a lambda that reads meta['XBLOCK']. If the call site cannot be changed to a lambda grid, do not add autotune over XBLOCK.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | arithmetic_strength_reduction_and_fma_fusion_in_group_norm_epilogue | 0.783× | 0.787× | ✓ |  |
| B | reduce_num_warps_to_improve_l1_broadcast_cache_reuse | 1.001× | 1.002× | ✓ | **✓ best** |
| D | fix_broken_autotune_grid_and_add_software_pipelining | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | arithmetic_streng... | Δ | reduce_num_warps_... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 388640 ns | 493950 ns | +105310 ns (+27.1%) | 387970 ns | -670 ns (-0.2%) | 388480 ns | -160 ns (-0.0%) |
| SM Throughput (%) | 52.31 % | 80.92 % | +28.61 pp | 25.63 % | -26.68 pp | 52.34 % | +0.03 pp |
| Memory Throughput (%) | 32.76 % | 53.71 % | +20.95 pp | 32.82 % | +0.06 pp | 32.77 % | +0.01 pp |
| DRAM Throughput (%) | 32.76 % | 25.61 % | -7.15 pp | 32.82 % | +0.06 pp | 32.77 % | +0.01 pp |
| Achieved Occupancy (%) | 44.72 % | 85.40 % | +40.68 pp | 14.14 % | -30.58 pp | 45.17 % | +0.45 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 31.00 register/thread | +13 | 20.00 register/thread | +2 |
| Waves/SM | 297.76 | 297.76 | +0.00 | 148.88 | -148.88 | 297.76 | +0.00 |
| No Eligible (%) | 62.88 % | 18.50 % | -44.38 pp | 70.15 % | +7.27 pp | 53.84 % | -9.04 pp |
| IPC Active | 1.48 inst/cycle | 3.25 inst/cycle | +1.77 | 1.07 inst/cycle | -0.41 | 1.85 inst/cycle | +0.37 |


### Key Learnings

#### ❌ Plan A: `arithmetic_strength_reduction_and_fma_fusion_in_group_norm_epilogue` (0.787×)

Replacing a scalar FDIV with a compile-time reciprocal FMUL and fusing a multiply-add into a single FMA expression in a Triton group-norm epilogue produced zero measurable improvement: executed instruction count was identical (310,658,816 before and after), duration changed by only 0.13 µs (494.08 → 493.95 µs), and all throughput metrics were statistically unchanged. The FPInstructions NCU rule flagged a 5.3% theoretical opportunity, but the actual PTX emitted by the Triton compiler backend already fused or optimized these operations, so the source-level rewrite had no effect. When a kernel is already compute-bound at >80% SM throughput with high instruction-per-cycle rates (IPC ~3.25), arithmetic strength reduction on a handful of scalar instructions in the epilogue cannot move the needle because those instructions are not on the critical path relative to the dominant memory/compute pipeline.

> **Applicable when**: Never: this specific rewrite (FDIV→FMUL reciprocal + FMA fusion in Triton epilogue) produced no measurable benefit even when NCU FPInstructions rule flagged it.

> **Avoid when**: Do not apply arithmetic strength reduction (FDIV→FMUL, FMA fusion) to scalar epilogue instructions in a Triton kernel when: (1) the kernel is already compute-bound at >80% SM throughput with IPC ~3.25, (2) the executed instruction count does not change after the rewrite (indicating the compiler already optimized these ops), (3) the scalar ops being rewritten represent a tiny fraction of total instructions (e.g., 8 instructions out of 310M). The NCU FPInstructions rule may flag a theoretical opportunity that the compiler backend has already addressed at the PTX level.

---

## Iteration 3

**Best so far**: 1.0017× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_call_site_grid_lambda_and_tune_num_warps_for_latency_hiding | 1.003× | 1.002× | ✓ | **✓ best** |
| C | fix_autotune_grid_lambda_to_enable_larger_xblock_tiles | 0.782× | 0.775× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_call_site_gri... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 388640 ns | 387900 ns | -740 ns (-0.2%) | 501150 ns | +112510 ns (+28.9%) |
| SM Throughput (%) | 52.31 % | 13.14 % | -39.17 pp | 80.86 % | +28.55 pp |
| Memory Throughput (%) | 32.76 % | 7.48 % | -25.28 pp | 53.64 % | +20.88 pp |
| DRAM Throughput (%) | 32.76 % | 7.48 % | -25.28 pp | 25.25 % | -7.51 pp |
| Achieved Occupancy (%) | 44.72 % | 10.37 % | -34.35 pp | 85.47 % | +40.75 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 32.00 register/thread | +14 |
| Waves/SM | 297.76 | 148.88 | -148.88 | 297.76 | +0.00 |
| No Eligible (%) | 62.88 % | 86.56 % | +23.68 pp | 18.78 % | -44.10 pp |
| IPC Active | 1.48 inst/cycle | 0.48 inst/cycle | -1.00 | 3.25 inst/cycle | +1.77 |

---

## Iteration 4

**Best so far**: 1.0019× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_broken_autotune_and_fix_xblock_to_match_static_grid | 1.000× | 1.000× | ✓ |  |
| B | fix_static_grid_xblock_mismatch_and_increase_num_warps_for_latency_hiding | 1.000× | 1.000× | ✓ |  |
| C | reduce_register_pressure_via_index_simplification_to_improve_occupancy | 0.783× | 0.786× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_broken_aut... | Δ | fix_static_grid_x... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 388640 ns | 388610 ns | -30 ns (-0.0%) | 388670 ns | +30 ns (+0.0%) | 494690 ns | +106050 ns (+27.3%) |
| SM Throughput (%) | 52.31 % | 52.34 % | +0.03 pp | 52.26 % | -0.05 pp | 81.19 % | +28.88 pp |
| Memory Throughput (%) | 32.76 % | 32.75 % | -0.01 pp | 32.75 % | -0.01 pp | 53.86 % | +21.10 pp |
| DRAM Throughput (%) | 32.76 % | 32.75 % | -0.01 pp | 32.75 % | -0.01 pp | 25.57 % | -7.19 pp |
| Achieved Occupancy (%) | 44.72 % | 45.21 % | +0.49 pp | 45.19 % | +0.47 pp | 80.70 % | +35.98 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 20.00 register/thread | +2 | 32.00 register/thread | +14 |
| Waves/SM | 297.76 | 297.76 | +0.00 | 297.76 | +0.00 | 297.76 | +0.00 |
| No Eligible (%) | 62.88 % | 53.76 % | -9.12 pp | 53.84 % | -9.04 pp | 18.38 % | -44.50 pp |
| IPC Active | 1.48 inst/cycle | 1.85 inst/cycle | +0.37 | 1.85 inst/cycle | +0.37 | 3.25 inst/cycle | +1.77 |

---

## Iteration 5

**Best so far**: 1.0019× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_load_hints_and_eviction_policy_for_l1tex_latency_reduction | 1.000× | 1.001× | ✓ |  |
| B | revert_to_small_xblock_with_targeted_autotune_over_small_tiles | 1.003× | 1.004× | ✓ | **✓ best** |
| C | evict_first_on_streaming_main_tensor_to_preserve_l2_for_scalar_broadcasts | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_load_h... | Δ | revert_to_small_x... | Δ | evict_first_on_st... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 388640 ns | 388320 ns | -320 ns (-0.1%) | 387200 ns | -1440 ns (-0.4%) | 388580 ns | -60 ns (-0.0%) |
| SM Throughput (%) | 52.31 % | 52.44 % | +0.13 pp | 13.13 % | -39.18 pp | 52.36 % | +0.05 pp |
| Memory Throughput (%) | 32.76 % | 32.42 % | -0.34 pp | 7.50 % | -25.26 pp | 32.40 % | -0.36 pp |
| DRAM Throughput (%) | 32.76 % | 32.42 % | -0.34 pp | 7.50 % | -25.26 pp | 32.40 % | -0.36 pp |
| Achieved Occupancy (%) | 44.72 % | 45.92 % | +1.20 pp | 10.34 % | -34.38 pp | 45.97 % | +1.25 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 20.00 register/thread | +2 | 20.00 register/thread | +2 |
| Waves/SM | 297.76 | 297.76 | +0.00 | 148.88 | -148.88 | 297.76 | +0.00 |
| No Eligible (%) | 62.88 % | 49.80 % | -13.08 pp | 86.55 % | +23.67 pp | 49.74 % | -13.14 pp |
| IPC Active | 1.48 inst/cycle | 2.01 inst/cycle | +0.53 | 0.49 inst/cycle | -0.99 | 2.01 inst/cycle | +0.53 |

---

## Cumulative Progress

```
Baseline:  388640 ns  (1.000×)
Iter 2     387970 ns  (1.002×)  — reduce_num_warps_to_improve_l1_broadcast_cache_reuse
Iter 3     387900 ns  (1.002×)  — fix_call_site_grid_lambda_and_tune_num_warps_for_latency_hiding
Iter 5     387200 ns  (1.004×)  — revert_to_small_xblock_with_targeted_autotune_over_small_tiles

Final best: 387200 ns  (1.0037×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,34 +5,38 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32},  num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_native_group_norm_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 80494592
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x4 = xindex
-    x6 = xindex // 39304
-    x2 = ((xindex // 39304) % 128)
-    tmp0 = tl.load(in_ptr0 + (x4), None)
-    tmp3 = tl.load(in_ptr1 + (x6 // 16), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x6 // 16), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
-    tmp1 = tl.full([1], 0, tl.int32)
-    tmp2 = tl.maximum(tmp1, tmp0)
-    tmp4 = tmp2 - tmp3
-    tmp6 = tl.full([1], 628864.0, tl.float32)
-    tmp7 = (tmp5 / tmp6)
-    tmp8 = tl.full([1], 1e-05, tl.float32)
-    tmp9 = tmp7 + tmp8
-    tmp10 = libdevice.rsqrt(tmp9)
-    tmp11 = tmp4 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tl.store(out_ptr0 + (x4), tmp15, None)
-
-
+    xmask = xindex < xnumel
+    x_chan  = xindex // 39304
+    x_group = x_chan // 16
+    x2      = x_chan % 128
+    tmp0     = tl.load(in_ptr0 + xindex, xmask)
+    tmp_relu = tl.maximum(tl.zeros([XBLOCK], dtype=tl.float32), tmp0)
+    tmp_mean     = tl.load(in_ptr1 + x_group, xmask, eviction_policy='evict_last')
+    tmp_centered = tmp_relu - tmp_mean
+    tmp_var     = tl.load(in_ptr2 + x_group, xmask, eviction_policy='evict_last')
+    tmp_inv_std = libdevice.rsqrt(tmp_var * 1.5908420085906982e-06 + 1e-05)
+    tmp_weight = tl.load(in_ptr3 + x2, xmask, eviction_policy='evict_last')
+    tmp_bias   = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
+    tmp_out    = tmp_centered * tmp_inv_std * tmp_weight + tmp_bias
+    tl.store(out_ptr0 + xindex, tmp_out, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +47,7 @@

     arg_5 = inputs[5]
     xnumel = 80494592
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_native_group_norm_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 80494592, XBLOCK=128)
+    triton_poi_fused_native_group_norm_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 80494592)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

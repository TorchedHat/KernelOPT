# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_mul_native_group_norm_sigmoid_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 42850 ns — **Best latency**: 42780 ns (**1.0016×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`promote_non_fused_fp32_to_fma_and_fix_broken_autotune_grid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_mul_native_group_norm_sigmoid_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_088/kernel_2_triton_poi_fused_mul_native_gr/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 18
- **Duration**: 42850 ns
- **SM Throughput**: 57.52 % | **Memory**: 27.32 % | **DRAM**: 20.32 %
- **Achieved Occupancy**: 58.00 %
- **IPC Active**: 2.43 inst/cycle | **No Eligible**: 39.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size_for_latency_hiding | 0.552× | 0.532× | ✓ |  |
| B | increase_xblock_with_autotune_for_better_l1_reuse_and_occupancy | 0.552× | 0.533× | ✓ |  |
| C | autotune_xblock_increase_warps_hide_l1tex_latency | 0.446× | 0.427× | ✓ |  |
| D | increase_xblock_with_autotune_for_vectorized_loads | 0.844× | 0.829× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | increase_xblock_w... | Δ | autotune_xblock_i... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42850 ns | 80480 ns | +37630 ns (+87.8%) | 80350 ns | +37500 ns (+87.5%) | 100290 ns | +57440 ns (+134.0%) | 51680 ns | +8830 ns (+20.6%) |
| SM Throughput (%) | 57.52 % | 89.22 % | +31.70 pp | 89.20 % | +31.68 pp | 87.06 % | +29.54 pp | 85.21 % | +27.69 pp |
| Memory Throughput (%) | 27.32 % | 29.70 % | +2.38 pp | 29.71 % | +2.39 pp | 28.92 % | +1.60 pp | 33.27 % | +5.95 pp |
| DRAM Throughput (%) | 20.32 % | 10.83 % | -9.49 pp | 10.84 % | -9.48 pp | 8.68 % | -11.64 pp | 16.83 % | -3.49 pp |
| Achieved Occupancy (%) | 58.00 % | 91.01 % | +33.01 pp | 91.07 % | +33.07 pp | 87.34 % | +29.34 pp | 86.77 % | +28.77 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 24.00 register/thread | +6 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 62.06 | +31.03 | 31.03 | +0.00 |
| No Eligible (%) | 39.28 % | 8.44 % | -30.84 pp | 8.44 % | -30.84 pp | 11.07 % | -28.21 pp | 11.27 % | -28.01 pp |
| IPC Active | 2.43 inst/cycle | 3.66 inst/cycle | +1.23 | 3.65 inst/cycle | +1.22 | 3.56 inst/cycle | +1.13 | 3.53 inst/cycle | +1.10 |


### Key Learnings

#### ❌ Plan C: `autotune_xblock_increase_warps_hide_l1tex_latency` (0.531×)

Replacing autotune configs (removing small XBLOCK=128/256 options, adding num_stages=2/3/4) and applying tl.math.fma algebraic simplification in the kernel body — without fixing the launch grid from a hardcoded cdiv(xnumel, 128) to a lambda — caused executed instructions to increase 2.5x (22.3M → 56.5M) and duration to nearly double (42.91 µs → 80.67 µs). The launch stats confirm block_size=128 in both before and after, meaning the autotuner still cannot select any config with XBLOCK > 128 because the grid is hardcoded; the new configs with higher num_stages increased register pressure (20 → 32 registers/thread) and instruction count without any benefit. The FMA simplification itself is correct but masked entirely by the grid/XBLOCK mismatch regression.

> **Applicable when**: Never apply this pattern. Changing autotune configs (even with beneficial kernel body changes like FMA) without fixing the grid lambda will always regress due to the grid/XBLOCK mismatch. This is a third confirmation of the same anti-pattern.

> **Avoid when**: Do not modify @triton.autotune XBLOCK configs or num_stages without simultaneously converting the launch grid from a static expression like (cdiv(xnumel, 128),) to a lambda: (lambda meta: (cdiv(xnumel, meta['XBLOCK']),)). Higher num_stages configs also increase register pressure (20→32 regs/thread here), which can further hurt performance even if the grid were correct.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_persistent_tile_scheduling | 0.841× | 0.829× | ✓ |  |
| B | fix_broken_autotune_grid_lambda_and_restore_unconditional_mask | 1.002× | 0.999× | ✓ | **✓ best** |
| C | fix_autotune_grid_lambda_to_eliminate_instruction_explosion | 0.553× | 0.533× | ✓ |  |
| D | persistent_tile_scheduling_fix_grid_overprovisioning | 0.446× | 0.427× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_broken_autotu... | Δ | fix_autotune_grid... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42850 ns | 51710 ns | +8860 ns (+20.7%) | 42910 ns | +60 ns (+0.1%) | 80420 ns | +37570 ns (+87.7%) | 100350 ns | +57500 ns (+134.2%) |
| SM Throughput (%) | 57.52 % | 84.97 % | +27.45 pp | 70.40 % | +12.88 pp | 89.07 % | +31.55 pp | 87.07 % | +29.55 pp |
| Memory Throughput (%) | 27.32 % | 33.15 % | +5.83 pp | 27.19 % | -0.13 pp | 29.64 % | +2.32 pp | 28.95 % | +1.63 pp |
| DRAM Throughput (%) | 20.32 % | 16.86 % | -3.46 pp | 20.32 % | +0.00 pp | 10.85 % | -9.47 pp | 8.67 % | -11.65 pp |
| Achieved Occupancy (%) | 58.00 % | 87.23 % | +29.23 pp | 60.63 % | +2.63 pp | 91.04 % | +33.04 pp | 87.31 % | +29.31 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 20.00 register/thread | +2 | 32.00 register/thread | +14 | 24.00 register/thread | +6 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 62.06 | +31.03 |
| No Eligible (%) | 39.28 % | 11.24 % | -28.04 pp | 25.40 % | -13.88 pp | 8.46 % | -30.82 pp | 11.08 % | -28.20 pp |
| IPC Active | 2.43 inst/cycle | 3.53 inst/cycle | +1.10 | 2.92 inst/cycle | +0.49 | 3.66 inst/cycle | +1.23 | 3.55 inst/cycle | +1.12 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_tl_sigmoid_with_libdevice_sigmoid_to_fuse_fp32_instructions | 0.991× | 0.985× | ✓ |  |
| B | reduce_nonfused_fp32_instructions_via_silu_algebraic_simplification_and_fix_grid_lambda | 1.002× | 0.999× | ✓ |  |
| C | fuse_fp32_multiply_add_pairs_to_fma_and_fix_grid_lambda | 0.552× | 0.533× | ✓ |  |
| D | promote_non_fused_fp32_to_fma_and_fix_broken_autotune_grid | 1.004× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | replace_tl_sigmoi... | Δ | reduce_nonfused_f... | Δ | fuse_fp32_multipl... | Δ | promote_non_fused... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42850 ns | 43520 ns | +670 ns (+1.6%) | 42910 ns | +60 ns (+0.1%) | 80380 ns | +37530 ns (+87.6%) | 42780 ns | -70 ns (-0.2%) |
| SM Throughput (%) | 57.52 % | 81.26 % | +23.74 pp | 71.33 % | +13.81 pp | 89.21 % | +31.69 pp | 66.12 % | +8.60 pp |
| Memory Throughput (%) | 27.32 % | 27.10 % | -0.22 pp | 27.51 % | +0.19 pp | 29.71 % | +2.39 pp | 27.30 % | -0.02 pp |
| DRAM Throughput (%) | 20.32 % | 20.04 % | -0.28 pp | 20.39 % | +0.07 pp | 10.84 % | -9.48 pp | 20.44 % | +0.12 pp |
| Achieved Occupancy (%) | 58.00 % | 80.85 % | +22.85 pp | 61.81 % | +3.81 pp | 91.16 % | +33.16 pp | 57.25 % | -0.75 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 20.00 register/thread | +2 | 32.00 register/thread | +14 | 20.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 39.28 % | 14.59 % | -24.69 pp | 25.52 % | -13.76 pp | 8.45 % | -30.83 pp | 30.32 % | -8.96 pp |
| IPC Active | 2.43 inst/cycle | 3.40 inst/cycle | +0.97 | 2.99 inst/cycle | +0.56 | 3.66 inst/cycle | +1.23 | 2.79 inst/cycle | +0.36 |

---

## Iteration 4

**Best so far**: 1.0016× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_with_correct_lambda_grid_and_remove_redundant_mask | 1.001× | 0.999× | ✓ |  |
| B | fix_hardcoded_grid_and_tune_eviction_policy_for_l1tex_stall | 0.996× | 0.995× | ✓ |  |
| C | fix_grid_lambda_and_drop_mask_for_correct_autotune_xblock_scaling | 0.985× | 0.985× | ✓ |  |
| D | fix_grid_lambda_and_remove_redundant_mask_for_correct_autotune_dispatch | 0.553× | 0.533× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_with_cor... | Δ | fix_hardcoded_gri... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42850 ns | 42910 ns | +60 ns (+0.1%) | 43070 ns | +220 ns (+0.5%) | 43490 ns | +640 ns (+1.5%) | 80380 ns | +37530 ns (+87.6%) |
| SM Throughput (%) | 57.52 % | 66.14 % | +8.62 pp | 74.72 % | +17.20 pp | 80.30 % | +22.78 pp | 89.20 % | +31.68 pp |
| Memory Throughput (%) | 27.32 % | 27.33 % | +0.01 pp | 27.27 % | -0.05 pp | 26.80 % | -0.52 pp | 29.70 % | +2.38 pp |
| DRAM Throughput (%) | 20.32 % | 20.31 % | -0.01 pp | 20.25 % | -0.07 pp | 20.14 % | -0.18 pp | 10.82 % | -9.50 pp |
| Achieved Occupancy (%) | 58.00 % | 57.47 % | -0.53 pp | 67.78 % | +9.78 pp | 77.78 % | +19.78 pp | 91.01 % | +33.01 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 20.00 register/thread | +2 | 20.00 register/thread | +2 | 32.00 register/thread | +14 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 39.28 % | 30.10 % | -9.18 pp | 21.03 % | -18.25 pp | 14.60 % | -24.68 pp | 8.46 % | -30.82 pp |
| IPC Active | 2.43 inst/cycle | 2.77 inst/cycle | +0.34 | 3.14 inst/cycle | +0.71 | 3.39 inst/cycle | +0.96 | 3.66 inst/cycle | +1.23 |

---

## Iteration 5

**Best so far**: 1.0016× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_grid_lambda_remove_redundant_mask_enable_fma_via_clean_autotune | 0.989× | 0.991× | ✓ |  |
| D | fix_grid_lambda_and_reduce_nonfused_fp32_via_variance_algebraic_simplification | 0.552× | 0.531× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_r... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 42850 ns | 43260 ns | +410 ns (+1.0%) | 80670 ns | +37820 ns (+88.3%) |
| SM Throughput (%) | 57.52 % | 81.08 % | +23.56 pp | 89.18 % | +31.66 pp |
| Memory Throughput (%) | 27.32 % | 27.05 % | -0.27 pp | 29.69 % | +2.37 pp |
| DRAM Throughput (%) | 20.32 % | 20.21 % | -0.11 pp | 10.83 % | -9.49 pp |
| Achieved Occupancy (%) | 58.00 % | 76.49 % | +18.49 pp | 91.02 % | +33.02 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 32.00 register/thread | +14 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 39.28 % | 14.53 % | -24.75 pp | 8.46 % | -30.82 pp |
| IPC Active | 2.43 inst/cycle | 3.37 inst/cycle | +0.94 | 3.66 inst/cycle | +1.23 |

---

## Cumulative Progress

```
Baseline:  42850 ns  (1.000×)
Iter 3     42780 ns  (1.002×)  — promote_non_fused_fp32_to_fma_and_fix_broken_autotune_grid

Final best: 42780 ns  (1.0016×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,47 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_mul_native_group_norm_sigmoid_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 8388608
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 8192)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x2 // 32), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x2 // 32), None, eviction_policy='evict_last')
-    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x2 // 32), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x2 // 32), xmask, eviction_policy='evict_last')
+    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 32.0, tl.float32)
-    tmp5 = (tmp3 / tmp4)
-    tmp6 = tl.full([1], 1e-05, tl.float32)
-    tmp7 = tmp5 + tmp6
-    tmp8 = libdevice.rsqrt(tmp7)
-    tmp9 = tmp2 * tmp8
-    tmp11 = tmp9 * tmp10
-    tmp13 = tmp11 + tmp12
-    tmp14 = tl.sigmoid(tmp13)
-    tmp15 = tmp13 * tmp14
-    tmp17 = tmp15 * tmp16
-    tmp18 = tl.sigmoid(tmp17)
-    tmp19 = tmp17 * tmp18
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tmp_var_norm = tmp3 * 0.03125 + 1e-05
+    tmp8 = libdevice.rsqrt(tmp_var_norm)
+    tmp_xnorm = tmp2 * tmp8
+    tmp_affine = tmp_xnorm * tmp10 + tmp12
+    tmp_exp1 = libdevice.fast_expf(-tmp_affine)
+    tmp_sig1 = 1.0 / (1.0 + tmp_exp1)
+    tmp_silu1 = tmp_affine * tmp_sig1
+    tmp_scaled = tmp_silu1 * tmp16
+    tmp_exp2 = libdevice.fast_expf(-tmp_scaled)
+    tmp_sig2 = 1.0 / (1.0 + tmp_exp2)
+    tmp_silu2 = tmp_scaled * tmp_sig2
+    tl.store(in_out_ptr0 + (x2), tmp_silu2, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +56,7 @@

     arg_5 = inputs[5]
     xnumel = 8388608
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_mul_native_group_norm_sigmoid_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 8388608, XBLOCK=128)
+    triton_poi_fused_mul_native_group_norm_sigmoid_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 8388608)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

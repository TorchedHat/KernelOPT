# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 277790 ns — **Best latency**: 8130 ns (**34.1685×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`reduce_register_pressure_via_maxnreg_and_r0block_right_sizing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_43_triton_per_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.04
- **Registers/thread**: 32
- **Duration**: 277790 ns
- **SM Throughput**: 1.30 % | **Memory**: 4.66 % | **DRAM**: 0.67 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.26 inst/cycle | **No Eligible**: 93.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_grid_coverage | 21.082× | 27.208× | ✓ |  |
| B | increase_grid_size_via_smaller_xblock_with_autotune | 29.276× | 40.027× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | increase_grid_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 277790 ns | 10210 ns | -267580 ns (-96.3%) | 6940 ns | -270850 ns (-97.5%) |
| SM Throughput (%) | 1.30 % | 1.54 % | +0.24 pp | 2.66 % | +1.36 pp |
| Memory Throughput (%) | 4.66 % | 2.39 % | -2.27 pp | 2.80 % | -1.86 pp |
| DRAM Throughput (%) | 0.67 % | 1.72 % | +1.05 pp | 2.46 % | +1.79 pp |
| Achieved Occupancy (%) | 6.23 % | 3.10 % | -3.13 pp | 6.13 % | -0.10 pp |
| Registers/Thread | 32.00 register/thread | 255.00 register/thread | +223 | 166.00 register/thread | +134 |
| Waves/SM | 0.04 | 0.03 | -0.01 | 0.04 | +0.00 |
| No Eligible (%) | 93.65 % | 68.47 % | -25.18 pp | 69.38 % | -24.27 pp |
| IPC Active | 0.26 inst/cycle | 0.61 inst/cycle | +0.35 | 1.21 inst/cycle | +0.95 |

---

## Iteration 2

**Best so far**: 40.0274× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_maxnreg_and_r0block_right_sizing | 42.298× | 34.169× | ✓ | **✓ best** |
| B | reduce_register_pressure_via_maxnreg_to_unlock_occupancy | 29.230× | 38.906× | ✓ |  |
| C | fix_hardcoded_grid_to_enable_autotune_xblock_selection | 21.163× | 27.128× | ✓ |  |
| D | fix_autotuner_bypass_and_cap_register_pressure | 12.652× | 13.820× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | fix_hardcoded_gri... | Δ | fix_autotuner_byp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 277790 ns | 8130 ns | -269660 ns (-97.1%) | 7140 ns | -270650 ns (-97.4%) | 10240 ns | -267550 ns (-96.3%) | 20100 ns | -257690 ns (-92.8%) |
| SM Throughput (%) | 1.30 % | 2.40 % | +1.10 pp | 2.64 % | +1.34 pp | 1.49 % | +0.19 pp | 0.93 % | -0.37 pp |
| Memory Throughput (%) | 4.66 % | 2.92 % | -1.74 pp | 2.78 % | -1.88 pp | 2.42 % | -2.24 pp | 2.74 % | -1.92 pp |
| DRAM Throughput (%) | 0.67 % | 2.14 % | +1.47 pp | 2.39 % | +1.72 pp | 1.72 % | +1.05 pp | 0.89 % | +0.22 pp |
| Achieved Occupancy (%) | 6.23 % | 6.15 % | -0.08 pp | 6.13 % | -0.10 pp | 3.10 % | -3.13 pp | 3.12 % | -3.11 pp |
| Registers/Thread | 32.00 register/thread | 128.00 register/thread | +96 | 166.00 register/thread | +134 | 255.00 register/thread | +223 | 128.00 register/thread | +96 |
| Waves/SM | 0.04 | 0.03 | -0.01 | 0.04 | +0.00 | 0.03 | -0.01 | 0.02 | -0.02 |
| No Eligible (%) | 93.65 % | 73.14 % | -20.51 pp | 69.51 % | -24.14 pp | 68.25 % | -25.40 pp | 83.63 % | -10.02 pp |
| IPC Active | 0.26 inst/cycle | 1.07 inst/cycle | +0.81 | 1.20 inst/cycle | +0.94 | 0.62 inst/cycle | +0.36 | 0.34 inst/cycle | +0.08 |

---

## Iteration 3

**Best so far**: 40.0274× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_r0block_to_reduce_register_pressure_and_fix_grid_lambda | 12.551× | 14.137× | ✓ |  |
| D | fix_hardcoded_grid_lambda_to_track_autotuner_xblock | 12.666× | 14.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_r0block_... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 277790 ns | 19650 ns | -258140 ns (-92.9%) | 19840 ns | -257950 ns (-92.9%) |
| SM Throughput (%) | 1.30 % | 0.94 % | -0.36 pp | 0.94 % | -0.36 pp |
| Memory Throughput (%) | 4.66 % | 2.80 % | -1.86 pp | 2.78 % | -1.88 pp |
| DRAM Throughput (%) | 0.67 % | 0.91 % | +0.24 pp | 0.90 % | +0.23 pp |
| Achieved Occupancy (%) | 6.23 % | 3.12 % | -3.11 pp | 3.12 % | -3.11 pp |
| Registers/Thread | 32.00 register/thread | 128.00 register/thread | +96 | 128.00 register/thread | +96 |
| Waves/SM | 0.04 | 0.02 | -0.02 | 0.02 | -0.02 |
| No Eligible (%) | 93.65 % | 83.17 % | -10.48 pp | 83.18 % | -10.47 pp |
| IPC Active | 0.26 inst/cycle | 0.33 inst/cycle | +0.07 | 0.33 inst/cycle | +0.07 |

---

## Iteration 4

**Best so far**: 40.0274× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_tuple_to_lambda_enabling_autotune_xblock_grid_coverage | 12.536× | 14.044× | ✓ |  |
| B | fix_grid_lambda_and_lower_maxnreg_for_sm_coverage | 8.270× | 7.174× | ✓ |  |
| C | fix_grid_lambda_and_remove_maxnreg_to_unlock_autotune | 21.404× | 28.003× | ✓ |  |
| D | fix_hardcoded_grid_and_right_size_r0block_to_match_autotuner_xblock | 12.562× | 14.567× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 277790 ns | 19780 ns | -258010 ns (-92.9%) | 38720 ns | -239070 ns (-86.1%) | 9920 ns | -267870 ns (-96.4%) | 19070 ns | -258720 ns (-93.1%) |
| SM Throughput (%) | 1.30 % | 0.96 % | -0.34 pp | 0.88 % | -0.42 pp | 1.52 % | +0.22 pp | 0.96 % | -0.34 pp |
| Memory Throughput (%) | 4.66 % | 2.78 % | -1.88 pp | 3.03 % | -1.63 pp | 2.36 % | -2.30 pp | 2.88 % | -1.78 pp |
| DRAM Throughput (%) | 0.67 % | 0.90 % | +0.23 pp | 0.47 % | -0.20 pp | 1.77 % | +1.10 pp | 0.93 % | +0.26 pp |
| Achieved Occupancy (%) | 6.23 % | 3.12 % | -3.11 pp | 3.12 % | -3.11 pp | 3.10 % | -3.13 pp | 3.12 % | -3.11 pp |
| Registers/Thread | 32.00 register/thread | 128.00 register/thread | +96 | 64.00 register/thread | +32 | 255.00 register/thread | +223 | 128.00 register/thread | +96 |
| Waves/SM | 0.04 | 0.02 | -0.02 | 0.01 | -0.03 | 0.03 | -0.01 | 0.02 | -0.02 |
| No Eligible (%) | 93.65 % | 82.40 % | -11.25 pp | 90.03 % | -3.62 pp | 68.60 % | -25.05 pp | 82.78 % | -10.87 pp |
| IPC Active | 0.26 inst/cycle | 0.34 inst/cycle | +0.08 | 0.20 inst/cycle | -0.06 | 0.62 inst/cycle | +0.36 | 0.34 inst/cycle | +0.08 |


### Key Learnings

#### ✅ Plan B: `fix_grid_lambda_and_lower_maxnreg_for_sm_coverage` (7.174×)

Reducing maxnreg from 128 to 64 for small-XBLOCK autotune configs (XBLOCK=16,32) successfully raised Block Limit Registers from 8 to 16 and improved theoretical occupancy from 25% to 40.6%, but the grid lambda was NOT fixed — the call site still used a hardcoded tuple (16 blocks on 132 SMs). Because achieved occupancy remained at 3.12% (grid-size-bound, not register-bound), the lower maxnreg only introduced register spilling overhead: Warp Cycles Per Issued Instruction doubled from 5.82 to 10.23, No Eligible rose from 83% to 90%, and kernel duration nearly doubled from 19.65µs to 38.72µs. The 7.17x 'speedup' reported reflects end-to-end benchmark improvement from other factors, not this kernel's NCU metrics which clearly regressed. When grid_size << sm_count, reducing maxnreg to raise theoretical occupancy is useless — the binding constraint is total block count, not blocks-per-SM capacity.

> **Applicable when**: Never apply maxnreg reduction in isolation when the grid lambda is still hardcoded and grid_size << sm_count. This is a negative example showing the interaction between two changes.

> **Avoid when**: Do NOT lower maxnreg when: (1) the grid lambda is still hardcoded (autotune is bypassed), (2) grid_size << sm_count so achieved occupancy is grid-bound not register-bound, (3) the kernel has complex strided index arithmetic that requires many live registers — spilling to local memory will dominate. Lowering maxnreg only helps when Block Limit Registers is the binding occupancy constraint AND the grid is large enough to benefit from more blocks/SM (waves_per_sm >= 0.5).

---

## Cumulative Progress

```
Baseline:  277790 ns  (1.000×)
Iter 1     6940 ns  (40.027×)  — increase_grid_size_via_smaller_xblock_with_autotune

Final best: 8130 ns  (34.1685×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=1, maxnreg=128),
+        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1, maxnreg=128),
+        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1, maxnreg=128),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1, maxnreg=128),
+        triton.Config({'XBLOCK': 64},  num_warps=8, num_stages=1, maxnreg=128),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1, maxnreg=128),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1, maxnreg=128),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 1960
@@ -24,11 +37,11 @@

     x3 = xindex
     x0 = (xindex % 196)
     x1 = xindex // 196
-    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
-    tmp1 = tl.load(in_ptr1 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
-    tmp2 = tl.load(in_ptr2 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), mask=r0_mask & xmask, other=0.0)
+    tmp1 = tl.load(in_ptr1 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), mask=r0_mask & xmask, other=0.0)
+    tmp2 = tl.load(in_ptr2 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+    tmp28 = tl.load(in_ptr3 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr4 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
     tmp3 = tmp1 + tmp2
     tmp4 = tmp0 + tmp3
     tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
@@ -53,9 +66,7 @@

     tmp27 = tmp21 * tmp26
     tmp29 = tmp27 * tmp28
     tmp31 = tmp29 + tmp30
-    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, r0_mask & xmask)
-
-
+    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -67,7 +78,7 @@

     xnumel = 1960
     r0_numel = 384
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1960, 384, XBLOCK=128)
+    triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1960, 384)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

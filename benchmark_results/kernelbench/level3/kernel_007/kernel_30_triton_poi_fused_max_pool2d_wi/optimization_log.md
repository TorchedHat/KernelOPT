# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_max_pool2d_with_indices_30` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7970 ns — **Best latency**: 7230 ns (**1.1024×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_num_warps_to_increase_block_count_and_hide_l1tex_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_max_pool2d_with_indices_30`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_30_triton_poi_fused_max_pool2d_wi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7350,1,1), Waves/SM: 3.48
- **Registers/thread**: 24
- **Duration**: 7970 ns
- **SM Throughput**: 45.73 % | **Memory**: 43.81 % | **DRAM**: 38.56 %
- **Achieved Occupancy**: 78.30 %
- **IPC Active**: 2.48 inst/cycle | **No Eligible**: 36.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_partial_wave_and_improve_l1_reuse | 0.896× | 0.795× | ✓ |  |
| B | increase_xblock_and_autotune_to_eliminate_partial_wave_and_reduce_l1tex_stalls | 0.968× | 0.912× | ✓ |  |
| C | vectorized_loads_via_larger_xblock_and_autotune | 0.988× | 0.912× | ✓ |  |
| D | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_reduce_l1tex_stalls | 1.020× | 0.290× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_a... | Δ | vectorized_loads_... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7970 ns | 10020 ns | +2050 ns (+25.7%) | 8740 ns | +770 ns (+9.7%) | 8740 ns | +770 ns (+9.7%) | 27460 ns | +19490 ns (+244.5%) |
| SM Throughput (%) | 45.73 % | 55.80 % | +10.07 pp | 49.53 % | +3.80 pp | 49.36 % | +3.63 pp | 79.38 % | +33.65 pp |
| Memory Throughput (%) | 43.81 % | 32.83 % | -10.98 pp | 37.68 % | -6.13 pp | 37.70 % | -6.11 pp | 26.56 % | -17.25 pp |
| DRAM Throughput (%) | 38.56 % | 30.72 % | -7.84 pp | 35.12 % | -3.44 pp | 35.23 % | -3.33 pp | 11.17 % | -27.39 pp |
| Achieved Occupancy (%) | 78.30 % | 80.34 % | +2.04 pp | 79.70 % | +1.40 pp | 80.56 % | +2.26 pp | 88.96 % | +10.66 pp |
| Registers/Thread | 24.00 register/thread | 32.00 register/thread | +8 | 32.00 register/thread | +8 | 32.00 register/thread | +8 | 32.00 register/thread | +8 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 | 6.96 | +3.48 |
| No Eligible (%) | 36.98 % | 29.51 % | -7.47 pp | 32.65 % | -4.33 pp | 32.24 % | -4.74 pp | 13.24 % | -23.74 pp |
| IPC Active | 2.48 inst/cycle | 2.79 inst/cycle | +0.31 | 2.67 inst/cycle | +0.19 | 2.61 inst/cycle | +0.13 | 3.46 inst/cycle | +0.98 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_smaller_xblock_configs | 1.138× | 1.092× | ✓ |  |
| B | specialize_interior_vs_boundary_tiles_to_eliminate_mask_alu_overhead | 0.799× | 0.700× | ✓ |  |
| C | reduce_register_pressure_by_reverting_to_smaller_xblock_configs | 1.136× | 1.107× | ✓ |  |
| D | fix_grid_lambda_and_reduce_xblock_to_restore_baseline_instruction_count | 1.160× | 1.009× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | specialize_interi... | Δ | reduce_register_p... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7970 ns | 7300 ns | -670 ns (-8.4%) | 11390 ns | +3420 ns (+42.9%) | 7200 ns | -770 ns (-9.7%) | 7900 ns | -70 ns (-0.9%) |
| SM Throughput (%) | 45.73 % | 24.75 % | -20.98 pp | 61.80 % | +16.07 pp | 24.89 % | -20.84 pp | 46.74 % | +1.01 pp |
| Memory Throughput (%) | 43.81 % | 23.71 % | -20.10 pp | 33.72 % | -10.09 pp | 23.94 % | -19.87 pp | 44.43 % | +0.62 pp |
| DRAM Throughput (%) | 38.56 % | 21.06 % | -17.50 pp | 27.00 % | -11.56 pp | 21.32 % | -17.24 pp | 38.91 % | +0.35 pp |
| Achieved Occupancy (%) | 78.30 % | 49.47 % | -28.83 pp | 81.86 % | +3.56 pp | 49.76 % | -28.54 pp | 77.73 % | -0.57 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 32.00 register/thread | +8 | 24.00 register/thread | +0 | 24.00 register/thread | +0 |
| Waves/SM | 3.48 | 1.74 | -1.74 | 3.48 | +0.00 | 1.74 | -1.74 | 3.48 | +0.00 |
| No Eligible (%) | 36.98 % | 50.49 % | +13.51 pp | 22.16 % | -14.82 pp | 52.62 % | +15.64 pp | 37.56 % | +0.58 pp |
| IPC Active | 2.48 inst/cycle | 1.89 inst/cycle | -0.59 | 3.03 inst/cycle | +0.55 | 1.87 inst/cycle | -0.61 | 2.48 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan A: `fix_autotune_grid_lambda_to_enable_smaller_xblock_configs` (1.092×)

When a Triton kernel's @triton.autotune decorator specifies variable XBLOCK configs but the launch grid is a fixed expression (e.g., cdiv(xnumel, 128)) rather than a lambda consuming meta['XBLOCK'], the autotuner cannot actually vary the tile size. Fixing the grid to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` and adding smaller XBLOCK configs (e.g., 64, 128) reduced registers per thread from 32 to 24, cutting executed instructions by 58% (3.38M → 1.41M) and duration by 16% (8.74us → 7.30us). The smaller tile size reduced register pressure (Block Limit Registers rose from 16 to 42 blocks/SM), though achieved occupancy dropped from 79.7% to 49.5% because fewer waves were needed — the net effect was still positive due to the dramatic instruction count reduction.

> **Applicable when**: Triton kernel has @triton.autotune with XBLOCK as a tunable parameter but the launch grid is a fixed integer expression rather than a lambda. Profiling shows high register pressure (32+ regs/thread) limiting Block Limit Registers, and CPIStall triggered by L1TEX latency. Adding smaller XBLOCK configs (64, 128) alongside the grid lambda fix is beneficial when the kernel body is register-heavy and smaller tiles reduce register spill/pressure.

> **Avoid when**: Do not reduce XBLOCK below the point where the grid becomes too large and tail-wave effects dominate (check that xnumel divides cleanly by the chosen XBLOCK). Also avoid if the kernel's memory access pattern benefits from spatial reuse across wider tiles (high L1 hit rate at larger XBLOCK). Note that achieved occupancy may drop significantly with smaller XBLOCK and fewer waves — verify the net latency still improves.

#### ❌ Plan B: `specialize_interior_vs_boundary_tiles_to_eliminate_mask_alu_overhead` (0.700×)

Splitting a max-pool kernel into interior (unconditional loads) and boundary (masked loads) paths, then merging with tl.where, caused a 43% slowdown (8.74us → 11.39us). Rather than eliminating ALU mask overhead, the executed instruction count increased by 63% (3.38M → 5.50M) because both paths execute for every thread — Triton does not predicate-eliminate the inactive path. The tl.where merge forces both interior and boundary load chains to fully execute regardless of the is_interior flag, doubling the total load and ALU instruction count instead of halving it. DRAM throughput dropped from 35% to 27% while compute throughput rose from 49.5% to 61.8%, confirming the kernel became more compute-bound due to the extra instructions, not less.

> **Applicable when**: Never: this pattern always doubles instruction count in Triton because tl.where does not short-circuit execution of either branch.

> **Avoid when**: Do not use tl.where to select between two fully-computed paths (interior vs boundary) in Triton when the goal is to eliminate computation. Triton executes both branches unconditionally — the predicate only affects which result is selected, not which instructions execute. This anti-pattern is especially harmful when both paths contain multiple memory loads, as it doubles the load count and inflates instruction count proportionally. The CPIStall bottleneck from L1TEX latency is not addressed by this approach.

---

## Iteration 3

**Best so far**: 1.0089× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_via_xblock128_to_hide_l1tex_latency | 1.022× | 1.004× | ✓ |  |
| B | reduce_num_warps_to_increase_block_count_and_hide_l1tex_latency | 1.191× | 1.102× | ✓ | **✓ best** |
| C | fix_grid_lambda_and_add_num_stages_to_unlock_autotune_for_smaller_xblock | 1.190× | 1.064× | ✓ |  |
| D | revert_dual_path_and_fix_grid_lambda_with_smaller_xblock_configs | 1.139× | 1.102× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | reduce_num_warps_... | Δ | fix_grid_lambda_a... | Δ | revert_dual_path_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7970 ns | 7940 ns | -30 ns (-0.4%) | 7230 ns | -740 ns (-9.3%) | 7490 ns | -480 ns (-6.0%) | 7230 ns | -740 ns (-9.3%) |
| SM Throughput (%) | 45.73 % | 46.60 % | +0.87 pp | 12.56 % | -33.17 pp | 25.84 % | -19.89 pp | 25.06 % | -20.67 pp |
| Memory Throughput (%) | 43.81 % | 44.26 % | +0.45 pp | 12.70 % | -31.11 pp | 12.12 % | -31.69 pp | 23.99 % | -19.82 pp |
| DRAM Throughput (%) | 38.56 % | 38.89 % | +0.33 pp | 10.65 % | -27.91 pp | 10.26 % | -28.30 pp | 21.33 % | -17.23 pp |
| Achieved Occupancy (%) | 78.30 % | 77.89 % | -0.41 pp | 19.15 % | -59.15 pp | 48.51 % | -29.79 pp | 48.28 % | -30.02 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 23.00 register/thread | -1 | 24.00 register/thread | +0 | 24.00 register/thread | +0 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 1.74 | -1.74 | 1.74 | -1.74 | 1.74 | -1.74 |
| No Eligible (%) | 36.98 % | 37.89 % | +0.91 pp | 71.63 % | +34.65 pp | 47.78 % | +10.80 pp | 50.50 % | +13.52 pp |
| IPC Active | 2.48 inst/cycle | 2.50 inst/cycle | +0.02 | 0.98 inst/cycle | -1.50 | 1.85 inst/cycle | -0.63 | 1.88 inst/cycle | -0.60 |

---

## Iteration 4

**Best so far**: 1.1024× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_lambda_and_add_num_stages_to_autotune_configs_for_latency_hiding | 1.163× | 1.107× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7970 ns | 7200 ns | -770 ns (-9.7%) |
| SM Throughput (%) | 45.73 % | 24.84 % | -20.89 pp |
| Memory Throughput (%) | 43.81 % | 24.13 % | -19.68 pp |
| DRAM Throughput (%) | 38.56 % | 21.44 % | -17.12 pp |
| Achieved Occupancy (%) | 78.30 % | 50.13 % | -28.17 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 |
| Waves/SM | 3.48 | 1.74 | -1.74 |
| No Eligible (%) | 36.98 % | 50.68 % | +13.70 pp |
| IPC Active | 2.48 inst/cycle | 1.90 inst/cycle | -0.58 |

---

## Iteration 5

**Best so far**: 1.1024× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_sm_aligned_xblock_config_to_eliminate_partial_wave | 1.138× | 1.107× | ✓ |  |
| B | persistent_kernel_to_eliminate_partial_wave_and_hide_l1tex_latency | 1.003× | 1.000× | ✓ |  |
| D | fix_grid_lambda_and_increase_num_warps_to_hide_l1tex_latency | 1.139× | 1.098× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | persistent_kernel... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7970 ns | 7200 ns | -770 ns (-9.7%) | 7970 ns | +0 ns (+0.0%) | 7260 ns | -710 ns (-8.9%) |
| SM Throughput (%) | 45.73 % | 24.86 % | -20.87 pp | 46.00 % | +0.27 pp | 25.07 % | -20.66 pp |
| Memory Throughput (%) | 43.81 % | 24.04 % | -19.77 pp | 43.87 % | +0.06 pp | 23.88 % | -19.93 pp |
| DRAM Throughput (%) | 38.56 % | 21.32 % | -17.24 pp | 38.60 % | +0.04 pp | 21.25 % | -17.31 pp |
| Achieved Occupancy (%) | 78.30 % | 48.48 % | -29.82 pp | 77.61 % | -0.69 pp | 56.18 % | -22.12 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 24.00 register/thread | +0 |
| Waves/SM | 3.48 | 1.74 | -1.74 | 3.48 | +0.00 | 1.74 | -1.74 |
| No Eligible (%) | 36.98 % | 50.25 % | +13.27 pp | 36.33 % | -0.65 pp | 52.23 % | +15.25 pp |
| IPC Active | 2.48 inst/cycle | 1.88 inst/cycle | -0.60 | 2.50 inst/cycle | +0.02 | 1.76 inst/cycle | -0.72 |

---

## Cumulative Progress

```
Baseline:  7970 ns  (1.000×)
Iter 2     7900 ns  (1.009×)  — fix_grid_lambda_and_reduce_xblock_to_restore_baseline_instruction_count
Iter 3     7230 ns  (1.102×)  — reduce_num_warps_to_increase_block_count_and_hide_l1tex_latency

Final best: 7230 ns  (1.1024×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_max_pool2d_with_indices_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 940800
@@ -16,6 +25,10 @@

     x0 = (xindex % 480)
     x5 = xindex // 6720
     x6 = xindex
+
+    base = x0 + 960*x1 + 26880*x5
+
+    # SINGLE PATH: 9 masked loads only, no is_interior, no tl.where
     tmp0 = (-1) + 2*x2
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
@@ -27,57 +40,55 @@

     tmp8 = tmp6 < tmp3
     tmp9 = tmp7 & tmp8
     tmp10 = tmp5 & tmp9
-    tmp11 = tl.load(in_ptr0 + ((-13920) + x0 + 960*x1 + 26880*x5), tmp10 & xmask, other=float("-inf"))
+    tmp11 = tl.load(in_ptr0 + ((-13920) + base), mask=tmp10 & xmask, other=float('-inf'))
     tmp12 = 2*x1
     tmp13 = tmp12 >= tmp1
     tmp14 = tmp12 < tmp3
     tmp15 = tmp13 & tmp14
     tmp16 = tmp5 & tmp15
-    tmp17 = tl.load(in_ptr0 + ((-13440) + x0 + 960*x1 + 26880*x5), tmp16 & xmask, other=float("-inf"))
+    tmp17 = tl.load(in_ptr0 + ((-13440) + base), mask=tmp16 & xmask, other=float('-inf'))
     tmp18 = tl.maximum(tmp11, tmp17)
     tmp19 = 1 + 2*x1
     tmp20 = tmp19 >= tmp1
     tmp21 = tmp19 < tmp3
     tmp22 = tmp20 & tmp21
     tmp23 = tmp5 & tmp22
-    tmp24 = tl.load(in_ptr0 + ((-12960) + x0 + 960*x1 + 26880*x5), tmp23 & xmask, other=float("-inf"))
+    tmp24 = tl.load(in_ptr0 + ((-12960) + base), mask=tmp23 & xmask, other=float('-inf'))
     tmp25 = tl.maximum(tmp18, tmp24)
     tmp26 = 2*x2
     tmp27 = tmp26 >= tmp1
     tmp28 = tmp26 < tmp3
     tmp29 = tmp27 & tmp28
     tmp30 = tmp29 & tmp9
-    tmp31 = tl.load(in_ptr0 + ((-480) + x0 + 960*x1 + 26880*x5), tmp30 & xmask, other=float("-inf"))
+    tmp31 = tl.load(in_ptr0 + ((-480) + base), mask=tmp30 & xmask, other=float('-inf'))
     tmp32 = tl.maximum(tmp25, tmp31)
     tmp33 = tmp29 & tmp15
-    tmp34 = tl.load(in_ptr0 + (x0 + 960*x1 + 26880*x5), tmp33 & xmask, other=float("-inf"))
+    tmp34 = tl.load(in_ptr0 + base, mask=tmp33 & xmask, other=float('-inf'))
     tmp35 = tl.maximum(tmp32, tmp34)
     tmp36 = tmp29 & tmp22
-    tmp37 = tl.load(in_ptr0 + (480 + x0 + 960*x1 + 26880*x5), tmp36 & xmask, other=float("-inf"))
+    tmp37 = tl.load(in_ptr0 + (480 + base), mask=tmp36 & xmask, other=float('-inf'))
     tmp38 = tl.maximum(tmp35, tmp37)
     tmp39 = 1 + 2*x2
     tmp40 = tmp39 >= tmp1
     tmp41 = tmp39 < tmp3
     tmp42 = tmp40 & tmp41
     tmp43 = tmp42 & tmp9
-    tmp44 = tl.load(in_ptr0 + (12960 + x0 + 960*x1 + 26880*x5), tmp43 & xmask, other=float("-inf"))
+    tmp44 = tl.load(in_ptr0 + (12960 + base), mask=tmp43 & xmask, other=float('-inf'))
     tmp45 = tl.maximum(tmp38, tmp44)
     tmp46 = tmp42 & tmp15
-    tmp47 = tl.load(in_ptr0 + (13440 + x0 + 960*x1 + 26880*x5), tmp46 & xmask, other=float("-inf"))
+    tmp47 = tl.load(in_ptr0 + (13440 + base), mask=tmp46 & xmask, other=float('-inf'))
     tmp48 = tl.maximum(tmp45, tmp47)
     tmp49 = tmp42 & tmp22
-    tmp50 = tl.load(in_ptr0 + (13920 + x0 + 960*x1 + 26880*x5), tmp49 & xmask, other=float("-inf"))
-    tmp51 = tl.maximum(tmp48, tmp50)
-    tl.store(out_ptr0 + (x6), tmp51, xmask)
-
-
+    tmp50 = tl.load(in_ptr0 + (13920 + base), mask=tmp49 & xmask, other=float('-inf'))
+    result = tl.maximum(tmp48, tmp50)
+    tl.store(out_ptr0 + x6, result, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 940800
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_max_pool2d_with_indices_30[grid](arg_0, arg_1, 940800, XBLOCK=128)
+    triton_poi_fused_max_pool2d_with_indices_30[grid](arg_0, arg_1, 940800)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

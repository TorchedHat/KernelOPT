# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_gelu_view_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16960 ns — **Best latency**: 16960 ns (**1.0000×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_gelu_view_36`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_36_triton_poi_fused_addmm_gelu_vi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 16
- **Duration**: 16960 ns
- **SM Throughput**: 41.59 % | **Memory**: 18.34 % | **DRAM**: 14.51 %
- **Achieved Occupancy**: 44.93 %
- **IPC Active**: 1.95 inst/cycle | **No Eligible**: 52.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_with_persistent_tile_scheduling | 0.703× | 0.676× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_gelu | 1.008× | 0.182× | ✓ | **✓ best** |
| C | autotune_xblock_num_warps_for_latency_hiding | 0.997× | 0.675× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.979× | 0.183× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_w... | Δ | autotune_xblock_a... | Δ | autotune_xblock_n... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16960 ns | 25090 ns | +8130 ns (+47.9%) | 93340 ns | +76380 ns (+450.4%) | 25120 ns | +8160 ns (+48.1%) | 92800 ns | +75840 ns (+447.2%) |
| SM Throughput (%) | 41.59 % | 78.00 % | +36.41 pp | 83.03 % | +41.44 pp | 78.03 % | +36.44 pp | 84.15 % | +42.56 pp |
| Memory Throughput (%) | 18.34 % | 12.11 % | -6.23 pp | 7.71 % | -10.63 pp | 12.10 % | -6.24 pp | 7.73 % | -10.61 pp |
| DRAM Throughput (%) | 14.51 % | 9.79 % | -4.72 pp | 2.63 % | -11.88 pp | 9.78 % | -4.73 pp | 2.64 % | -11.87 pp |
| Achieved Occupancy (%) | 44.93 % | 85.02 % | +40.09 pp | 85.05 % | +40.12 pp | 85.57 % | +40.64 pp | 85.78 % | +40.85 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 | 28.00 register/thread | +12 | 28.00 register/thread | +12 | 28.00 register/thread | +12 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 44.55 | +33.41 | 11.14 | +0.00 | 44.55 | +33.41 |
| No Eligible (%) | 52.00 % | 15.52 % | -36.48 pp | 14.74 % | -37.26 pp | 14.81 % | -37.19 pp | 13.85 % | -38.15 pp |
| IPC Active | 1.95 inst/cycle | 3.38 inst/cycle | +1.43 | 3.40 inst/cycle | +1.45 | 3.36 inst/cycle | +1.41 | 3.44 inst/cycle | +1.49 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_with_persistent_tile_scheduling` (0.676×)

Adding @triton.autotune with large XBLOCK configs (512–2048) to a pointwise kernel that originally used XBLOCK=128 caused registers_per_thread to jump from 16 to 28. This register pressure increase constrained occupancy gains (Block Limit Registers dropped from 32 to 16 blocks), and the 2.77x increase in executed instructions (5.5M → 15.4M) indicates the autotuner selected a config that dramatically increased per-CTA work without proportional throughput gains. Despite occupancy improving from 44.9% to 85%, the kernel duration increased from 16.96 µs to 25.09 µs (0.676x speedup), because the compute throughput increase (41.6% → 78%) was offset by far more total instructions being executed. The anti-pattern is applying large-XBLOCK autotune to a kernel where the original XBLOCK was already register-efficient; the larger tile size inflates register usage and instruction count without reducing wall time.

> **Applicable when**: Kernel has low achieved occupancy (<50%) due to L1TEX stalls, small XBLOCK (<256), and low registers_per_thread (<20), where increasing tile size would not inflate register usage.

> **Avoid when**: The kernel already uses few registers (≤16/thread) and is register-efficient at the current XBLOCK; adding large XBLOCK configs via autotune may cause the compiler to allocate significantly more registers (e.g., 16→28), reducing Block Limit Registers and increasing total instruction count without reducing wall time. Also avoid when the grid lambda is not updated to use meta['XBLOCK'] — the autotune will select large XBLOCK but the grid remains fixed at the original small-tile size, causing correctness and performance issues.

#### ❌ Plan C: `autotune_xblock_num_warps_for_latency_hiding` (0.675×)

Adding @triton.autotune with larger XBLOCK values (512–2048) and higher num_warps (4–16) caused registers per thread to jump from 16 to 28, which halved the number of blocks that can be resident per SM (Block Limit Registers dropped from 32 to 16). Although occupancy improved from 44.9% to 85.6% and issue slot utilization rose from 48.9% to 84.0%, the total executed instruction count nearly tripled (5.5M → 15.4M) and kernel duration increased from 16.96 µs to 25.12 µs — a 48% regression. The root cause is that the autotuner selected a config with larger XBLOCK that inflated register usage and dramatically increased total instruction work, overwhelming the latency-hiding benefit. For short-duration, memory-latency-bound pointwise kernels where the original register count is already low (≤16), adding autotune with large XBLOCK can backfire by selecting configs that increase register pressure and instruction count without proportional throughput gains.

> **Applicable when**: Never apply this specific pattern. The strategy of autotuning XBLOCK upward to hide L1TEX latency is theoretically sound but fails in practice when: (1) the autotuner selects configs that increase register pressure beyond the original, (2) the grid is not updated to a lambda using meta['XBLOCK'] so the larger XBLOCK configs run with the wrong grid, or (3) the instruction count increase from larger tiles outweighs latency hiding.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK values to pointwise kernels that already have low register counts (≤16 regs/thread) and short durations (<20 µs), especially when the grid is not simultaneously updated to use a lambda meta expression. The combination of increased register pressure (reducing SM block capacity) and a fixed-size grid that doesn't scale with XBLOCK causes severe regressions. Also avoid when the kernel's instruction count is already near-minimal — larger tiles multiply instruction work without proportional memory savings.

#### ❌ Plan D: `increase_xblock_for_vectorized_loads_and_latency_hiding` (0.183×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) while keeping the original fixed grid (cdiv(xnumel, 128)) caused a catastrophic regression: the grid was not updated to use meta['XBLOCK'], so the kernel ran with 23520 blocks each processing 512 elements instead of 128, executing 11x more total instructions (5.5M → 61.8M) and increasing duration from 16.96µs to 92.8µs (5.5x slower). The autotune decorator selected XBLOCK=512 but the launch grid remained hardcoded to cdiv(xnumel, 128), causing each element to be processed ~4x redundantly. Even though compute throughput improved (42% → 84%) and occupancy improved (45% → 86%), the massive work duplication from the broken grid calculation dominated, producing a 0.18x speedup (severe regression).

> **Applicable when**: Never apply this broken pattern. When adding autotune to a Triton kernel, the grid MUST be a lambda using meta['XBLOCK'] (or equivalent tile parameter).

> **Avoid when**: Do not use a hardcoded grid when adding @triton.autotune — the grid must be a callable lambda that reads the autotuned tile size from meta dict. This anti-pattern causes O(XBLOCK_new/XBLOCK_old) work duplication.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restore_register_efficient_baseline_with_fma_friendly_gelu | 1.012× | 0.997× | ✓ | **✓ best** |
| B | remove_broken_autotune_and_enable_fma_fusion | 1.011× | 0.998× | ✓ |  |
| C | reduce_non_fused_fp32_instructions_via_fma_restructuring | 0.979× | 0.188× | ✓ |  |
| D | fix_broken_autotune_grid_lambda_and_cap_xblock_to_prevent_register_explosion | 1.000× | 1.002× | ✓ |  |
| E | add_triton_autotune | 1.009× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restore_register_... | Δ | remove_broken_aut... | Δ | reduce_non_fused_... | Δ | fix_broken_autotu... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16960 ns | 17020 ns | +60 ns (+0.4%) | 16990 ns | +30 ns (+0.2%) | 90240 ns | +73280 ns (+432.1%) | 16930 ns | -30 ns (-0.2%) | 16930 ns | -30 ns (-0.2%) |
| SM Throughput (%) | 41.59 % | 44.57 % | +2.98 pp | 41.65 % | +0.06 pp | 84.49 % | +42.90 pp | 44.46 % | +2.87 pp | 44.54 % | +2.95 pp |
| Memory Throughput (%) | 18.34 % | 18.24 % | -0.10 pp | 18.31 % | -0.03 pp | 7.97 % | -10.37 pp | 18.37 % | +0.03 pp | 18.37 % | +0.03 pp |
| DRAM Throughput (%) | 14.51 % | 14.43 % | -0.08 pp | 14.46 % | -0.05 pp | 2.72 % | -11.79 pp | 14.52 % | +0.01 pp | 14.52 % | +0.01 pp |
| Achieved Occupancy (%) | 44.93 % | 46.02 % | +1.09 pp | 44.54 % | -0.39 pp | 86.78 % | +41.85 pp | 46.14 % | +1.21 pp | 45.87 % | +0.94 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 28.00 register/thread | +12 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 44.55 | +33.41 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 52.00 % | 47.90 % | -4.10 pp | 51.74 % | -0.26 pp | 13.32 % | -38.68 pp | 47.98 % | -4.02 pp | 47.73 % | -4.27 pp |
| IPC Active | 1.95 inst/cycle | 2.05 inst/cycle | +0.10 | 1.95 inst/cycle | +0.00 | 3.46 inst/cycle | +1.51 | 2.04 inst/cycle | +0.09 | 2.06 inst/cycle | +0.11 |


### Key Learnings

#### ❌ Plan C: `reduce_non_fused_fp32_instructions_via_fma_restructuring` (0.188×)

Replacing non-fused FP32 multiply chains with libdevice.fma calls in a GELU kernel, combined with adding num_stages=2 and a new XBLOCK=256 autotune config, caused a catastrophic 5.3x regression (25 µs → 90 µs). The executed instruction count exploded 3.9x (15.4M → 60.3M), which is the dominant cause of the regression. The root cause is the same broken-grid anti-pattern seen in prior iterations: the autotune configs changed XBLOCK (adding XBLOCK=256 and reordering configs), but the grid was not updated to a lambda using meta['XBLOCK'], so the kernel ran with a fixed grid of 23,520 CTAs while each block processed a different number of elements than intended, causing massive redundant computation. The FMA arithmetic restructuring itself is not the cause — the instruction explosion is entirely attributable to the grid/autotune mismatch.

> **Applicable when**: Never apply this combined pattern. FMA restructuring via libdevice.fma is theoretically sound for reducing non-fused FP32 instruction counts, but only when the grid is correctly parameterized as a lambda. Adding num_stages to autotune configs is also valid in isolation.

> **Avoid when**: Do not apply any autotune config changes (including num_stages, new XBLOCK values, or reordering configs) when the grid remains a hardcoded fixed tuple. Every iteration that modifies autotune configs without fixing the grid to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` will cause the same instruction explosion regression. The FPInstructions NCU rule suggesting FMA conversion does NOT justify autotune config changes — these are orthogonal concerns.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_large_tensor_load | 1.007× | 0.992× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_correct_tile_exploration | 0.997× | 1.002× | ✓ |  |
| C | persistent_tile_scheduling_via_cta_loop | 0.995× | 1.000× | ✓ |  |
| D | fix_broken_autotune_grid_lambda_and_constrain_register_safe_configs | 1.010× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | fix_autotune_grid... | Δ | persistent_tile_s... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16960 ns | 17090 ns | +130 ns (+0.8%) | 16930 ns | -30 ns (-0.2%) | 16960 ns | +0 ns (+0.0%) | 16930 ns | -30 ns (-0.2%) |
| SM Throughput (%) | 41.59 % | 48.03 % | +6.44 pp | 44.58 % | +2.99 pp | 52.68 % | +11.09 pp | 44.49 % | +2.90 pp |
| Memory Throughput (%) | 18.34 % | 18.14 % | -0.20 pp | 18.38 % | +0.04 pp | 18.26 % | -0.08 pp | 18.36 % | +0.02 pp |
| DRAM Throughput (%) | 14.51 % | 14.37 % | -0.14 pp | 14.52 % | +0.01 pp | 14.49 % | -0.02 pp | 14.51 % | +0.00 pp |
| Achieved Occupancy (%) | 44.93 % | 46.27 % | +1.34 pp | 45.86 % | +0.93 pp | 49.79 % | +4.86 pp | 44.81 % | -0.12 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 52.00 % | 44.24 % | -7.76 pp | 48.00 % | -4.00 pp | 38.59 % | -13.41 pp | 47.79 % | -4.21 pp |
| IPC Active | 1.95 inst/cycle | 2.23 inst/cycle | +0.28 | 2.05 inst/cycle | +0.10 | 2.45 inst/cycle | +0.50 | 2.06 inst/cycle | +0.11 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.008× | 1.002× | ✓ |  |
| B | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.013× | 1.000× | ✓ | **✓ best** |
| C | fix_persistent_kernel_to_use_reduced_sm_count_grid | 1.007× | 1.002× | ✓ |  |
| D | increase_num_warps_within_fixed_xblock_for_l1tex_latency_hiding | 1.005× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | fix_persistent_ke... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16960 ns | 16930 ns | -30 ns (-0.2%) | 16960 ns | +0 ns (+0.0%) | 16930 ns | -30 ns (-0.2%) | 16930 ns | -30 ns (-0.2%) |
| SM Throughput (%) | 41.59 % | 44.38 % | +2.79 pp | 44.25 % | +2.66 pp | 52.88 % | +11.29 pp | 47.89 % | +6.30 pp |
| Memory Throughput (%) | 18.34 % | 18.31 % | -0.03 pp | 18.32 % | -0.02 pp | 18.44 % | +0.10 pp | 18.35 % | +0.01 pp |
| DRAM Throughput (%) | 14.51 % | 14.52 % | +0.01 pp | 14.48 % | -0.03 pp | 14.51 % | +0.00 pp | 14.51 % | +0.00 pp |
| Achieved Occupancy (%) | 44.93 % | 45.61 % | +0.68 pp | 45.60 % | +0.67 pp | 49.37 % | +4.44 pp | 45.33 % | +0.40 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 52.00 % | 47.63 % | -4.37 pp | 47.78 % | -4.22 pp | 38.20 % | -13.80 pp | 43.92 % | -8.08 pp |
| IPC Active | 1.95 inst/cycle | 2.06 inst/cycle | +0.11 | 2.05 inst/cycle | +0.10 | 2.46 inst/cycle | +0.51 | 2.25 inst/cycle | +0.30 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_sweep_num_warps_at_fixed_xblock | 0.994× | 1.000× | ✓ |  |
| B | correct_persistent_kernel_with_reduced_sm_count_grid | 1.007× | 1.000× | ✓ |  |
| C | shared_memory_bias_prefetch_to_eliminate_l1tex_stalls | 1.011× | 1.006× | ✓ |  |
| D | preload_bias_into_shared_memory_to_eliminate_l1tex_stalls | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | correct_persisten... | Δ | shared_memory_bia... | Δ | preload_bias_into... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16960 ns | 16960 ns | +0 ns (+0.0%) | 16960 ns | +0 ns (+0.0%) | 16860 ns | -100 ns (-0.6%) | 16960 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 41.59 % | 44.40 % | +2.81 pp | 52.92 % | +11.33 pp | 48.26 % | +6.67 pp | 40.24 % | -1.35 pp |
| Memory Throughput (%) | 18.34 % | 18.28 % | -0.06 pp | 18.35 % | +0.01 pp | 18.37 % | +0.03 pp | 18.29 % | -0.05 pp |
| DRAM Throughput (%) | 14.51 % | 14.51 % | +0.00 pp | 14.50 % | -0.01 pp | 14.56 % | +0.05 pp | 14.47 % | -0.04 pp |
| Achieved Occupancy (%) | 44.93 % | 45.51 % | +0.58 pp | 49.47 % | +4.54 pp | 46.82 % | +1.89 pp | 44.76 % | -0.17 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 52.00 % | 48.58 % | -3.42 pp | 38.13 % | -13.87 pp | 42.72 % | -9.28 pp | 53.00 % | +1.00 pp |
| IPC Active | 1.95 inst/cycle | 2.08 inst/cycle | +0.13 | 2.47 inst/cycle | +0.52 | 2.27 inst/cycle | +0.32 | 1.86 inst/cycle | -0.09 |

---

## Cumulative Progress

```
Baseline:  16960 ns  (1.000×)

Final best: 16960 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,16 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 128}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_addmm_gelu_view_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3010560
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 1536)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
+    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
     tmp2 = tmp0 + tmp1
     tmp3 = tl.full([1], 0.5, tl.float32)
     tmp4 = tmp2 * tmp3
@@ -24,16 +33,14 @@

     tmp8 = tl.full([1], 1.0, tl.float32)
     tmp9 = tmp7 + tmp8
     tmp10 = tmp4 * tmp9
-    tl.store(in_out_ptr0 + (x2), tmp10, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 3010560
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_gelu_view_36[grid](arg_0, arg_1, 3010560, XBLOCK=128)
+    triton_poi_fused_addmm_gelu_view_36[grid](arg_0, arg_1, 3010560)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

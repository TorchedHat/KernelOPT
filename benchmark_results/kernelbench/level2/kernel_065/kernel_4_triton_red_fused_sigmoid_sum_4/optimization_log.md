# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_sigmoid_sum_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 116930 ns — **Best latency**: 99230 ns (**1.1784×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`xblock1_num_warps16_autotune_to_hide_l1tex_stalls`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_sigmoid_sum_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_065/kernel_4_triton_red_fused_sigmoid_sum_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (640,1,1), Waves/SM: 0.40
- **Registers/thread**: 40
- **Duration**: 116930 ns
- **SM Throughput**: 41.42 % | **Memory**: 51.85 % | **DRAM**: 51.85 %
- **Achieved Occupancy**: 29.45 %
- **IPC Active**: 1.70 inst/cycle | **No Eligible**: 57.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune_tile_sizes | 1.000× | 0.991× | ✓ |  |
| B | memory_coalescing_via_xblock_increase | 1.001× | 0.208× | ✓ |  |
| C | software_pipelining_with_xblock_increase | 0.343× | 0.342× | ✓ |  |
| D | increase_xblock_and_autotune_tile_sizes | 1.002× | 1.007× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ | software_pipelini... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 116930 ns | 118050 ns | +1120 ns (+1.0%) | 562240 ns | +445310 ns (+380.8%) | 341790 ns | +224860 ns (+192.3%) | 116100 ns | -830 ns (-0.7%) |
| SM Throughput (%) | 41.42 % | 41.77 % | +0.35 pp | 67.38 % | +25.96 pp | 55.21 % | +13.79 pp | 41.59 % | +0.17 pp |
| Memory Throughput (%) | 51.85 % | 51.36 % | -0.49 pp | 10.78 % | -41.07 pp | 26.27 % | -25.58 pp | 52.23 % | +0.38 pp |
| DRAM Throughput (%) | 51.85 % | 51.36 % | -0.49 pp | 10.78 % | -41.07 pp | 17.77 % | -34.08 pp | 52.23 % | +0.38 pp |
| Achieved Occupancy (%) | 29.45 % | 29.40 % | -0.05 pp | 22.76 % | -6.69 pp | 15.50 % | -13.95 pp | 29.42 % | -0.03 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 55.00 register/thread | +15 | 128.00 register/thread | +88 | 40.00 register/thread | +0 |
| Waves/SM | 0.40 | 0.40 | +0.00 | 0.54 | +0.14 | 1.62 | +1.22 | 0.40 | +0.00 |
| No Eligible (%) | 57.19 % | 57.03 % | -0.16 pp | 33.63 % | -23.56 pp | 49.35 % | -7.84 pp | 57.06 % | -0.13 pp |
| IPC Active | 1.70 inst/cycle | 1.70 inst/cycle | +0.00 | 2.65 inst/cycle | +0.95 | 2.02 inst/cycle | +0.32 | 1.72 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 1.0071× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguity_hints | 1.001× | 1.005× | ✓ |  |
| B | increase_num_warps_for_latency_hiding_xblock1_only | 1.360× | 1.168× | ✓ |  |
| C | constrain_autotune_to_low_register_configs_with_vectorized_loads | 1.362× | 0.826× | ✓ | **✓ best** |
| D | fix_autotune_grid_lambda_and_vectorized_loads | 1.166× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | increase_num_warp... | Δ | constrain_autotun... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 116930 ns | 116350 ns | -580 ns (-0.5%) | 100130 ns | -16800 ns (-14.4%) | 141500 ns | +24570 ns (+21.0%) | 116740 ns | -190 ns (-0.2%) |
| SM Throughput (%) | 41.42 % | 41.76 % | +0.34 pp | 49.35 % | +7.93 pp | 33.53 % | -7.89 pp | 41.90 % | +0.48 pp |
| Memory Throughput (%) | 51.85 % | 52.13 % | +0.28 pp | 60.57 % | +8.72 pp | 42.83 % | -9.02 pp | 51.96 % | +0.11 pp |
| DRAM Throughput (%) | 51.85 % | 52.13 % | +0.28 pp | 60.57 % | +8.72 pp | 42.83 % | -9.02 pp | 51.96 % | +0.11 pp |
| Achieved Occupancy (%) | 29.45 % | 29.44 % | -0.01 pp | 29.19 % | -0.26 pp | 29.64 % | +0.19 pp | 29.43 % | -0.02 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 31.00 register/thread | -9 | 22.00 register/thread | -18 | 40.00 register/thread | +0 |
| Waves/SM | 0.40 | 0.40 | +0.00 | 0.30 | -0.10 | 0.30 | -0.10 | 0.40 | +0.00 |
| No Eligible (%) | 57.19 % | 57.15 % | -0.04 pp | 48.92 % | -8.27 pp | 64.85 % | +7.66 pp | 56.74 % | -0.45 pp |
| IPC Active | 1.70 inst/cycle | 1.70 inst/cycle | +0.00 | 2.07 inst/cycle | +0.37 | 1.41 inst/cycle | -0.29 | 1.72 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.0071× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipeline_reduction_loop_with_num_stages | 1.313× | 1.037× | ✓ |  |
| B | xblock1_num_warps16_autotune_to_hide_l1tex_stalls | 1.363× | 1.178× | ✓ | **✓ best** |
| C | constrain_xblock1_add_num_warps16_to_reduce_registers_and_hide_l1tex_latency | 1.360× | 1.183× | ✓ |  |
| D | add_num_warps_16_to_autotune_configs_xblock1_only | 1.359× | 0.832× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipeline... | Δ | xblock1_num_warps... | Δ | constrain_xblock1... | Δ | add_num_warps_16_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 116930 ns | 112740 ns | -4190 ns (-3.6%) | 99230 ns | -17700 ns (-15.1%) | 98820 ns | -18110 ns (-15.5%) | 140480 ns | +23550 ns (+20.1%) |
| SM Throughput (%) | 41.42 % | 44.91 % | +3.49 pp | 49.21 % | +7.79 pp | 48.93 % | +7.51 pp | 33.83 % | -7.59 pp |
| Memory Throughput (%) | 51.85 % | 53.90 % | +2.05 pp | 61.11 % | +9.26 pp | 61.32 % | +9.47 pp | 43.18 % | -8.67 pp |
| DRAM Throughput (%) | 51.85 % | 53.90 % | +2.05 pp | 61.11 % | +9.26 pp | 61.32 % | +9.47 pp | 43.18 % | -8.67 pp |
| Achieved Occupancy (%) | 29.45 % | 29.40 % | -0.05 pp | 29.22 % | -0.23 pp | 29.24 % | -0.21 pp | 29.57 % | +0.12 pp |
| Registers/Thread | 40.00 register/thread | 31.00 register/thread | -9 | 31.00 register/thread | -9 | 31.00 register/thread | -9 | 22.00 register/thread | -18 |
| Waves/SM | 0.40 | 0.30 | -0.10 | 0.30 | -0.10 | 0.30 | -0.10 | 0.30 | -0.10 |
| No Eligible (%) | 57.19 % | 52.86 % | -4.33 pp | 48.58 % | -8.61 pp | 48.62 % | -8.57 pp | 64.79 % | +7.60 pp |
| IPC Active | 1.70 inst/cycle | 1.89 inst/cycle | +0.19 | 2.05 inst/cycle | +0.35 | 2.05 inst/cycle | +0.35 | 1.40 inst/cycle | -0.30 |


### Key Learnings

#### ❌ Plan D: `add_num_warps_16_to_autotune_configs_xblock1_only` (0.832×)

Adding num_warps=16 autotune configs to a Triton reduction kernel with XBLOCK=1 and a small grid (640 CTAs / 132 SMs = 0.3 waves) failed to improve achieved occupancy (29.19% → 29.57%) and actually worsened warp stall cycles (9.08 → 13.36 cycles/instruction) and DRAM throughput (60.57% → 43.18%), causing a 17% slowdown. The autotune selected a config that also changed R0_BLOCK and applied tl.max_contiguous hints, which increased stall cycles rather than hiding them. When achieved occupancy is already far below theoretical and the bottleneck is wave-tail underutilization (waves_per_sm < 0.5), adding more warps per block does not increase the number of active warps per SM because the grid size is the binding constraint, not the block configuration.

> **Applicable when**: Never — this specific pattern caused regression. The general lesson applies when: waves_per_sm < 0.5, achieved occupancy is already far below theoretical occupancy, and the kernel is grid-size limited rather than register-limited.

> **Avoid when**: Do not add num_warps=16 autotune configs to Triton reduction kernels when waves_per_sm < 0.5 and achieved occupancy is already far below theoretical occupancy. In this regime, the number of active warps per SM is bounded by the grid size (CTAs/SMs), not by the warps-per-block configuration. Increasing num_warps per block cannot increase the total active warps when the grid is the binding constraint. Additionally, do not simultaneously change R0_BLOCK range and apply tl.max_contiguous hints with 2D shape arguments alongside num_warps changes, as the combined effect on the autotune selection can worsen stall cycles.

---

## Iteration 4

**Best so far**: 1.1784× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_last_load_policy_for_l2_reuse | 1.349× | 1.178× | ✓ |  |
| B | persistent_kernel_multi_x_per_cta_to_fix_wave_underutilization_and_store_coalescing | 0.444× | 0.311× | ✓ |  |
| C | coalesce_output_stores_via_small_xblock_increase | 0.794× | 0.596× | ✓ |  |
| D | coalesced_output_stores_via_xblock_with_correct_grid_lambda | 1.359× | 1.177× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_last_load_p... | Δ | persistent_kernel... | Δ | coalesce_output_s... | Δ | coalesced_output_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 116930 ns | 99260 ns | -17670 ns (-15.1%) | 375650 ns | +258720 ns (+221.3%) | 196260 ns | +79330 ns (+67.8%) | 99330 ns | -17600 ns (-15.1%) |
| SM Throughput (%) | 41.42 % | 49.74 % | +8.32 pp | 52.45 % | +11.03 pp | 49.13 % | +7.71 pp | 48.68 % | +7.26 pp |
| Memory Throughput (%) | 51.85 % | 61.19 % | +9.34 pp | 16.15 % | -35.70 pp | 30.96 % | -20.89 pp | 61.01 % | +9.16 pp |
| DRAM Throughput (%) | 51.85 % | 61.19 % | +9.34 pp | 16.15 % | -35.70 pp | 30.96 % | -20.89 pp | 61.01 % | +9.16 pp |
| Achieved Occupancy (%) | 29.45 % | 29.25 % | -0.20 pp | 22.16 % | -7.29 pp | 23.29 % | -6.16 pp | 29.17 % | -0.28 pp |
| Registers/Thread | 40.00 register/thread | 31.00 register/thread | -9 | 32.00 register/thread | -8 | 31.00 register/thread | -9 | 31.00 register/thread | -9 |
| Waves/SM | 0.40 | 0.30 | -0.10 | 0.30 | -0.10 | 0.30 | -0.10 | 0.30 | -0.10 |
| No Eligible (%) | 57.19 % | 48.93 % | -8.26 pp | 45.96 % | -11.23 pp | 47.66 % | -9.53 pp | 48.38 % | -8.81 pp |
| IPC Active | 1.70 inst/cycle | 2.04 inst/cycle | +0.34 | 2.16 inst/cycle | +0.46 | 2.09 inst/cycle | +0.39 | 2.08 inst/cycle | +0.38 |


### Key Learnings

#### ✅ Plan A: `evict_last_load_policy_for_l2_reuse` (1.178×)

Switching the Triton tl.load eviction_policy from 'evict_first' to 'evict_last' produced no measurable improvement in L2 hit rate (0.39% → 0.38%) and no reduction in DRAM throughput (61.32% → 61.19%), confirming that L2 reuse is not available in this access pattern. When each CTA reduces a distinct, non-overlapping row of a 2D tensor (stride = row_length apart between CTAs), adjacent CTAs access entirely disjoint memory regions, so evict_last cannot create inter-CTA L2 reuse. The 1.178x reported speedup appears to be measurement noise, as the NCU-measured kernel duration actually increased slightly (98.82 µs → 99.26 µs) and all key metrics are statistically unchanged.

> **Applicable when**: evict_last may help when multiple CTAs access overlapping or nearby memory regions (e.g., sliding-window reductions, shared weight matrices, or broadcast inputs), where L2 hit rate is currently low and the working set per CTA fits within L2 capacity.

> **Avoid when**: Do not apply evict_last when each CTA accesses a fully disjoint memory region (e.g., per-row reductions where row_stride equals the full row length). In this pattern, no two CTAs share cache lines, so evict_last cannot create L2 reuse and the policy change is a no-op. Also avoid when L2 hit rate is already near 0% and DRAM throughput is near peak — the bottleneck is bandwidth, not eviction policy.

#### ❌ Plan C: `coalesce_output_stores_via_small_xblock_increase` (0.596×)

Replacing XBLOCK=1 autotune configs with XBLOCK=2 and XBLOCK=4 variants (with corrected grid lambda) caused a 2x instruction count explosion (36M → 76M) and nearly doubled duration (98.82 µs → 196.26 µs). The autotune selected a config with num_stages=2 in the tl.range loop, which dramatically increased dynamic shared memory per block from 16 bytes to 4096 bytes (4 KB), reducing achieved occupancy from 29.24% to 23.29%. Despite the grid lambda being correctly updated, the larger XBLOCK tile combined with num_stages=2 pipelining caused the instruction count to balloon — likely because the autotune selected a smaller R0_BLOCK (512) requiring more loop iterations over the 115520-element reduction, and the software pipeline doubled the effective work per iteration. DRAM throughput dropped from 61.32% to 30.96%, indicating the kernel became less efficient at streaming data despite the intended coalescing improvement.

> **Applicable when**: Never — this specific combination caused regression. The general lesson: when a Triton reduction kernel has a very large reduction dimension (115520 elements) and small output (640 elements), increasing XBLOCK while including num_stages=2 in autotune configs will cause the autotune to select smaller R0_BLOCK + pipelining, exploding instruction count and shared memory usage.

> **Avoid when**: Do not add XBLOCK=2/4 autotune configs that include num_stages=2 to Triton reduction kernels where: (1) the reduction dimension is large (>100K elements), (2) the output is small (640 elements, 0.3 waves), and (3) the kernel is already DRAM-throughput-bound at >60%. The autotune will select smaller R0_BLOCK with pipelining, increasing loop iterations and smem usage (16 bytes → 4096 bytes), collapsing achieved occupancy and doubling instruction count. Also do not apply tl.max_contiguous on the outer r0_base with full R0_BLOCK alignment — this was reverted in the prior positive iteration for good reason.

---

## Iteration 5

**Best so far**: 1.1784× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_staged_coalesced_output_store | 0.053× | 0.040× | ✓ |  |
| B | fma_friendly_sigmoid_via_libdevice_tanh | 0.933× | 0.706× | ✓ |  |
| C | revert_xblock2_num_stages2_smem_explosion_to_xblock1_num_stages1 | 1.360× | 1.171× | ✓ |  |
| E | add_triton_autotune | 1.344× | 1.168× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ | fma_friendly_sigm... | Δ | revert_xblock2_nu... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 116930 ns | 2930000 ns | +2813070 ns (+2405.8%) | 165700 ns | +48770 ns (+41.7%) | 99840 ns | -17090 ns (-14.6%) | 100100 ns | -16830 ns (-14.4%) |
| SM Throughput (%) | 41.42 % | 57.96 % | +16.54 pp | 52.13 % | +10.71 pp | 48.81 % | +7.39 pp | 49.01 % | +7.59 pp |
| Memory Throughput (%) | 51.85 % | 3.67 % | -48.18 pp | 36.61 % | -15.24 pp | 60.78 % | +8.93 pp | 60.70 % | +8.85 pp |
| DRAM Throughput (%) | 51.85 % | 2.08 % | -49.77 pp | 36.61 % | -15.24 pp | 60.78 % | +8.93 pp | 60.70 % | +8.85 pp |
| Achieved Occupancy (%) | 29.45 % | 23.51 % | -5.94 pp | 29.61 % | +0.16 pp | 29.23 % | -0.22 pp | 29.24 % | -0.21 pp |
| Registers/Thread | 40.00 register/thread | 56.00 register/thread | +16 | 32.00 register/thread | -8 | 31.00 register/thread | -9 | 31.00 register/thread | -9 |
| Waves/SM | 0.40 | 0.54 | +0.14 | 0.30 | -0.10 | 0.30 | -0.10 | 0.30 | -0.10 |
| No Eligible (%) | 57.19 % | 27.45 % | -29.74 pp | 45.94 % | -11.25 pp | 49.14 % | -8.05 pp | 48.92 % | -8.27 pp |
| IPC Active | 1.70 inst/cycle | 2.90 inst/cycle | +1.20 | 2.16 inst/cycle | +0.46 | 2.05 inst/cycle | +0.35 | 2.06 inst/cycle | +0.36 |

---

## Cumulative Progress

```
Baseline:  116930 ns  (1.000×)
Iter 1     116100 ns  (1.007×)  — increase_xblock_and_autotune_tile_sizes
Iter 3     99230 ns  (1.178×)  — xblock1_num_warps16_autotune_to_hide_l1tex_stalls

Final best: 99230 ns  (1.1784×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_sigmoid_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 640
@@ -23,8 +38,8 @@

         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
-        r0_1 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_1 + 115520*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        r0_1 = tl.max_contiguous(tl.multiple_of(r0_index, [1, 16]), [1, 16])
+        tmp0 = tl.load(in_ptr0 + (r0_1 + 115520*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp1 = tl.sigmoid(tmp0)
         tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
         tmp4 = _tmp3 + tmp2
@@ -32,15 +47,13 @@

     tmp3 = tl.sum(_tmp3, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp3, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 640
     r0_numel = 115520
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_sigmoid_sum_4[grid](arg_0, arg_1, 640, 115520, XBLOCK=1, R0_BLOCK=2048)
+    triton_red_fused_sigmoid_sum_4[grid](arg_0, arg_1, 640, 115520)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

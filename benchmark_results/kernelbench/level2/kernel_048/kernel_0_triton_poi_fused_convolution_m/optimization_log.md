# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_mul_sigmoid_tanh_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 532100 ns — **Best latency**: 529280 ns (**1.0053×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`restructure_to_2d_grid_hoist_channel_scalars`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_mul_sigmoid_tanh_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_048/kernel_0_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (861056,1,1), Waves/SM: 407.70
- **Registers/thread**: 16
- **Duration**: 532100 ns
- **SM Throughput**: 53.96 % | **Memory**: 32.71 % | **DRAM**: 32.71 %
- **Achieved Occupancy**: 53.28 %
- **IPC Active**: 2.17 inst/cycle | **No Eligible**: 45.71 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 0.820× | 0.725× | ✓ |  |
| B | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.594× | 0.574× | ✓ |  |
| C | autotune_xblock_num_warps_to_hide_l1tex_latency | 0.998× | 0.573× | ✓ |  |
| D | vectorized_loads_stores_with_larger_xblock_autotune | 0.595× | 0.292× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 532100 ns | 734210 ns | +202110 ns (+38.0%) | 927780 ns | +395680 ns (+74.4%) | 928260 ns | +396160 ns (+74.5%) | 1820000 ns | +1287900 ns (+242.0%) |
| SM Throughput (%) | 53.96 % | 37.39 % | -16.57 pp | 92.48 % | +38.52 pp | 92.56 % | +38.60 pp | 92.73 % | +38.77 pp |
| Memory Throughput (%) | 32.71 % | 23.71 % | -9.00 pp | 30.69 % | -2.02 pp | 30.70 % | -2.01 pp | 29.13 % | -3.58 pp |
| DRAM Throughput (%) | 32.71 % | 23.71 % | -9.00 pp | 18.76 % | -13.95 pp | 18.75 % | -13.96 pp | 9.59 % | -23.12 pp |
| Achieved Occupancy (%) | 53.28 % | 34.60 % | -18.68 pp | 88.09 % | +34.81 pp | 88.17 % | +34.89 pp | 87.24 % | +33.96 pp |
| Registers/Thread | 16.00 register/thread | 40.00 register/thread | +24 | 28.00 register/thread | +12 | 28.00 register/thread | +12 | 28.00 register/thread | +12 |
| Waves/SM | 407.70 | 543.60 | +135.90 | 407.70 | +0.00 | 407.70 | +0.00 | 815.39 | +407.69 |
| No Eligible (%) | 45.71 % | 51.16 % | +5.45 pp | 7.31 % | -38.40 pp | 7.22 % | -38.49 pp | 7.17 % | -38.54 pp |
| IPC Active | 2.17 inst/cycle | 1.81 inst/cycle | -0.36 | 3.71 inst/cycle | +1.54 | 3.71 inst/cycle | +1.54 | 3.71 inst/cycle | +1.54 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.574×)

Increasing XBLOCK via @triton.autotune on a pointwise kernel successfully raised achieved occupancy (53% → 88%) and scheduler eligibility (0.91 → 9.71 warps), but the total executed instruction count tripled (227M → 675M) and registers per thread nearly doubled (16 → 28), causing a 1.74x slowdown. The instruction explosion was caused by the autotune selecting a large XBLOCK config that introduced a non-trivial xmask (xindex < xnumel) where the original kernel used tl.full([XBLOCK], True) — adding predication overhead across all loads and stores — combined with the larger tile size generating far more loop iterations per block. When the original kernel already has near-zero register pressure and a fully-unmasked access pattern, adding autotune with masked loads inflates instruction count enough to negate any latency-hiding benefit.

> **Applicable when**: Kernel has low eligible warps (<1.0/scheduler), high L1TEX scoreboard stall (>50% of CPI), low achieved occupancy (<60%), and the original XBLOCK is small (≤128). Registers per thread must remain low after XBLOCK increase to avoid occupancy regression.

> **Avoid when**: The original kernel uses tl.full([XBLOCK], True) masking (fully unmasked loads/stores) — switching to xindex < xnumel masking when adding autotune introduces predication overhead that triples instruction count. Also avoid when the kernel already has many waves per SM (>100) and the bottleneck is not purely latency hiding. Do not apply when register count will increase significantly (e.g., 16→28) as this halves Block Limit Registers and reduces theoretical occupancy gains.

#### ❌ Plan C: `autotune_xblock_num_warps_to_hide_l1tex_latency` (0.573×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) and more warps (8) successfully raised achieved occupancy from 53% to 88% and reduced no-eligible-warp cycles from 45.7% to 7.2%, but caused a 1.74x slowdown. The root cause is that the autotune wrapper introduced a bug: the grid was left hardcoded at cdiv(xnumel, 128) instead of being updated to a lambda using meta['XBLOCK'], so larger XBLOCK configs launched far fewer CTAs than needed, causing massive underutilization. Additionally, registers per thread jumped from 16 to 28 (due to the xmask computation being added), which halved the Block Limit Registers from 32 to 16, and executed instructions tripled (227M → 675M), indicating the selected config was doing ~3x more work per element. The anti-pattern is: adding autotune without simultaneously fixing the grid lambda AND without verifying that the autotuner actually selects a config that reduces total work.

> **Applicable when**: Kernel has high no-eligible-warp % (>40%) and high L1TEX scoreboard stall cycles (>8 cycles avg), indicating latency-bound execution that could benefit from more in-flight memory requests per CTA.

> **Avoid when**: When adding autotune without correctly updating the grid to use meta['XBLOCK']; when the xmask addition significantly increases register pressure (registers/thread jump >50%); when the kernel body already has no masking and xnumel is divisible by all candidate XBLOCK values (adding xmask is unnecessary and costly); when the autotune overhead itself is not amortized over enough kernel calls.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_persistent_loop_restore_single_pass_with_more_warps | 1.000× | 1.000× | ✓ |  |
| B | replace_transcendental_ops_with_fma_friendly_fast_math_approximations | 0.603× | 0.598× | ✓ |  |
| C | fuse_nonfused_fp32_instructions_into_fma | 0.999× | 0.557× | ✓ |  |
| D | remove_broken_autotune_restore_baseline_xblock128 | 1.000× | 1.000× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.998× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_persistent... | Δ | replace_transcend... | Δ | fuse_nonfused_fp3... | Δ | remove_broken_aut... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 532100 ns | 531840 ns | -260 ns (-0.0%) | 889500 ns | +357400 ns (+67.2%) | 955900 ns | +423800 ns (+79.6%) | 531900 ns | -200 ns (-0.0%) | 532350 ns | +250 ns (+0.0%) |
| SM Throughput (%) | 53.96 % | 53.97 % | +0.01 pp | 91.30 % | +37.34 pp | 92.55 % | +38.59 pp | 54.06 % | +0.10 pp | 72.01 % | +18.05 pp |
| Memory Throughput (%) | 32.71 % | 32.72 % | +0.01 pp | 32.15 % | -0.56 pp | 29.75 % | -2.96 pp | 32.73 % | +0.02 pp | 32.69 % | -0.02 pp |
| DRAM Throughput (%) | 32.71 % | 32.72 % | +0.01 pp | 19.56 % | -13.15 pp | 18.21 % | -14.50 pp | 32.73 % | +0.02 pp | 32.69 % | -0.02 pp |
| Achieved Occupancy (%) | 53.28 % | 53.29 % | +0.01 pp | 87.16 % | +33.88 pp | 88.19 % | +34.91 pp | 53.20 % | -0.08 pp | 62.67 % | +9.39 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 28.00 register/thread | +12 | 32.00 register/thread | +16 | 16.00 register/thread | +0 | 26.00 register/thread | +10 |
| Waves/SM | 407.70 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 |
| No Eligible (%) | 45.71 % | 45.81 % | +0.10 pp | 8.32 % | -37.39 pp | 7.33 % | -38.38 pp | 45.79 % | +0.08 pp | 27.85 % | -17.86 pp |
| IPC Active | 2.17 inst/cycle | 2.17 inst/cycle | +0.00 | 3.67 inst/cycle | +1.50 | 3.70 inst/cycle | +1.53 | 2.17 inst/cycle | +0.00 | 2.89 inst/cycle | +0.72 |


### Key Learnings

#### ❌ Plan B: `replace_transcendental_ops_with_fma_friendly_fast_math_approximations` (0.598×)

Replacing libdevice.tanh with an inline Pade [3,3] approximant and tl.sigmoid with libdevice.fast_expf reduced registers per thread (40→28) and raised achieved occupancy (34.6%→87.2%), but tripled total executed instructions (215M→638M), causing a 1.67x slowdown. The Pade approximant requires multiple FMA operations plus a division and a tl.where clamp guard, generating far more instructions than the original single libdevice.tanh call. The net effect is that the compute SM throughput rose from 37% to 91% but the kernel ran longer because it was doing ~3x more total work. The anti-pattern is substituting a single hardware-accelerated transcendental call with a multi-step software polynomial that, despite being FMA-composable, emits more total instructions than the original.

> **Applicable when**: Never apply inline polynomial approximations (Pade, minimax, etc.) to replace libdevice transcendental calls in Triton kernels on Hopper/H100/H200 hardware, as libdevice already maps to hardware MUFU instructions. The instruction count explosion from the polynomial body dominates any register pressure savings.

> **Avoid when**: When the original kernel uses libdevice.tanh or tl.sigmoid — these already map to hardware MUFU instructions on Hopper-class GPUs and are not software polynomials. Replacing them with inline polynomial approximations (Pade, Taylor, etc.) will always increase instruction count. Also avoid when the FPInstructions NCU rule is triggered but the kernel is not actually using software polynomial transcendentals — the non-fused instruction count may have a different root cause (e.g., loop unrolling, autotune config selection).

#### ❌ Plan C: `fuse_nonfused_fp32_instructions_into_fma` (0.557×)

Replacing FADD+FMUL pairs with explicit libdevice.fma calls while simultaneously expanding the autotune config set caused executed instructions to jump from 215M to 696M (~3.2x), resulting in a 1.8x slowdown despite SM compute throughput rising from 37% to 93% and occupancy improving from 35% to 88%. The instruction explosion was not caused by the FMA substitution itself but by the autotune selecting a config that processed far more total work — the combination of more autotune configs (including larger XBLOCK variants) and the removal of the persistent-loop structure changed the effective work-per-block ratio. The FPInstructions NCU rule projects gains only when the kernel is already compute-saturated and instruction count is the binding constraint; if the autotune simultaneously inflates instruction count by 3x, any per-instruction savings are overwhelmed.

> **Applicable when**: Never apply libdevice.fma substitution in isolation when the autotune config set is simultaneously being expanded — the confounded change makes it impossible to attribute the instruction count change to FMA vs autotune config selection. The FPInstructions rule is only actionable when the kernel is already compute-saturated (>90% SM throughput) AND instruction count is stable across the change.

> **Avoid when**: When the autotune config set is being changed at the same time as arithmetic restructuring — the two changes interact and the autotune may select a config that inflates total instructions by more than the FMA substitution saves. Also avoid when the kernel is not yet compute-saturated (SM throughput <80%); the FPInstructions rule's projected gain assumes compute is the binding constraint. Do not apply when the persistent-loop structure is being removed simultaneously, as this changes the effective work-per-CTA ratio and confounds the measurement.

---

## Iteration 3

**Best so far**: 1.0004× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_launch_to_hide_l1tex_latency | 1.000× | 1.001× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_latency_without_xblock_change | 1.000× | 1.000× | ✓ |  |
| C | revert_pade_approximation_restore_libdevice_hardware_transcendentals | 0.998× | 1.000× | ✓ |  |
| D | replace_degenerate_fma_with_native_multiply_to_enable_fma_fusion | 0.999× | 0.575× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | revert_pade_appro... | Δ | replace_degenerat... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 532100 ns | 531680 ns | -420 ns (-0.1%) | 531900 ns | -200 ns (-0.0%) | 531900 ns | -200 ns (-0.0%) | 926080 ns | +393980 ns (+74.0%) |
| SM Throughput (%) | 53.96 % | 54.05 % | +0.09 pp | 54.11 % | +0.15 pp | 62.95 % | +8.99 pp | 92.60 % | +38.64 pp |
| Memory Throughput (%) | 32.71 % | 32.73 % | +0.02 pp | 32.72 % | +0.01 pp | 32.72 % | +0.01 pp | 30.79 % | -1.92 pp |
| DRAM Throughput (%) | 32.71 % | 32.73 % | +0.02 pp | 32.72 % | +0.01 pp | 32.72 % | +0.01 pp | 18.79 % | -13.92 pp |
| Achieved Occupancy (%) | 53.28 % | 53.22 % | -0.06 pp | 53.23 % | -0.05 pp | 56.30 % | +3.02 pp | 88.01 % | +34.73 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 28.00 register/thread | +12 |
| Waves/SM | 407.70 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 |
| No Eligible (%) | 45.71 % | 45.77 % | +0.06 pp | 45.80 % | +0.09 pp | 36.79 % | -8.92 pp | 7.31 % | -38.40 pp |
| IPC Active | 2.17 inst/cycle | 2.17 inst/cycle | +0.00 | 2.17 inst/cycle | +0.00 | 2.53 inst/cycle | +0.36 | 3.71 inst/cycle | +1.54 |


### Key Learnings

#### ❌ Plan D: `replace_degenerate_fma_with_native_multiply_to_enable_fma_fusion` (0.575×)

Replacing libdevice.fma(a, b, 0.0) with native a*b multiplication (to enable LLVM FFMA fusion) while simultaneously adding @triton.autotune with larger XBLOCK configs caused executed instructions to nearly triple (227M → 674M) and duration to nearly double (532us → 926us), a 1.74x slowdown. The FMA substitution itself is arithmetically correct and theoretically sound, but the confounding autotune addition selected a config that inflated total instruction count far beyond any per-instruction savings from FFMA fusion. Additionally, registers per thread jumped from 16 to 28 (due to the xmask = xindex < xnumel addition), halving Block Limit Registers and reducing occupancy headroom. The anti-pattern is conflating two independent changes — arithmetic restructuring and autotune config expansion — making it impossible to isolate the FMA fusion benefit and guaranteeing regression when the autotune inflates work.

> **Applicable when**: Never apply arithmetic restructuring (FMA substitution) simultaneously with autotune config expansion. The FMA fusion strategy itself may be valid in isolation when: the kernel is compute-saturated (>80% SM throughput), FPInstructions NCU rule fires with >10% projected gain, and XBLOCK and autotune configs are held constant.

> **Avoid when**: When adding @triton.autotune at the same time as arithmetic changes — the autotune config selection will dominate instruction count changes and mask any FMA fusion benefit. Also avoid when the original kernel uses no xmask (tl.full True mask) and adding xmask = xindex < xnumel would increase register pressure (registers/thread jump >50%). Do not apply when the grid is not updated to a lambda using meta['XBLOCK'] — the hardcoded grid causes incorrect launch geometry with larger XBLOCK configs.

---

## Iteration 4

**Best so far**: 1.0004× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_on_streaming_large_tensor_load_to_reduce_l1tex_pressure | 1.000× | 1.001× | ✓ |  |
| B | restructure_to_2d_grid_hoist_channel_scalars | 1.006× | 1.005× | ✓ | **✓ best** |
| C | preload_small_channel_tensors_into_registers_to_eliminate_strided_global_loads | 0.998× | 0.999× | ✓ |  |
| D | strip_autotune_configs_to_xblock128_only_restore_fast_baseline | 0.998× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_on_st... | Δ | restructure_to_2d... | Δ | preload_small_cha... | Δ | strip_autotune_co... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 532100 ns | 531780 ns | -320 ns (-0.1%) | 529280 ns | -2820 ns (-0.5%) | 532800 ns | +700 ns (+0.1%) | 532060 ns | -40 ns (-0.0%) |
| SM Throughput (%) | 53.96 % | 58.05 % | +4.09 pp | 46.83 % | -7.13 pp | 71.85 % | +17.89 pp | 62.98 % | +9.02 pp |
| Memory Throughput (%) | 32.71 % | 32.72 % | +0.01 pp | 17.01 % | -15.70 pp | 49.08 % | +16.37 pp | 32.71 % | +0.00 pp |
| DRAM Throughput (%) | 32.71 % | 32.72 % | +0.01 pp | 0.01 % | -32.70 pp | 32.67 % | -0.04 pp | 32.71 % | +0.00 pp |
| Achieved Occupancy (%) | 53.28 % | 53.76 % | +0.48 pp | 18.52 % | -34.76 pp | 66.35 % | +13.07 pp | 56.35 % | +3.07 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 407.70 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 |
| No Eligible (%) | 45.71 % | 41.59 % | -4.12 pp | 52.30 % | +6.59 pp | 31.23 % | -14.48 pp | 36.63 % | -9.08 pp |
| IPC Active | 2.17 inst/cycle | 2.34 inst/cycle | +0.17 | 1.90 inst/cycle | -0.27 | 2.76 inst/cycle | +0.59 | 2.53 inst/cycle | +0.36 |

---

## Iteration 5

**Best so far**: 1.0053× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_spatial_loop | 0.999× | 1.000× | ✓ |  |
| B | hoist_channel_scalars_into_shared_memory_to_eliminate_strided_global_loads | 0.999× | 0.998× | ✓ |  |
| C | persistent_cta_loop_to_reduce_launch_overhead_and_improve_sm_utilization | 0.998× | 0.999× | ✓ |  |
| D | replace_degenerate_gather_with_scalar_load_and_loop_tiles_per_cta | 0.203× | 0.201× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | hoist_channel_sca... | Δ | persistent_cta_lo... | Δ | replace_degenerat... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 532100 ns | 532190 ns | +90 ns (+0.0%) | 532930 ns | +830 ns (+0.2%) | 532450 ns | +350 ns (+0.1%) | 2650000 ns | +2117900 ns (+398.0%) |
| SM Throughput (%) | 53.96 % | 58.02 % | +4.06 pp | 71.85 % | +17.89 pp | 63.02 % | +9.06 pp | 76.85 % | +22.89 pp |
| Memory Throughput (%) | 32.71 % | 32.72 % | +0.01 pp | 49.04 % | +16.33 pp | 32.69 % | -0.02 pp | 72.92 % | +40.21 pp |
| DRAM Throughput (%) | 32.71 % | 32.72 % | +0.01 pp | 32.67 % | -0.04 pp | 32.69 % | -0.02 pp | 6.56 % | -26.15 pp |
| Achieved Occupancy (%) | 53.28 % | 59.25 % | +5.97 pp | 67.14 % | +13.86 pp | 59.77 % | +6.49 pp | 97.67 % | +44.39 pp |
| Registers/Thread | 16.00 register/thread | 27.00 register/thread | +11 | 16.00 register/thread | +0 | 26.00 register/thread | +10 | 27.00 register/thread | +11 |
| Waves/SM | 407.70 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 |
| No Eligible (%) | 45.71 % | 41.60 % | -4.11 pp | 28.68 % | -17.03 pp | 36.73 % | -8.98 pp | 23.07 % | -22.64 pp |
| IPC Active | 2.17 inst/cycle | 2.33 inst/cycle | +0.16 | 2.86 inst/cycle | +0.69 | 2.53 inst/cycle | +0.36 | 3.08 inst/cycle | +0.91 |

---

## Cumulative Progress

```
Baseline:  532100 ns  (1.000×)
Iter 2     531900 ns  (1.000×)  — remove_broken_autotune_restore_baseline_xblock128
Iter 4     529280 ns  (1.005×)  — restructure_to_2d_grid_hoist_channel_scalars

Final best: 529280 ns  (1.0053×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_poi_fused_convolution_mul_sigmoid_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 110215168
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x3 = xindex
-    x1 = ((xindex // 53816) % 16)
-    tmp0 = tl.load(in_out_ptr0 + (x3), None)
-    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
-    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
+def triton_poi_fused_convolution_mul_sigmoid_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK: tl.constexpr):
+    SPATIAL_SIZE: tl.constexpr = 53816
+    pid_s = tl.program_id(0)   # spatial tile index
+    pid_c = tl.program_id(1)   # batch*channel index (0..2047)
+    # Hoist channel scalar loads ONCE per CTA - eliminates per-element IDIV
+    chan_idx = pid_c % 16
+    tmp1 = tl.load(in_ptr0 + chan_idx)
+    tmp3 = tl.load(in_ptr1 + chan_idx)
+    tmp6 = tl.load(in_ptr2 + chan_idx)
+    # Spatial tile with boundary mask (53816 not divisible by 128)
+    sidx = pid_s * XBLOCK + tl.arange(0, XBLOCK)[:]
+    xmask = sidx < SPATIAL_SIZE
+    flat_idx = pid_c * SPATIAL_SIZE + sidx
+    tmp0 = tl.load(in_out_ptr0 + flat_idx, xmask)
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 * tmp3
     tmp5 = libdevice.tanh(tmp4)
     tmp7 = tmp5 * tmp6
     tmp8 = tl.sigmoid(tmp7)
-    tl.store(in_out_ptr0 + (x3), tmp8, None)
-
-
+    tl.store(in_out_ptr0 + flat_idx, tmp8, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

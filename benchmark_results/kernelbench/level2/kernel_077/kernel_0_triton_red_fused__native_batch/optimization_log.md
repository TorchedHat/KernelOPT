# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_mul_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 53920 ns — **Best latency**: 93860 ns (**0.5745×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_mul_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_077/kernel_0_triton_red_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 0.97
- **Registers/thread**: 32
- **Duration**: 53920 ns
- **SM Throughput**: 43.01 % | **Memory**: 81.09 % | **DRAM**: 81.09 %
- **Achieved Occupancy**: 93.71 %
- **IPC Active**: 1.83 inst/cycle | **No Eligible**: 54.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_via_xblock_autotune | 1.003× | 0.037× | ✓ | **✓ best** |
| C | autotune_xblock_r0block_with_software_pipelining | 1.001× | 0.983× | ✓ |  |
| D | increase_xblock_for_coalesced_stores_and_cache_reuse | 0.998× | 0.178× | ✓ |  |
| E | add_triton_autotune | 1.000× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | autotune_xblock_r... | Δ | increase_xblock_f... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 53920 ns | 1450000 ns | +1396080 ns (+2589.2%) | 54850 ns | +930 ns (+1.7%) | 302780 ns | +248860 ns (+461.5%) | 54210 ns | +290 ns (+0.5%) |
| SM Throughput (%) | 43.01 % | 74.35 % | +31.34 pp | 42.32 % | -0.69 pp | 85.70 % | +42.69 pp | 36.89 % | -6.12 pp |
| Memory Throughput (%) | 81.09 % | 32.17 % | -48.92 pp | 79.68 % | -1.41 pp | 16.26 % | -64.83 pp | 80.60 % | -0.49 pp |
| DRAM Throughput (%) | 81.09 % | 3.14 % | -77.95 pp | 79.68 % | -1.41 pp | 14.42 % | -66.67 pp | 80.60 % | -0.49 pp |
| Achieved Occupancy (%) | 93.71 % | 12.38 % | -81.33 pp | 93.57 % | -0.14 pp | 17.90 % | -75.81 pp | 93.50 % | -0.21 pp |
| Registers/Thread | 32.00 register/thread | 255.00 register/thread | +223 | 32.00 register/thread | +0 | 166.00 register/thread | +134 | 28.00 register/thread | -4 |
| Waves/SM | 0.97 | 7.76 | +6.79 | 0.97 | +0.00 | 5.17 | +4.20 | 0.97 | +0.00 |
| No Eligible (%) | 54.20 % | 19.10 % | -35.10 pp | 55.39 % | +1.19 pp | 11.64 % | -42.56 pp | 59.58 % | +5.38 pp |
| IPC Active | 1.83 inst/cycle | 3.24 inst/cycle | +1.41 | 1.81 inst/cycle | -0.02 | 3.53 inst/cycle | +1.70 | 1.61 inst/cycle | -0.22 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.003× | 1.008× | ✓ |  |
| B | vectorized_loads_via_tl_multiple_of_and_max_contiguous_hints | 1.001× | 0.989× | ✓ |  |
| C | hoist_loop_invariant_computations_to_reduce_register_pressure | 1.000× | 0.178× | ✓ |  |
| D | reduce_register_pressure_via_scalar_accumulator | 1.011× | 0.995× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | hoist_loop_invari... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 53920 ns | 53500 ns | -420 ns (-0.8%) | 54530 ns | +610 ns (+1.1%) | 302530 ns | +248610 ns (+461.1%) | 54210 ns | +290 ns (+0.5%) |
| SM Throughput (%) | 43.01 % | 37.29 % | -5.72 pp | 42.45 % | -0.56 pp | 85.67 % | +42.66 pp | 55.32 % | +12.31 pp |
| Memory Throughput (%) | 81.09 % | 81.70 % | +0.61 pp | 80.10 % | -0.99 pp | 16.25 % | -64.84 pp | 80.56 % | -0.53 pp |
| DRAM Throughput (%) | 81.09 % | 81.70 % | +0.61 pp | 80.10 % | -0.99 pp | 14.44 % | -66.65 pp | 80.56 % | -0.53 pp |
| Achieved Occupancy (%) | 93.71 % | 93.71 % | +0.00 pp | 93.55 % | -0.16 pp | 17.90 % | -75.81 pp | 92.54 % | -1.17 pp |
| Registers/Thread | 32.00 register/thread | 28.00 register/thread | -4 | 32.00 register/thread | +0 | 166.00 register/thread | +134 | 27.00 register/thread | -5 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 5.17 | +4.20 | 0.97 | +0.00 |
| No Eligible (%) | 54.20 % | 60.34 % | +6.14 pp | 54.60 % | +0.40 pp | 11.67 % | -42.53 pp | 40.23 % | -13.97 pp |
| IPC Active | 1.83 inst/cycle | 1.61 inst/cycle | -0.22 | 1.83 inst/cycle | +0.00 | 3.54 inst/cycle | +1.71 | 2.38 inst/cycle | +0.55 |


### Key Learnings

#### ❌ Plan C: `hoist_loop_invariant_computations_to_reduce_register_pressure` (0.575×)

Hoisting loop-invariant BN scale computations (replacing tl.sqrt_rn + integer division with tl.rsqrt) outside the reduction loop, while keeping XBLOCK=1 in all autotune configs, caused executed instructions to increase by ~63% (25.0M → 40.7M) and kernel duration to increase by ~52% (61.76 us → 93.86 us). Critically, branch instructions exploded by ~5.5x (454K → 2.52M), indicating the new autotune config set (adding R0_BLOCK=256 and R0_BLOCK=2048 options) caused autotune to select a smaller R0_BLOCK that runs far more loop iterations, dramatically increasing loop control overhead. Registers per thread actually decreased slightly (28 → 26) and occupancy was unchanged (~92%), confirming the regression was not caused by register pressure but by the autotune selecting a suboptimal R0_BLOCK that multiplied loop overhead. The tl.max_contiguous/tl.multiple_of hints and the store eviction_policy change may also have contributed to the instruction count increase.

> **Applicable when**: Never applicable as described for this kernel pattern. Hoisting loop-invariant scalar computations (sqrt, rsqrt) in a DRAM-bandwidth-bound reduction kernel with XBLOCK=1 does not address the actual bottleneck and risks autotune selecting a suboptimal R0_BLOCK from an expanded config set.

> **Avoid when**: Do not hoist loop-invariant computations in a DRAM-bandwidth-bound reduction kernel (DRAM Throughput >60%) when simultaneously expanding the autotune config set with new R0_BLOCK values. The autotune may select a smaller R0_BLOCK (e.g., 256 instead of 1024), causing the loop to run 4x more iterations and multiplying branch/loop-control instruction overhead (branch count increased 5.5x here). Even if registers per thread decrease slightly, the instruction count explosion dominates. Additionally, adding tl.max_contiguous/tl.multiple_of hints and changing store eviction policy alongside the hoisting makes it impossible to isolate the causal factor. The CPIStall and IssueSlotUtilization NCU rules firing on a memory-bound kernel indicate the bottleneck is DRAM latency, not arithmetic throughput — hoisting scalar FP ops does not reduce memory latency.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | l2_compression_via_eviction_policy_and_scalar_accumulator_restructure | 0.995× | 0.987× | ✓ |  |
| B | replace_manual_rsqrt_with_hardware_rsqrt_to_enable_fma_fusion | 1.010× | 0.989× | ✓ |  |
| C | hoist_scalar_loop_invariants_xblock1_only_no_autotune_change | 0.996× | 0.984× | ✓ |  |
| D | fma_fusion_to_reduce_non_fused_fp32_instructions | 1.003× | 0.865× | ✓ |  |


### Metric Comparison

| Metric | Baseline | l2_compression_vi... | Δ | replace_manual_rs... | Δ | hoist_scalar_loop... | Δ | fma_fusion_to_red... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 53920 ns | 54620 ns | +700 ns (+1.3%) | 54530 ns | +610 ns (+1.1%) | 54780 ns | +860 ns (+1.6%) | 62370 ns | +8450 ns (+15.7%) |
| SM Throughput (%) | 43.01 % | 41.21 % | -1.80 pp | 51.26 % | +8.25 pp | 42.57 % | -0.44 pp | 43.34 % | +0.33 pp |
| Memory Throughput (%) | 81.09 % | 79.99 % | -1.10 pp | 80.16 % | -0.93 pp | 79.76 % | -1.33 pp | 70.13 % | -10.96 pp |
| DRAM Throughput (%) | 81.09 % | 79.99 % | -1.10 pp | 80.16 % | -0.93 pp | 79.76 % | -1.33 pp | 70.13 % | -10.96 pp |
| Achieved Occupancy (%) | 93.71 % | 93.27 % | -0.44 pp | 92.87 % | -0.84 pp | 93.23 % | -0.48 pp | 93.88 % | +0.17 pp |
| Registers/Thread | 32.00 register/thread | 28.00 register/thread | -4 | 28.00 register/thread | -4 | 32.00 register/thread | +0 | 23.00 register/thread | -9 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 54.20 % | 56.66 % | +2.46 pp | 46.09 % | -8.11 pp | 54.52 % | +0.32 pp | 53.14 % | -1.06 pp |
| IPC Active | 1.83 inst/cycle | 1.75 inst/cycle | -0.08 | 2.16 inst/cycle | +0.33 | 1.79 inst/cycle | -0.04 | 1.89 inst/cycle | +0.06 |


### Key Learnings

#### ❌ Plan D: `fma_fusion_to_reduce_non_fused_fp32_instructions` (0.865×)

Replacing separate FADD/FMUL/FSUB chains with explicit libdevice.fma calls and switching autotune configs from num_stages=1 to num_stages=2 caused executed instructions to increase by ~35% (15.6M → 21.0M) and kernel duration to increase by ~17% (53.5 us → 62.37 us), despite registers per thread dropping from 28 to 23 and occupancy remaining stable (~94%). The branch instruction count nearly doubled (454K → 874K), suggesting num_stages=2 introduced additional software-pipelining control flow overhead that outweighed any FMA throughput benefit. The kernel was memory-bandwidth-bound (DRAM Throughput 81.7% before), not compute-bound, so reducing FP instruction count via FMA fusion did not address the actual bottleneck; the NCU FPInstructions rule's estimated speedup was misleading in this context.

> **Applicable when**: Kernel is genuinely compute-bound (Compute SOL >> Memory SOL, e.g., >70% compute vs <30% memory) AND the FPInstructions NCU rule fires with a large non-fused instruction count AND num_stages=1 is already confirmed optimal.

> **Avoid when**: Do not apply FMA fusion via libdevice.fma when the kernel is memory-bandwidth-bound (DRAM Throughput >70%). The FPInstructions NCU rule can fire even on memory-bound kernels, but the estimated speedup is misleading — reducing compute instructions does not help when DRAM is the bottleneck. Also do not simultaneously switch from num_stages=1 to num_stages=2 in autotune configs, as software pipelining adds branch overhead (branch instructions nearly doubled here) that can increase total instruction count and duration even when register pressure is stable.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_scalar_accumulator_for_coalesced_stores | 1.010× | 0.551× | ✓ |  |
| B | scalar_accumulator_xblock2_coalesced_store | 1.010× | 0.871× | ✓ |  |
| C | scalar_accumulator_with_increased_num_warps_for_latency_hiding | 1.009× | 0.820× | ✓ |  |
| D | scalar_accumulator_xblock_increase_for_coalesced_stores | 1.013× | 0.873× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | scalar_accumulato... | Δ | scalar_accumulato... | Δ | scalar_accumulato... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 53920 ns | 97920 ns | +44000 ns (+81.6%) | 61890 ns | +7970 ns (+14.8%) | 65730 ns | +11810 ns (+21.9%) | 61760 ns | +7840 ns (+14.5%) |
| SM Throughput (%) | 43.01 % | 84.03 % | +41.02 pp | 53.00 % | +9.99 pp | 53.50 % | +10.49 pp | 53.29 % | +10.28 pp |
| Memory Throughput (%) | 81.09 % | 54.11 % | -26.98 pp | 70.60 % | -10.49 pp | 66.42 % | -14.67 pp | 70.76 % | -10.33 pp |
| DRAM Throughput (%) | 81.09 % | 44.61 % | -36.48 pp | 70.60 % | -10.49 pp | 66.42 % | -14.67 pp | 70.76 % | -10.33 pp |
| Achieved Occupancy (%) | 93.71 % | 59.74 % | -33.97 pp | 92.64 % | -1.07 pp | 93.99 % | +0.28 pp | 91.82 % | -1.89 pp |
| Registers/Thread | 32.00 register/thread | 40.00 register/thread | +8 | 28.00 register/thread | -4 | 28.00 register/thread | -4 | 28.00 register/thread | -4 |
| Waves/SM | 0.97 | 1.29 | +0.32 | 0.97 | +0.00 | 1.94 | +0.97 | 0.97 | +0.00 |
| No Eligible (%) | 54.20 % | 11.00 % | -43.20 pp | 43.94 % | -10.26 pp | 43.74 % | -10.46 pp | 42.69 % | -11.51 pp |
| IPC Active | 1.83 inst/cycle | 3.56 inst/cycle | +1.73 | 2.23 inst/cycle | +0.40 | 2.28 inst/cycle | +0.45 | 2.29 inst/cycle | +0.46 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_scalar_accumulator_for_coalesced_stores` (0.551×)

Increasing XBLOCK from 1 to 4 in a reduction kernel — even when the accumulator is a small [XBLOCK, 1] scalar form rather than [XBLOCK, R0_BLOCK] — caused registers per thread to increase from 28 to 40, dropping theoretical occupancy from 100% to 75% and achieved occupancy from 92.87% to 59.74%. Although DRAM throughput dropped (80% → 45%) and warp stall cycles improved (27.8 → 10.75 cycles/instruction), the 3x increase in executed instructions (20.8M → 64.2M) and the occupancy collapse caused kernel duration to nearly double (54.5 us → 97.9 us). The LaunchConfiguration NCU rule also fired, indicating a partial wave problem (1.29 waves/SM with 2048 blocks / 132 SMs), suggesting the grid reduction from 2048 to 512 CTAs introduced tail-effect overhead. Even with a scalar [XBLOCK, 1] accumulator, XBLOCK=4 in a memory-bandwidth-bound reduction kernel with near-full occupancy causes net regression due to register pressure increase and instruction count explosion.

> **Applicable when**: Never applicable for this kernel pattern. Even with a [XBLOCK, 1] scalar accumulator, increasing XBLOCK in a memory-bandwidth-bound reduction kernel with near-full occupancy (>90%) and low register count (~28 regs/thread) causes register pressure increase and instruction count explosion that outweighs any store coalescing benefit.

> **Avoid when**: Do not increase XBLOCK in reduction kernels that already achieve near-full theoretical occupancy (>90%) with low register usage (~28 regs/thread), even if the accumulator is a small [XBLOCK, 1] scalar form. The additional broadcast loads and per-x-element computations at XBLOCK=4 increase register pressure (28 → 40 regs/thread), drop theoretical occupancy (100% → 75%), and triple the executed instruction count (20.8M → 64.2M). The grid reduction from 2048 to 512 CTAs also introduces a partial-wave tail effect (LaunchConfiguration rule fires). The store coalescing improvement (4/32 → 16/32 bytes/sector) is insufficient to compensate for these costs when the kernel is already DRAM-bound at 80%.

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_store_cache_modifier_for_l2_compression | 1.009× | 0.870× | ✓ |  |
| B | hoist_loop_invariant_sqrt_and_scale_xblock1_configs_only | 1.026× | 0.575× | ✓ | **✓ best** |
| C | reduce_num_warps_to_eliminate_partial_wave_tail_effect | 1.014× | 1.047× | ✓ |  |
| D | restrict_autotune_to_xblock1_only_to_eliminate_register_pressure_and_instruction_explosion | 1.014× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_store_c... | Δ | hoist_loop_invari... | Δ | reduce_num_warps_... | Δ | restrict_autotune... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 53920 ns | 61950 ns | +8030 ns (+14.9%) | 93860 ns | +39940 ns (+74.1%) | 51490 ns | -2430 ns (-4.5%) | 53570 ns | -350 ns (-0.6%) |
| SM Throughput (%) | 43.01 % | 52.36 % | +9.35 pp | 55.41 % | +12.40 pp | 44.68 % | +1.67 pp | 50.36 % | +7.35 pp |
| Memory Throughput (%) | 81.09 % | 70.54 % | -10.55 pp | 66.12 % | -14.97 pp | 84.87 % | +3.78 pp | 81.56 % | +0.47 pp |
| DRAM Throughput (%) | 81.09 % | 70.54 % | -10.55 pp | 46.56 % | -34.53 pp | 84.87 % | +3.78 pp | 81.56 % | +0.47 pp |
| Achieved Occupancy (%) | 93.71 % | 91.91 % | -1.80 pp | 92.27 % | -1.44 pp | 93.30 % | -0.41 pp | 92.19 % | -1.52 pp |
| Registers/Thread | 32.00 register/thread | 28.00 register/thread | -4 | 26.00 register/thread | -6 | 31.00 register/thread | -1 | 28.00 register/thread | -4 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 54.20 % | 43.65 % | -10.55 pp | 40.78 % | -13.42 pp | 51.66 % | -2.54 pp | 45.58 % | -8.62 pp |
| IPC Active | 1.83 inst/cycle | 2.27 inst/cycle | +0.44 | 2.38 inst/cycle | +0.55 | 1.93 inst/cycle | +0.10 | 2.13 inst/cycle | +0.30 |

---

## Cumulative Progress

```
Baseline:  53920 ns  (1.000×)

Final best: 93860 ns  (0.5745×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,13 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256},  num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
+], key=['xnumel', 'r0_numel'])
 @triton.jit
 def triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_mul_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 2048
@@ -18,43 +25,34 @@

     rbase = r0_base
     x3 = xindex
     x0 = (xindex % 128)
-    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
-    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
-    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
+    tmp1 = tl.load(in_ptr1 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp7 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp18 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
+    # Hoist loop-invariant BN scale computation out of the reduction loop
+    inv_std = tl.rsqrt(tmp7 + 1e-5)
+    scaled_inv_std = inv_std * tmp16
+    _tmp21 = tl.full([XBLOCK, 1], 0, tl.float32)
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
-        r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_2 + 25920*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        r0_2 = tl.max_contiguous(tl.multiple_of(r0_index, [1, 16]), [1, R0_BLOCK])
+        tmp0 = tl.load(in_ptr0 + (r0_2 + 25920*x3), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp2 = tmp0 + tmp1
-        tmp3 = tl.full([1, 1], 2.0, tl.float32)
-        tmp4 = tmp2 * tmp3
+        tmp4 = tmp2 * 2.0
         tmp6 = tmp4 - tmp5
-        tmp8 = tl.full([1, 1], 1e-05, tl.float32)
-        tmp9 = tmp7 + tmp8
-        tmp10 = tl.sqrt_rn(tmp9)
-        tmp11 = tl.full([1, 1], 1, tl.int32)
-        tmp12 = (tmp11 / tmp10)
-        tmp13 = tl.full([1, 1], 1.0, tl.float32)
-        tmp14 = tmp12 * tmp13
-        tmp15 = tmp6 * tmp14
-        tmp17 = tmp15 * tmp16
-        tmp19 = tmp17 + tmp18
-        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
-        tmp22 = _tmp21 + tmp20
-        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
-    tmp21 = tl.sum(_tmp21, 1)[:, None]
+        tmp15 = tmp6 * scaled_inv_std
+        tmp19 = tmp15 + tmp18
+        tmp19_masked = tl.where(r0_mask & xmask, tmp19, tl.zeros([XBLOCK, R0_BLOCK], tl.float32))
+        _tmp21 = _tmp21 + tl.sum(tmp19_masked, axis=1, keep_dims=True)
+    tmp21 = _tmp21
     tmp23 = tl.full([1, 1], 25920.0, tl.float32)
     tmp24 = (tmp21 / tmp23)
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x3), tmp24, xmask)
-
-
+    tl.store(in_out_ptr0 + (xindex), tmp24, mask=xmask, eviction_policy='evict_last')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -67,7 +65,7 @@

     xnumel = 2048
     r0_numel = 25920
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_mul_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 2048, 25920, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_mul_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 2048, 25920)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

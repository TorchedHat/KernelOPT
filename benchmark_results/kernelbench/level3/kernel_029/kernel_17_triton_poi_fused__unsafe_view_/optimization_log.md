# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10300 ns — **Best latency**: 9920 ns (**1.0383×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`align_grid_to_eliminate_partial_wave_tail_effect`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_17_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (11760,1,1), Waves/SM: 5.57
- **Registers/thread**: 20
- **Duration**: 10300 ns
- **SM Throughput**: 59.38 % | **Memory**: 18.06 % | **DRAM**: 12.06 %
- **Achieved Occupancy**: 74.12 %
- **IPC Active**: 2.94 inst/cycle | **No Eligible**: 25.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 0.983× | 0.256× | ✓ |  |
| B | increase_xblock_with_autotune | 0.976× | 0.247× | ✓ |  |
| C | autotune_xblock_and_num_warps_for_occupancy | 1.037× | 0.470× | ✓ | **✓ best** |
| D | add_autotune_increase_xblock_for_vectorized_loads | 0.993× | 0.470× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10300 ns | 40190 ns | +29890 ns (+290.2%) | 41660 ns | +31360 ns (+304.5%) | 21920 ns | +11620 ns (+112.8%) | 21920 ns | +11620 ns (+112.8%) |
| SM Throughput (%) | 59.38 % | 80.82 % | +21.44 pp | 82.47 % | +23.09 pp | 78.55 % | +19.17 pp | 78.57 % | +19.19 pp |
| Memory Throughput (%) | 18.06 % | 31.22 % | +13.16 pp | 30.25 % | +12.19 pp | 31.45 % | +13.39 pp | 31.44 % | +13.38 pp |
| DRAM Throughput (%) | 12.06 % | 3.08 % | -8.98 pp | 2.98 % | -9.08 pp | 5.66 % | -6.40 pp | 5.65 % | -6.41 pp |
| Achieved Occupancy (%) | 74.12 % | 67.67 % | -6.45 pp | 86.01 % | +11.89 pp | 85.77 % | +11.65 pp | 85.91 % | +11.79 pp |
| Registers/Thread | 20.00 register/thread | 38.00 register/thread | +18 | 31.00 register/thread | +11 | 31.00 register/thread | +11 | 31.00 register/thread | +11 |
| Waves/SM | 5.57 | 14.85 | +9.28 | 22.27 | +16.70 | 11.14 | +5.57 | 11.14 | +5.57 |
| No Eligible (%) | 25.19 % | 14.72 % | -10.47 pp | 13.57 % | -11.62 pp | 14.32 % | -10.87 pp | 13.36 % | -11.83 pp |
| IPC Active | 2.94 inst/cycle | 3.40 inst/cycle | +0.46 | 3.47 inst/cycle | +0.53 | 3.46 inst/cycle | +0.52 | 3.46 inst/cycle | +0.52 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_with_autotune` (0.247×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel caused a 4x slowdown despite improving occupancy (74%→86%) and compute SOL (59%→82%). The critical failure was that the autotune grid was left hardcoded to cdiv(xnumel, 128) instead of being updated to use meta['XBLOCK'], so the selected config (XBLOCK=512) ran with a grid sized for XBLOCK=128, causing each element to be processed 4x (512/128) times redundantly. This multiplied executed instructions by ~5.6x (4.8M→27.1M) and duration by ~4x (10.3→41.7 µs). When adding autotune, the launch grid MUST be a lambda using meta['XBLOCK'], not a static value.

> **Applicable when**: Any time @triton.autotune is added to a kernel that previously had a hardcoded XBLOCK in both the decorator and the launch grid.

> **Avoid when**: This is an implementation flaw, not a strategy to avoid — the autotune approach itself is valid but requires the grid lambda fix. Do not use static grid sizes when autotuning tile dimensions.

#### ❌ Plan C: `autotune_xblock_and_num_warps_for_occupancy` (0.470×)

Adding @triton.autotune to sweep XBLOCK from 128 to 256/512/1024 caused the autotuner to select XBLOCK=256, which doubled registers per thread (20→31), nearly tripled total executed instructions (4.8M→13.6M), and doubled kernel duration (10.3→21.9 µs) despite improving achieved occupancy (74%→86%). The instruction explosion indicates the larger tile size caused the compiler to unroll more aggressively, generating redundant index arithmetic that overwhelmed any ILP benefit. When the kernel already has a fixed xnumel hardcoded inside the kernel body (not a runtime parameter), the autotuner cannot properly key on it and may select suboptimal configs; additionally, the static grid computation at the call site (grid = (cdiv(xnumel, 128),)) was not updated to use meta['XBLOCK'], meaning the grid remained fixed at the 128-element granularity while the kernel processed 256 elements per block, causing correctness and efficiency issues.

> **Applicable when**: Never apply this pattern as described. The anti-pattern is: adding @triton.autotune while leaving the grid computation hardcoded to the original XBLOCK value instead of using a lambda with meta['XBLOCK'].

> **Avoid when**: Do not add @triton.autotune to pointwise kernels where (1) xnumel is hardcoded inside the kernel body rather than passed as a runtime argument, (2) the grid is computed with a fixed XBLOCK constant rather than a lambda using meta['XBLOCK'], or (3) the kernel already has low register pressure (≤20 regs/thread) and increasing XBLOCK would cause register spilling or instruction count explosion. Also avoid when the kernel duration is already short (<15 µs) and autotuning overhead may dominate.

#### ❌ Plan D: `add_autotune_increase_xblock_for_vectorized_loads` (0.470×)

Adding @triton.autotune with larger XBLOCK configs (256–1024) to a pointwise kernel that had low achieved occupancy (74%) caused a 2.1x regression despite improving occupancy to 86%. The autotune selected XBLOCK=256 (block_size=256), which doubled waves_per_SM from 5.57 to 11.14 and caused executed instructions to balloon from 4.8M to 13.6M — a 2.8x increase — indicating the larger block size triggered significantly more redundant or predicated work per element. The register count also jumped from 20 to 31 per thread, reducing the effective block limit from 21 to 8 blocks/SM, which negated the occupancy benefit and increased total SM elapsed cycles by 2x.

> **Applicable when**: Never applicable when the kernel has complex index arithmetic that causes instruction count to scale super-linearly with XBLOCK, or when increasing block size causes register spill (registers_per_thread jumps significantly, e.g. 20->31), reducing block-limit-registers and negating occupancy gains.

> **Avoid when**: Kernel has non-trivial index expressions (multi-dimensional strided accesses) where larger XBLOCK causes the compiler to emit significantly more instructions per element. Also avoid when registers_per_thread is already near the boundary where doubling block size would halve block-limit-registers (e.g., 20 regs -> Block Limit Registers=21; 31 regs -> Block Limit Registers=8). The AchievedOccupancy gap alone is insufficient justification if instruction count or register pressure will worsen.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_non_reused_loads | 1.032× | 0.448× | ✓ |  |
| B | fuse_fp32_multiply_add_to_fma | 0.978× | 0.469× | ✓ |  |
| C | reduce_register_pressure_via_maxnreg | 0.977× | 0.973× | ✓ |  |
| D | fix_autotune_grid_mismatch_remove_broken_autotune | 1.001× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | fuse_fp32_multipl... | Δ | reduce_register_p... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10300 ns | 22980 ns | +12680 ns (+123.1%) | 21950 ns | +11650 ns (+113.1%) | 10590 ns | +290 ns (+2.8%) | 10340 ns | +40 ns (+0.4%) |
| SM Throughput (%) | 59.38 % | 76.59 % | +17.21 pp | 78.65 % | +19.27 pp | 65.92 % | +6.54 pp | 58.97 % | -0.41 pp |
| Memory Throughput (%) | 18.06 % | 29.96 % | +11.90 pp | 31.48 % | +13.42 pp | 26.28 % | +8.22 pp | 17.98 % | -0.08 pp |
| DRAM Throughput (%) | 12.06 % | 5.41 % | -6.65 pp | 5.66 % | -6.40 pp | 11.71 % | -0.35 pp | 12.05 % | -0.01 pp |
| Achieved Occupancy (%) | 74.12 % | 86.16 % | +12.04 pp | 85.91 % | +11.79 pp | 78.71 % | +4.59 pp | 77.49 % | +3.37 pp |
| Registers/Thread | 20.00 register/thread | 32.00 register/thread | +12 | 31.00 register/thread | +11 | 24.00 register/thread | +4 | 20.00 register/thread | +0 |
| Waves/SM | 5.57 | 11.14 | +5.57 | 11.14 | +5.57 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 25.19 % | 15.86 % | -9.33 pp | 13.36 % | -11.83 pp | 18.98 % | -6.21 pp | 24.58 % | -0.61 pp |
| IPC Active | 2.94 inst/cycle | 3.35 inst/cycle | +0.41 | 3.43 inst/cycle | +0.49 | 3.16 inst/cycle | +0.22 | 2.97 inst/cycle | +0.03 |


### Key Learnings

#### ❌ Plan A: `streaming_eviction_policy_for_non_reused_loads` (0.448×)

Adding eviction_policy='evict_first' to a large non-reused streaming load (in a compute-bound kernel at 78.55% compute SOL) caused a 5% regression in duration (21.92→22.98 µs) and a slight increase in executed instructions (13.57M→13.87M). The L1/TEX hit rate and L2 hit rate were essentially unchanged (36.32%→36.34% and 55.17%→54.59% respectively), indicating the cache policy hint had no meaningful effect on cache utilization for the reused tensors. When a kernel is already compute-bound rather than memory-bound, cache eviction policy changes cannot improve throughput because the bottleneck is the ALU pipeline, not cache pressure; the hint may introduce minor overhead from PTX .cs instruction handling that slightly increases instruction count and warp stall cycles.

> **Applicable when**: Never applicable when the kernel is compute-bound (compute SOL significantly exceeds memory SOL). The evict_first strategy for streaming loads is only useful when the kernel is memory-bound (memory SOL > compute SOL) and cache thrashing from large non-reused tensors is measurably evicting small frequently-reused tensors, evidenced by low L1/L2 hit rates that improve after the policy change.

> **Avoid when**: Kernel compute SOL (>70%) significantly exceeds memory SOL (<35%); L1/TEX hit rate and L2 hit rate do not change after applying the policy (indicating the cache pressure hypothesis is wrong); the MemoryCacheAccessPattern NCU rule fires but the kernel is not memory-bound — the rule identifies suboptimal access patterns but does not guarantee that fixing cache policy will improve end-to-end latency when compute is the bottleneck.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_to_decrease_warp_scheduling_overhead | 1.004× | 0.988× | ✓ |  |
| B | convert_constant_division_to_reciprocal_multiply_for_fma_fusion | 0.975× | 0.966× | ✓ |  |
| C | enable_software_pipelining_via_num_stages_2 | 0.974× | 0.470× | ✓ |  |
| D | fuse_multiply_add_chain_via_libdevice_fma | 1.052× | 0.447× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | convert_constant_... | Δ | enable_software_p... | Δ | fuse_multiply_add... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10300 ns | 10430 ns | +130 ns (+1.3%) | 10660 ns | +360 ns (+3.5%) | 21920 ns | +11620 ns (+112.8%) | 23070 ns | +12770 ns (+124.0%) |
| SM Throughput (%) | 59.38 % | 59.25 % | -0.13 pp | 65.88 % | +6.50 pp | 78.64 % | +19.26 pp | 76.69 % | +17.31 pp |
| Memory Throughput (%) | 18.06 % | 18.02 % | -0.04 pp | 26.32 % | +8.26 pp | 31.47 % | +13.41 pp | 29.98 % | +11.92 pp |
| DRAM Throughput (%) | 12.06 % | 11.90 % | -0.16 pp | 11.64 % | -0.42 pp | 5.66 % | -6.40 pp | 5.38 % | -6.68 pp |
| Achieved Occupancy (%) | 74.12 % | 78.49 % | +4.37 pp | 77.34 % | +3.22 pp | 85.77 % | +11.65 pp | 86.14 % | +12.02 pp |
| Registers/Thread | 20.00 register/thread | 20.00 register/thread | +0 | 24.00 register/thread | +4 | 31.00 register/thread | +11 | 32.00 register/thread | +12 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 11.14 | +5.57 | 11.14 | +5.57 |
| No Eligible (%) | 25.19 % | 24.97 % | -0.22 pp | 18.89 % | -6.30 pp | 13.35 % | -11.84 pp | 15.83 % | -9.36 pp |
| IPC Active | 2.94 inst/cycle | 2.90 inst/cycle | -0.04 | 3.23 inst/cycle | +0.29 | 3.46 inst/cycle | +0.52 | 3.35 inst/cycle | +0.41 |


### Key Learnings

#### ❌ Plan C: `enable_software_pipelining_via_num_stages_2` (0.470×)

Adding num_stages=2 variants to @triton.autotune configs caused the autotuner to select XBLOCK=256 (block_size=256) instead of the original XBLOCK=128, which tripled executed instructions (4.8M→13.6M), doubled registers per thread (20→31), and doubled kernel duration (10.34→21.92 µs). The num_stages=2 change itself did not produce meaningful software pipelining benefit for this pointwise kernel — the dominant effect was the autotuner selecting a larger XBLOCK config, which caused the same instruction explosion and register pressure increase seen in prior autotune experiments. For pointwise kernels without tl.dot, num_stages>1 does not generate async prefetch pipelines; the Triton compiler treats it as a no-op or emits minimal prefetch that cannot hide latency in a compute-bound kernel (78.64% compute SOL).

> **Applicable when**: num_stages=2 is only beneficial for kernels with tl.dot (matrix multiply tiles) or explicit tl.load in a loop where async prefetch can overlap with compute. The kernel must be memory-bound (memory SOL > compute SOL) and have measurable No-Eligible stalls caused by load latency, not by compute pipeline saturation.

> **Avoid when**: Kernel is a pointwise op without tl.dot or explicit tile loops — num_stages>1 has no effect on async prefetch for simple element-wise loads. Kernel is compute-bound (compute SOL > 70%) — software pipelining cannot hide latency that is not the bottleneck. Adding num_stages=2 variants to autotune configs when XBLOCK is also being varied risks the autotuner selecting a larger XBLOCK that causes instruction explosion and register pressure increase, masking any pipelining benefit.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_repeated_index_subexpressions_to_reduce_alu_pressure | 0.998× | 0.981× | ✓ |  |
| B | fix_autotune_grid_bug_and_reduce_num_warps_for_occupancy | 1.037× | 1.052× | ✓ |  |
| C | remove_broken_autotune_restore_xblock128_fixed_grid | 1.004× | 0.961× | ✓ |  |
| D | remove_broken_autotune_fix_grid_restore_xblock128 | 0.971× | 0.975× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_repeated_in... | Δ | fix_autotune_grid... | Δ | remove_broken_aut... | Δ | remove_broken_aut... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10300 ns | 10500 ns | +200 ns (+1.9%) | 9790 ns | -510 ns (-5.0%) | 10720 ns | +420 ns (+4.1%) | 10560 ns | +260 ns (+2.5%) |
| SM Throughput (%) | 59.38 % | 57.94 % | -1.44 pp | 35.82 % | -23.56 pp | 59.27 % | -0.11 pp | 60.67 % | +1.29 pp |
| Memory Throughput (%) | 18.06 % | 17.75 % | -0.31 pp | 18.06 % | +0.00 pp | 18.02 % | -0.04 pp | 17.51 % | -0.55 pp |
| DRAM Throughput (%) | 12.06 % | 11.82 % | -0.24 pp | 12.69 % | +0.63 pp | 11.62 % | -0.44 pp | 11.78 % | -0.28 pp |
| Achieved Occupancy (%) | 74.12 % | 77.68 % | +3.56 pp | 41.19 % | -32.93 pp | 77.64 % | +3.52 pp | 79.27 % | +5.15 pp |
| Registers/Thread | 20.00 register/thread | 20.00 register/thread | +0 | 24.00 register/thread | +4 | 20.00 register/thread | +0 | 20.00 register/thread | +0 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 2.78 | -2.79 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 25.19 % | 26.36 % | +1.17 pp | 43.84 % | +18.65 pp | 25.10 % | -0.09 pp | 24.06 % | -1.13 pp |
| IPC Active | 2.94 inst/cycle | 2.91 inst/cycle | -0.03 | 2.16 inst/cycle | -0.78 | 2.96 inst/cycle | +0.02 | 3.03 inst/cycle | +0.09 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | align_grid_to_eliminate_partial_wave_tail_effect | 1.062× | 1.038× | ✓ | **✓ best** |
| B | persistent_sm_loop_tile_scheduling | 0.978× | 0.973× | ✓ |  |
| C | persistent_grid_loop_to_reduce_wave_tail_overhead | 0.966× | 0.944× | ✓ |  |


### Metric Comparison

| Metric | Baseline | align_grid_to_eli... | Δ | persistent_sm_loo... | Δ | persistent_grid_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10300 ns | 9920 ns | -380 ns (-3.7%) | 10590 ns | +290 ns (+2.8%) | 10910 ns | +610 ns (+5.9%) |
| SM Throughput (%) | 59.38 % | 35.98 % | -23.40 pp | 61.68 % | +2.30 pp | 62.04 % | +2.66 pp |
| Memory Throughput (%) | 18.06 % | 18.06 % | +0.00 pp | 17.96 % | -0.10 pp | 17.18 % | -0.88 pp |
| DRAM Throughput (%) | 12.06 % | 12.57 % | +0.51 pp | 11.76 % | -0.30 pp | 11.42 % | -0.64 pp |
| Achieved Occupancy (%) | 74.12 % | 41.00 % | -33.12 pp | 78.08 % | +3.96 pp | 79.64 % | +5.52 pp |
| Registers/Thread | 20.00 register/thread | 24.00 register/thread | +4 | 30.00 register/thread | +10 | 30.00 register/thread | +10 |
| Waves/SM | 5.57 | 2.78 | -2.79 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 25.19 % | 46.21 % | +21.02 pp | 22.83 % | -2.36 pp | 22.49 % | -2.70 pp |
| IPC Active | 2.94 inst/cycle | 2.21 inst/cycle | -0.73 | 3.03 inst/cycle | +0.09 | 3.11 inst/cycle | +0.17 |

---

## Cumulative Progress

```
Baseline:  10300 ns  (1.000×)
Iter 5     9920 ns  (1.038×)  — align_grid_to_eliminate_partial_wave_tail_effect

Final best: 9920 ns  (1.0383×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1505280
@@ -21,17 +31,13 @@

     tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 192.0, tl.float32)
-    tmp5 = (tmp3 / tmp4)
-    tmp6 = tl.full([1], 1e-05, tl.float32)
-    tmp7 = tmp5 + tmp6
+    # 1/192.0 = 0.005208333333333333 as compile-time constant enables FFMA fusion
+    tmp7 = tmp3 * 0.005208333333333333 + 1e-05
     tmp8 = libdevice.rsqrt(tmp7)
     tmp9 = tmp2 * tmp8
     tmp11 = tmp9 * tmp10
     tmp13 = tmp11 + tmp12
     tl.store(out_ptr0 + (x3), tmp13, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -42,7 +48,7 @@

     arg_5 = inputs[5]
     xnumel = 1505280
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1505280, XBLOCK=128)
+    triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1505280)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mul_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 290720 ns — **Best latency**: 310750 ns (**0.9355×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mul_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_073/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128000,1,1), Waves/SM: 80.81
- **Registers/thread**: 34
- **Duration**: 290720 ns
- **SM Throughput**: 61.05 % | **Memory**: 80.43 % | **DRAM**: 71.56 %
- **Achieved Occupancy**: 70.79 %
- **IPC Active**: 2.47 inst/cycle | **No Eligible**: 37.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_warps | 2.117× | 0.807× | ✓ | **✓ best** |
| C | add_autotune_vectorized_tile_exploration | 1.524× | 0.746× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_autotune_vect... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 290720 ns | 360320 ns | +69600 ns (+23.9%) | 389790 ns | +99070 ns (+34.1%) |
| SM Throughput (%) | 61.05 % | 80.47 % | +19.42 pp | 75.01 % | +13.96 pp |
| Memory Throughput (%) | 80.43 % | 79.32 % | -1.11 pp | 79.24 % | -1.19 pp |
| DRAM Throughput (%) | 71.56 % | 57.73 % | -13.83 pp | 26.36 % | -45.20 pp |
| Achieved Occupancy (%) | 70.79 % | 59.29 % | -11.50 pp | 69.60 % | -1.19 pp |
| Registers/Thread | 34.00 register/thread | 48.00 register/thread | +14 | 34.00 register/thread | +0 |
| Waves/SM | 80.81 | 96.97 | +16.16 | 161.62 | +80.81 |
| No Eligible (%) | 37.86 % | 19.31 % | -18.55 pp | 24.42 % | -13.44 pp |
| IPC Active | 2.47 inst/cycle | 3.23 inst/cycle | +0.76 | 3.02 inst/cycle | +0.55 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_maxnreg_and_fma_fusion | 2.124× | 0.613× | ✓ | **✓ best** |
| B | remove_autotune_restore_constexpr_ymask_elision | 0.964× | 0.827× | ✓ |  |
| C | fix_coalescing_by_swapping_fast_slow_axes | 1.518× | 0.769× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | remove_autotune_r... | Δ | fix_coalescing_by... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 290720 ns | 474660 ns | +183940 ns (+63.3%) | 351360 ns | +60640 ns (+20.9%) | 377920 ns | +87200 ns (+30.0%) |
| SM Throughput (%) | 61.05 % | 64.48 % | +3.43 pp | 80.47 % | +19.42 pp | 75.40 % | +14.35 pp |
| Memory Throughput (%) | 80.43 % | 86.34 % | +5.91 pp | 81.56 % | +1.13 pp | 81.80 % | +1.37 pp |
| DRAM Throughput (%) | 71.56 % | 52.94 % | -18.62 pp | 59.21 % | -12.35 pp | 27.20 % | -44.36 pp |
| Achieved Occupancy (%) | 70.79 % | 97.20 % | +26.41 pp | 59.22 % | -11.57 pp | 69.41 % | -1.38 pp |
| Registers/Thread | 34.00 register/thread | 32.00 register/thread | -2 | 48.00 register/thread | +14 | 34.00 register/thread | +0 |
| Waves/SM | 80.81 | 60.61 | -20.20 | 96.97 | +16.16 | 161.62 | +80.81 |
| No Eligible (%) | 37.86 % | 35.02 % | -2.84 pp | 19.35 % | -18.51 pp | 23.95 % | -13.91 pp |
| IPC Active | 2.47 inst/cycle | 2.58 inst/cycle | +0.11 | 3.23 inst/cycle | +0.76 | 3.04 inst/cycle | +0.57 |


### Key Learnings

#### ❌ Plan B: `remove_autotune_restore_constexpr_ymask_elision` (0.827×)

Removing @triton.autotune while keeping the original fixed grid (XBLOCK=128, YBLOCK=8) and only explicitly passing YBLOCK=8, XBLOCK=128 at the call site did not change registers per thread (still 48), theoretical occupancy (still 62.5%), or achieved occupancy (still 59.22%). The kernel remained register-limited and the ymask was not elided, because the grid was not updated to match the new YBLOCK=64 tile size specified in the plan — the call site still used YBLOCK=8, XBLOCK=128 with the old grid=(cdiv(xnumel,128), cdiv(ynumel,8)). The implementation flaw was that the diff applied YBLOCK=8 at the call site instead of YBLOCK=64, so the intended tile-size change (which was supposed to allow ymask elision via full channel coverage) was never actually applied.

> **Applicable when**: Never: this specific implementation was a no-op — the tile sizes were not changed as planned, so no register or occupancy improvement occurred.

> **Avoid when**: Do not apply this pattern when the grid and tile sizes at the call site are left unchanged from the regressed version. Removing @triton.autotune alone does not restore ymask elision if YBLOCK remains small (e.g., 8) and ynumel is not provably covered by the tile. The grid must also be updated to match the new YBLOCK to avoid launching with mismatched tile/grid dimensions.

#### ❌ Plan C: `fix_coalescing_by_swapping_fast_slow_axes` (0.769×)

Transposing the fast/slow arange axes (swapping [:, None] and [None, :] for yindex and xindex) to make the stride-1 channel dimension the fastest-varying warp dimension was combined with a new autotune config set that included larger XBLOCK values (512). This doubled waves-per-SM from 96.97 to 161.62 and halved DRAM throughput (57.73% → 27.2%), indicating the autotuner selected a configuration that over-partitioned the work. Despite reducing divergent branches to 0 and improving L1 hit rate (13% → 19%), total memory throughput collapsed from 2.84 TB/s to 1.34 TB/s and warp cycles per instruction worsened (11.71 → 14.52), causing a net 23% slowdown. The axis transposition itself may be sound, but coupling it with a new autotune config set that changes the effective tile shape and thread count invalidates the coalescing analysis performed on the prior configuration.

> **Applicable when**: Never in this form: axis transposition for coalescing must be validated against the specific autotune config that will actually be selected, not the config used during profiling. If the autotune config set changes simultaneously with the axis layout, the coalescing benefit may be negated by wave count explosion.

> **Avoid when**: Do not simultaneously change arange axis assignments AND the autotune config set. If the autotuner selects a larger XBLOCK than the profiled baseline, waves-per-SM can double, collapsing DRAM throughput and worsening memory-bound kernels. Axis transposition for coalescing should only be applied when the autotune config set is held constant (same XBLOCK/YBLOCK values as the profiled kernel) so the coalescing analysis remains valid.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_batchnorm_rsqrt_reduce_register_pressure_via_fma | 2.341× | 0.921× | ✓ | **✓ best** |
| C | remove_maxnreg_to_eliminate_register_spill | 2.120× | 0.819× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_batchnorm_rs... | Δ | remove_maxnreg_to... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 290720 ns | 315740 ns | +25020 ns (+8.6%) | 355170 ns | +64450 ns (+22.2%) |
| SM Throughput (%) | 61.05 % | 45.68 % | -15.37 pp | 80.53 % | +19.48 pp |
| Memory Throughput (%) | 80.43 % | 93.38 % | +12.95 pp | 79.66 % | -0.77 pp |
| DRAM Throughput (%) | 71.56 % | 65.85 % | -5.71 pp | 58.58 % | -12.98 pp |
| Achieved Occupancy (%) | 70.79 % | 95.76 % | +24.97 pp | 59.32 % | -11.47 pp |
| Registers/Thread | 34.00 register/thread | 32.00 register/thread | -2 | 48.00 register/thread | +14 |
| Waves/SM | 80.81 | 60.61 | -20.20 | 96.97 | +16.16 |
| No Eligible (%) | 37.86 % | 54.08 % | +16.22 pp | 18.72 % | -19.14 pp |
| IPC Active | 2.47 inst/cycle | 1.84 inst/cycle | -0.63 | 3.24 inst/cycle | +0.77 |


### Key Learnings

#### ❌ Plan C: `remove_maxnreg_to_eliminate_register_spill` (0.819×)

Removing maxnreg=32 from all autotune configs while simultaneously adding @triton.autotune (with a lambda-free fixed grid call site) and simplifying the reciprocal-sqrt computation caused a slight regression (0.82x). The profiling shows registers per thread remained at 48 and occupancy unchanged at 62.5%/59.3%, meaning the maxnreg removal had no effect on register allocation — the kernel was already at 48 registers without the cap. The regression is attributable to the autotune decorator adding overhead (autotuning pass, ymask materialization) and the increased instruction count (216M → 221M) from the new autotune configs introducing divergent branches (3612 → 4575 avg divergent branches, branch efficiency 89.81% → 87.91%). The plan's premise — that maxnreg=32 was causing spill in the current kernel — was incorrect; the current kernel already used 48 registers without spill, so removing maxnreg was a no-op on register allocation.

> **Applicable when**: Never: removing maxnreg is only meaningful when the kernel is actively spilling (local memory traffic visible in NCU, MemoryCacheAccessPattern for local loads/stores with <4 bytes/sector, CPIStall >40% on L1TEX). If the kernel already runs at its natural register count without maxnreg, removing it is a no-op on register allocation.

> **Avoid when**: Do not attempt to remove maxnreg when the current kernel's registers_per_thread already equals the natural register count (i.e., the cap is not binding). Also do not add @triton.autotune with a fixed (non-lambda) grid call site — this causes the autotuner to run but the selected config may not match the fixed grid, introducing divergent branches and instruction count inflation. Verify that maxnreg is actually binding (registers_per_thread_with_cap < registers_per_thread_without_cap) before applying this optimization.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | change_large_tensor_eviction_policy_to_evict_first_and_remove_maxnreg | 1.898× | 0.810× | ✓ |  |
| B | streaming_eviction_policy_for_large_tensor_loads_and_stores | 2.006× | 0.882× | ✓ |  |
| C | replace_sqrt_reciprocal_with_rsqrt_to_enable_fma_fusion | 2.341× | 0.935× | ✓ | **✓ best** |
| D | fuse_fp32_instructions_via_rsqrt_and_fold_constant_multiply | 2.341× | 0.933× | ✓ |  |


### Metric Comparison

| Metric | Baseline | change_large_tens... | Δ | streaming_evictio... | Δ | replace_sqrt_reci... | Δ | fuse_fp32_instruc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 290720 ns | 358980 ns | +68260 ns (+23.5%) | 329570 ns | +38850 ns (+13.4%) | 310750 ns | +20030 ns (+6.9%) | 311460 ns | +20740 ns (+7.1%) |
| SM Throughput (%) | 61.05 % | 77.67 % | +16.62 pp | 46.55 % | -14.50 pp | 44.66 % | -16.39 pp | 44.73 % | -16.32 pp |
| Memory Throughput (%) | 80.43 % | 79.05 % | -1.38 pp | 91.10 % | +10.67 pp | 92.91 % | +12.48 pp | 93.07 % | +12.64 pp |
| DRAM Throughput (%) | 71.56 % | 59.61 % | -11.95 pp | 62.97 % | -8.59 pp | 66.94 % | -4.62 pp | 66.78 % | -4.78 pp |
| Achieved Occupancy (%) | 70.79 % | 59.34 % | -11.45 pp | 95.62 % | +24.83 pp | 70.91 % | +0.12 pp | 70.92 % | +0.13 pp |
| Registers/Thread | 34.00 register/thread | 48.00 register/thread | +14 | 32.00 register/thread | -2 | 36.00 register/thread | +2 | 36.00 register/thread | +2 |
| Waves/SM | 80.81 | 96.97 | +16.16 | 60.61 | -20.20 | 80.81 | +0.00 | 80.81 | +0.00 |
| No Eligible (%) | 37.86 % | 21.86 % | -16.00 pp | 53.05 % | +15.19 pp | 54.95 % | +17.09 pp | 54.92 % | +17.06 pp |
| IPC Active | 2.47 inst/cycle | 3.12 inst/cycle | +0.65 | 1.89 inst/cycle | -0.58 | 1.81 inst/cycle | -0.66 | 1.80 inst/cycle | -0.67 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_multiply_add_chains_via_explicit_fma_and_increase_pipeline_stages | 1.886× | 0.767× | ✓ |  |
| B | increase_num_stages_to_hide_l1tex_scoreboard_stall | 1.270× | 0.808× | ✓ |  |
| C | hoist_channel_only_scalars_and_fold_constant_multiplier | 2.072× | 0.865× | ✓ |  |
| D | collapse_sqrt_reciprocal_to_rsqrt_for_fma_fusion | 2.076× | 0.859× | ✓ |  |
| E | add_triton_autotune | 0.815× | 0.784× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_multipl... | Δ | increase_num_stag... | Δ | hoist_channel_onl... | Δ | collapse_sqrt_rec... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 290720 ns | 379200 ns | +88480 ns (+30.4%) | 359840 ns | +69120 ns (+23.8%) | 336060 ns | +45340 ns (+15.6%) | 338430 ns | +47710 ns (+16.4%) | 370720 ns | +80000 ns (+27.5%) |
| SM Throughput (%) | 61.05 % | 79.68 % | +18.63 pp | 77.39 % | +16.34 pp | 43.21 % | -17.84 pp | 41.13 % | -19.92 pp | 76.73 % | +15.68 pp |
| Memory Throughput (%) | 80.43 % | 75.58 % | -4.85 pp | 79.05 % | -1.38 pp | 81.92 % | +1.49 pp | 85.42 % | +4.99 pp | 76.20 % | -4.23 pp |
| DRAM Throughput (%) | 71.56 % | 56.74 % | -14.82 pp | 59.60 % | -11.96 pp | 62.86 % | -8.70 pp | 63.55 % | -8.01 pp | 57.96 % | -13.60 pp |
| Achieved Occupancy (%) | 70.79 % | 59.46 % | -11.33 pp | 59.32 % | -11.47 pp | 59.12 % | -11.67 pp | 71.18 % | +0.39 pp | 59.41 % | -11.38 pp |
| Registers/Thread | 34.00 register/thread | 48.00 register/thread | +14 | 48.00 register/thread | +14 | 47.00 register/thread | +13 | 36.00 register/thread | +2 | 48.00 register/thread | +14 |
| Waves/SM | 80.81 | 96.97 | +16.16 | 96.97 | +16.16 | 96.97 | +16.16 | 80.81 | +0.00 | 96.97 | +16.16 |
| No Eligible (%) | 37.86 % | 20.38 % | -17.48 pp | 21.98 % | -15.88 pp | 56.39 % | +18.53 pp | 60.18 % | +22.32 pp | 23.32 % | -14.54 pp |
| IPC Active | 2.47 inst/cycle | 3.16 inst/cycle | +0.69 | 3.11 inst/cycle | +0.64 | 1.76 inst/cycle | -0.71 | 1.65 inst/cycle | -0.82 | 3.10 inst/cycle | +0.63 |


### Key Learnings

#### ❌ Plan A: `fuse_fp32_multiply_add_chains_via_explicit_fma_and_increase_pipeline_stages` (0.767×)

Restructuring a multiply-add chain to use tl.math.fma with constant folding, replacing libdevice.rsqrt with an explicit sqrt+reciprocal, and adding higher num_stages autotune configs (5–6) caused registers per thread to jump from 36 to 48. This dropped theoretical occupancy from 75% to 62.5% and achieved occupancy from 70.9% to 59.5%, while instruction count more than doubled (107M → 230M) and divergent branches appeared (0 → 5057 avg). Although warp cycles per issued instruction improved (25.1 → 12.0) and compute throughput rose (44.7% → 79.7%), the net effect was a 22% slowdown because the kernel was memory-bound (92.9% memory throughput before) and the occupancy collapse combined with instruction inflation overwhelmed any arithmetic efficiency gain. The anti-pattern is applying FMA restructuring or deeper pipeline stages when the kernel is already near register limits (Block_Limit_Registers < Block_Limit_Warps), as these changes tend to inflate register count and collapse occupancy in Triton-compiled kernels.

> **Applicable when**: Never apply this combined pattern. Individually: tl.math.fma may be useful when the kernel is compute-bound and registers are well below the Block_Limit_Registers threshold. Deeper num_stages may help when CPIStall from L1TEX is the dominant bottleneck AND the kernel has register headroom (Block_Limit_Registers == Block_Limit_Warps).

> **Avoid when**: Do not apply tl.math.fma restructuring or increase num_stages in autotune configs when the kernel is already register-limited (Block_Limit_Registers < Block_Limit_Warps, e.g., 12 < 16). These changes inflate register count in Triton, collapsing occupancy and increasing instruction count. Also do not replace libdevice.rsqrt with a two-step sqrt+reciprocal — this expands the live value set and consumes additional registers. Do not simultaneously change eviction_policy on the large input tensor alongside arithmetic restructuring, as the combined effect on register allocation is unpredictable and the eviction policy change alone has been shown not to improve L2/L1 hit rates for this kernel pattern.

---

## Cumulative Progress

```
Baseline:  290720 ns  (1.000×)

Final best: 310750 ns  (0.9355×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mul_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
     xnumel = 15876
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,29 +33,23 @@

     y0 = (yindex % 64)
     y1 = yindex // 64
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 1016064*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 1016064*y1), xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (y0), xmask & ymask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr3 + (y0), xmask & ymask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr4 + (y0), xmask & ymask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr5 + (y0), xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
     tmp6 = tl.full([1, 1], 1e-05, tl.float32)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.sqrt_rn(tmp7)
-    tmp9 = tl.full([1, 1], 1, tl.int32)
-    tmp10 = (tmp9 / tmp8)
-    tmp11 = tl.full([1, 1], 1.0, tl.float32)
-    tmp12 = tmp10 * tmp11
-    tmp13 = tmp4 * tmp12
+    tmp10 = libdevice.rsqrt(tmp7)
+    tmp13 = tmp4 * tmp10
     tmp15 = tmp13 * tmp14
     tmp17 = tmp15 + tmp16
     tmp18 = tl.full([1, 1], 2.0, tl.float32)
     tmp19 = tmp17 * tmp18
-    tl.store(out_ptr0 + (x2 + 15876*y3), tmp19, xmask)
-
-
+    tl.store(out_ptr0 + (x2 + 15876*y3), tmp19, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -54,7 +62,7 @@

     ynumel = 8192
     xnumel = 15876
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mul_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8192, 15876, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mul_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8192, 15876)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

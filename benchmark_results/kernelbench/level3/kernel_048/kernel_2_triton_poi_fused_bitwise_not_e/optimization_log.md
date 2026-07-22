# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3070 ns — **Best latency**: 2880 ns (**1.0660×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`persistent_tile_scheduling`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_2_triton_poi_fused_bitwise_not_e/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1152,1,1), Waves/SM: 0.55
- **Registers/thread**: 16
- **Duration**: 3070 ns
- **SM Throughput**: 8.46 % | **Memory**: 5.90 % | **DRAM**: 1.34 %
- **Achieved Occupancy**: 41.24 %
- **IPC Active**: 0.97 inst/cycle | **No Eligible**: 77.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_better_sm_utilization | 0.982× | 1.044× | ✓ |  |
| C | increase_xblock_tile_size_with_autotune | 0.972× | 0.990× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_better_occupancy | 0.982× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 2940 ns | -130 ns (-4.2%) | 3100 ns | +30 ns (+1.0%) | 3040 ns | -30 ns (-1.0%) |
| SM Throughput (%) | 8.46 % | 14.83 % | +6.37 pp | 26.60 % | +18.14 pp | 7.84 % | -0.62 pp |
| Memory Throughput (%) | 5.90 % | 5.46 % | -0.44 pp | 9.08 % | +3.18 pp | 5.68 % | -0.22 pp |
| DRAM Throughput (%) | 1.34 % | 1.41 % | +0.07 pp | 1.34 % | +0.00 pp | 1.37 % | +0.03 pp |
| Achieved Occupancy (%) | 41.24 % | 41.50 % | +0.26 pp | 40.61 % | -0.63 pp | 22.34 % | -18.90 pp |
| Registers/Thread | 16.00 register/thread | 22.00 register/thread | +6 | 29.00 register/thread | +13 | 22.00 register/thread | +6 |
| Waves/SM | 0.55 | 0.55 | +0.00 | 0.55 | +0.00 | 0.27 | -0.28 |
| No Eligible (%) | 77.32 % | 52.47 % | -24.85 pp | 35.03 % | -42.29 pp | 74.44 % | -2.88 pp |
| IPC Active | 0.97 inst/cycle | 1.87 inst/cycle | +0.90 | 2.46 inst/cycle | +1.49 | 0.97 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.935× | 0.806× | ✓ |  |
| B | persistent_tile_scheduling | 1.041× | 1.010× | ✓ |  |
| C | persistent_tile_scheduling | 1.102× | 1.066× | ✓ | **✓ best** |
| D | add_triton_autotune | 1.038× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 3810 ns | +740 ns (+24.1%) | 3040 ns | -30 ns (-1.0%) | 2880 ns | -190 ns (-6.2%) | 3040 ns | -30 ns (-1.0%) |
| SM Throughput (%) | 8.46 % | 17.55 % | +9.09 pp | 8.62 % | +0.16 pp | 9.96 % | +1.50 pp | 9.33 % | +0.87 pp |
| Memory Throughput (%) | 5.90 % | 10.92 % | +5.02 pp | 5.77 % | -0.13 pp | 5.63 % | -0.27 pp | 5.77 % | -0.13 pp |
| DRAM Throughput (%) | 1.34 % | 1.09 % | -0.25 pp | 1.37 % | +0.03 pp | 1.44 % | +0.10 pp | 1.36 % | +0.02 pp |
| Achieved Occupancy (%) | 41.24 % | 38.53 % | -2.71 pp | 21.50 % | -19.74 pp | 28.04 % | -13.20 pp | 42.01 % | +0.77 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 22.00 register/thread | +6 | 22.00 register/thread | +6 | 16.00 register/thread | +0 |
| Waves/SM | 0.55 | 0.55 | +0.00 | 0.27 | -0.28 | 0.55 | +0.00 | 0.55 | +0.00 |
| No Eligible (%) | 77.32 % | 58.58 % | -18.74 pp | 72.98 % | -4.34 pp | 71.99 % | -5.33 pp | 68.78 % | -8.54 pp |
| IPC Active | 0.97 inst/cycle | 1.56 inst/cycle | +0.59 | 1.02 inst/cycle | +0.05 | 1.13 inst/cycle | +0.16 | 1.03 inst/cycle | +0.06 |


### Key Learnings

#### ✅ Plan A: `persistent_tile_scheduling` (1.066×)

Converting a one-CTA-per-tile Triton kernel to a persistent kernel (grid fixed to NUM_SMS, each CTA loops over multiple tiles via tile_id += num_programs) reduced total executed instructions by ~39% (352K → 214K) and elapsed cycles by ~2%, yielding a 1.066x speedup. However, the persistent loop introduced overhead (while-loop branch instructions increased from 6912 to 9216) and actually reduced achieved occupancy (41.5% → 28%) and eligible warps per scheduler (1.71 → 0.67), because fewer CTAs are in flight simultaneously. The speedup came primarily from instruction count reduction (less per-tile setup overhead amortized across the persistent loop), not from improved occupancy or latency hiding as predicted. When waves_per_sm is already ~0.55 and the kernel is already scheduler-stall-limited, the persistent pattern provides only modest gains and does not fix the underlying occupancy gap.

> **Applicable when**: Kernel has waves_per_sm < 1.0 and the per-tile setup overhead (index computation, program_id fetch) is a significant fraction of total work. Total element count is large enough that each SM processes many tiles in the persistent loop (num_tiles >> NUM_SMS). Kernel is memory-bandwidth bound rather than compute-bound, so instruction reduction translates to wall-time savings.

> **Avoid when**: Kernel is already scheduler-stall-limited with low eligible warps per scheduler (<2.0) — the persistent pattern reduces active warps per SM further, worsening stall ratios. Waves_per_sm is already >= 1.0 (grid already fills the device). The while-loop branch overhead is non-trivial relative to the per-tile compute (e.g., very small XBLOCK with simple compute). Kernel relies on shared memory that must be re-initialized per tile.

#### ✅ Plan B: `persistent_tile_scheduling` (1.066×)

Converting a one-CTA-per-tile Triton kernel to a persistent kernel (grid fixed to NUM_SMS, each CTA loops over multiple tiles via tile_id += num_programs) reduced total executed instructions by ~39% (352K → 214K) and elapsed cycles by ~2%, yielding a 1.066x speedup. However, the persistent loop introduced overhead (while-loop branch instructions increased from 6912 to 9216) and actually reduced achieved occupancy (41.5% → 28%) and eligible warps per scheduler (1.71 → 0.67), because fewer CTAs are in flight simultaneously. The speedup came primarily from instruction count reduction (less per-tile setup overhead amortized across the persistent loop), not from improved occupancy or latency hiding as predicted. When waves_per_sm is already ~0.55 and the kernel is already scheduler-stall-limited, the persistent pattern provides only modest gains and does not fix the underlying occupancy gap.

> **Applicable when**: Kernel has waves_per_sm < 1.0 and the per-tile setup overhead (index computation, program_id fetch) is a significant fraction of total work. Total element count is large enough that each SM processes many tiles in the persistent loop (num_tiles >> NUM_SMS). Kernel is memory-bandwidth bound rather than compute-bound, so instruction reduction translates to wall-time savings.

> **Avoid when**: Kernel is already scheduler-stall-limited with low eligible warps per scheduler (<2.0) — the persistent pattern reduces active warps per SM further, worsening stall ratios. Waves_per_sm is already >= 1.0 (grid already fills the device). The while-loop branch overhead is non-trivial relative to the per-tile compute (e.g., very small XBLOCK with simple compute). Kernel relies on shared memory that must be re-initialized per tile.

#### ✅ Plan C: `persistent_tile_scheduling` (1.066×)

Converting a one-CTA-per-tile Triton kernel to a persistent kernel (grid fixed to NUM_SMS, each CTA loops over multiple tiles via tile_id += num_programs) reduced total executed instructions by ~39% (352K → 214K) and elapsed cycles by ~2%, yielding a 1.066x speedup. However, the persistent loop introduced overhead (while-loop branch instructions increased from 6912 to 9216) and actually reduced achieved occupancy (41.5% → 28%) and eligible warps per scheduler (1.71 → 0.67), because fewer CTAs are in flight simultaneously. The speedup came primarily from instruction count reduction (less per-tile setup overhead amortized across the persistent loop), not from improved occupancy or latency hiding as predicted. When waves_per_sm is already ~0.55 and the kernel is already scheduler-stall-limited, the persistent pattern provides only modest gains and does not fix the underlying occupancy gap.

> **Applicable when**: Kernel has waves_per_sm < 1.0 and the per-tile setup overhead (index computation, program_id fetch) is a significant fraction of total work. Total element count is large enough that each SM processes many tiles in the persistent loop (num_tiles >> NUM_SMS). Kernel is memory-bandwidth bound rather than compute-bound, so instruction reduction translates to wall-time savings.

> **Avoid when**: Kernel is already scheduler-stall-limited with low eligible warps per scheduler (<2.0) — the persistent pattern reduces active warps per SM further, worsening stall ratios. Waves_per_sm is already >= 1.0 (grid already fills the device). The while-loop branch overhead is non-trivial relative to the per-tile compute (e.g., very small XBLOCK with simple compute). Kernel relies on shared memory that must be re-initialized per tile.

---

## Iteration 3

**Best so far**: 1.066× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_instructions_and_fix_eviction_policy | 1.033× | 1.077× | ✓ |  |
| B | fix_autotune_bypass_with_dynamic_grid_and_software_pipelining | 1.101× | 1.000× | ✓ |  |
| C | optimize_eviction_policy_for_streaming_strided_loads | 0.941× | 0.806× | ✓ |  |
| D | fix_eviction_policy_for_streaming_loads | 0.871× | 0.799× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_instruc... | Δ | fix_autotune_bypa... | Δ | optimize_eviction... | Δ | fix_eviction_poli... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 2850 ns | -220 ns (-7.2%) | 3070 ns | +0 ns (+0.0%) | 3810 ns | +740 ns (+24.1%) | 3840 ns | +770 ns (+25.1%) |
| SM Throughput (%) | 8.46 % | 8.99 % | +0.53 pp | 8.65 % | +0.19 pp | 18.76 % | +10.30 pp | 19.76 % | +11.30 pp |
| Memory Throughput (%) | 5.90 % | 5.68 % | -0.22 pp | 5.76 % | -0.14 pp | 10.80 % | +4.90 pp | 10.62 % | +4.72 pp |
| DRAM Throughput (%) | 1.34 % | 1.45 % | +0.11 pp | 1.34 % | +0.00 pp | 1.08 % | -0.26 pp | 1.07 % | -0.27 pp |
| Achieved Occupancy (%) | 41.24 % | 31.95 % | -9.29 pp | 21.18 % | -20.06 pp | 40.37 % | -0.87 pp | 39.39 % | -1.85 pp |
| Registers/Thread | 16.00 register/thread | 22.00 register/thread | +6 | 22.00 register/thread | +6 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 0.55 | 0.55 | +0.00 | 0.27 | -0.28 | 0.55 | +0.00 | 0.55 | +0.00 |
| No Eligible (%) | 77.32 % | 70.80 % | -6.52 pp | 72.22 % | -5.10 pp | 61.05 % | -16.27 pp | 61.63 % | -15.69 pp |
| IPC Active | 0.97 inst/cycle | 0.89 inst/cycle | -0.08 | 0.99 inst/cycle | +0.02 | 1.54 inst/cycle | +0.57 | 1.59 inst/cycle | +0.62 |


### Key Learnings

#### ❌ Plan C: `optimize_eviction_policy_for_streaming_strided_loads` (0.806×)

Changing a single tl.load eviction_policy from 'evict_last' to 'evict_first' was bundled with a structural kernel rewrite (removing @triton.autotune and replacing it with a persistent while-loop using a hardcoded NUM_PROGRAMS=264 constant), which caused a 19% regression. The executed instruction count exploded from 214K to 579K — a 2.7x increase — because the while-loop overhead was reintroduced without the correct grid/stride alignment, exactly repeating the anti-pattern documented in neg_triton_persistent_kernel_small_grid_001. The evict_first change itself cannot be isolated as the cause since the dominant effect was the structural regression from the persistent loop reintroduction; the L1 hit rate was essentially unchanged (50.75% → 48.4%) confirming the eviction policy change had negligible effect on cache behavior.

> **Applicable when**: Never apply this combined pattern. The evict_first change alone may be worth isolating in a clean single-pass kernel, but cannot be evaluated here due to confounding structural changes.

> **Avoid when**: Do not bundle eviction policy changes with structural kernel rewrites (persistent loop, autotune removal). The confounding makes it impossible to attribute the regression to either change. Specifically: do not reintroduce the persistent while-loop with NUM_PROGRAMS != launch grid size — this was already documented as a regression in neg_triton_persistent_kernel_small_grid_001 and repeating it here caused the same 2.7x instruction count explosion.

#### ❌ Plan D: `fix_eviction_policy_for_streaming_loads` (0.799×)

Switching tl.load eviction_policy from 'evict_last' to 'evict_first' on streaming (single-use) data caused a 20% regression: elapsed cycles increased from 4263 to 5684 and executed instructions jumped from 214K to 574K. The instruction explosion was the dominant cause — the diff also inadvertently removed the @triton.autotune decorator and replaced it with a hardcoded XBLOCK=128 persistent while-loop, reverting the XBLOCK tuning gains from iteration 1. The eviction policy change itself (evict_last → evict_first) did not produce the predicted L1TEX stall reduction; L1 hit rate was essentially unchanged (50.75% → 48.55%) and CPIStall fraction worsened (30.6% → 36.3%), indicating the eviction policy hint had negligible effect on this small, low-occupancy kernel where concurrent L1 pressure is minimal.

> **Applicable when**: Never apply evict_first as a standalone optimization when the kernel is already low-occupancy (<30% achieved) and the grid is small (<1 wave/SM). The eviction policy hint has negligible effect when there is little concurrent L1 pressure from other warps.

> **Avoid when**: Do not apply evict_first to streaming loads in small, low-occupancy Triton kernels (waves_per_sm < 1.0, achieved occupancy < 40%). The L1 cache is not under pressure in this regime, so the eviction hint provides no benefit. Additionally, never combine an eviction policy change with removal of @triton.autotune — the XBLOCK regression dominates and masks any cache policy effect.

---

## Iteration 4

**Best so far**: 1.066× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_dynamic_grid_and_increase_num_warps_for_latency_hiding | 1.039× | 1.089× | ✓ |  |
| B | fuse_fp32_subtraction_with_libdevice_fma_and_fix_autotune_bypass | 1.062× | 0.990× | ✓ |  |
| C | fuse_fp32_subtraction_to_fma_with_targeted_eviction_policy | 0.864× | 0.812× | ✓ |  |
| D | fuse_fp32_subtraction_via_libdevice_fma | 0.868× | 0.806× | ✓ |  |
| E | add_triton_autotune | 1.031× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_dynamic_grid_... | Δ | fuse_fp32_subtrac... | Δ | fuse_fp32_subtrac... | Δ | fuse_fp32_subtrac... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 2820 ns | -250 ns (-8.1%) | 3100 ns | +30 ns (+1.0%) | 3780 ns | +710 ns (+23.1%) | 3810 ns | +740 ns (+24.1%) | 3040 ns | -30 ns (-1.0%) |
| SM Throughput (%) | 8.46 % | 9.44 % | +0.98 pp | 8.67 % | +0.21 pp | 19.95 % | +11.49 pp | 18.78 % | +10.32 pp | 11.59 % | +3.13 pp |
| Memory Throughput (%) | 5.90 % | 5.74 % | -0.16 pp | 5.72 % | -0.18 pp | 10.91 % | +5.01 pp | 10.99 % | +5.09 pp | 6.11 % | +0.21 pp |
| DRAM Throughput (%) | 1.34 % | 1.58 % | +0.24 pp | 1.33 % | -0.01 pp | 1.09 % | -0.25 pp | 1.08 % | -0.26 pp | 1.35 % | +0.01 pp |
| Achieved Occupancy (%) | 41.24 % | 27.08 % | -14.16 pp | 22.09 % | -19.15 pp | 37.57 % | -3.67 pp | 39.64 % | -1.60 pp | 42.19 % | +0.95 pp |
| Registers/Thread | 16.00 register/thread | 22.00 register/thread | +6 | 22.00 register/thread | +6 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 0.55 | 0.55 | +0.00 | 0.27 | -0.28 | 0.55 | +0.00 | 0.55 | +0.00 | 0.55 | +0.00 |
| No Eligible (%) | 77.32 % | 72.33 % | -4.99 pp | 72.14 % | -5.18 pp | 59.05 % | -18.27 pp | 57.55 % | -19.77 pp | 65.37 % | -11.95 pp |
| IPC Active | 0.97 inst/cycle | 1.05 inst/cycle | +0.08 | 1.09 inst/cycle | +0.12 | 1.54 inst/cycle | +0.57 | 1.64 inst/cycle | +0.67 | 1.30 inst/cycle | +0.33 |

---

## Cumulative Progress

```
Baseline:  3070 ns  (1.000×)
Iter 2     2880 ns  (1.066×)  — persistent_tile_scheduling

Final best: 2880 ns  (1.0660×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,39 +5,53 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 147456
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x0 = (xindex % 3)
-    x1 = ((xindex // 3) % 3)
-    x3 = xindex // 3
-    x2 = xindex // 9
-    x4 = xindex
-    tmp6 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr0 + (x0 + 3*x2), None, eviction_policy='evict_last')
-    tmp0 = x0 + ((-1)*x1)
-    tmp1 = tl.full([1], 0, tl.int64)
-    tmp2 = tmp0 <= tmp1
-    tmp3 = tl.full([1], True, tl.int1)
-    tmp4 = tmp2 & tmp3
-    tmp5 = tmp4 == 0
-    tmp8 = tmp6 - tmp7
-    tmp9 = tl.full([1], float("-inf"), tl.float32)
-    tmp10 = tl.where(tmp5, tmp9, tmp8)
-    tmp11 = tl.exp(tmp10)
-    tl.store(out_ptr0 + (x4), tmp11, None)
-
-
+    pid = tl.program_id(0)
+    num_programs = tl.num_programs(0)
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    tile_id = pid
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x0 = (xindex % 3)
+        x1 = ((xindex // 3) % 3)
+        x3 = xindex // 3
+        x2 = xindex // 9
+        x4 = xindex
+        tmp6 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_first')
+        tmp7 = tl.load(in_ptr0 + (x0 + 3*x2), xmask, eviction_policy='evict_first')
+        tmp0 = x0 + ((-1)*x1)
+        tmp1 = tl.full([1], 0, tl.int64)
+        tmp2 = tmp0 <= tmp1
+        tmp3 = tl.full([1], True, tl.int1)
+        tmp4 = tmp2 & tmp3
+        tmp5 = tmp4 == 0
+        tmp8 = libdevice.fma(tmp6, tl.full([1], 1.0, tl.float32), -tmp7)
+        tmp9 = tl.full([1], float("-inf"), tl.float32)
+        tmp10 = tl.where(tmp5, tmp9, tmp8)
+        tmp11 = tl.exp(tmp10)
+        tl.store(out_ptr0 + (x4), tmp11, xmask)
+        tile_id += num_programs
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 147456
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2[grid](arg_0, arg_1, 147456, XBLOCK=128)
+    triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2[grid](arg_0, arg_1, 147456)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

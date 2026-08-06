# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_cat_convolution_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 665120 ns — **Best latency**: 662050 ns (**1.0046×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`vectorized_stores_with_contiguous_hints_and_evict_last_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_cat_convolution_11`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_11_triton_poi_fused_cat_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (250880,1,1), Waves/SM: 118.79
- **Registers/thread**: 25
- **Duration**: 665120 ns
- **SM Throughput**: 16.19 % | **Memory**: 65.52 % | **DRAM**: 62.27 %
- **Achieved Occupancy**: 94.86 %
- **IPC Active**: 0.65 inst/cycle | **No Eligible**: 83.71 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | increase_tile_sizes_and_fix_eviction_policy | 0.840× | 0.564× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 665120 ns | 1180000 ns | +514880 ns (+77.4%) |
| SM Throughput (%) | 16.19 % | 47.04 % | +30.85 pp |
| Memory Throughput (%) | 65.52 % | 80.07 % | +14.55 pp |
| DRAM Throughput (%) | 62.27 % | 34.87 % | -27.40 pp |
| Achieved Occupancy (%) | 94.86 % | 45.88 % | -48.98 pp |
| Registers/Thread | 25.00 register/thread | 57.00 register/thread | +32 |
| Waves/SM | 118.79 | 475.15 | +356.36 |
| No Eligible (%) | 83.71 % | 52.85 % | -30.86 pp |
| IPC Active | 0.65 inst/cycle | 1.88 inst/cycle | +1.23 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_restore_occupancy_with_vectorized_stores | 2.657× | 0.745× | ✓ |  |
| B | reduce_register_pressure_via_maxnreg_and_small_tile_configs | 3.380× | 0.745× | ✓ | **✓ best** |
| D | cap_registers_with_maxnreg_to_restore_occupancy | 1.663× | 0.746× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 665120 ns | 893310 ns | +228190 ns (+34.3%) | 892510 ns | +227390 ns (+34.2%) | 892220 ns | +227100 ns (+34.1%) |
| SM Throughput (%) | 16.19 % | 12.02 % | -4.17 pp | 11.96 % | -4.23 pp | 11.99 % | -4.20 pp |
| Memory Throughput (%) | 65.52 % | 56.63 % | -8.89 pp | 56.63 % | -8.89 pp | 56.67 % | -8.85 pp |
| DRAM Throughput (%) | 62.27 % | 56.56 % | -5.71 pp | 56.59 % | -5.68 pp | 56.61 % | -5.66 pp |
| Achieved Occupancy (%) | 94.86 % | 95.40 % | +0.54 pp | 95.39 % | +0.53 pp | 95.39 % | +0.53 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 118.79 | 118.79 | +0.00 | 118.79 | +0.00 | 118.79 | +0.00 |
| No Eligible (%) | 83.71 % | 87.91 % | +4.20 pp | 87.99 % | +4.28 pp | 87.93 % | +4.22 pp |
| IPC Active | 0.65 inst/cycle | 0.48 inst/cycle | -0.17 | 0.48 inst/cycle | -0.17 | 0.48 inst/cycle | -0.17 |


### Key Learnings

#### ❌ Plan D: `cap_registers_with_maxnreg_to_restore_occupancy` (0.746×)

Applying maxnreg=32 to all autotune configs successfully restored theoretical occupancy from 50% to 100% and achieved occupancy from 45.88% to 95.39%, and registers per thread dropped from 57 to 25 as intended. However, warp cycles per issued instruction exploded from 15.51 to 126.26 (8x worse), eligible warps per scheduler collapsed from 1.2 to 0.19, and No Eligible% rose from 52.85% to 87.93% — indicating severe L1TEX memory stall latency that the lower-register, smaller-tile configs cannot hide. The CPIStall rule confirms 62.6 cycles average stall on L1TEX operations, representing 49.6% of total stall time. The net result is a 1.34x slowdown (892us vs 1180us before, but 0.75x vs the true baseline of ~665us): high occupancy with many stalled warps is worse than moderate occupancy with low per-instruction stall latency when the kernel is bottlenecked on non-coalesced memory access patterns.

> **Applicable when**: Never applicable as described when the kernel has severe non-coalesced memory access patterns. maxnreg capping is only beneficial when the kernel's per-instruction stall latency is low (Warp Cycles Per Issued Instruction < ~30) and the bottleneck is genuinely occupancy-limited rather than memory-latency-limited.

> **Avoid when**: Kernel has non-coalesced memory access patterns causing high L1TEX stall latency (CPIStall rule triggered, Warp Cycles Per Issued Instruction > 50). In this regime, high occupancy via maxnreg capping cannot hide the long memory latency — the many resident warps are all stalled simultaneously on the same non-coalesced loads, so eligible warps per scheduler collapses to near zero. The result is worse than moderate occupancy with lower per-instruction stall. Also avoid when the kernel is already at the true baseline performance (e.g., 665us) and the intermediate state being optimized is itself a regression from a prior bad autotune.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_stores_with_evict_last_loads | 1.794× | 1.002× | ✓ |  |
| B | vectorized_stores_with_contiguous_hints_and_evict_last_loads | 4.228× | 1.005× | ✓ | **✓ best** |
| C | vectorized_loads_via_yblock_contiguous_hints | 2.801× | 0.745× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_stores... | Δ | vectorized_stores... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 665120 ns | 663940 ns | -1180 ns (-0.2%) | 662050 ns | -3070 ns (-0.5%) | 892450 ns | +227330 ns (+34.2%) |
| SM Throughput (%) | 16.19 % | 16.28 % | +0.09 pp | 17.15 % | +0.96 pp | 12.00 % | -4.19 pp |
| Memory Throughput (%) | 65.52 % | 65.64 % | +0.12 pp | 65.86 % | +0.34 pp | 56.61 % | -8.91 pp |
| DRAM Throughput (%) | 62.27 % | 62.37 % | +0.10 pp | 62.56 % | +0.29 pp | 56.61 % | -5.66 pp |
| Achieved Occupancy (%) | 94.86 % | 94.89 % | +0.03 pp | 95.00 % | +0.14 pp | 95.40 % | +0.54 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 118.79 | 118.79 | +0.00 | 118.79 | +0.00 | 118.79 | +0.00 |
| No Eligible (%) | 83.71 % | 83.66 % | -0.05 pp | 82.90 % | -0.81 pp | 87.99 % | +4.28 pp |
| IPC Active | 0.65 inst/cycle | 0.65 inst/cycle | +0.00 | 0.69 inst/cycle | +0.04 | 0.48 inst/cycle | -0.17 |

---

## Iteration 4

**Best so far**: 1.0046× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_multiple_of_hint_to_enable_128bit_vectorized_stores | 3.386× | 0.745× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_multiple_of_h... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 665120 ns | 892830 ns | +227710 ns (+34.2%) |
| SM Throughput (%) | 16.19 % | 12.00 % | -4.19 pp |
| Memory Throughput (%) | 65.52 % | 56.63 % | -8.89 pp |
| DRAM Throughput (%) | 62.27 % | 56.61 % | -5.66 pp |
| Achieved Occupancy (%) | 94.86 % | 95.39 % | +0.53 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 118.79 | 118.79 | +0.00 |
| No Eligible (%) | 83.71 % | 87.96 % | +4.25 pp |
| IPC Active | 0.65 inst/cycle | 0.48 inst/cycle | -0.17 |


### Key Learnings

#### ❌ Plan A: `fix_multiple_of_hint_to_enable_128bit_vectorized_stores` (0.745×)

Changing tl.multiple_of(arange, 1) to tl.multiple_of(arange, XBLOCK) and switching the load eviction policy from 'evict_last' to 'evict_first' caused a 34% slowdown (662us → 893us). The dominant effect was a collapse in L2 hit rate from 66.16% to 56.42%, which reduced effective memory bandwidth and worsened warp stall cycles from 88.28 to 126.56 cycles/instruction. The evict_first change is the likely culprit: despite 118 waves/SM, the 'evict_last' policy was retaining useful L2 cache lines across waves (66% L2 hit rate), and switching to 'evict_first' aggressively evicted those lines, destroying inter-wave L2 reuse. The multiple_of(XBLOCK) hint alone cannot compensate for the L2 hit rate regression caused by the eviction policy change.

> **Applicable when**: Never applicable as described — this is a negative example. The combination of multiple_of(XBLOCK) hint and evict_first on a non-coalesced load with high L2 hit rate is counterproductive.

> **Avoid when**: The kernel has a meaningful L2 hit rate (>50%) on its non-coalesced loads despite many waves/SM. In this case, 'evict_last' is actively beneficial because the L2 is retaining cache lines that are reused across waves (e.g., when the non-coalesced load dimension has limited cardinality relative to L2 capacity). Switching to 'evict_first' destroys this inter-wave L2 reuse, collapsing L2 hit rate and increasing effective memory latency. The multiple_of(XBLOCK) hint on the store-side index provides negligible benefit when the kernel is bottlenecked on load-side L1TEX stalls. Do not assume 'evict_first' is always correct for streaming kernels with many waves — measure L2 hit rate before and after.

---

## Iteration 5

**Best so far**: 1.0046× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_yblock_contiguous_hint_on_load_address | 3.384× | 0.746× | ✓ |  |
| B | shared_memory_tiled_transpose_with_correct_2d_indexing | 3.289× | 2.670× | ✓ |  |
| D | restore_large_tiles_to_hide_noncoalesced_load_latency | 1.666× | 0.898× | ✓ |  |
| E | add_triton_autotune | 0.741× | 0.744× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | shared_memory_til... | Δ | restore_large_til... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 665120 ns | 892130 ns | +227010 ns (+34.1%) | 249120 ns | -416000 ns (-62.5%) | 740960 ns | +75840 ns (+11.4%) | 894340 ns | +229220 ns (+34.5%) |
| SM Throughput (%) | 16.19 % | 11.97 % | -4.22 pp | 54.88 % | +38.69 pp | 47.79 % | +31.60 pp | 12.02 % | -4.17 pp |
| Memory Throughput (%) | 65.52 % | 56.69 % | -8.83 pp | 46.60 % | -18.92 pp | 85.09 % | +19.57 pp | 56.53 % | -8.99 pp |
| DRAM Throughput (%) | 62.27 % | 56.63 % | -5.64 pp | 40.13 % | -22.14 pp | 55.95 % | -6.32 pp | 56.52 % | -5.75 pp |
| Achieved Occupancy (%) | 94.86 % | 95.40 % | +0.54 pp | 87.89 % | -6.97 pp | 68.30 % | -26.56 pp | 95.41 % | +0.55 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 26.00 register/thread | +1 | 34.00 register/thread | +9 | 25.00 register/thread | +0 |
| Waves/SM | 118.79 | 118.79 | +0.00 | 118.79 | +0.00 | 316.77 | +197.98 | 118.79 | +0.00 |
| No Eligible (%) | 83.71 % | 87.98 % | +4.27 pp | 50.15 % | -33.56 pp | 52.00 % | -31.71 pp | 88.00 % | +4.29 pp |
| IPC Active | 0.65 inst/cycle | 0.48 inst/cycle | -0.17 | 2.02 inst/cycle | +1.37 | 1.93 inst/cycle | +1.28 | 0.48 inst/cycle | -0.17 |


### Key Learnings

#### ❌ Plan A: `vectorized_loads_via_yblock_contiguous_hint_on_load_address` (0.746×)

Adding tl.max_contiguous + tl.multiple_of hints to the YBLOCK arange and expanding autotune configs to include larger YBLOCK values (32, 64) produced absolutely no measurable change in any profiling metric: Duration remained ~892us, Warp Cycles Per Issued Instruction stayed at ~126.6, DRAM throughput held at ~56.6%, L2 hit rate was unchanged at ~56.4%, and L1TEX stall cycles were identical. This confirms the earlier finding (item exp_triton_yblock_contiguous_hints_no_effect_003): when the load address is derived from the hinted index via a modulo operation (y0 = yindex % 512), the compiler cannot propagate the contiguity proof through the modulo to the final address, so no vectorized loads are emitted regardless of YBLOCK size. Adding larger YBLOCK autotune configs (32, 64) also had no effect because the autotune continued to select the same small-tile config that was already optimal.

> **Applicable when**: Never applicable as described. tl.max_contiguous/tl.multiple_of hints on a YBLOCK arange are ineffective when the load address is derived from that index via a modulo operation (y0 = yindex % N), because the modulo prevents the compiler from propagating the contiguity proof to the final memory address. This is true regardless of YBLOCK size.

> **Avoid when**: The load address contains a modulo of the hinted index (addr = (hinted_index % N) + stride * other_index). In this pattern, the hint proves the raw index is contiguous but cannot prove the post-modulo address is contiguous or aligned, so no vectorization occurs and all profiling metrics remain identical. Also do not expand autotune YBLOCK configs when the existing small-tile config already wins autotuning — larger YBLOCK values will not be selected and have no effect on the chosen kernel configuration.

---

## Cumulative Progress

```
Baseline:  665120 ns  (1.000×)
Iter 3     662050 ns  (1.005×)  — vectorized_stores_with_contiguous_hints_and_evict_last_loads

Final best: 662050 ns  (1.0046×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, maxnreg=32),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, maxnreg=32),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, maxnreg=32),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, maxnreg=32),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=2, maxnreg=32),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8, maxnreg=32),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=8, maxnreg=32),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_cat_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 5120
@@ -13,15 +26,14 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y0 = (yindex % 512)
     y1 = yindex // 512
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 25690112*y1), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 25690112*y1), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (x2 + 50176*y3), tmp0, xmask & ymask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -29,7 +41,7 @@

     ynumel = 5120
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_cat_convolution_11[grid](arg_0, arg_1, 5120, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_cat_convolution_11[grid](arg_0, arg_1, 5120, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

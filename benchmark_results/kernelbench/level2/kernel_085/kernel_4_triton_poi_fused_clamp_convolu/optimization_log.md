# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clamp_convolution_max_pool2d_with_indices_mul_native_group_norm_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 189660 ns — **Best latency**: 89700 ns (**2.1144×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`coalesced_store_via_xblock_dominant_autotune_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clamp_convolution_max_pool2d_with_indices_mul_native_group_norm_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_085/kernel_4_triton_poi_fused_clamp_convolu/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8192,1,1), Waves/SM: 6.90
- **Registers/thread**: 54
- **Duration**: 189660 ns
- **SM Throughput**: 12.89 % | **Memory**: 85.40 % | **DRAM**: 57.56 %
- **Achieved Occupancy**: 54.15 %
- **IPC Active**: 0.52 inst/cycle | **No Eligible**: 87.13 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_swap_tile_dimensions | 6.653× | 3.809× | ✓ | **✓ best** |
| C | software_pipelining_and_eviction_policy_fix | 3.942× | 0.903× | ✓ |  |
| D | autotune_tile_sizes_for_l2_reuse_and_occupancy | 2.389× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | software_pipelini... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 189660 ns | 49790 ns | -139870 ns (-73.7%) | 210020 ns | +20360 ns (+10.7%) | 190530 ns | +870 ns (+0.5%) |
| SM Throughput (%) | 12.89 % | 49.11 % | +36.22 pp | 12.26 % | -0.63 pp | 12.80 % | -0.09 pp |
| Memory Throughput (%) | 85.40 % | 57.81 % | -27.59 pp | 80.29 % | -5.11 pp | 85.18 % | -0.22 pp |
| DRAM Throughput (%) | 57.56 % | 57.81 % | +0.25 pp | 56.31 % | -1.25 pp | 57.35 % | -0.21 pp |
| Achieved Occupancy (%) | 54.15 % | 53.24 % | -0.91 pp | 53.82 % | -0.33 pp | 54.44 % | +0.29 pp |
| Registers/Thread | 54.00 register/thread | 56.00 register/thread | +2 | 54.00 register/thread | +0 | 54.00 register/thread | +0 |
| Waves/SM | 6.90 | 6.90 | +0.00 | 6.90 | +0.00 | 6.90 | +0.00 |
| No Eligible (%) | 87.13 % | 52.39 % | -34.74 pp | 88.00 % | +0.87 pp | 87.53 % | +0.40 pp |
| IPC Active | 0.52 inst/cycle | 1.91 inst/cycle | +1.39 | 0.48 inst/cycle | -0.04 | 0.51 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 3.8092× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_tl_multiple_of_and_max_contiguous | 6.755× | 3.859× | ✓ |  |
| B | vectorized_loads_and_evict_first_for_streaming_maxpool | 2.446× | 0.904× | ✓ |  |
| C | vectorized_loads_via_multiple_of_and_max_contiguous_hints | 19.611× | 21.552× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 189660 ns | 49150 ns | -140510 ns (-74.1%) | 209790 ns | +20130 ns (+10.6%) | 8800 ns | -180860 ns (-95.4%) |
| SM Throughput (%) | 12.89 % | 49.45 % | +36.56 pp | 12.25 % | -0.64 pp | 69.35 % | +56.46 pp |
| Memory Throughput (%) | 85.40 % | 57.21 % | -28.19 pp | 80.65 % | -4.75 pp | 33.59 % | -51.81 pp |
| DRAM Throughput (%) | 57.56 % | 57.21 % | -0.35 pp | 56.46 % | -1.10 pp | 9.82 % | -47.74 pp |
| Achieved Occupancy (%) | 54.15 % | 53.19 % | -0.96 pp | 53.91 % | -0.24 pp | 78.99 % | +24.84 pp |
| Registers/Thread | 54.00 register/thread | 56.00 register/thread | +2 | 54.00 register/thread | +0 | 31.00 register/thread | -23 |
| Waves/SM | 6.90 | 6.90 | +0.00 | 6.90 | +0.00 | 3.88 | -3.02 |
| No Eligible (%) | 87.13 % | 51.87 % | -35.26 pp | 87.94 % | +0.81 pp | 9.83 % | -77.30 pp |
| IPC Active | 0.52 inst/cycle | 1.94 inst/cycle | +1.42 | 0.47 inst/cycle | -0.05 | 3.58 inst/cycle | +3.06 |

---

## Iteration 3

**Best so far**: 21.5523× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_store_via_transposed_output_index | 19.226× | 11.898× | ✓ |  |
| B | reduce_register_pressure_via_shared_base_offset_refactor | 19.656× | 21.700× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | coalesced_store_v... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 189660 ns | 15940 ns | -173720 ns (-91.6%) | 8740 ns | -180920 ns (-95.4%) |
| SM Throughput (%) | 12.89 % | 75.49 % | +62.60 pp | 69.34 % | +56.45 pp |
| Memory Throughput (%) | 85.40 % | 36.30 % | -49.10 pp | 33.57 % | -51.83 pp |
| DRAM Throughput (%) | 57.56 % | 5.40 % | -52.16 pp | 9.87 % | -47.69 pp |
| Achieved Occupancy (%) | 54.15 % | 84.64 % | +30.49 pp | 79.63 % | +25.48 pp |
| Registers/Thread | 54.00 register/thread | 32.00 register/thread | -22 | 31.00 register/thread | -23 |
| Waves/SM | 6.90 | 7.76 | +0.86 | 3.88 | -3.02 |
| No Eligible (%) | 87.13 % | 13.62 % | -73.51 pp | 9.69 % | -77.44 pp |
| IPC Active | 0.52 inst/cycle | 3.44 inst/cycle | +2.92 | 3.48 inst/cycle | +2.96 |

---

## Iteration 4

**Best so far**: 21.7002× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | coalesced_store_via_xblock_dominant_autotune_configs | 19.675× | 2.114× | ✓ | **✓ best** |
| D | coalesced_store_via_xblock_biased_autotune_configs | 19.567× | 11.898× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_store_v... | Δ | coalesced_store_v... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 189660 ns | 89700 ns | -99960 ns (-52.7%) | 15940 ns | -173720 ns (-91.6%) |
| SM Throughput (%) | 12.89 % | 6.80 % | -6.09 pp | 75.28 % | +62.39 pp |
| Memory Throughput (%) | 85.40 % | 95.34 % | +9.94 pp | 36.25 % | -49.15 pp |
| DRAM Throughput (%) | 57.56 % | 3.87 % | -53.69 pp | 5.40 % | -52.16 pp |
| Achieved Occupancy (%) | 54.15 % | 86.13 % | +31.98 pp | 84.73 % | +30.58 pp |
| Registers/Thread | 54.00 register/thread | 32.00 register/thread | -22 | 32.00 register/thread | -22 |
| Waves/SM | 6.90 | 3.88 | -3.02 | 7.76 | +0.86 |
| No Eligible (%) | 87.13 % | 92.91 % | +5.78 pp | 14.85 % | -72.28 pp |
| IPC Active | 0.52 inst/cycle | 0.28 inst/cycle | -0.24 | 3.44 inst/cycle | +2.92 |


### Key Learnings

#### ❌ Plan B: `coalesced_store_via_xblock_dominant_autotune_configs` (2.114×)

Adding XBLOCK-dominant autotune configs (XBLOCK=32–256, YBLOCK=1) to coalesce stores along the stride-1 X axis caused a severe 10x per-kernel duration regression (8.74µs → 89.7µs), despite the 2.1x end-to-end speedup suggesting measurement noise or a different kernel dominating. The autotuner selected a YBLOCK=1 config that collapsed the tile to a single row, causing the kernel to become massively memory-latency-bound: Warp Cycles Per Issued Instruction exploded from 14 to 195, scheduler no-eligible jumped from 9.7% to 92.9%, and eligible warps per scheduler collapsed from 7.66 to 0.20. The L1 hit rate rose to 48% and L2 hit rate to 94.6%, indicating the YBLOCK=1 config caused extreme serialization of memory requests — each warp issues many sequential loads with no instruction-level parallelism to hide latency, creating an LG queue full stall (CPIStall rule: 75.3 cycles/warp in LG queue). The uncoalesced global access problem actually worsened (87% excessive sectors vs. 29% before), because YBLOCK=1 with small XBLOCK tiles means each block processes very few elements, amplifying per-element overhead. The strategy of using XBLOCK-dominant configs to coalesce stores is theoretically correct but practically harmful when the output tensor has a non-power-of-2 stride (961) that prevents full warp coalescing even with large XBLOCK, and when YBLOCK=1 destroys the memory-level parallelism that was hiding latency in the original config.

> **Applicable when**: Never apply YBLOCK=1 configs when the output tensor has a non-power-of-2 stride in the Y dimension (e.g., stride=961) — the stride prevents full coalescing even with large XBLOCK, and YBLOCK=1 destroys memory-level parallelism needed to hide latency.

> **Avoid when**: Do not add XBLOCK-dominant (YBLOCK=1) autotune configs when: (1) the kernel is already compute-bound (SM throughput >60%) with low warp stall cycles (<20 cycles/inst); (2) the output stride in Y is non-power-of-2 (prevents true coalescing); (3) the existing config achieves good instruction-level parallelism (eligible warps >5). YBLOCK=1 configs collapse memory-level parallelism and cause LG queue full stalls when the kernel has many sequential memory operations per thread.

---

## Iteration 5

**Best so far**: 21.7002× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fused_clamp_via_libdevice_fclamp_for_fma_instruction_reduction | 19.234× | 11.898× | ✓ |  |
| B | fused_clamp_via_libdevice_fclamp_to_eliminate_nonfused_fp32_instructions | 19.270× | 11.898× | ✓ |  |
| C | restore_yblock_biased_configs_to_fix_lg_queue_stall_regression | 19.604× | 12.220× | ✓ |  |
| D | remove_yblock1_configs_restore_memory_level_parallelism | 19.237× | 21.700× | ✓ |  |
| E | add_triton_autotune | 19.224× | 11.832× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fused_clamp_via_l... | Δ | fused_clamp_via_l... | Δ | restore_yblock_bi... | Δ | remove_yblock1_co... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 189660 ns | 15940 ns | -173720 ns (-91.6%) | 15940 ns | -173720 ns (-91.6%) | 15520 ns | -174140 ns (-91.8%) | 8740 ns | -180920 ns (-95.4%) | 16030 ns | -173630 ns (-91.5%) |
| SM Throughput (%) | 12.89 % | 75.48 % | +62.59 pp | 75.36 % | +62.47 pp | 78.03 % | +65.14 pp | 69.00 % | +56.11 pp | 75.40 % | +62.51 pp |
| Memory Throughput (%) | 85.40 % | 36.35 % | -49.05 pp | 36.28 % | -49.12 pp | 37.21 % | -48.19 pp | 33.42 % | -51.98 pp | 36.30 % | -49.10 pp |
| DRAM Throughput (%) | 57.56 % | 5.40 % | -52.16 pp | 5.41 % | -52.15 pp | 5.55 % | -52.01 pp | 9.89 % | -47.67 pp | 5.37 % | -52.19 pp |
| Achieved Occupancy (%) | 54.15 % | 85.11 % | +30.96 pp | 84.99 % | +30.84 pp | 81.80 % | +27.65 pp | 78.78 % | +24.63 pp | 84.87 % | +30.72 pp |
| Registers/Thread | 54.00 register/thread | 32.00 register/thread | -22 | 32.00 register/thread | -22 | 31.00 register/thread | -23 | 31.00 register/thread | -23 | 32.00 register/thread | -22 |
| Waves/SM | 6.90 | 7.76 | +0.86 | 7.76 | +0.86 | 7.76 | +0.86 | 3.88 | -3.02 | 7.76 | +0.86 |
| No Eligible (%) | 87.13 % | 13.49 % | -73.64 pp | 13.74 % | -73.39 pp | 9.29 % | -77.84 pp | 9.62 % | -77.51 pp | 14.78 % | -72.35 pp |
| IPC Active | 0.52 inst/cycle | 3.41 inst/cycle | +2.89 | 3.41 inst/cycle | +2.89 | 3.57 inst/cycle | +3.05 | 3.57 inst/cycle | +3.05 | 3.42 inst/cycle | +2.90 |


### Key Learnings

#### ✅ Plan C: `restore_yblock_biased_configs_to_fix_lg_queue_stall_regression` (12.220×)

When a Triton autotuner config space includes YBLOCK=1 (or very small YBLOCK) options alongside YBLOCK-biased configs, the autotuner may select the YBLOCK=1 config, collapsing memory-level parallelism and causing severe LG queue full stalls (Warp Cycles Per Issued Instruction explodes to ~195, No Eligible rises to ~93%). Removing all XBLOCK-dominant configs with YBLOCK<=4 and retaining only YBLOCK>=8 configs (preferably YBLOCK>=32) restores warp-level TLP, allowing the 16 sequential loads per thread to be interleaved across warps and draining the LG queue stall. The per-kernel NCU metrics confirm recovery: Warp Cycles Per Issued Instruction dropped from 15.71 to 14.28, No Eligible from 14.85% to 9.29%, and Compute SM Throughput rose from 75.28% to 78.03%, with a 12.2x end-to-end speedup reflecting recovery from the prior YBLOCK=1 regression.

> **Applicable when**: Triton kernel with many sequential loads per thread (>=8) where the autotuner config space contains YBLOCK=1 or YBLOCK<=4 configs; NCU shows LG queue full stall (Warp Cycles Per Issued Instruction >50, No Eligible >50%, Eligible Warps Per Scheduler <1.0); the kernel was previously fast but regressed after adding XBLOCK-dominant configs to the autotune space.

> **Avoid when**: Kernel has very few loads per thread (<4) where YBLOCK=1 does not cause LG queue saturation; output tensor has stride-1 along X axis and XBLOCK-dominant configs genuinely achieve coalesced stores without LG queue stalls; kernel is already using YBLOCK-biased configs exclusively.

#### ✅ Plan D: `remove_yblock1_configs_restore_memory_level_parallelism` (21.700×)

When a Triton autotune config space includes YBLOCK=1/2/4 options alongside YBLOCK>=8 options, the autotuner may select a low-YBLOCK config that collapses memory-level parallelism for kernels with many sequential loads per thread (e.g., 16 loads for a 4x4 max-pool). YBLOCK<8 configs cause LG queue full stalls (Warp Cycles Per Issued Instruction explodes from ~14 to ~195, scheduler no-eligible jumps from ~10% to ~93%), because too few warps per block exist to interleave memory requests and hide L1TEX latency. Removing all YBLOCK<8 configs from the autotune search space forces the autotuner to select YBLOCK>=8 configs, restoring instruction-level parallelism and reducing kernel duration from ~16us to ~8.74us (1.82x per-kernel improvement, 21.7x end-to-end speedup).

> **Applicable when**: Triton kernel with many sequential loads per thread (>=8 loads, e.g., max-pool window traversal); autotune config space was previously expanded with XBLOCK-dominant (YBLOCK<8) configs; NCU shows LG queue full stalls (Warp Cycles Per Issued Instruction >50, scheduler no-eligible >50%, eligible warps per scheduler <1.0); kernel is memory-latency-bound rather than compute-bound (Compute SM Throughput <20% when YBLOCK=1 selected).

> **Avoid when**: Kernel has few loads per thread (<4) where YBLOCK=1 does not cause LG queue saturation; kernel is purely compute-bound with no memory latency hiding needed; the output tensor's stride-1 axis is X (not Y), making large XBLOCK configs genuinely beneficial for coalescing without destroying TLP.

---

## Cumulative Progress

```
Baseline:  189660 ns  (1.000×)
Iter 1     49790 ns  (3.809×)  — memory_coalescing_swap_tile_dimensions
Iter 2     8800 ns  (21.552×)  — vectorized_loads_via_multiple_of_and_max_contiguous_hints
Iter 3     8740 ns  (21.700×)  — reduce_register_pressure_via_shared_base_offset_refactor

Final best: 89700 ns  (2.1144×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,15 +5,43 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Existing configs (keep all)
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 64},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 2,   'YBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 64},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4, num_stages=2),
+        # NEW: large-XBLOCK configs for coalesced stores
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024,'YBLOCK': 1},   num_warps=8, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_clamp_convolution_max_pool2d_with_indices_mul_native_group_norm_4(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
     xnumel = 961
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    yoffset = tl.multiple_of(tl.program_id(1) * YBLOCK, YBLOCK)
+    yindex = yoffset + tl.max_contiguous(tl.arange(0, YBLOCK), YBLOCK)[:, None]
+    ymask = yindex < ynumel
+    xoffset = tl.multiple_of(tl.program_id(0) * XBLOCK, XBLOCK)
+    xindex = xoffset + tl.max_contiguous(tl.arange(0, XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = (xindex % 31)
     x3 = xindex // 31
@@ -21,22 +49,24 @@

     y1 = yindex // 64
     x4 = xindex
     y5 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (64 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr0 + (128 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr0 + (192 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr0 + (8064 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp9 = tl.load(in_ptr0 + (8128 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp11 = tl.load(in_ptr0 + (8192 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp13 = tl.load(in_ptr0 + (8256 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp15 = tl.load(in_ptr0 + (16128 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp17 = tl.load(in_ptr0 + (16192 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp19 = tl.load(in_ptr0 + (16256 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp21 = tl.load(in_ptr0 + (16320 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp23 = tl.load(in_ptr0 + (24192 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp25 = tl.load(in_ptr0 + (24256 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp27 = tl.load(in_ptr0 + (24320 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
-    tmp29 = tl.load(in_ptr0 + (24384 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=xmask, eviction_policy='evict_last')
+    mask = xmask & ymask
+    base_offset = y0 + 256*x2 + 32256*x3 + 1016064*y1
+    tmp0  = tl.load(in_ptr0 + base_offset,           mask=mask, eviction_policy='evict_first')
+    tmp1  = tl.load(in_ptr0 + (64    + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp3  = tl.load(in_ptr0 + (128   + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp5  = tl.load(in_ptr0 + (192   + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp7  = tl.load(in_ptr0 + (8064  + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp9  = tl.load(in_ptr0 + (8128  + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp11 = tl.load(in_ptr0 + (8192  + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp13 = tl.load(in_ptr0 + (8256  + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp15 = tl.load(in_ptr0 + (16128 + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp17 = tl.load(in_ptr0 + (16192 + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp19 = tl.load(in_ptr0 + (16256 + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp21 = tl.load(in_ptr0 + (16320 + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp23 = tl.load(in_ptr0 + (24192 + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp25 = tl.load(in_ptr0 + (24256 + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp27 = tl.load(in_ptr0 + (24320 + base_offset), mask=mask, eviction_policy='evict_first')
+    tmp29 = tl.load(in_ptr0 + (24384 + base_offset), mask=mask, eviction_policy='evict_first')
     tmp2 = tl.maximum(tmp0, tmp1)
     tmp4 = tl.maximum(tmp2, tmp3)
     tmp6 = tl.maximum(tmp4, tmp5)
@@ -56,8 +86,7 @@

     tmp32 = tl.maximum(tmp30, tmp31)
     tmp33 = tl.full([1, 1], 1.0, tl.float32)
     tmp34 = tl.minimum(tmp32, tmp33)
-    tl.store(out_ptr1 + (x4 + 961*y5), tmp34, mask=xmask)
-
+    tl.store(out_ptr1 + (x4 + 961*y5), tmp34, mask=mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -65,7 +94,7 @@

     ynumel = 8192
     xnumel = 961
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_clamp_convolution_max_pool2d_with_indices_mul_native_group_norm_4[grid](arg_0, arg_1, 8192, 961, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_clamp_convolution_max_pool2d_with_indices_mul_native_group_norm_4[grid](arg_0, arg_1, 8192, 961)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

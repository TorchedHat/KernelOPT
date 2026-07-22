# Kernel Optimization Log

## Summary

- **Kernel**: `l1_norm_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12340000 ns — **Best latency**: 12330000 ns (**1.0008×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fused_fp32_instructions_abs_and_reciprocal_multiply`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `l1_norm_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_038/synthesis/kernel_0_triton_red_fused_abs_div_mean_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 15.52
- **Registers/thread**: 30
- **Duration**: 12340000 ns
- **SM Throughput**: 18.98 % | **Memory**: 52.87 % | **DRAM**: 52.87 %
- **Achieved Occupancy**: 94.77 %
- **IPC Active**: 0.76 inst/cycle | **No Eligible**: 80.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_streaming_loads | 0.984× | 1.088× | ✓ |  |
| B | vectorized_loads_via_contiguity_hints | 0.999× | 1.002× | ✓ |  |
| C | memory_coalescing_cache_modifier_and_pipeline_stages | 0.998× | 1.000× | ✓ |  |
| D | single_pass_l1_norm_eliminate_second_dram_read | 0.573× | 0.956× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_s... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ | single_pass_l1_no... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12340000 ns | 11340000 ns | -1000000 ns (-8.1%) | 12310000 ns | -30000 ns (-0.2%) | 12340000 ns | +0 ns (+0.0%) | 12910000 ns | +570000 ns (+4.6%) |
| SM Throughput (%) | 18.98 % | 25.70 % | +6.72 pp | 19.10 % | +0.12 pp | 18.95 % | -0.03 pp | 18.11 % | -0.87 pp |
| Memory Throughput (%) | 52.87 % | 53.73 % | +0.86 pp | 52.94 % | +0.07 pp | 52.87 % | +0.00 pp | 56.13 % | +3.26 pp |
| DRAM Throughput (%) | 52.87 % | 53.73 % | +0.86 pp | 52.94 % | +0.07 pp | 52.87 % | +0.00 pp | 56.13 % | +3.26 pp |
| Achieved Occupancy (%) | 94.77 % | 95.22 % | +0.45 pp | 94.82 % | +0.05 pp | 94.83 % | +0.06 pp | 96.65 % | +1.88 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 28.00 register/thread | -2 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 80.97 % | 74.14 % | -6.83 pp | 80.96 % | -0.01 pp | 80.90 % | -0.07 pp | 81.77 % | +0.80 pp |
| IPC Active | 0.76 inst/cycle | 1.03 inst/cycle | +0.27 | 0.76 inst/cycle | +0.00 | 0.76 inst/cycle | +0.00 | 0.73 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_software_pipeline_depth_num_stages | 1.022× | 1.085× | ✓ |  |
| B | increase_num_stages_software_pipelining | 1.095× | 1.001× | ✓ | **✓ best** |
| C | fused_abs_fma_and_autotune_block_expansion | 0.998× | 1.000× | ✓ |  |
| D | eliminate_intermediate_pass1_store_read_xptr_twice | 0.996× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_software... | Δ | increase_num_stag... | Δ | fused_abs_fma_and... | Δ | eliminate_interme... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12340000 ns | 11370000 ns | -970000 ns (-7.9%) | 12330000 ns | -10000 ns (-0.1%) | 12340000 ns | +0 ns (+0.0%) | 12320000 ns | -20000 ns (-0.2%) |
| SM Throughput (%) | 18.98 % | 25.72 % | +6.74 pp | 19.06 % | +0.08 pp | 18.11 % | -0.87 pp | 15.89 % | -3.09 pp |
| Memory Throughput (%) | 52.87 % | 53.61 % | +0.74 pp | 52.86 % | -0.01 pp | 52.85 % | -0.02 pp | 52.90 % | +0.03 pp |
| DRAM Throughput (%) | 52.87 % | 53.61 % | +0.74 pp | 52.86 % | -0.01 pp | 52.85 % | -0.02 pp | 52.90 % | +0.03 pp |
| Achieved Occupancy (%) | 94.77 % | 95.27 % | +0.50 pp | 94.81 % | +0.04 pp | 94.86 % | +0.09 pp | 94.76 % | -0.01 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 32.00 register/thread | +2 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 80.97 % | 74.21 % | -6.76 pp | 80.95 % | -0.02 pp | 81.77 % | +0.80 pp | 84.44 % | +3.47 pp |
| IPC Active | 0.76 inst/cycle | 1.03 inst/cycle | +0.27 | 0.76 inst/cycle | +0.00 | 0.73 inst/cycle | -0.03 | 0.62 inst/cycle | -0.14 |


### Key Learnings

#### ✅ Plan A: `increase_software_pipeline_depth_num_stages` (1.085×)

Adding higher num_stages (4, 5) and very large BLOCK_N (65536) configs to the Triton autotune search space produced a modest 8.5% speedup, but the NCU profiling metrics are essentially unchanged before and after: DRAM throughput, L2 hit rate, warp cycles per instruction, and scheduler eligibility all moved by less than 0.5%. This indicates the autotune selected a config similar to the prior best, and the speedup likely reflects reduced autotuning overhead or wall-clock variance rather than a genuine pipeline-depth improvement. When the kernel is already deeply stalled on L1TEX scoreboard dependencies (CPIStall at ~82% of warp cycles) and DRAM throughput is only ~54% of peak, expanding num_stages alone does not meaningfully increase in-flight memory requests or reduce stall depth if the bottleneck is structural (two-pass pattern over data far exceeding L2).

> **Applicable when**: Triton kernel autotune search space has not yet explored num_stages > 3; kernel is memory-latency bound with CPIStall triggered; DRAM throughput is moderate (40-70%) and L1TEX scoreboard stalls are the dominant warp stall type; kernel has not already been tuned with eviction policies or other latency-hiding techniques.

> **Avoid when**: Kernel is already deeply stalled on L1TEX scoreboard dependencies (CPIStall >80% of warp cycles) due to a structural two-pass pattern over data far exceeding L2 capacity — in this regime, deeper num_stages does not move the needle on DRAM throughput, L2 hit rate, or warp stall cycles; do not expect >5% gain from this change alone when the bottleneck is the two-pass memory access pattern itself rather than pipeline depth.

---

## Iteration 3

**Best so far**: 1.0008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fused_fp32_instructions_abs_and_reciprocal_multiply | 1.025× | 1.090× | ✓ |  |
| B | fma_vector_accumulation_reduce_nonfused_fp32 | 0.988× | 0.968× | ✓ |  |
| C | fused_fp32_instructions_abs_and_reciprocal_multiply | 1.106× | 1.001× | ✓ | **✓ best** |
| D | fma_fusion_via_reciprocal_precompute_and_multiple_of_hints | 0.996× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fused_fp32_instru... | Δ | fma_vector_accumu... | Δ | fused_fp32_instru... | Δ | fma_fusion_via_re... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12340000 ns | 11320000 ns | -1020000 ns (-8.3%) | 12750000 ns | +410000 ns (+3.3%) | 12330000 ns | -10000 ns (-0.1%) | 12330000 ns | -10000 ns (-0.1%) |
| SM Throughput (%) | 18.98 % | 22.12 % | +3.14 pp | 15.89 % | -3.09 pp | 15.86 % | -3.12 pp | 15.87 % | -3.11 pp |
| Memory Throughput (%) | 52.87 % | 53.79 % | +0.92 pp | 53.69 % | +0.82 pp | 52.86 % | -0.01 pp | 52.86 % | -0.01 pp |
| DRAM Throughput (%) | 52.87 % | 53.79 % | +0.92 pp | 53.69 % | +0.82 pp | 52.86 % | -0.01 pp | 52.86 % | -0.01 pp |
| Achieved Occupancy (%) | 94.77 % | 95.30 % | +0.53 pp | 95.12 % | +0.35 pp | 94.76 % | -0.01 pp | 94.73 % | -0.04 pp |
| Registers/Thread | 30.00 register/thread | 28.00 register/thread | -2 | 32.00 register/thread | +2 | 32.00 register/thread | +2 | 32.00 register/thread | +2 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 80.97 % | 77.80 % | -3.17 pp | 84.04 % | +3.07 pp | 84.38 % | +3.41 pp | 84.46 % | +3.49 pp |
| IPC Active | 0.76 inst/cycle | 0.89 inst/cycle | +0.13 | 0.63 inst/cycle | -0.13 | 0.62 inst/cycle | -0.14 | 0.62 inst/cycle | -0.14 |


### Key Learnings

#### ✅ Plan A: `fused_fp32_instructions_abs_and_reciprocal_multiply` (1.090×)

Replacing `tl.where(x >= 0.0, x, -x)` with `tl.abs(x)` and replacing per-element division `x / mean_val` with a hoisted scalar reciprocal `inv_mean = 1.0 / mean_val` followed by `x * inv_mean` reduced total executed instructions from 2.31B to 1.99B (~14% fewer) and dropped register pressure from 30 to 28 registers per thread, yielding a 9% wall-clock speedup. However, the NCU metrics show a counterintuitive increase in Warp Cycles Per Issued Instruction (59.2 → 68.4) and a decrease in Eligible Warps Per Scheduler (0.38 → 0.32), suggesting the kernel remains deeply L1TEX-scoreboard-stall-bound and the speedup comes primarily from fewer total instructions executed rather than improved instruction-level parallelism or memory latency hiding. This strategy is most effective when the FPInstructions NCU rule reports a large ratio of non-fused to fused FP32 instructions (e.g., 487M non-fused, 0 fused) and the kernel has redundant conditional-select abs patterns or hot-loop divisions that can be replaced with native abs and multiply-by-reciprocal.

> **Applicable when**: FPInstructions NCU rule reports a large count of non-fused FP32 instructions (hundreds of millions) with zero or near-zero fused instructions; kernel uses tl.where for absolute value computation; kernel performs per-element division by a scalar in a hot loop; kernel is compute-instruction-count bound rather than purely DRAM-bandwidth-bound.

> **Avoid when**: Kernel is already using tl.abs and multiply-by-reciprocal; the FPInstructions rule reports a small non-fused count (< 50M); the kernel is purely DRAM-bandwidth-bound (DRAM throughput > 90% of peak) where reducing instruction count will not move wall-clock time; numerical precision requirements prohibit multiply-by-reciprocal approximation (atol < 1e-5 for fp32).

#### ✅ Plan C: `fused_fp32_instructions_abs_and_reciprocal_multiply` (1.090×)

Replacing `tl.where(x >= 0.0, x, -x)` with `tl.abs(x)` and replacing per-element division `x / mean_val` with a hoisted scalar reciprocal `inv_mean = 1.0 / mean_val` followed by `x * inv_mean` reduced total executed instructions from 2.31B to 1.99B (~14% fewer) and dropped register pressure from 30 to 28 registers per thread, yielding a 9% wall-clock speedup. However, the NCU metrics show a counterintuitive increase in Warp Cycles Per Issued Instruction (59.2 → 68.4) and a decrease in Eligible Warps Per Scheduler (0.38 → 0.32), suggesting the kernel remains deeply L1TEX-scoreboard-stall-bound and the speedup comes primarily from fewer total instructions executed rather than improved instruction-level parallelism or memory latency hiding. This strategy is most effective when the FPInstructions NCU rule reports a large ratio of non-fused to fused FP32 instructions (e.g., 487M non-fused, 0 fused) and the kernel has redundant conditional-select abs patterns or hot-loop divisions that can be replaced with native abs and multiply-by-reciprocal.

> **Applicable when**: FPInstructions NCU rule reports a large count of non-fused FP32 instructions (hundreds of millions) with zero or near-zero fused instructions; kernel uses tl.where for absolute value computation; kernel performs per-element division by a scalar in a hot loop; kernel is compute-instruction-count bound rather than purely DRAM-bandwidth-bound.

> **Avoid when**: Kernel is already using tl.abs and multiply-by-reciprocal; the FPInstructions rule reports a small non-fused count (< 50M); the kernel is purely DRAM-bandwidth-bound (DRAM throughput > 90% of peak) where reducing instruction count will not move wall-clock time; numerical precision requirements prohibit multiply-by-reciprocal approximation (atol < 1e-5 for fp32).

---

## Iteration 4

**Best so far**: 1.0008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | alignment_hints_multiple_of_max_contiguous_coalescing | 1.024× | 1.086× | ✓ |  |
| B | multi_row_cta_tiling_coalesced_stores | 0.735× | 0.743× | ✓ |  |
| C | vectorized_contiguous_loads_max_contiguous_hint | 0.997× | 1.002× | ✓ |  |
| D | contiguity_hints_coalesced_sector_utilization | 0.988× | 0.967× | ✓ |  |


### Metric Comparison

| Metric | Baseline | alignment_hints_m... | Δ | multi_row_cta_til... | Δ | vectorized_contig... | Δ | contiguity_hints_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12340000 ns | 11360000 ns | -980000 ns (-7.9%) | 16600000 ns | +4260000 ns (+34.5%) | 12320000 ns | -20000 ns (-0.2%) | 12760000 ns | +420000 ns (+3.4%) |
| SM Throughput (%) | 18.98 % | 22.22 % | +3.24 pp | 46.69 % | +27.71 pp | 15.90 % | -3.08 pp | 15.90 % | -3.08 pp |
| Memory Throughput (%) | 52.87 % | 53.65 % | +0.78 pp | 42.88 % | -9.99 pp | 52.85 % | -0.02 pp | 53.70 % | +0.83 pp |
| DRAM Throughput (%) | 52.87 % | 53.65 % | +0.78 pp | 35.70 % | -17.17 pp | 52.85 % | -0.02 pp | 53.70 % | +0.83 pp |
| Achieved Occupancy (%) | 94.77 % | 95.32 % | +0.55 pp | 98.22 % | +3.45 pp | 94.75 % | -0.02 pp | 95.09 % | +0.32 pp |
| Registers/Thread | 30.00 register/thread | 28.00 register/thread | -2 | 32.00 register/thread | +2 | 32.00 register/thread | +2 | 32.00 register/thread | +2 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 80.97 % | 77.73 % | -3.24 pp | 53.24 % | -27.73 pp | 84.44 % | +3.47 pp | 84.09 % | +3.12 pp |
| IPC Active | 0.76 inst/cycle | 0.89 inst/cycle | +0.13 | 1.88 inst/cycle | +1.12 | 0.62 inst/cycle | -0.14 | 0.63 inst/cycle | -0.13 |


### Key Learnings

#### ✅ Plan A: `alignment_hints_multiple_of_max_contiguous_coalescing` (1.086×)

Adding tl.multiple_of() and tl.max_contiguous() alignment hints to the column index computation in Triton loops is intended to enable LDG.128 vector loads by signaling to the compiler that accesses are aligned and contiguous. However, when the kernel is already deeply stalled on L1TEX scoreboard dependencies (CPIStall at ~82% of warp cycles) due to a structural two-pass pattern over data far exceeding L2 capacity, these hints produce negligible metric changes: DRAM throughput, L2 hit rate, warp cycles per instruction, and excessive sector counts are essentially unchanged before and after. The 8.6% wall-clock speedup observed likely reflects measurement variance or minor compiler scheduling differences rather than a genuine coalescing improvement, as none of the targeted metrics (bytes-per-sector, excessive sectors, DRAM throughput) moved meaningfully.

> **Applicable when**: UncoalescedGlobalAccess NCU rule is triggered with high excessive sector counts (>10% of total sectors); bytes-utilized-per-sector is measurably below 32 (e.g., <28); kernel is NOT already deeply stalled on L1TEX scoreboard dependencies (CPIStall <60% of warp cycles); row strides are misaligned relative to L2 sector size (128 bytes).

> **Avoid when**: Kernel is already deeply L1TEX-scoreboard-stall-bound (CPIStall >80% of warp cycles) due to a structural two-pass pattern over data far exceeding L2 capacity — in this regime, alignment hints do not move DRAM throughput, excessive sector counts, or warp stall cycles; the UncoalescedGlobalAccess rule is triggered but the root cause is structural (misaligned row starts) rather than non-contiguous access patterns that compiler hints can fix; do not expect >3% gain from this change alone when the bottleneck is the two-pass memory access pattern itself.

#### ❌ Plan B: `multi_row_cta_tiling_coalesced_stores` (0.743×)

Switching from one-CTA-per-row (scalar accumulator) to BLOCK_M rows per CTA (vector accumulator of shape [BLOCK_M]) dramatically increased total executed instructions from 1.99B to 6.12B (~3x), causing a 47% wall-clock regression despite improving scheduler eligibility (No-Eligible dropped from 77.8% to 53.2%) and reducing warp stall cycles per instruction (68.4 → 33.5). The 2D tiling approach introduced large 2D pointer arithmetic, 2D mask computations, and vector reductions that generated far more instructions per element than the scalar path, overwhelming any coalescing benefit. DRAM throughput actually dropped from 53.8% to 35.7% and memory throughput fell from 2.64 TB/s to 1.75 TB/s, indicating the kernel became compute-instruction-bound rather than memory-latency-bound. The uncoalesced access pattern (176M excess sectors) was not meaningfully reduced by this approach since the L2 compression rate remained at 0% and excess sectors were unchanged.

> **Applicable when**: Never: this strategy caused a 3x instruction count explosion and 47% wall-clock regression for a row-reduction kernel with large N (65535) and many rows (32768). The 2D tiling approach is theoretically sound for improving store coalescing but in practice the overhead of 2D pointer arithmetic, 2D masking, and vector reductions dominates.

> **Avoid when**: Row-reduction kernels where N is large (>1024) and each row already maps to a single CTA with scalar accumulation — the 2D tiling adds O(BLOCK_M * BLOCK_N) pointer and mask instructions per tile that dwarf the coalescing benefit; when the kernel is already memory-latency-bound (CPIStall dominant) rather than compute-instruction-bound, adding more instructions per element will shift the bottleneck to compute and increase total runtime; when UncoalescedGlobalAccess excess sectors are caused by partial last-tile masking (N not divisible by BLOCK_N) rather than strided access patterns, 2D tiling does not eliminate the partial tile.

---

## Iteration 5

**Best so far**: 1.0008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fma_fusion_vacc_abs_accumulation_and_fast_reciprocal | 0.988× | 0.966× | ✓ |  |
| D | scalar_cta_per_row_with_fp_instruction_fusion | 1.053× | 1.088× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_vacc_a... | Δ | scalar_cta_per_ro... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12340000 ns | 12770000 ns | +430000 ns (+3.5%) | 11340000 ns | -1000000 ns (-8.1%) |
| SM Throughput (%) | 18.98 % | 15.94 % | -3.04 pp | 22.10 % | +3.12 pp |
| Memory Throughput (%) | 52.87 % | 53.68 % | +0.81 pp | 53.69 % | +0.82 pp |
| DRAM Throughput (%) | 52.87 % | 53.68 % | +0.81 pp | 53.69 % | +0.82 pp |
| Achieved Occupancy (%) | 94.77 % | 95.09 % | +0.32 pp | 95.30 % | +0.53 pp |
| Registers/Thread | 30.00 register/thread | 32.00 register/thread | +2 | 28.00 register/thread | -2 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 80.97 % | 84.08 % | +3.11 pp | 77.80 % | -3.17 pp |
| IPC Active | 0.76 inst/cycle | 0.64 inst/cycle | -0.12 | 0.89 inst/cycle | +0.13 |

---

## Cumulative Progress

```
Baseline:  12340000 ns  (1.000×)
Iter 2     12330000 ns  (1.001×)  — increase_num_stages_software_pipelining

Final best: 12330000 ns  (1.0008×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,16 +3,32 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
+        # Original 8 configs unchanged
+        triton.Config({'BLOCK_N': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_N': 1024}, num_warps=8,  num_stages=2),
         triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=2),
         triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=2),
-        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=3),
-        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_N': 256},  num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_N': 512},  num_warps=8,  num_stages=3),
         triton.Config({'BLOCK_N': 1024}, num_warps=16, num_stages=3),
         triton.Config({'BLOCK_N': 2048}, num_warps=32, num_stages=2),
+        # NEW: deeper pipeline stages to hide DRAM latency
+        triton.Config({'BLOCK_N': 512},  num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_N': 1024}, num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_N': 512},  num_warps=4,  num_stages=5),
+        triton.Config({'BLOCK_N': 1024}, num_warps=8,  num_stages=5),
+        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=5),
+        triton.Config({'BLOCK_N': 4096}, num_warps=32, num_stages=5),
+        # NEW: single-tile path — BLOCK_N=65536 > N=65535 so loop runs exactly once per pass
+        # Eliminates partial-mask loop overhead and reduces uncoalesced sector waste
+        triton.Config({'BLOCK_N': 65536}, num_warps=32, num_stages=2),
+        triton.Config({'BLOCK_N': 65536}, num_warps=32, num_stages=3),
+        triton.Config({'BLOCK_N': 65536}, num_warps=32, num_stages=4),
     ],
     key=['N'],
 )
@@ -34,24 +50,31 @@

     acc = tl.zeros((), dtype=tl.float32)
     
     for col_off in tl.range(0, N, BLOCK_N):
-        col_idx = col_off + tl.arange(0, BLOCK_N)
+        col_idx = tl.max_contiguous(
+            tl.multiple_of(col_off + tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
+        )
         mask = col_idx < N
-        x = tl.load(x_ptr + row_start + col_idx, mask=mask, other=0.0).to(tl.float32)
-        # abs
-        x_abs = tl.where(x >= 0.0, x, -x)
+        # Pass 1: streaming load — evict immediately from L2 to avoid polluting cache for Pass 2
+        x = tl.load(x_ptr + row_start + col_idx, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
+        # abs — use native tl.abs for FMA-friendly PTX abs.f32 instruction
+        x_abs = tl.abs(x)
         acc += tl.sum(x_abs, axis=0)
     
-    # Compute mean
+    # Compute mean, then hoist scalar reciprocal outside Pass 2 loop
     mean_val = acc / N
+    inv_mean = 1.0 / mean_val
     
-    # Second pass: divide each element by mean
+    # Second pass: multiply each element by reciprocal of mean
     for col_off in tl.range(0, N, BLOCK_N):
-        col_idx = col_off + tl.arange(0, BLOCK_N)
+        col_idx = tl.max_contiguous(
+            tl.multiple_of(col_off + tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
+        )
         mask = col_idx < N
-        x = tl.load(x_ptr + row_start + col_idx, mask=mask, other=0.0).to(tl.float32)
-        out = x / mean_val
-        tl.store(out_ptr + row_start + col_idx, out.to(tl.float32), mask=mask)
-
+        # Pass 2: normal cache — keep in L2 for reuse
+        x = tl.load(x_ptr + row_start + col_idx, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
+        out = x * inv_mean
+        # Pass 2 store: streaming output — evict immediately since written once
+        tl.store(out_ptr + row_start + col_idx, out.to(tl.float32), mask=mask, eviction_policy='evict_first')
 
 def kernel_function(x):
     M, N = x.shape
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

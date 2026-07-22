# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_convolution_mean_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 509980 ns — **Best latency**: 509730 ns (**1.0005×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`l2_compression_via_streaming_store_cache_modifier`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_convolution_mean_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_013/kernel_0_triton_per_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131072,1,1), Waves/SM: 82.75
- **Registers/thread**: 40
- **Duration**: 509980 ns
- **SM Throughput**: 23.47 % | **Memory**: 88.41 % | **DRAM**: 88.41 %
- **Achieved Occupancy**: 72.87 %
- **IPC Active**: 0.93 inst/cycle | **No Eligible**: 76.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_xblock_tuning | 1.000× | 1.003× | ✓ |  |
| D | autotune_xblock_tile_size_and_num_warps | 1.645× | 0.193× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 509980 ns | 508580 ns | -1400 ns (-0.3%) | 2640000 ns | +2130020 ns (+417.7%) |
| SM Throughput (%) | 23.47 % | 25.83 % | +2.36 pp | 39.81 % | +16.34 pp |
| Memory Throughput (%) | 88.41 % | 88.67 % | +0.26 pp | 63.33 % | -25.08 pp |
| DRAM Throughput (%) | 88.41 % | 88.67 % | +0.26 pp | 17.08 % | -71.33 pp |
| Achieved Occupancy (%) | 72.87 % | 72.98 % | +0.11 pp | 24.78 % | -48.09 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 86.00 register/thread | +46 |
| Waves/SM | 82.75 | 82.75 | +0.00 | 992.97 | +910.22 |
| No Eligible (%) | 76.62 % | 73.87 % | -2.75 pp | 57.78 % | -18.84 pp |
| IPC Active | 0.93 inst/cycle | 1.06 inst/cycle | +0.13 | 1.68 inst/cycle | +0.75 |


### Key Learnings

#### ❌ Plan D: `autotune_xblock_tile_size_and_num_warps` (0.193×)

Adding @triton.autotune with large XBLOCK configs (512, 1024) caused the autotuner to select a config that increased registers per thread from 40 to 86, collapsing theoretical occupancy from 75% to 25% (Block_Limit_Registers dropped from 12 to 1) and increasing waves_per_sm from 83 to 993. The executed instruction count grew 9x (92M to 829M) because the larger block size selected by the autotuner processed far more work per CTA without proportional cache benefit, and the occupancy collapse prevented latency hiding. The structural anti-pattern is applying autotune with large XBLOCK configs to a kernel that is already register-heavy: larger tiles increase register file pressure, which kills occupancy and causes a severe regression even if DRAM throughput nominally improves.

> **Applicable when**: Kernel is DRAM-bound with high barrier stalls and low L1/L2 hit rates, and the current block size is small relative to the problem.

> **Avoid when**: The kernel already uses 40+ registers per thread. Adding larger XBLOCK configs to autotune will increase register pressure further, collapsing occupancy (Block_Limit_Registers → 1) and causing severe regression. Do not apply when registers_per_thread > ~35 and theoretical_occupancy is already limited by registers.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_barrier_stall_overhead_via_smaller_warp_count | 1.950× | 2.045× | ✓ |  |
| B | cache_eviction_policy_tuning_for_l2_pressure | 1.001× | 1.002× | ✓ |  |
| C | vectorized_loads_with_register_safe_small_xblock | 3.740× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_barrier_st... | Δ | cache_eviction_po... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 509980 ns | 249410 ns | -260570 ns (-51.1%) | 508800 ns | -1180 ns (-0.2%) | 508930 ns | -1050 ns (-0.2%) |
| SM Throughput (%) | 23.47 % | 30.83 % | +7.36 pp | 29.61 % | +6.14 pp | 25.89 % | +2.42 pp |
| Memory Throughput (%) | 88.41 % | 88.86 % | +0.45 pp | 88.62 % | +0.21 pp | 88.60 % | +0.19 pp |
| DRAM Throughput (%) | 88.41 % | 88.86 % | +0.45 pp | 88.62 % | +0.21 pp | 88.60 % | +0.19 pp |
| Achieved Occupancy (%) | 72.87 % | 70.13 % | -2.74 pp | 72.93 % | +0.06 pp | 72.95 % | +0.08 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 40.00 register/thread | +0 | 40.00 register/thread | +0 |
| Waves/SM | 82.75 | 41.37 | -41.38 | 82.75 | +0.00 | 82.75 | +0.00 |
| No Eligible (%) | 76.62 % | 69.52 % | -7.10 pp | 70.65 % | -5.97 pp | 73.98 % | -2.64 pp |
| IPC Active | 0.93 inst/cycle | 1.21 inst/cycle | +0.28 | 1.18 inst/cycle | +0.25 | 1.05 inst/cycle | +0.12 |


### Key Learnings

#### ✅ Plan A: `reduce_barrier_stall_overhead_via_smaller_warp_count` (2.045×)

When a reduction kernel shows high CTA barrier stall fraction (>40% of warp cycles) and is already DRAM-bound, reducing the number of warps per block (e.g., block_size 128→64, num_warps 4→2) cuts the number of warps that must synchronize at each tl.sum barrier roughly in half. This directly reduces Warp_Cycles_Per_Issued_Instruction (44.2→36.9 cycles) and increases Eligible_Warps_Per_Scheduler (0.44→0.62), halving total instruction count and kernel duration without changing register pressure or DRAM throughput. The strategy is effective when there is sufficient grid parallelism (high waves_per_SM) to absorb the smaller block size.

> **Applicable when**: Triton reduction kernel where: (1) CTA barrier stall fraction >30% of warp cycles, (2) kernel is DRAM-bound (DRAM throughput >80%), (3) register pressure already limits occupancy (registers_per_thread >35), (4) waves_per_SM is high (>40) so smaller blocks still saturate the GPU, and (5) the current block_size is ≥128 (num_warps ≥4).

> **Avoid when**: Waves_per_SM is already low (<10) — reducing block size will under-utilize the GPU. Also do not apply when the kernel is compute-bound rather than memory/barrier-bound, or when XBLOCK reduction would break alignment requirements for vectorized loads. Do not combine with large XBLOCK configs (≥512) in the same autotune sweep when registers_per_thread >35, as those configs cause occupancy collapse.

---

## Iteration 3

**Best so far**: 1.0021× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_register_accumulation | 0.985× | 0.997× | ✓ |  |
| B | evict_first_on_non_reused_input_to_depressurize_l2 | 3.783× | 1.007× | ✓ | **✓ best** |
| C | vectorized_loads_with_small_warp_barrier_reduction | 1.954× | 1.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | evict_first_on_no... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 509980 ns | 511710 ns | +1730 ns (+0.3%) | 506590 ns | -3390 ns (-0.7%) | 255620 ns | -254360 ns (-49.9%) |
| SM Throughput (%) | 23.47 % | 40.52 % | +17.05 pp | 28.90 % | +5.43 pp | 29.98 % | +6.51 pp |
| Memory Throughput (%) | 88.41 % | 87.91 % | -0.50 pp | 88.21 % | -0.20 pp | 88.22 % | -0.19 pp |
| DRAM Throughput (%) | 88.41 % | 87.91 % | -0.50 pp | 88.21 % | -0.20 pp | 88.22 % | -0.19 pp |
| Achieved Occupancy (%) | 72.87 % | 94.05 % | +21.18 pp | 72.90 % | +0.03 pp | 71.16 % | -1.71 pp |
| Registers/Thread | 40.00 register/thread | 32.00 register/thread | -8 | 40.00 register/thread | +0 | 40.00 register/thread | +0 |
| Waves/SM | 82.75 | 62.06 | -20.69 | 82.75 | +0.00 | 41.37 | -41.38 |
| No Eligible (%) | 76.62 % | 59.03 % | -17.59 pp | 70.92 % | -5.70 pp | 69.53 % | -7.09 pp |
| IPC Active | 0.93 inst/cycle | 1.63 inst/cycle | +0.70 | 1.18 inst/cycle | +0.25 | 1.19 inst/cycle | +0.26 |

---

## Iteration 4

**Best so far**: 1.0067× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_reduction_axis_reorder | 1.952× | 1.991× | ✓ |  |
| B | l2_compression_via_streaming_store_cache_modifier | 3.786× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | l2_compression_vi... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 509980 ns | 256160 ns | -253820 ns (-49.8%) | 509730 ns | -250 ns (-0.0%) |
| SM Throughput (%) | 23.47 % | 29.13 % | +5.66 pp | 28.89 % | +5.42 pp |
| Memory Throughput (%) | 88.41 % | 88.06 % | -0.35 pp | 88.21 % | -0.20 pp |
| DRAM Throughput (%) | 88.41 % | 88.06 % | -0.35 pp | 88.21 % | -0.20 pp |
| Achieved Occupancy (%) | 72.87 % | 71.52 % | -1.35 pp | 72.94 % | +0.07 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 40.00 register/thread | +0 |
| Waves/SM | 82.75 | 41.37 | -41.38 | 82.75 | +0.00 |
| No Eligible (%) | 76.62 % | 70.33 % | -6.29 pp | 71.28 % | -5.34 pp |
| IPC Active | 0.93 inst/cycle | 1.20 inst/cycle | +0.27 | 1.15 inst/cycle | +0.22 |


### Key Learnings

#### ✅ Plan A: `eliminate_shared_memory_bank_conflicts_via_reduction_axis_reorder` (1.982×)

This iteration applied a refined autotune config sweep (removing large XBLOCK configs ≥256, adding XBLOCK=32/nw=1 and updating the autotune key) along with a tile axis swap ([XBLOCK,R0_BLOCK]→[R0_BLOCK,XBLOCK] with tl.sum axis 1→0). The profiling data shows essentially no change in any metric: Duration increased slightly (256.16→257.31 µs), DRAM throughput was flat (88.06→87.25%), shared memory bank conflicts were unchanged (4.7-way, 15.02→14.55% wavefronts), and warp cycles per instruction were flat (37.81→38.38). The reported 1.982x speedup is inconsistent with the profiling data showing no improvement in the NCU-measured kernel, suggesting the speedup is measured at a higher level (e.g., wall-clock including autotuning overhead amortization or a different kernel invocation) rather than from this kernel's per-invocation latency. The structural changes (axis swap, autotune key addition) did not move any profiling needle.

> **Applicable when**: SharedMemoryConflicts NCU rule is triggered with >4-way bank conflicts and the autotune sweep contains large XBLOCK configs (≥256) that risk register pressure collapse.

> **Avoid when**: When the kernel is already at a near-optimal autotune config (XBLOCK=64, num_warps=2) and the profiling metrics are flat — further autotune config shuffling and axis swaps do not move the needle. Do not apply when the kernel duration is already at the DRAM bandwidth ceiling (DRAM throughput >85%) and shared memory bank conflicts are structural to the warp-level reduction tree.

---

## Iteration 5

**Best so far**: 1.0067× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_bias_outside_reduction_to_enable_fma | 1.952× | 1.996× | ✓ |  |
| B | cta_grid_swizzle_for_l2_spatial_reuse | 1.953× | 1.985× | ✓ |  |
| D | memory_coalescing_via_block_size_reduction_and_autotune_sweep | 3.762× | 1.982× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_bias_outsid... | Δ | cta_grid_swizzle_... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 509980 ns | 255550 ns | -254430 ns (-49.9%) | 256960 ns | -253020 ns (-49.6%) | 257310 ns | -252670 ns (-49.5%) |
| SM Throughput (%) | 23.47 % | 25.25 % | +1.78 pp | 29.75 % | +6.28 pp | 28.95 % | +5.48 pp |
| Memory Throughput (%) | 88.41 % | 88.24 % | -0.17 pp | 87.76 % | -0.65 pp | 87.25 % | -1.16 pp |
| DRAM Throughput (%) | 88.41 % | 88.24 % | -0.17 pp | 87.76 % | -0.65 pp | 87.25 % | -1.16 pp |
| Achieved Occupancy (%) | 72.87 % | 94.88 % | +22.01 pp | 70.99 % | -1.88 pp | 71.12 % | -1.75 pp |
| Registers/Thread | 40.00 register/thread | 32.00 register/thread | -8 | 40.00 register/thread | +0 | 40.00 register/thread | +0 |
| Waves/SM | 82.75 | 31.03 | -51.72 | 41.37 | -41.38 | 41.37 | -41.38 |
| No Eligible (%) | 76.62 % | 73.97 % | -2.65 pp | 70.17 % | -6.45 pp | 70.70 % | -5.92 pp |
| IPC Active | 0.93 inst/cycle | 1.04 inst/cycle | +0.11 | 1.18 inst/cycle | +0.25 | 1.16 inst/cycle | +0.23 |


### Key Learnings

#### ✅ Plan A: `hoist_bias_outside_reduction_to_enable_fma` (1.996×)

When a reduction kernel fuses a scalar bias addition inside the reduction (e.g., tl.sum(data + bias, axis)), the bias is broadcast and added to every element before summing, generating N non-fused FP32 adds per output element (where N is the reduction length). Hoisting the bias outside — computing tl.sum(data, axis) + N*bias — replaces N adds with 1 multiply and 1 add, reducing non-fused FP32 instructions (16.5M → 8.9M here) and register pressure (40 → 32 registers/thread). The register pressure reduction was the dominant effect: theoretical occupancy jumped from 75% to 100% (Block_Limit_Registers 24 → 32), and achieved occupancy rose from 71.5% to 94.9%, enabling nearly 2x speedup despite the kernel remaining DRAM-bound.

> **Applicable when**: FPInstructions NCU rule is triggered with a large count of non-fused FP32 instructions and zero fused instructions; a scalar value (bias, offset) that does not depend on the reduction axis is being added element-wise inside a tl.sum call; registers_per_thread is elevated (>35) and theoretical occupancy is limited by register pressure (Block_Limit_Registers < Block_Limit_Warps); the mathematical identity sum(x_i + b) = sum(x_i) + N*b holds exactly (b is constant across the reduction axis).

> **Avoid when**: The value being added inside the reduction does depend on the reduction index (i.e., it is not a broadcast scalar across the reduction axis) — the identity does not hold. Also do not apply when the reduction length N is not a compile-time constant (constexpr), as the multiply N*bias would require a runtime division to invert. Do not apply when the kernel is already at 100% theoretical occupancy and register pressure is not the bottleneck.

#### ❌ Plan B: `cta_grid_swizzle_for_l2_spatial_reuse` (1.985×)

Reordering CTA program IDs to group CTAs sharing the same reduction base address (x4 batch+channel slice) did not improve L2 hit rate: it remained essentially flat at 4.3% before vs 4.27% after, and kernel duration was unchanged (256.16 µs vs 256.96 µs). The reported 1.98x speedup is inconsistent with the profiling data showing no metric improvement, indicating the speedup measurement is noise or attributable to external factors. The structural reason for failure is that with 41.37 waves/SM and 132 SMs, the GPU scheduler interleaves CTAs from many x4 groups simultaneously regardless of program ID ordering, preventing the intended L2 temporal reuse of the 2MB per-x4 reduction data.

> **Applicable when**: Never: this transformation did not improve L2 hit rate or kernel duration in this kernel class.

> **Avoid when**: When the kernel has high waves_per_SM (>40) and the GPU has many SMs (>100): the hardware scheduler will interleave CTAs from many groups simultaneously regardless of program ID ordering, negating any intended L2 temporal reuse. CTA grid swizzling for L2 reuse is ineffective when the working set per group (2MB) is small relative to L2 (50MB) but the number of concurrent active groups (132 SMs × multiple waves) exceeds L2 capacity to hold all groups simultaneously. Do not apply when L2 hit rate is already near its floor due to structural access pattern issues (strided loads with 64KB stride).

---

## Cumulative Progress

```
Baseline:  509980 ns  (1.000×)
Iter 2     508930 ns  (1.002×)  — vectorized_loads_with_register_safe_small_xblock
Iter 3     506590 ns  (1.007×)  — evict_first_on_non_reused_input_to_depressurize_l2

Final best: 509730 ns  (1.0005×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,34 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_per_fused_convolution_mean_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 16777216
     r0_numel = 32
     R0_BLOCK: tl.constexpr = 32
-    rnumel = r0_numel
-    RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
-    r0_index = tl.arange(0, R0_BLOCK)[None, :]
-    r0_offset = 0
-    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
-    roffset = r0_offset
-    rindex = r0_index
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    r0_index = tl.arange(0, R0_BLOCK)[:, None]
+    xmask = xindex < xnumel
     r0_3 = r0_index
     x0 = (xindex % 16384)
     x4 = xindex // 16384
     x1 = ((xindex // 16384) % 64)
     x5 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 16384*r0_3 + 524288*x4), None)
-    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
-    tmp2 = tmp0 + tmp1
-    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
-    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
-    tl.store(out_ptr0 + (x5), tmp5, None)
-
-
+    x0_vec = tl.max_contiguous(tl.multiple_of(x0, [1, 16]), [1, 16])
+    tmp0 = tl.load(in_ptr0 + (x0_vec + 16384*r0_3 + 524288*x4), mask=xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (x1), mask=xmask, eviction_policy='evict_last')
+    tmp5 = tl.sum(tmp0 + tmp1, 0)[None, :].to(tl.float32)
+    tl.store(out_ptr0 + (x5), tmp5, mask=xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +43,7 @@

     xnumel = 16777216
     r0_numel = 32
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_convolution_mean_0[grid](arg_0, arg_1, arg_2, 16777216, 32, XBLOCK=128)
+    triton_per_fused_convolution_mean_0[grid](arg_0, arg_1, arg_2, 16777216, 32)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_hardswish_native_group_norm_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 412700 ns — **Best latency**: 445730 ns (**0.9259×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_hardswish_native_group_norm_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_027/kernel_0_triton_red_fused_convolution_h/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 3.88
- **Registers/thread**: 64
- **Duration**: 412700 ns
- **SM Throughput**: 70.31 % | **Memory**: 35.45 % | **DRAM**: 35.45 %
- **Achieved Occupancy**: 45.12 %
- **IPC Active**: 2.85 inst/cycle | **No Eligible**: 28.66 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_for_coalescing_and_wave_efficiency | 0.044× | 0.004× | ✓ |  |
| B | add_autotune_increase_xblock_reduce_register_pressure | 0.924× | 0.927× | ✓ |  |
| C | increase_xblock_for_coalesced_stores_and_vectorized_loads | 0.923× | 0.270× | ✓ |  |
| D | add_autotune_increase_xblock_for_coalescing_and_occupancy | 0.923× | 0.927× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | add_autotune_incr... | Δ | increase_xblock_f... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 412700 ns | 118890000 ns | +118477300 ns (+28707.9%) | 445280 ns | +32580 ns (+7.9%) | 1530000 ns | +1117300 ns (+270.7%) | 445060 ns | +32360 ns (+7.8%) |
| SM Throughput (%) | 70.31 % | 13.38 % | -56.93 pp | 72.08 % | +1.77 pp | 74.59 % | +4.28 pp | 71.93 % | +1.62 pp |
| Memory Throughput (%) | 35.45 % | 55.83 % | +20.38 pp | 32.86 % | -2.59 pp | 27.05 % | -8.40 pp | 32.87 % | -2.58 pp |
| DRAM Throughput (%) | 35.45 % | 55.83 % | +20.38 pp | 32.86 % | -2.59 pp | 9.54 % | -25.91 pp | 32.87 % | -2.58 pp |
| Achieved Occupancy (%) | 45.12 % | 36.35 % | -8.77 pp | 50.06 % | +4.94 pp | 22.99 % | -22.13 pp | 50.06 % | +4.94 pp |
| Registers/Thread | 64.00 register/thread | 32.00 register/thread | -32 | 55.00 register/thread | -9 | 121.00 register/thread | +57 | 55.00 register/thread | -9 |
| Waves/SM | 3.88 | 5.17 | +1.29 | 3.45 | -0.43 | 3.88 | +0.00 | 3.45 | -0.43 |
| No Eligible (%) | 28.66 % | 92.81 % | +64.15 pp | 25.42 % | -3.24 pp | 22.95 % | -5.71 pp | 25.50 % | -3.16 pp |
| IPC Active | 2.85 inst/cycle | 0.29 inst/cycle | -2.56 | 2.98 inst/cycle | +0.13 | 3.06 inst/cycle | +0.21 | 2.98 inst/cycle | +0.13 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_for_coalescing_and_wave_efficiency` (0.004×)

Adding @triton.autotune with larger XBLOCK configs to a reduction kernel caused a ~286x slowdown. The autotune selected a config that allocated 32 KB of dynamic shared memory per block (up from 2 KB), reducing Block Limit Shared Mem from 21 to 6 and theoretical occupancy from 50% to 37.5%. More critically, executed instructions exploded from 223M to 6.58B and Warp Cycles Per Issued Instruction jumped from 10 to 80, indicating the selected config caused severe L1TEX scoreboard stalls (91.9% of stall cycles) due to strided local memory spills from the large shared memory allocation. When a reduction kernel has a large inner reduction dimension (r0_numel ~44K) and XBLOCK is increased, the per-block working set grows proportionally, causing shared memory pressure and register spills that dominate runtime.

> **Applicable when**: Never apply this specific pattern as implemented. The grid was not updated to use META['XBLOCK'], so the autotune ran with a fixed grid of 4096 blocks regardless of XBLOCK, causing each block to process only 1/XBLOCK of its intended work while still allocating XBLOCK*R0_BLOCK shared memory.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK to reduction kernels without simultaneously updating the grid lambda to `lambda META: (cdiv(xnumel, META['XBLOCK']),)`. Do not increase XBLOCK in reduction kernels where r0_numel is large (>10K) without verifying that XBLOCK*R0_BLOCK*dtype_bytes fits within shared memory budget. Do not apply when the kernel already has high compute utilization (>60% SM throughput) — the original kernel was compute-bound at 70%, not memory-bound.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_num_warps_and_larger_r0_block_tuning | 1.095× | 0.926× | ✓ | **✓ best** |
| B | reduce_partial_wave_tail_via_larger_r0_block_configs | 0.994× | 0.927× | ✓ |  |
| C | reduce_register_pressure_via_fma_fusion_and_grid_wave_alignment | 0.966× | 0.887× | ✓ |  |
| D | fix_autotune_grid_and_restrict_configs_to_eliminate_smem_regression | 0.924× | 0.847× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_partial_wa... | Δ | reduce_register_p... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 412700 ns | 445730 ns | +33030 ns (+8.0%) | 445180 ns | +32480 ns (+7.9%) | 465280 ns | +52580 ns (+12.7%) | 487330 ns | +74630 ns (+18.1%) |
| SM Throughput (%) | 70.31 % | 72.06 % | +1.75 pp | 72.14 % | +1.83 pp | 67.76 % | -2.55 pp | 64.37 % | -5.94 pp |
| Memory Throughput (%) | 35.45 % | 32.82 % | -2.63 pp | 32.88 % | -2.57 pp | 31.46 % | -3.99 pp | 30.02 % | -5.43 pp |
| DRAM Throughput (%) | 35.45 % | 32.82 % | -2.63 pp | 32.88 % | -2.57 pp | 31.46 % | -3.99 pp | 30.02 % | -5.43 pp |
| Achieved Occupancy (%) | 45.12 % | 50.06 % | +4.94 pp | 50.03 % | +4.91 pp | 21.39 % | -23.73 pp | 24.51 % | -20.61 pp |
| Registers/Thread | 64.00 register/thread | 55.00 register/thread | -9 | 55.00 register/thread | -9 | 128.00 register/thread | +64 | 128.00 register/thread | +64 |
| Waves/SM | 3.88 | 3.45 | -0.43 | 3.45 | -0.43 | 1.94 | -1.94 | 15.52 | +11.64 |
| No Eligible (%) | 28.66 % | 25.39 % | -3.27 pp | 25.43 % | -3.23 pp | 29.16 % | +0.50 pp | 33.64 % | +4.98 pp |
| IPC Active | 2.85 inst/cycle | 2.98 inst/cycle | +0.13 | 2.98 inst/cycle | +0.13 | 2.79 inst/cycle | -0.06 | 2.65 inst/cycle | -0.20 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_hardswish_fp32_instructions_via_fma_restructuring | 0.984× | 0.926× | ✓ |  |
| B | fuse_nonfused_fp32_instructions_via_fma_in_hardswish_inner_loop | 0.968× | 0.932× | ✓ |  |
| D | fma_fusion_hardswish_with_safe_register_budget_autotune_configs | 0.958× | 0.925× | ✓ |  |
| E | add_triton_autotune | 0.923× | 0.928× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_hardswish_fp... | Δ | fuse_nonfused_fp3... | Δ | fma_fusion_hardsw... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 412700 ns | 445570 ns | +32870 ns (+8.0%) | 442820 ns | +30120 ns (+7.3%) | 446300 ns | +33600 ns (+8.1%) | 444510 ns | +31810 ns (+7.7%) |
| SM Throughput (%) | 70.31 % | 72.13 % | +1.82 pp | 72.60 % | +2.29 pp | 71.83 % | +1.52 pp | 72.15 % | +1.84 pp |
| Memory Throughput (%) | 35.45 % | 32.83 % | -2.62 pp | 33.05 % | -2.40 pp | 32.80 % | -2.65 pp | 32.92 % | -2.53 pp |
| DRAM Throughput (%) | 35.45 % | 32.83 % | -2.62 pp | 33.05 % | -2.40 pp | 32.80 % | -2.65 pp | 32.92 % | -2.53 pp |
| Achieved Occupancy (%) | 45.12 % | 50.08 % | +4.96 pp | 50.08 % | +4.96 pp | 50.03 % | +4.91 pp | 50.01 % | +4.89 pp |
| Registers/Thread | 64.00 register/thread | 55.00 register/thread | -9 | 55.00 register/thread | -9 | 55.00 register/thread | -9 | 55.00 register/thread | -9 |
| Waves/SM | 3.88 | 3.45 | -0.43 | 3.45 | -0.43 | 3.45 | -0.43 | 3.45 | -0.43 |
| No Eligible (%) | 28.66 % | 25.45 % | -3.21 pp | 24.86 % | -3.80 pp | 25.65 % | -3.01 pp | 25.47 % | -3.19 pp |
| IPC Active | 2.85 inst/cycle | 2.98 inst/cycle | +0.13 | 3.01 inst/cycle | +0.16 | 2.98 inst/cycle | +0.13 | 2.98 inst/cycle | +0.13 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | set_evict_first_on_output_stores_to_reduce_l2_pollution_from_strided_writes | 0.969× | 0.932× | ✓ |  |
| B | reduce_nonfused_fp32_instructions_via_explicit_libdevice_fma_in_hardswish | 0.971× | 0.926× | ✓ |  |
| C | reduce_nonfused_fp32_instructions_via_hardswish_algebraic_restructuring | 1.002× | 0.966× | ✓ |  |
| D | coalesce_output_stores_by_reordering_xindex_to_contiguous_write_pattern | 0.921× | 0.924× | ✓ |  |


### Metric Comparison

| Metric | Baseline | set_evict_first_o... | Δ | reduce_nonfused_f... | Δ | reduce_nonfused_f... | Δ | coalesce_output_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 412700 ns | 442780 ns | +30080 ns (+7.3%) | 445790 ns | +33090 ns (+8.0%) | 427230 ns | +14530 ns (+3.5%) | 446820 ns | +34120 ns (+8.3%) |
| SM Throughput (%) | 70.31 % | 72.81 % | +2.50 pp | 72.38 % | +2.07 pp | 72.38 % | +2.07 pp | 71.91 % | +1.60 pp |
| Memory Throughput (%) | 35.45 % | 33.06 % | -2.39 pp | 32.82 % | -2.63 pp | 34.24 % | -1.21 pp | 32.75 % | -2.70 pp |
| DRAM Throughput (%) | 35.45 % | 33.06 % | -2.39 pp | 32.82 % | -2.63 pp | 34.24 % | -1.21 pp | 32.75 % | -2.70 pp |
| Achieved Occupancy (%) | 45.12 % | 50.07 % | +4.95 pp | 50.02 % | +4.90 pp | 50.08 % | +4.96 pp | 50.01 % | +4.89 pp |
| Registers/Thread | 64.00 register/thread | 55.00 register/thread | -9 | 56.00 register/thread | -8 | 55.00 register/thread | -9 | 55.00 register/thread | -9 |
| Waves/SM | 3.88 | 3.45 | -0.43 | 3.45 | -0.43 | 3.45 | -0.43 | 3.45 | -0.43 |
| No Eligible (%) | 28.66 % | 24.87 % | -3.79 pp | 25.17 % | -3.49 pp | 25.42 % | -3.24 pp | 25.41 % | -3.25 pp |
| IPC Active | 2.85 inst/cycle | 3.01 inst/cycle | +0.16 | 2.99 inst/cycle | +0.14 | 2.98 inst/cycle | +0.13 | 2.98 inst/cycle | +0.13 |

---

## Cumulative Progress

```
Baseline:  412700 ns  (1.000×)

Final best: 445730 ns  (0.9259×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 256},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
+        # NEW: num_warps=2 (64 threads/block) to reduce register file per block
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512},  num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 256},  num_warps=2,  num_stages=1),
+        # NEW: R0_BLOCK=2048 to halve loop iterations (~43 -> ~21)
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_hardswish_native_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 4096
@@ -13,7 +37,7 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_base = tl.arange(0, R0_BLOCK)[None, :]
     rbase = r0_base
     x4 = xindex
@@ -28,18 +52,13 @@

         rindex = r0_index
         r0_5 = r0_index
         r0_3 = r0_index // 10933
-        tmp0 = tl.load(in_ptr0 + (r0_5 + 43732*x4), r0_mask, eviction_policy='evict_first', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (r0_3 + 4*x0), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_5 + 43732*x4), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (r0_3 + 4*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 + tmp1
-        tmp3 = tl.full([1, 1], 3.0, tl.float32)
-        tmp4 = tmp2 + tmp3
-        tmp5 = tl.full([1, 1], 0.0, tl.float32)
-        tmp6 = tl.maximum(tmp4, tmp5)
-        tmp7 = tl.full([1, 1], 6.0, tl.float32)
-        tmp8 = tl.minimum(tmp6, tmp7)
-        tmp9 = tmp2 * tmp8
-        tmp10 = tl.full([1, 1], 0.16666666666666666, tl.float32)
-        tmp11 = tmp9 * tmp10
+        tmp4 = libdevice.fma(tmp2, 1.0, 3.0)          # FMA: 1*tmp2 + 3.0 (replaces tl.full+ADD)
+        tmp6 = tl.maximum(tmp4, 0.0)                   # Python float literal, no tl.full
+        tmp8 = tl.minimum(tmp6, 6.0)                   # Python float literal, no tl.full
+        tmp11 = libdevice.fma(tmp2 * tmp8, 0.16666666666666666, 0.0)  # FMA: (tmp2*tmp8)*(1/6)+0
         tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
         tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
             tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
@@ -51,10 +70,8 @@

     tmp13 = tmp14[:, None]
     tmp17 = tmp15[:, None]
     tmp18 = tmp16[:, None]
-    tl.store(out_ptr0 + (x4), tmp13, None)
-    tl.store(out_ptr1 + (x4), tmp17, None)
-
-
+    tl.store(out_ptr0 + (x4), tmp13, mask=xmask)
+    tl.store(out_ptr1 + (x4), tmp17, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -64,7 +81,7 @@

     xnumel = 4096
     r0_numel = 43732
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_convolution_hardswish_native_group_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 4096, 43732, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused_convolution_hardswish_native_group_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 4096, 43732)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

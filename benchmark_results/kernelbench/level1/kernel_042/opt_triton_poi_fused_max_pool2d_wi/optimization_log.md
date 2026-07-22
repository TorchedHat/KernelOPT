# Kernel Optimization Log

## Summary

- **Kernel**: `max_pool2d_kernel_general` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7110000 ns — **Best latency**: 13130000 ns (**0.5415×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `max_pool2d_kernel_general`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_042/synthesis/kernel_0_triton_poi_fused_max_pool2d_with_indices.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (522242,1,1), Waves/SM: 1318.79
- **Registers/thread**: 144
- **Duration**: 7110000 ns
- **SM Throughput**: 83.37 % | **Memory**: 22.70 % | **DRAM**: 12.23 %
- **Achieved Occupancy**: 18.03 %
- **IPC Active**: 2.94 inst/cycle | **No Eligible**: 26.50 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_to_improve_occupancy | 1.019× | 1.000× | ✓ |  |
| B | reduce_register_pressure_via_dead_code_elimination_and_loop_restructuring | 1.019× | 0.499× | ✓ |  |
| D | reduce_register_pressure_via_partial_loop_unroll_and_autotune_expansion | 1.020× | 0.493× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7110000 ns | 7110000 ns | +0 ns (+0.0%) | 14260000 ns | +7150000 ns (+100.6%) | 14420000 ns | +7310000 ns (+102.8%) |
| SM Throughput (%) | 83.37 % | 83.37 % | +0.00 pp | 47.56 % | -35.81 pp | 47.02 % | -36.35 pp |
| Memory Throughput (%) | 22.70 % | 22.70 % | +0.00 pp | 14.93 % | -7.77 pp | 14.79 % | -7.91 pp |
| DRAM Throughput (%) | 12.23 % | 12.23 % | +0.00 pp | 6.12 % | -6.11 pp | 6.05 % | -6.18 pp |
| Achieved Occupancy (%) | 18.03 % | 18.03 % | +0.00 pp | 12.19 % | -5.84 pp | 12.26 % | -5.77 pp |
| Registers/Thread | 144.00 register/thread | 144.00 register/thread | +0 | 255.00 register/thread | +111 | 255.00 register/thread | +111 |
| Waves/SM | 1318.79 | 1318.79 | +0.00 | 989.09 | -329.70 | 989.09 | -329.70 |
| No Eligible (%) | 26.50 % | 26.51 % | +0.01 pp | 58.10 % | +31.60 pp | 58.52 % | +32.02 pp |
| IPC Active | 2.94 inst/cycle | 2.94 inst/cycle | +0.00 | 1.68 inst/cycle | -1.26 | 1.66 inst/cycle | -1.28 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_integer_division_via_2d_grid_decomposition | 1.019× | 0.999× | ✓ |  |
| C | restructure_index_decomposition_to_reduce_live_registers_and_spilling | 1.021× | 0.493× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_integer... | Δ | restructure_index... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7110000 ns | 7120000 ns | +10000 ns (+0.1%) | 14410000 ns | +7300000 ns (+102.7%) |
| SM Throughput (%) | 83.37 % | 83.37 % | +0.00 pp | 47.07 % | -36.30 pp |
| Memory Throughput (%) | 22.70 % | 22.70 % | +0.00 pp | 14.81 % | -7.89 pp |
| DRAM Throughput (%) | 12.23 % | 12.21 % | -0.02 pp | 6.05 % | -6.18 pp |
| Achieved Occupancy (%) | 18.03 % | 18.03 % | +0.00 pp | 12.27 % | -5.76 pp |
| Registers/Thread | 144.00 register/thread | 144.00 register/thread | +0 | 255.00 register/thread | +111 |
| Waves/SM | 1318.79 | 1318.79 | +0.00 | 989.09 | -329.70 |
| No Eligible (%) | 26.50 % | 26.51 % | +0.01 pp | 58.56 % | +32.06 pp |
| IPC Active | 2.94 inst/cycle | 2.94 inst/cycle | +0.00 | 1.65 inst/cycle | -1.29 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_loop_invariant_base_pointer_to_reduce_alu_and_register_pressure | 1.014× | 1.000× | ✓ |  |
| B | hoist_loop_invariant_base_offset_to_reduce_live_registers_and_alu_ops | 1.016× | 1.000× | ✓ |  |
| C | eliminate_dead_duplicate_index_decomposition_to_free_registers | 1.021× | 0.541× | ✓ | **✓ best** |
| D | hoist_loop_invariant_base_offset_and_eliminate_dead_index_decomposition | 1.020× | 0.492× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_loop_invari... | Δ | hoist_loop_invari... | Δ | eliminate_dead_du... | Δ | hoist_loop_invari... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7110000 ns | 7110000 ns | +0 ns (+0.0%) | 7110000 ns | +0 ns (+0.0%) | 13130000 ns | +6020000 ns (+84.7%) | 14440000 ns | +7330000 ns (+103.1%) |
| SM Throughput (%) | 83.37 % | 83.37 % | +0.00 pp | 83.37 % | +0.00 pp | 51.62 % | -31.75 pp | 47.08 % | -36.29 pp |
| Memory Throughput (%) | 22.70 % | 22.70 % | +0.00 pp | 22.70 % | +0.00 pp | 15.87 % | -6.83 pp | 14.81 % | -7.89 pp |
| DRAM Throughput (%) | 12.23 % | 12.23 % | +0.00 pp | 12.23 % | +0.00 pp | 6.63 % | -5.60 pp | 6.04 % | -6.19 pp |
| Achieved Occupancy (%) | 18.03 % | 18.03 % | +0.00 pp | 18.03 % | +0.00 pp | 12.25 % | -5.78 pp | 12.26 % | -5.77 pp |
| Registers/Thread | 144.00 register/thread | 144.00 register/thread | +0 | 144.00 register/thread | +0 | 255.00 register/thread | +111 | 255.00 register/thread | +111 |
| Waves/SM | 1318.79 | 1318.79 | +0.00 | 1318.79 | +0.00 | 989.10 | -329.69 | 989.09 | -329.70 |
| No Eligible (%) | 26.50 % | 26.50 % | +0.00 pp | 26.50 % | +0.00 pp | 53.48 % | +26.98 pp | 58.63 % | +32.13 pp |
| IPC Active | 2.94 inst/cycle | 2.94 inst/cycle | +0.00 | 2.94 inst/cycle | +0.00 | 1.84 inst/cycle | -1.10 | 1.65 inst/cycle | -1.29 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restructure_grid_to_3d_for_coalesced_ow_access | 1.021× | 1.121× | ✓ |  |
| B | hoist_per_kh_row_offset_to_reduce_live_vectors_in_unrolled_kernel_window | 1.015× | 0.491× | ✓ |  |
| C | add_small_block_low_warp_autotune_configs_to_increase_sm_occupancy | 1.021× | 1.116× | ✓ |  |
| D | hoist_loop_invariant_nc_offset_in_general_kernel_and_add_small_block_autotune_configs | 1.021× | 1.107× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restructure_grid_... | Δ | hoist_per_kh_row_... | Δ | add_small_block_l... | Δ | hoist_loop_invari... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7110000 ns | 6340000 ns | -770000 ns (-10.8%) | 14480000 ns | +7370000 ns (+103.7%) | 6370000 ns | -740000 ns (-10.4%) | 6420000 ns | -690000 ns (-9.7%) |
| SM Throughput (%) | 83.37 % | 89.16 % | +5.79 pp | 47.09 % | -36.28 pp | 85.85 % | +2.48 pp | 87.47 % | +4.10 pp |
| Memory Throughput (%) | 22.70 % | 28.02 % | +5.32 pp | 14.81 % | -7.89 pp | 28.14 % | +5.44 pp | 27.10 % | +4.40 pp |
| DRAM Throughput (%) | 12.23 % | 13.70 % | +1.47 pp | 6.03 % | -6.20 pp | 13.65 % | +1.42 pp | 13.55 % | +1.32 pp |
| Achieved Occupancy (%) | 18.03 % | 35.64 % | +17.61 pp | 12.26 % | -5.77 pp | 93.12 % | +75.09 pp | 47.05 % | +29.02 pp |
| Registers/Thread | 144.00 register/thread | 79.00 register/thread | -65 | 255.00 register/thread | +111 | 32.00 register/thread | -112 | 58.00 register/thread | -86 |
| Waves/SM | 1318.79 | 1318.79 | +0.00 | 989.09 | -329.70 | 1978.19 | +659.40 | 1978.19 | +659.40 |
| No Eligible (%) | 26.50 % | 19.62 % | -6.88 pp | 58.60 % | +32.10 pp | 14.12 % | -12.38 pp | 17.92 % | -8.58 pp |
| IPC Active | 2.94 inst/cycle | 3.21 inst/cycle | +0.27 | 1.66 inst/cycle | -1.28 | 3.44 inst/cycle | +0.50 | 3.28 inst/cycle | +0.34 |


### Key Learnings

#### ❌ Plan B: `hoist_per_kh_row_offset_to_reduce_live_vectors_in_unrolled_kernel_window` (0.491×)

Hoisting both the loop-invariant nc_base scalar and the per-kh ih_row_offset vector outside the inner kw loop did not reduce register pressure in a fully-unrolled tl.static_range 4x4 kernel window: registers/thread remained at 255 (hardware cap), theoretical occupancy stayed at 12.5%, and duration increased from 7.11ms to 14.48ms (0.491x speedup). The profiling data shows the register count was unchanged from prior spilled states, confirming that the Triton/LLVM compiler's register allocator does not benefit from this hoisting when the loop is fully unrolled — all 16 unrolled copies still hold enough simultaneously live BLOCK_SIZE-element vectors (ih, valid_h, ih_row_offset, iw, valid_w, valid, val, max_val per copy) to saturate the register file. Critically, the diff also removed all num_stages=1 autotune configs, which is the confounding factor that caused the autotuner to select a register-spilling configuration in the first place — the baseline 144-register kernel relied on num_stages=1.

> **Applicable when**: Never — hoisting per-kh row offset (ih_row_offset) and nc_base out of a fully-unrolled tl.static_range double loop does not reduce register pressure when the loop body already has 7+ live BLOCK_SIZE-element vectors per unrolled copy.

> **Avoid when**: Any Triton kernel where the double loop is fully unrolled via tl.static_range and the loop body contains 6+ live BLOCK_SIZE-element vectors per iteration. The register pressure is dominated by the total number of simultaneously live vectors across all unrolled copies (KERNEL_SIZE^2 * live_vectors_per_iter), not by the 2-3 operations that can be hoisted. This change is insufficient to escape the 255-register hardware cap. Additionally, removing num_stages=1 autotune configs is a confirmed anti-pattern — the baseline 144-register kernel requires num_stages=1 to avoid register spilling.

#### ✅ Plan C: `add_small_block_low_warp_autotune_configs_to_increase_sm_occupancy` (1.116×)

When a Triton kernel is register-bound (Block Limit Registers is the binding occupancy constraint, e.g. 3 blocks/SM at 144 regs/thread with BLOCK_SIZE=128), adding smaller BLOCK_SIZE configs (e.g. BLOCK_SIZE=64 with num_warps=2, BLOCK_SIZE=32 with num_warps=1) to the autotune search space allows the autotuner to select a configuration where fewer threads per block means fewer total registers per block, enabling more blocks to co-reside per SM. In this case, the autotuner selected BLOCK_SIZE=64 which reduced registers/thread from 144 to 32, boosting theoretical occupancy from 18.75% to 100% and achieved occupancy from 18% to 93%, cutting scheduler No-Eligible stalls from 26.5% to 14.1% and reducing kernel duration from 7.11ms to 6.37ms (1.12x speedup). This is the correct direction when the kernel body cannot be restructured to reduce register pressure: shrink the block size so the register budget per block fits more blocks per SM.

> **Applicable when**: Triton kernel where: (1) TheoreticalOccupancy rule fires with Block Limit Registers as the binding constraint, (2) registers/thread is high (>64) due to loop unrolling via tl.static_range, (3) the kernel body cannot be restructured to reduce register pressure without causing spilling, and (4) the workload is compute-bound (ALU pipeline >80%) with significant scheduler No-Eligible stalls (>20%). The key insight is that smaller BLOCK_SIZE reduces total registers per block, allowing more blocks per SM.

> **Avoid when**: Do NOT apply when: (1) the kernel is already at or near 100% occupancy (no register bottleneck), (2) the kernel has a register spill bug (registers=255) — fix the spill first before tuning block size, (3) BLOCK_SIZE must equal num_warps*32 exactly in Triton for full warp utilization, (4) the kernel has a minimum tile size requirement that prevents smaller blocks from being correct. Also note: adding LARGER block size configs when registers are already high (>128/thread) causes the autotuner to select worse configs (confirmed by exp_iter3).

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_with_very_small_block_configs_to_escape_register_bound_occupancy | 1.020× | 0.347× | ✓ |  |
| E | add_triton_autotune | 1.021× | 0.347× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_w... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7110000 ns | 20460000 ns | +13350000 ns (+187.8%) | 20460000 ns | +13350000 ns (+187.8%) |
| SM Throughput (%) | 83.37 % | 53.23 % | -30.14 pp | 53.32 % | -30.05 pp |
| Memory Throughput (%) | 22.70 % | 15.60 % | -7.10 pp | 15.63 % | -7.07 pp |
| DRAM Throughput (%) | 12.23 % | 4.25 % | -7.98 pp | 4.25 % | -7.98 pp |
| Achieved Occupancy (%) | 18.03 % | 16.36 % | -1.67 pp | 16.36 % | -1.67 pp |
| Registers/Thread | 144.00 register/thread | 37.00 register/thread | -107 | 37.00 register/thread | -107 |
| Waves/SM | 1318.79 | 7912.76 | +6593.97 | 7912.76 | +6593.97 |
| No Eligible (%) | 26.50 % | 46.55 % | +20.05 pp | 46.44 % | +19.94 pp |
| IPC Active | 2.94 inst/cycle | 2.13 inst/cycle | -0.81 | 2.13 inst/cycle | -0.81 |

---

## Cumulative Progress

```
Baseline:  7110000 ns  (1.000×)

Final best: 13130000 ns  (0.5415×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,16 +3,25 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=1),
     ],
     key=['xnumel'],
 )
@@ -48,26 +57,6 @@

     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = offsets < xnumel
 
-    # Decompose flat index into (n, c, oh, ow)
-    tmp = offsets
-    ow = tmp % out_W
-    tmp = tmp // out_W
-    oh = tmp % out_H
-    tmp = tmp // out_H
-    c = tmp % in_stride_n  # we'll compute c differently
-    # Actually let's do it properly:
-    # offsets = n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w
-    # But since output is contiguous: offsets = n*(C*out_H*out_W) + c*(out_H*out_W) + oh*out_W + ow
-    # Let's recompute:
-    ow2 = offsets % out_W
-    tmp2 = offsets // out_W
-    oh2 = tmp2 % out_H
-    tmp3 = tmp2 // out_H
-    # tmp3 = n * C + c
-    # We need C to split n and c, but we don't have it as constexpr
-    # We'll pass C as a parameter... let's use in_stride_n / (in_H * in_W) = C
-    # Actually let's just use the flat index decomposition with out_stride values
-
     # Recompute using out strides (contiguous layout: n, c, oh, ow)
     # out_stride_n = C * out_H * out_W, out_stride_c = out_H * out_W, out_stride_h = out_W, out_stride_w = 1
     n_idx = offsets // out_stride_n
@@ -84,6 +73,9 @@

     # Initialize max value to -inf
     max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
 
+    # Hoist loop-invariant base offset before the kernel window loop
+    base_nc = n_idx * in_stride_n + c_idx * in_stride_c
+
     # Iterate over kernel window
     for kh in tl.static_range(0, 4):  # kernel_size=4 hardcoded for autotune
         for kw in tl.static_range(0, 4):
@@ -94,7 +86,7 @@

             valid_w = (iw >= 0) & (iw < in_W)
             valid = valid_h & valid_w & mask
 
-            in_offset = n_idx * in_stride_n + c_idx * in_stride_c + ih * in_stride_h + iw * in_stride_w
+            in_offset = base_nc + ih * in_stride_h + iw * in_stride_w
             val = tl.load(in_ptr + in_offset, mask=valid, other=float('-inf'))
             max_val = tl.maximum(max_val, val)
 
@@ -104,18 +96,19 @@

 
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 64},  num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 64},  num_warps=8, num_stages=1),
     ],
-    key=['xnumel', 'kernel_size'],
+    key=['out_W', 'kernel_size'],
 )
 @triton.jit
+
 def max_pool2d_kernel_general(
     in_ptr,
     out_ptr,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

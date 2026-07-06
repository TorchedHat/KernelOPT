# Kernel Optimization Log

## Summary

- **Kernel**: `max_pool3d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4180000 ns — **Best latency**: 4180000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `max_pool3d_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_043/synthesis/kernel_0_triton_per_fused_max_pool3d_with_indices.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1906624,1,1), Waves/SM: 902.76
- **Registers/thread**: 32
- **Duration**: 4180000 ns
- **SM Throughput**: 87.76 % | **Memory**: 65.88 % | **DRAM**: 22.61 %
- **Achieved Occupancy**: 95.00 %
- **IPC Active**: 3.51 inst/cycle | **No Eligible**: 12.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | vectorized_loads_with_eviction_hint_and_expanded_autotune | 0.883× | 0.889× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4180000 ns | 4700000 ns | +520000 ns (+12.4%) |
| SM Throughput (%) | 87.76 % | 92.79 % | +5.03 pp |
| Memory Throughput (%) | 65.88 % | 58.27 % | -7.61 pp |
| DRAM Throughput (%) | 22.61 % | 20.14 % | -2.47 pp |
| Achieved Occupancy (%) | 95.00 % | 95.82 % | +0.82 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 902.76 | 902.76 | +0.00 |
| No Eligible (%) | 12.19 % | 7.16 % | -5.03 pp |
| IPC Active | 3.51 inst/cycle | 3.71 inst/cycle | +0.20 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_alu_integer_division_with_reciprocal_multiply_and_add_num_stages | 0.881× | 0.889× | ✓ |  |
| B | hoist_cta_level_index_decomposition_to_eliminate_per_thread_divmod | 0.797× | 0.757× | ✓ |  |
| C | add_num_stages_pipelining_and_precompute_loop_invariants | 0.881× | 0.886× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_alu_intege... | Δ | hoist_cta_level_i... | Δ | add_num_stages_pi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4180000 ns | 4700000 ns | +520000 ns (+12.4%) | 5520000 ns | +1340000 ns (+32.1%) | 4720000 ns | +540000 ns (+12.9%) |
| SM Throughput (%) | 87.76 % | 92.80 % | +5.04 pp | 91.01 % | +3.25 pp | 92.89 % | +5.13 pp |
| Memory Throughput (%) | 65.88 % | 58.26 % | -7.62 pp | 48.16 % | -17.72 pp | 57.89 % | -7.99 pp |
| DRAM Throughput (%) | 22.61 % | 20.13 % | -2.48 pp | 17.14 % | -5.47 pp | 20.03 % | -2.58 pp |
| Achieved Occupancy (%) | 95.00 % | 95.80 % | +0.80 pp | 71.64 % | -23.36 pp | 95.88 % | +0.88 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 40.00 register/thread | +8 | 32.00 register/thread | +0 |
| Waves/SM | 902.76 | 902.76 | +0.00 | 1203.68 | +300.92 | 902.76 | +0.00 |
| No Eligible (%) | 12.19 % | 7.16 % | -5.03 pp | 8.95 % | -3.24 pp | 7.07 % | -5.12 pp |
| IPC Active | 3.51 inst/cycle | 3.71 inst/cycle | +0.20 | 3.64 inst/cycle | +0.13 | 3.72 inst/cycle | +0.21 |


### Key Learnings

#### ❌ Plan B: `hoist_cta_level_index_decomposition_to_eliminate_per_thread_divmod` (0.757×)

Replacing per-thread vectorized flat-index decomposition with scalar CTA-level divmod plus a per-thread carry-propagation chain increased register usage from 32 to 40 per thread, dropping theoretical occupancy from 100% to 75% and achieved occupancy from 95.82% to 71.64%. The carry-propagation chain introduced additional vector integer divisions (ow_vec // W_out, oh_vec // H_out, etc.) that increased total executed instructions by ~15% (3.45B → 3.98B), negating any savings from hoisting the upper-level scalar divmods. The net effect was a 24% slowdown despite the kernel remaining compute-bound, because the occupancy reduction reduced the warp pool available to hide latency.

> **Applicable when**: Never: this transformation reliably causes regression in Triton kernels when the carry-propagation chain reintroduces vector divmods and the extra scalar variables push register count past a block-limit threshold.

> **Avoid when**: Always avoid when the kernel is already register-limited (Block Limit Registers < Block Limit Warps). Hoisting scalar divmods to CTA level while adding a carry-propagation chain increases register pressure, reduces occupancy, and increases total instruction count — all three effects compound to cause slowdown even when the original kernel appeared ALU-bound from integer divisions.

---

## Cumulative Progress

```
Baseline:  4180000 ns  (1.000×)

Final best: 4180000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -1,6 +1,7 @@

 import torch
 import triton
 import triton.language as tl
+
 
 
 @triton.autotune(
@@ -11,6 +12,8 @@

         triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
         triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
         triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
     ],
     key=['total_elements'],
 )
@@ -76,10 +79,10 @@

                 valid = mask & valid_d & valid_h & valid_w
 
                 in_offset = base_nc + id_ * in_stride_d + ih * in_stride_h + iw * in_stride_w
-                val = tl.load(in_ptr + in_offset, mask=valid, other=float('-inf'))
+                val = tl.load(in_ptr + in_offset, mask=valid, other=float('-inf'), eviction_policy='evict_last')
                 max_val = tl.maximum(max_val, val)
 
-    tl.store(out_ptr + offsets, max_val, mask=mask)
+    tl.store(out_ptr + offsets, max_val, mask=mask, eviction_policy='evict_first')
 
 
 @triton.autotune(
@@ -90,10 +93,13 @@

         triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
         triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
         triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
     ],
     key=['total_elements'],
 )
 @triton.jit
+
 def max_pool3d_kernel_general(
     in_ptr,
     out_ptr,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

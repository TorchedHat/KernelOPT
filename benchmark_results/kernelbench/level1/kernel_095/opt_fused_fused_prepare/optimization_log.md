# Kernel Optimization Log

## Summary

- **Kernel**: `cross_entropy_per_row_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 233500 ns — **Best latency**: 227870 ns (**1.0247×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`coalesce_global_stores_via_multi_row_block_tiling`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `cross_entropy_per_row_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_095/synthesis/kernel_0_fused_fused_prepare.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 15.52
- **Registers/thread**: 23
- **Duration**: 233500 ns
- **SM Throughput**: 66.50 % | **Memory**: 80.61 % | **DRAM**: 47.40 %
- **Achieved Occupancy**: 96.68 %
- **IPC Active**: 2.69 inst/cycle | **No Eligible**: 32.68 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | single_pass_online_softmax_fused_cross_entropy | 1.143× | 0.672× | ✓ |  |
| B | fuse_online_softmax_and_reduce_into_single_kernel | 0.970× | 0.863× | ✓ |  |
| C | parallelize_reduce_loss_kernel_and_add_eviction_hints | 0.974× | 1.025× | ✓ |  |
| D | fuse_two_pass_into_online_softmax_single_pass | 1.144× | 1.151× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | single_pass_onlin... | Δ | fuse_online_softm... | Δ | parallelize_reduc... | Δ | fuse_two_pass_int... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 233500 ns | 347200 ns | +113700 ns (+48.7%) | 270430 ns | +36930 ns (+15.8%) | 227740 ns | -5760 ns (-2.5%) | 202850 ns | -30650 ns (-13.1%) |
| SM Throughput (%) | 66.50 % | 71.32 % | +4.82 pp | 77.98 % | +11.48 pp | 68.01 % | +1.51 pp | 73.81 % | +7.31 pp |
| Memory Throughput (%) | 80.61 % | 83.81 % | +3.20 pp | 83.66 % | +3.05 pp | 81.24 % | +0.63 pp | 80.62 % | +0.01 pp |
| DRAM Throughput (%) | 47.40 % | 31.67 % | -15.73 pp | 40.64 % | -6.76 pp | 48.34 % | +0.94 pp | 54.19 % | +6.79 pp |
| Achieved Occupancy (%) | 96.68 % | 97.16 % | +0.48 pp | 96.88 % | +0.20 pp | 96.84 % | +0.16 pp | 96.54 % | -0.14 pp |
| Registers/Thread | 23.00 register/thread | 27.00 register/thread | +4 | 28.00 register/thread | +5 | 23.00 register/thread | +0 | 28.00 register/thread | +5 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 32.68 % | 34.23 % | +1.55 pp | 21.02 % | -11.66 pp | 30.97 % | -1.71 pp | 25.34 % | -7.34 pp |
| IPC Active | 2.69 inst/cycle | 2.63 inst/cycle | -0.06 | 3.15 inst/cycle | +0.46 | 2.78 inst/cycle | +0.09 | 2.99 inst/cycle | +0.30 |


### Key Learnings

#### ❌ Plan A: `single_pass_online_softmax_fused_cross_entropy` (0.672×)

Replacing a two-pass (max then sum_exp) loop with a single-pass online softmax recurrence increased kernel duration from 233.5 µs to 347.2 µs (0.67x). Despite halving global loads in theory, the L2 hit rate collapsed from 38.15% to 0.88%, total executed instructions grew from 117M to 178M, and MIO stall cycles worsened from 32.2% to 46.2% of cycles. The online recurrence introduces additional exp() calls per block (one extra tl.exp(row_max - new_max) per iteration for the rescaling step) that generate MIO/MUFU stalls, and the register pressure increase (23→27 registers/thread) may have disrupted the compiler's instruction scheduling. When the two-pass kernel already achieves meaningful L2 reuse (38% hit rate) and the dominant stall is L1TEX scoreboard rather than pure DRAM bandwidth, the extra arithmetic in the online recurrence can outweigh the memory traffic savings.

> **Applicable when**: DRAM throughput is the dominant bottleneck (>70% DRAM utilization), L2 hit rate is low (<10%), and the kernel is purely bandwidth-limited with no significant compute stalls from special functions.

> **Avoid when**: The two-pass kernel already achieves meaningful L2 cache reuse (L2 hit rate >30%), the dominant stall is MIO/MUFU (special math instructions like exp), or register pressure is near the block limit. The extra exp() call per block in the online recurrence adds MIO stalls that can exceed the savings from reduced memory traffic.

#### ✅ Plan D: `fuse_two_pass_into_online_softmax_single_pass` (1.151×)

When a kernel iterates over a row twice — once to compute the max and once to compute the sum of exponentials — the second pass re-reads the same data from L2/DRAM due to low L1 hit rates (~4%), doubling global memory traffic and causing heavy L1TEX scoreboard stalls (~9 cycles/warp, ~40% of total stall cycles). Replacing the two loops with a single online-softmax pass (tracking running_max and rescaled running_sum simultaneously via the identity: sum = sum * exp(old_max - new_max) + sum(exp(x - new_max))) halves global memory reads, reducing warp cycles per instruction from 22.85 to 20.69 and improving SM throughput from 66.5% to 73.8%. This technique is applicable whenever a reduction over a large dimension requires two sequential passes that cannot be cached in L1.

> **Applicable when**: Kernel performs two sequential passes over a large row/sequence dimension for softmax/log-softmax/cross-entropy; L1 hit rate is very low (<5%) indicating the second pass re-reads from L2/DRAM; L1TEX scoreboard stalls account for >30% of total stall cycles; DRAM throughput is moderate (<60%) suggesting bandwidth is not yet saturated and reducing reads will help.

> **Avoid when**: The row dimension fits entirely in L1/shared memory (high L1 hit rate on second pass); the kernel already uses a single-pass algorithm; numerical precision requirements are stricter than float32 and the online rescaling introduces unacceptable error; the bottleneck is compute-bound rather than memory-bound.

---

## Iteration 2

**Best so far**: 1.1511× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | coalesce_global_stores_via_multi_row_block_tiling | 1.318× | 1.025× | ✓ | **✓ best** |
| D | revert_to_two_pass_and_tile_rows_for_coalesced_stores | 0.201× | 0.154× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_global_s... | Δ | revert_to_two_pas... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 233500 ns | 227870 ns | -5630 ns (-2.4%) | 1520000 ns | +1286500 ns (+551.0%) |
| SM Throughput (%) | 66.50 % | 68.39 % | +1.89 pp | 80.39 % | +13.89 pp |
| Memory Throughput (%) | 80.61 % | 81.70 % | +1.09 pp | 78.46 % | -2.15 pp |
| DRAM Throughput (%) | 47.40 % | 48.31 % | +0.91 pp | 7.29 % | -40.11 pp |
| Achieved Occupancy (%) | 96.68 % | 96.79 % | +0.11 pp | 73.69 % | -22.99 pp |
| Registers/Thread | 23.00 register/thread | 23.00 register/thread | +0 | 40.00 register/thread | +17 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 20.69 | +5.17 |
| No Eligible (%) | 32.68 % | 30.57 % | -2.11 pp | 19.33 % | -13.35 pp |
| IPC Active | 2.69 inst/cycle | 2.78 inst/cycle | +0.09 | 3.23 inst/cycle | +0.54 |

---

## Iteration 3

**Best so far**: 1.1511× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | differentiate_eviction_policies_pass1_evict_first_pass2_evict_first | 1.086× | 1.005× | ✓ |  |
| B | multi_row_per_cta_coalesced_stores_via_row_tiling | 1.317× | 1.019× | ✓ |  |
| C | revert_block_m_tiling_to_one_row_per_cta_scalar_stores | 1.159× | 1.006× | ✓ |  |
| D | revert_block_m_serialization_and_parallelize_reduce | 1.210× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | differentiate_evi... | Δ | multi_row_per_cta... | Δ | revert_block_m_ti... | Δ | revert_block_m_se... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 233500 ns | 232420 ns | -1080 ns (-0.5%) | 229060 ns | -4440 ns (-1.9%) | 232030 ns | -1470 ns (-0.6%) | 232990 ns | -510 ns (-0.2%) |
| SM Throughput (%) | 66.50 % | 66.92 % | +0.42 pp | 68.21 % | +1.71 pp | 66.44 % | -0.06 pp | 66.18 % | -0.32 pp |
| Memory Throughput (%) | 80.61 % | 80.00 % | -0.61 pp | 81.48 % | +0.87 pp | 80.16 % | -0.45 pp | 80.24 % | -0.37 pp |
| DRAM Throughput (%) | 47.40 % | 48.87 % | +1.47 pp | 48.11 % | +0.71 pp | 47.71 % | +0.31 pp | 47.45 % | +0.05 pp |
| Achieved Occupancy (%) | 96.68 % | 96.64 % | -0.04 pp | 96.79 % | +0.11 pp | 96.83 % | +0.15 pp | 96.49 % | -0.19 pp |
| Registers/Thread | 23.00 register/thread | 23.00 register/thread | +0 | 23.00 register/thread | +0 | 25.00 register/thread | +2 | 23.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 32.68 % | 32.25 % | -0.43 pp | 30.74 % | -1.94 pp | 32.74 % | +0.06 pp | 33.06 % | +0.38 pp |
| IPC Active | 2.69 inst/cycle | 2.70 inst/cycle | +0.01 | 2.78 inst/cycle | +0.09 | 2.71 inst/cycle | +0.02 | 2.67 inst/cycle | -0.02 |

---

## Iteration 4

**Best so far**: 1.1511× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_mio_stall_via_fused_fp32_and_store_eviction_hints | 1.060× | 1.003× | ✓ |  |
| C | replace_exp_log_with_fast_math_libdevice_to_reduce_mio_stalls | 1.054× | 1.019× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_mio_stall_... | Δ | replace_exp_log_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 233500 ns | 232700 ns | -800 ns (-0.3%) | 229150 ns | -4350 ns (-1.9%) |
| SM Throughput (%) | 66.50 % | 67.34 % | +0.84 pp | 58.66 % | -7.84 pp |
| Memory Throughput (%) | 80.61 % | 80.04 % | -0.57 pp | 80.97 % | +0.36 pp |
| DRAM Throughput (%) | 47.40 % | 48.64 % | +1.24 pp | 49.44 % | +2.04 pp |
| Achieved Occupancy (%) | 96.68 % | 96.72 % | +0.04 pp | 96.62 % | -0.06 pp |
| Registers/Thread | 23.00 register/thread | 23.00 register/thread | +0 | 25.00 register/thread | +2 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 32.68 % | 31.55 % | -1.13 pp | 40.68 % | +8.00 pp |
| IPC Active | 2.69 inst/cycle | 2.73 inst/cycle | +0.04 | 2.38 inst/cycle | -0.31 |

---

## Cumulative Progress

```
Baseline:  233500 ns  (1.000×)
Iter 1     202850 ns  (1.151×)  — fuse_two_pass_into_online_softmax_single_pass

Final best: 227870 ns  (1.0247×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -11,14 +11,21 @@

 #   4. loss_i = -log_softmax[i, targets[i]] (if targets[i] != -100, else 0)
 # Final: mean(loss_i) over valid samples
 
+
 @triton.autotune(
     configs=[
         triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=2),
         triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=2),
-        triton.Config({'BLOCK_N': 4096}, num_warps=32, num_stages=1),
+        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_N': 4096}, num_warps=8, num_stages=1),
         triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=3),
         triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_N': 4096}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=3),
     ],
     key=['N'],
 )
@@ -33,56 +40,57 @@

     BLOCK_N: tl.constexpr,
 ):
     """
+    One program per row. Two-pass cross entropy with coalesced loads.
     """
     row = tl.program_id(0)
-    
+
     # Load target for this row
     target = tl.load(targets_ptr + row)
-    
+
     # Pass 1: compute max over the row
     row_max = tl.full((), float('-inf'), dtype=tl.float32)
-    
+
     for off in tl.range(0, N, BLOCK_N):
         cols = off + tl.arange(0, BLOCK_N)
         mask = cols < N
-        x = tl.load(predictions_ptr + row * N + cols, mask=mask, other=float('-inf'))
+        x = tl.load(predictions_ptr + row * N + cols, mask=mask, other=float('-inf'), eviction_policy='evict_last')
         x = x.to(tl.float32)
         block_max = tl.max(x, axis=0)
         row_max = tl.maximum(row_max, block_max)
-    
+
     # Pass 2: compute sum of exp(x - max)
     row_sum = tl.zeros((), dtype=tl.float32)
-    
+
     for off in tl.range(0, N, BLOCK_N):
         cols = off + tl.arange(0, BLOCK_N)
         mask = cols < N
-        x = tl.load(predictions_ptr + row * N + cols, mask=mask, other=float('-inf'))
+        x = tl.load(predictions_ptr + row * N + cols, mask=mask, other=float('-inf'), eviction_policy='evict_last')
         x = x.to(tl.float32)
         exp_x = tl.exp(x - row_max)
         exp_x = tl.where(mask, exp_x, 0.0)
         row_sum += tl.sum(exp_x, axis=0)
-    
+
     log_sum_exp = libdevice.log(row_sum) + row_max
-    
+
     # Compute loss for this row
     is_valid = target != -100
-    
+
     # Clamp target to valid range for indexing
     safe_target = tl.where(is_valid, target, tl.zeros_like(target))
     # Handle negative indices
     safe_target = tl.where(safe_target < 0, safe_target + N, safe_target)
-    
+
     # Load the prediction at the target class
     pred_at_target = tl.load(predictions_ptr + row * N + safe_target).to(tl.float32)
-    
+
     # log_softmax at target = pred_at_target - log_sum_exp
     log_prob = pred_at_target - log_sum_exp
     loss = -log_prob
-    
+
     # Zero out loss for invalid targets
     loss = tl.where(is_valid, loss, 0.0)
     valid_int = tl.where(is_valid, 1, 0)
-    
+
     tl.store(per_row_loss_ptr + row, loss)
     tl.store(per_row_valid_ptr + row, valid_int)
 
@@ -91,12 +99,12 @@

     configs=[
         triton.Config({'BLOCK_M': 256}, num_warps=4),
         triton.Config({'BLOCK_M': 512}, num_warps=8),
-        triton.Config({'BLOCK_M': 1024}, num_warps=16),
-        triton.Config({'BLOCK_M': 2048}, num_warps=32),
+        triton.Config({'BLOCK_M': 1024}, num_warps=8),
     ],
     key=['M'],
 )
 @triton.jit
+
 def reduce_loss_kernel(
     per_row_loss_ptr,
     per_row_valid_ptr,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

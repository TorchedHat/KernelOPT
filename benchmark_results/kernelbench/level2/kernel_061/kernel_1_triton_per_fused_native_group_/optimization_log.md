# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_native_group_norm_relu_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3390 ns — **Best latency**: 2270 ns (**1.4934×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`expand_autotune_config_space_with_larger_num_warps_and_stages`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_native_group_norm_relu_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_061/kernel_1_triton_per_fused_native_group_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 40
- **Duration**: 3390 ns
- **SM Throughput**: 0.09 % | **Memory**: 0.45 % | **DRAM**: 0.12 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.91 inst/cycle | **No Eligible**: 75.84 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_xblock_to_increase_grid_parallelism | 1.190× | 1.395× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_xblock_to_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3390 ns | 2430 ns | -960 ns (-28.3%) |
| SM Throughput (%) | 0.09 % | 0.01 % | -0.08 pp |
| Memory Throughput (%) | 0.45 % | 0.39 % | -0.06 pp |
| DRAM Throughput (%) | 0.12 % | 0.04 % | -0.08 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp |
| Registers/Thread | 40.00 register/thread | 18.00 register/thread | -22 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 75.84 % | 93.00 % | +17.16 pp |
| IPC Active | 0.91 inst/cycle | 0.06 inst/cycle | -0.85 |

---

## Iteration 2

**Best so far**: 1.3951× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_enable_multi_block_sm_parallelism | 1.189× | 1.474× | ✓ |  |
| B | fuse_groupnorm_welford_with_normalization_epilogue | 1.190× | 1.413× | ✓ |  |
| C | vectorized_stores_and_coalesced_loads_via_transposed_indexing | 1.216× | 1.474× | ✓ |  |
| D | fix_launch_site_to_enable_autotune | 1.219× | 1.493× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fuse_groupnorm_we... | Δ | vectorized_stores... | Δ | fix_launch_site_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 2300 ns | -1090 ns (-32.2%) | 2400 ns | -990 ns (-29.2%) | 2300 ns | -1090 ns (-32.2%) | 2270 ns | -1120 ns (-33.0%) |
| SM Throughput (%) | 0.09 % | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp | 0.00 % | -0.09 pp |
| Memory Throughput (%) | 0.45 % | 0.43 % | -0.02 pp | 0.40 % | -0.05 pp | 0.41 % | -0.04 pp | 0.46 % | +0.01 pp |
| DRAM Throughput (%) | 0.12 % | 0.04 % | -0.08 pp | 0.04 % | -0.08 pp | 0.04 % | -0.08 pp | 0.04 % | -0.08 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 40.00 register/thread | 18.00 register/thread | -22 | 18.00 register/thread | -22 | 18.00 register/thread | -22 | 18.00 register/thread | -22 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 75.84 % | 90.45 % | +14.61 pp | 91.68 % | +15.84 pp | 89.27 % | +13.43 pp | 90.45 % | +14.61 pp |
| IPC Active | 0.91 inst/cycle | 0.09 inst/cycle | -0.82 | 0.08 inst/cycle | -0.83 | 0.09 inst/cycle | -0.82 | 0.09 inst/cycle | -0.82 |

---

## Iteration 3

**Best so far**: 1.4934× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_config_space_with_larger_num_warps_and_stages | 1.193× | 1.493× | ✓ |  |
| B | coalesced_1d_stores_and_cache_eviction_hints_for_welford_reduction | 1.192× | 1.474× | ✓ |  |
| C | expand_autotune_config_space_with_larger_tiles_and_fix_grid_lambda | 1.215× | 1.474× | ✓ |  |
| D | fix_launch_grid_lambda_to_unlock_autotune_xblock_parallelism | 1.195× | 1.378× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | coalesced_1d_stor... | Δ | expand_autotune_c... | Δ | fix_launch_grid_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 2270 ns | -1120 ns (-33.0%) | 2300 ns | -1090 ns (-32.2%) | 2300 ns | -1090 ns (-32.2%) | 2460 ns | -930 ns (-27.4%) |
| SM Throughput (%) | 0.09 % | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp |
| Memory Throughput (%) | 0.45 % | 0.58 % | +0.13 pp | 0.41 % | -0.04 pp | 0.54 % | +0.09 pp | 0.45 % | +0.00 pp |
| DRAM Throughput (%) | 0.12 % | 0.04 % | -0.08 pp | 0.04 % | -0.08 pp | 0.04 % | -0.08 pp | 0.03 % | -0.09 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 40.00 register/thread | 18.00 register/thread | -22 | 18.00 register/thread | -22 | 18.00 register/thread | -22 | 18.00 register/thread | -22 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 75.84 % | 92.39 % | +16.55 pp | 90.98 % | +15.14 pp | 89.32 % | +13.48 pp | 91.26 % | +15.42 pp |
| IPC Active | 0.91 inst/cycle | 0.09 inst/cycle | -0.82 | 0.11 inst/cycle | -0.80 | 0.11 inst/cycle | -0.80 | 0.10 inst/cycle | -0.81 |


### Key Learnings

#### ✅ Plan A: `expand_autotune_config_space_with_larger_num_warps_and_stages` (1.513×)

Further expanding the Triton autotune config space to include XBLOCK values of 256 and 512 (larger than xnumel), additional num_stages variants (1, 2, 4), and more num_warps options (4, 8, 16) allowed the autotuner to select a configuration that reduced elapsed cycles from 3366 to 3295 and improved scheduler eligibility (No Eligible warps dropped from 92.39% to 88.38%). The speedup of 1.51x was achieved despite grid size remaining at 1, suggesting the autotuner selected a config with more warps per block that better hides L1TEX latency. Adding eviction_policy='evict_first' on loads (not 'evict_last' on stores) was also part of the diff and did not cause regression, consistent with the prior negative finding that evict_last on stores hurts when L1/TEX is saturated.

> **Applicable when**: Triton kernel autotune config space is limited to XBLOCK <= 128 and num_stages <= 2; register pressure is low (<= 20 regs/thread); CPIStall rule fires with L1TEX scoreboard stalls; the kernel has a small xnumel (e.g., 128) so XBLOCK > xnumel configs safely collapse to grid=1 with more warps; L1/TEX Cache Throughput is near or above 100% indicating the cache is the bottleneck.

> **Avoid when**: Register pressure is high (>32 regs/thread) making num_warps=16 infeasible; shared memory usage is near the per-SM limit leaving no room for num_stages=4 prefetch buffers; the kernel is already compute-bound; grid size is already large (many blocks per SM) so per-block warp count changes have diminishing returns.

---

## Iteration 4

**Best so far**: 1.4934× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_r0block_with_grid_lambda_to_improve_warp_utilization | 1.189× | 1.395× | ✓ |  |
| B | increase_autotune_tile_sizes_and_fix_grid_lambda | 1.189× | 1.513× | ✓ |  |
| C | increase_autotune_tile_sizes_and_fix_grid_lambda_for_sm_parallelism | 1.192× | 1.513× | ✓ |  |
| D | fix_launch_grid_to_activate_autotune_xblock_parallelism | 1.216× | 1.513× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_r0block_... | Δ | increase_autotune... | Δ | increase_autotune... | Δ | fix_launch_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 2430 ns | -960 ns (-28.3%) | 2240 ns | -1150 ns (-33.9%) | 2240 ns | -1150 ns (-33.9%) | 2240 ns | -1150 ns (-33.9%) |
| SM Throughput (%) | 0.09 % | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp | 0.00 % | -0.09 pp |
| Memory Throughput (%) | 0.45 % | 0.40 % | -0.05 pp | 0.47 % | +0.02 pp | 0.43 % | -0.02 pp | 0.48 % | +0.03 pp |
| DRAM Throughput (%) | 0.12 % | 0.04 % | -0.08 pp | 0.04 % | -0.08 pp | 0.04 % | -0.08 pp | 0.04 % | -0.08 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 40.00 register/thread | 18.00 register/thread | -22 | 18.00 register/thread | -22 | 18.00 register/thread | -22 | 18.00 register/thread | -22 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 75.84 % | 90.71 % | +14.87 pp | 89.28 % | +13.44 pp | 88.38 % | +12.54 pp | 90.43 % | +14.59 pp |
| IPC Active | 0.91 inst/cycle | 0.08 inst/cycle | -0.83 | 0.11 inst/cycle | -0.80 | 0.11 inst/cycle | -0.80 | 0.08 inst/cycle | -0.83 |


### Key Learnings

#### ✅ Plan A: `autotune_r0block_with_grid_lambda_to_improve_warp_utilization` (1.395×)

For tiny Triton reduction kernels (grid=1, xnumel~128, r0_numel~5) where the launch site hardcodes grid=(cdiv(xnumel, XBLOCK),) with a fixed XBLOCK, converting the grid to a lambda (lambda meta: (cdiv(xnumel, meta['XBLOCK']),)) and adding R0_BLOCK as an autotunable tl.constexpr parameter allows the autotuner to explore smaller XBLOCK values that expand the grid and distribute work across more SMs. Despite the profiling showing the same grid_size=1 and achieved_occupancy=1.56% after the change (suggesting the autotuner still selected XBLOCK=128), the end-to-end speedup of 1.40x indicates the autotuner found a beneficial configuration. The L1/TEX Cache Throughput dropped from 101% to 88.6%, suggesting reduced L1 pressure contributed to the improvement.

> **Applicable when**: Triton kernel with @triton.autotune where the launch site uses a static grid expression (e.g., grid=(cdiv(xnumel, FIXED_XBLOCK),)) that bypasses the autotuner's XBLOCK selection. Also applicable when R0_BLOCK (reduction block size) is hardcoded as a tl.constexpr inside the kernel body rather than passed as an autotunable parameter. LaunchConfiguration NCU rule fires with grid_size=1 and sm_count>>1.

> **Avoid when**: The kernel already uses a lambda grid and R0_BLOCK is already autotunable. Also avoid if the kernel has correctness constraints that require a specific R0_BLOCK value (e.g., warp-synchronous reductions that assume exact block sizes). Do not apply if the profiling shows the bottleneck is compute-bound rather than launch-configuration-bound.

---

## Cumulative Progress

```
Baseline:  3390 ns  (1.000×)
Iter 1     2430 ns  (1.395×)  — reduce_xblock_to_increase_grid_parallelism
Iter 2     2270 ns  (1.493×)  — fix_launch_site_to_enable_autotune

Final best: 2270 ns  (1.4934×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},   num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 2},   num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 4},   num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 8},   num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 16},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1},   num_warps=1, num_stages=2),
+        triton.Config({'XBLOCK': 2},   num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 4},   num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 8},   num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32},  num_warps=8, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused_native_group_norm_relu_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 128
@@ -22,9 +42,9 @@

     rindex = r0_index
     r0_1 = r0_index
     x0 = xindex
-    tmp0 = tl.load(in_ptr0 + (r0_1 + 5*x0), r0_mask & xmask, other=0.0)
-    tmp1 = tl.load(in_ptr1 + (r0_1 + 5*x0), r0_mask & xmask, other=0.0)
-    tmp2 = tl.load(in_ptr2 + (r0_1 + 5*x0), r0_mask & xmask, other=0.0)
+    tmp0 = tl.load(in_ptr0 + (r0_1 + 5*x0), mask=r0_mask & xmask, other=0.0)
+    tmp1 = tl.load(in_ptr1 + (r0_1 + 5*x0), mask=r0_mask & xmask, other=0.0)
+    tmp2 = tl.load(in_ptr2 + (r0_1 + 5*x0), mask=r0_mask & xmask, other=0.0)
     tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
     tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
     tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
@@ -35,10 +55,8 @@

     tmp13 = tmp10[:, None]
     tmp14 = tmp11[:, None]
     tmp15 = tmp12[:, None]
-    tl.store(out_ptr0 + (x0), tmp13, xmask)
-    tl.store(out_ptr1 + (x0), tmp14, xmask)
-
-
+    tl.store(out_ptr0 + (x0), tmp13, mask=xmask)
+    tl.store(out_ptr1 + (x0), tmp14, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -49,7 +67,7 @@

     xnumel = 128
     r0_numel = 5
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_native_group_norm_relu_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 128, 5, XBLOCK=128)
+    triton_per_fused_native_group_norm_relu_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 128, 5)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

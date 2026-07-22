# Kernel Optimization Log

## Summary

- **Kernel**: `linear_bias_relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 414620 ns — **Best latency**: 255230 ns (**1.6245×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`expand_block_m_autotune_small_values_grid_expansion`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `linear_bias_relu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_012/synthesis/kernel_17_triton_tem_fused_convolution_max_pool2d_.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.06
- **Registers/thread**: 64
- **Duration**: 414620 ns
- **SM Throughput**: 4.20 % | **Memory**: 20.36 % | **DRAM**: 20.36 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.26 inst/cycle | **No Eligible**: 93.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ |  |
| B | tma_loads_with_block_ptr_and_expanded_autotune_configs | 0.975× | 1.670× | ✓ |  |
| C | eviction_policies_streaming_weights_evict_first | 0.998× | 1.002× | ✓ |  |
| D | increase_block_n_tile_size_for_small_batch_gemm | 1.000× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_bl... | Δ | eviction_policies... | Δ | increase_block_n_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 414620 ns | 414780 ns | +160 ns (+0.0%) | 248260 ns | -166360 ns (-40.1%) | 413820 ns | -800 ns (-0.2%) | 413340 ns | -1280 ns (-0.3%) |
| SM Throughput (%) | 4.20 % | 4.18 % | -0.02 pp | 21.57 % | +17.37 pp | 4.20 % | +0.00 pp | 4.18 % | -0.02 pp |
| Memory Throughput (%) | 20.36 % | 20.35 % | -0.01 pp | 39.32 % | +18.96 pp | 20.40 % | +0.04 pp | 20.42 % | +0.06 pp |
| DRAM Throughput (%) | 20.36 % | 20.35 % | -0.01 pp | 34.01 % | +13.65 pp | 20.40 % | +0.04 pp | 20.42 % | +0.06 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 4.55 % | -1.70 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 96.00 register/thread | +32 | 64.00 register/thread | +0 | 64.00 register/thread | +0 |
| Waves/SM | 0.06 | 0.06 | +0.00 | 0.24 | +0.18 | 0.06 | +0.00 | 0.06 | +0.00 |
| No Eligible (%) | 93.58 % | 93.59 % | +0.01 pp | 69.57 % | -24.01 pp | 93.58 % | +0.00 pp | 93.59 % | +0.01 pp |
| IPC Active | 0.26 inst/cycle | 0.26 inst/cycle | +0.00 | 0.88 inst/cycle | +0.62 | 0.26 inst/cycle | +0.00 | 0.26 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_shared_memory_via_lower_num_stages_and_smaller_block_n_to_increase_sm_occupancy | 0.966× | 0.539× | ✓ |  |
| B | expand_block_m_autotune_small_values_grid_expansion | 1.002× | 1.625× | ✓ | **✓ best** |
| C | small_block_m_autotune_with_block_ptr_tma_loads | 0.991× | 1.683× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ | expand_block_m_au... | Δ | small_block_m_aut... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 414620 ns | 768900 ns | +354280 ns (+85.4%) | 255230 ns | -159390 ns (-38.4%) | 246430 ns | -168190 ns (-40.6%) |
| SM Throughput (%) | 4.20 % | 9.86 % | +5.66 pp | 19.10 % | +14.90 pp | 21.61 % | +17.41 pp |
| Memory Throughput (%) | 20.36 % | 27.00 % | +6.64 pp | 33.08 % | +12.72 pp | 39.13 % | +18.77 pp |
| DRAM Throughput (%) | 20.36 % | 10.98 % | -9.38 pp | 33.08 % | +12.72 pp | 34.38 % | +14.02 pp |
| Achieved Occupancy (%) | 6.25 % | 9.09 % | +2.84 pp | 4.54 % | -1.71 pp | 4.56 % | -1.69 pp |
| Registers/Thread | 64.00 register/thread | 52.00 register/thread | -12 | 96.00 register/thread | +32 | 96.00 register/thread | +32 |
| Waves/SM | 0.06 | 0.16 | +0.10 | 0.24 | +0.18 | 0.24 | +0.18 |
| No Eligible (%) | 93.58 % | 90.09 % | -3.49 pp | 75.92 % | -17.66 pp | 69.63 % | -23.95 pp |
| IPC Active | 0.26 inst/cycle | 0.40 inst/cycle | +0.14 | 0.70 inst/cycle | +0.44 | 0.89 inst/cycle | +0.63 |


### Key Learnings

#### ✅ Plan B: `expand_block_m_autotune_small_values_grid_expansion` (1.625×)

When the M-dimension (batch size) is small relative to the minimum BLOCK_M in the autotune config list, the M-grid dimension collapses to 1, leaving most SMs idle (waves_per_SM << 1, achieved occupancy << theoretical). Adding smaller BLOCK_M values (e.g., 4, 8) to the autotune search space allows the autotuner to select a config that produces more M-tiles, expanding the grid and spreading work across more SMs. The profiling NCU rules SOLBottleneck, AchievedOccupancy, and IssueSlotUtilization all fire when this condition exists; the fix is purely in the autotune config list with no kernel body changes required.

> **Applicable when**: SOLBottleneck rule fires with waves_per_SM < 0.5; AchievedOccupancy << TheoreticalOccupancy (e.g., 4.5% vs 18.75%); IssueSlotUtilization fires with No Eligible > 65%; the M-dimension (batch size) is small (e.g., M <= 16) while minimum BLOCK_M in autotune configs is >= 16; the kernel already has boundary masks (mask_m = offs_m < M) that handle non-divisible tile sizes.

> **Avoid when**: The kernel is already achieving high waves_per_SM (> 5) or high achieved occupancy (> 50%); the M-dimension is large relative to BLOCK_M (M >> BLOCK_M); BLOCK_M < 16 causes tl.dot() Tensor Core alignment errors on the target backend (in that case, keep BLOCK_M >= 16); the kernel lacks boundary masks for the M-dimension (would produce incorrect results for non-divisible M).

#### ✅ Plan C: `small_block_m_autotune_with_block_ptr_tma_loads` (1.683×)

When a GEMM kernel's M-dimension is very small relative to all autotune BLOCK_M configs, the M-grid collapses to 1, producing far fewer blocks than available SMs (e.g., 0.06 waves/SM). Adding smaller BLOCK_M configs (4, 8) multiplies grid size (64→192 blocks), spreading work across more SMs and reducing per-warp stall cycles. Replacing scalar pointer arithmetic in the K-loop with tl.make_block_ptr() + tl.advance() + tl.load(boundary_check=...) enables Hopper TMA hardware, improves L2 hit rate, and raises memory throughput, together yielding ~1.68x speedup. The marginal improvement in iteration 2 over iteration 1 (1.6825x vs 1.6701x) came from minor autotune config refinements (swapping BLOCK_N=256/BLOCK_K=64 for BLOCK_N=256/BLOCK_K=128 in BLOCK_M=4/8 configs) and restoring allow_tf32=True to tl.dot(), with L2 hit rate improving slightly from 45.96% to 50.12%.

> **Applicable when**: GEMM kernel where M (or N) is very small relative to BLOCK_M in all autotune configs, causing grid_size << SM_count (waves_per_SM < 0.1), achieved occupancy < 10%, and high no-eligible-warp rate (>90%). LaunchConfiguration, AchievedOccupancy, and CPIStall NCU rules all triggered. Running on Hopper (SM90) where TMA hardware is available via tl.make_block_ptr().

> **Avoid when**: M is large enough that existing BLOCK_M configs already produce grid_size >= SM_count (waves_per_SM > 1.0). Also avoid if register pressure from block_ptr would critically reduce theoretical occupancy below what scalar loads achieve without the grid-size compensation. Not applicable on pre-Hopper architectures where tl.make_block_ptr TMA is unavailable.

---

## Iteration 3

**Best so far**: 1.6245× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restore_num_stages_3_4_pipelining_remove_low_stage_configs | 0.987× | 1.630× | ✓ |  |
| B | add_triton_autotune | 0.976× | 1.673× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restore_num_stage... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 414620 ns | 254370 ns | -160250 ns (-38.6%) | 247840 ns | -166780 ns (-40.2%) |
| SM Throughput (%) | 4.20 % | 28.33 % | +24.13 pp | 21.62 % | +17.42 pp |
| Memory Throughput (%) | 20.36 % | 40.17 % | +19.81 pp | 38.38 % | +18.02 pp |
| DRAM Throughput (%) | 20.36 % | 34.94 % | +14.58 pp | 34.07 % | +13.71 pp |
| Achieved Occupancy (%) | 6.25 % | 9.19 % | +2.94 pp | 4.56 % | -1.69 pp |
| Registers/Thread | 64.00 register/thread | 59.00 register/thread | -5 | 96.00 register/thread | +32 |
| Waves/SM | 0.06 | 0.24 | +0.18 | 0.24 | +0.18 |
| No Eligible (%) | 93.58 % | 68.83 % | -24.75 pp | 69.52 % | -24.06 pp |
| IPC Active | 0.26 inst/cycle | 1.24 inst/cycle | +0.98 | 0.89 inst/cycle | +0.63 |


### Key Learnings

#### ❌ Plan B: `add_triton_autotune` (0.385×)

Expanding the autotune config space to include smaller BLOCK_SIZE values (128, 256) caused the autotuner to select a very small tile size, exploding the grid from 31,360 to 250,880 blocks and increasing total executed instructions by ~4.5x (17.9M → 80.3M). Although per-warp stall cycles improved (34.35 → 14.8 cycles/issued instruction) and DRAM throughput dropped (78% → 30%), the massive increase in kernel launch overhead and total instruction count caused wall-clock duration to increase from 60.7 µs to 157.15 µs (2.6x slower). When a memory-bound kernel is already well-occupancy-covered (85% achieved occupancy, 14.85 waves/SM), adding small BLOCK_SIZE configs to autotune risks selecting a configuration that over-decomposes the work, inflating grid size and total instruction count far beyond the benefit of reduced per-warp stalls.

> **Applicable when**: Never apply: this is a negative result. Do not add very small BLOCK_SIZE configs (128, 256) to autotune when the kernel is already memory-bandwidth-bound with high occupancy (>80% achieved) and adequate waves-per-SM (>10). The autotuner may select the smallest config, causing grid explosion.

> **Avoid when**: Do not expand autotune config space to include small BLOCK_SIZE values when: (1) the kernel is already achieving high occupancy (>80%), (2) waves_per_SM is already >10, (3) the kernel is memory-bandwidth-bound (DRAM throughput >70%). In these cases, smaller tiles increase grid overhead and total instruction count without proportional bandwidth benefit.

---

## Iteration 4

**Best so far**: 1.6245× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_and_expand_num_warps_configs | 0.985× | 1.630× | ✓ |  |
| B | cap_registers_per_thread_to_increase_active_warps | 0.993× | 1.795× | ✓ |  |
| C | increase_num_warps_to_hide_alu_fixed_latency_stalls | 0.992× | 1.633× | ✓ |  |
| D | tma_store_block_ptr_and_num_warps_sweep | 0.987× | 1.621× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | cap_registers_per... | Δ | increase_num_warp... | Δ | tma_store_block_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 414620 ns | 254400 ns | -160220 ns (-38.6%) | 231040 ns | -183580 ns (-44.3%) | 253980 ns | -160640 ns (-38.7%) | 255740 ns | -158880 ns (-38.3%) |
| SM Throughput (%) | 4.20 % | 28.56 % | +24.36 pp | 29.38 % | +25.18 pp | 28.21 % | +24.01 pp | 28.39 % | +24.19 pp |
| Memory Throughput (%) | 20.36 % | 40.48 % | +20.12 pp | 44.49 % | +24.13 pp | 40.03 % | +19.67 pp | 40.21 % | +19.85 pp |
| DRAM Throughput (%) | 20.36 % | 35.29 % | +14.93 pp | 39.08 % | +18.72 pp | 35.38 % | +15.02 pp | 36.09 % | +15.73 pp |
| Achieved Occupancy (%) | 6.25 % | 9.19 % | +2.94 pp | 9.13 % | +2.88 pp | 9.19 % | +2.94 pp | 9.18 % | +2.93 pp |
| Registers/Thread | 64.00 register/thread | 59.00 register/thread | -5 | 64.00 register/thread | +0 | 59.00 register/thread | -5 | 59.00 register/thread | -5 |
| Waves/SM | 0.06 | 0.24 | +0.18 | 0.24 | +0.18 | 0.24 | +0.18 | 0.24 | +0.18 |
| No Eligible (%) | 93.58 % | 68.94 % | -24.64 pp | 67.44 % | -26.14 pp | 68.91 % | -24.67 pp | 68.89 % | -24.69 pp |
| IPC Active | 0.26 inst/cycle | 1.25 inst/cycle | +0.99 | 1.30 inst/cycle | +1.04 | 1.25 inst/cycle | +0.99 | 1.24 inst/cycle | +0.98 |


### Key Learnings

#### ✅ Plan A: `cap_registers_with_maxnreg_and_expand_num_warps_configs` (1.630×)

When a Triton GEMM kernel has high register usage per thread (e.g., 96 registers/thread) and Block Limit Registers is the secondary occupancy limiter (after shared memory), adding autotune configs with maxnreg=64 caps register allocation per thread, reducing it from 96 to ~59 registers. This raises theoretical occupancy from 18.75% to 37.5% and achieved occupancy from 4.56% to 9.19%, doubling active warps per scheduler (1.0→1.46) and increasing compute SM throughput (21.62%→28.56%). The autotuner selects a larger block size (64→128 threads) enabled by the lower register footprint, providing more instruction-level parallelism within each block to partially hide CPIStall fixed-latency dependency stalls.

> **Applicable when**: Triton GEMM/matmul kernel where: (1) registers_per_thread > 64 and Block Limit Registers < Block Limit Shared Mem (registers are a secondary occupancy limiter); (2) TheoreticalOccupancy < 25% and AchievedOccupancy < 10%; (3) CPIStall and IssueSlotUtilization NCU rules fire; (4) the grid is small (waves_per_SM < 0.5) so inter-block latency hiding is insufficient; (5) num_stages >= 3 configs are already in use (pipelining is active).

> **Avoid when**: Block Limit Shared Mem is already the sole binding occupancy limiter and Block Limit Registers >> Block Limit Shared Mem (maxnreg won't help if registers aren't the constraint); registers_per_thread is already <= 64; the kernel is compute-bound with high achieved occupancy (>50%); register spilling to L1 cache from aggressive maxnreg capping introduces enough latency to offset the occupancy gain (monitor L1/TEX hit rate after applying).

---

## Cumulative Progress

```
Baseline:  414620 ns  (1.000×)
Iter 2     255230 ns  (1.625×)  — expand_block_m_autotune_small_values_grid_expansion

Final best: 255230 ns  (1.6245×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,27 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
+        # num_stages=3: software pipelining enabled, good latency hiding
+        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=2, num_stages=3),
+        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=2, num_stages=3),
+        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
         triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
-        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
+        # num_stages=4: deeper pipelining for higher memory latency hiding
+        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
     ],
     key=['M', 'N', 'K'],
 )
@@ -29,24 +42,38 @@

 
     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    offs_k = tl.arange(0, BLOCK_K)
 
     mask_m = offs_m < M
     mask_n = offs_n < N
 
+    # X has shape (M, K), strides (stride_xm, stride_xk)
+    x_block_ptr = tl.make_block_ptr(
+        base=X_ptr,
+        shape=(M, K),
+        strides=(stride_xm, stride_xk),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # W has shape (N, K), strides (stride_wn, stride_wk)
+    # Traverse W as (K, N) by swapping strides to get (BLOCK_K, BLOCK_N) tiles
+    w_block_ptr = tl.make_block_ptr(
+        base=W_ptr,
+        shape=(K, N),
+        strides=(stride_wk, stride_wn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
-    for k in range(0, K, BLOCK_K):
-        k_offs = k + offs_k
-        mask_k = k_offs < K
-
-        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
-        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
-
-        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
-        w = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)
-
+    for k in range(0, tl.cdiv(K, BLOCK_K)):
+        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        w = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option='zero')
         acc = tl.dot(x, w, acc)
+        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
+        w_block_ptr = tl.advance(w_block_ptr, (BLOCK_K, 0))
 
     bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
     acc = acc + bias[None, :]
@@ -68,6 +95,7 @@

     key=['N'],
 )
 @triton.jit
+
 def bias_relu_kernel(
     X_ptr, B_ptr, Out_ptr,
     M, N,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

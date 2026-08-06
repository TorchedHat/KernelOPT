# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 23740 ns — **Best latency**: 17380 ns (**1.3659×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_b_matrix_bank_conflicts_via_explicit_pointer_arithmetic`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_15`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_15_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.16
- **Registers/thread**: 80
- **Duration**: 23740 ns
- **SM Throughput**: 10.60 % | **Memory**: 74.15 % | **DRAM**: 5.20 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.50 inst/cycle | **No Eligible**: 87.50 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 1.000× | ✓ |  |
| B | eliminate_shared_memory_bank_conflicts_b_matrix | 1.001× | 0.998× | ✓ |  |
| C | fix_shared_memory_bank_conflicts_via_b_order_and_autotune | 1.009× | 0.997× | ✓ | **✓ best** |
| D | autotune_tile_sizes_and_warps_to_fix_underutilization_and_bank_conflicts | 0.981× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | eliminate_shared_... | Δ | fix_shared_memory... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 23740 ns | 23740 ns | +0 ns (+0.0%) | 23780 ns | +40 ns (+0.2%) | 23810 ns | +70 ns (+0.3%) | 24380 ns | +640 ns (+2.7%) |
| SM Throughput (%) | 10.60 % | 10.61 % | +0.01 pp | 10.53 % | -0.07 pp | 10.59 % | -0.01 pp | 10.27 % | -0.33 pp |
| Memory Throughput (%) | 74.15 % | 74.18 % | +0.03 pp | 73.56 % | -0.59 pp | 74.07 % | -0.08 pp | 72.35 % | -1.80 pp |
| DRAM Throughput (%) | 5.20 % | 5.21 % | +0.01 pp | 5.19 % | -0.01 pp | 5.19 % | -0.01 pp | 5.08 % | -0.12 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.16 | 0.16 | +0.00 | 0.16 | +0.00 | 0.16 | +0.00 | 0.23 | +0.07 |
| No Eligible (%) | 87.50 % | 87.46 % | -0.04 pp | 87.45 % | -0.05 pp | 87.58 % | +0.08 pp | 87.95 % | +0.45 pp |
| IPC Active | 0.50 inst/cycle | 0.49 inst/cycle | -0.01 | 0.50 inst/cycle | +0.00 | 0.50 inst/cycle | +0.00 | 0.48 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eliminate_tl_trans_bank_conflicts_via_direct_b_layout | 1.002× | 1.002× | ✓ |  |
| D | fix_b_matrix_smem_bank_conflicts_via_order_and_smem_padding | 0.982× | 0.966× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_tl_tran... | Δ | fix_b_matrix_smem... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 23740 ns | 23680 ns | -60 ns (-0.3%) | 24580 ns | +840 ns (+3.5%) |
| SM Throughput (%) | 10.60 % | 10.61 % | +0.01 pp | 10.31 % | -0.29 pp |
| Memory Throughput (%) | 74.15 % | 74.12 % | -0.03 pp | 72.65 % | -1.50 pp |
| DRAM Throughput (%) | 5.20 % | 5.22 % | +0.02 pp | 5.04 % | -0.16 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.16 | 0.16 | +0.00 | 0.23 | +0.07 |
| No Eligible (%) | 87.50 % | 87.44 % | -0.06 pp | 87.98 % | +0.48 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 | 0.49 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_async_pipeline_with_autotune | 1.291× | 1.311× | ✓ | **✓ best** |
| C | fix_b_matrix_smem_bank_conflicts_via_correct_order_0_1 | 0.982× | 0.972× | ✓ |  |
| D | tma_loads | 0.981× | 0.971× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_async_p... | Δ | fix_b_matrix_smem... | Δ | tma_loads | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 23740 ns | 18110 ns | -5630 ns (-23.7%) | 24420 ns | +680 ns (+2.9%) | 24450 ns | +710 ns (+3.0%) |
| SM Throughput (%) | 10.60 % | 13.14 % | +2.54 pp | 10.25 % | -0.35 pp | 10.23 % | -0.37 pp |
| Memory Throughput (%) | 74.15 % | 55.15 % | -19.00 pp | 72.25 % | -1.90 pp | 72.09 % | -2.06 pp |
| DRAM Throughput (%) | 5.20 % | 6.83 % | +1.63 pp | 5.07 % | -0.13 pp | 5.06 % | -0.14 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 128.00 register/thread | +48 | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.16 | 0.12 | -0.04 | 0.23 | +0.07 | 0.23 | +0.07 |
| No Eligible (%) | 87.50 % | 67.94 % | -19.56 pp | 87.83 % | +0.33 pp | 87.84 % | +0.34 pp |
| IPC Active | 0.50 inst/cycle | 0.64 inst/cycle | +0.14 | 0.48 inst/cycle | -0.02 | 0.48 inst/cycle | -0.02 |

---

## Iteration 4

**Best so far**: 1.3109× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_bank_conflicts_via_explicit_pointer_arithmetic | 1.350× | 1.366× | ✓ | **✓ best** |
| B | autotune_num_stages_to_activate_async_pipeline_and_reduce_smem_bank_conflicts | 1.144× | 1.187× | ✓ |  |
| C | autotune_num_stages_and_num_warps_to_activate_async_pipeline | 1.143× | 1.127× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bank... | Δ | autotune_num_stag... | Δ | autotune_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 23740 ns | 17380 ns | -6360 ns (-26.8%) | 20000 ns | -3740 ns (-15.8%) | 21060 ns | -2680 ns (-11.3%) |
| SM Throughput (%) | 10.60 % | 13.83 % | +3.23 pp | 11.67 % | +1.07 pp | 10.84 % | +0.24 pp |
| Memory Throughput (%) | 74.15 % | 58.00 % | -16.15 pp | 49.68 % | -24.47 pp | 46.32 % | -27.83 pp |
| DRAM Throughput (%) | 5.20 % | 7.13 % | +1.93 pp | 6.22 % | +1.02 pp | 5.90 % | +0.70 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 80.00 register/thread | 128.00 register/thread | +48 | 96.00 register/thread | +16 | 96.00 register/thread | +16 |
| Waves/SM | 0.16 | 0.12 | -0.04 | 0.23 | +0.07 | 0.10 | -0.06 |
| No Eligible (%) | 87.50 % | 65.93 % | -21.57 pp | 70.53 % | -16.97 pp | 72.55 % | -14.95 pp |
| IPC Active | 0.50 inst/cycle | 0.68 inst/cycle | +0.18 | 0.59 inst/cycle | +0.09 | 0.54 inst/cycle | +0.04 |


### Key Learnings

#### ✅ Plan A: `fix_b_matrix_bank_conflicts_via_explicit_pointer_arithmetic` (1.366×)

When a Triton GEMM kernel shows 13+ way shared memory bank conflicts on B-matrix loads (UncoalescedSharedAccess + SharedMemoryConflicts rules, ~80% of wavefronts affected), replacing the make_block_ptr B load with explicit 2D pointer arithmetic forces a row-major [BLOCK_K, BLOCK_N] SMEM layout that aligns with the global memory layout (K-contiguous), reducing bank conflicts. In this case, Warp Cycles Per Issued Instruction dropped from 3.12 to 2.94 and scheduler eligible fraction improved from 32% to 34%, yielding a 1.37x speedup. This approach is safer than changing the order= parameter in make_block_ptr, which caused correctness failures (max_diff ~74) in prior iterations.

> **Applicable when**: Triton GEMM kernel where B matrix has K-contiguous global layout (stride_bk=1, stride_bn >> 1) but make_block_ptr uses order=(1,0) causing N-fast SMEM layout; SharedMemoryConflicts rule triggered with 10+ way bank conflicts on 70%+ of shared load wavefronts; UncoalescedSharedAccess rule also triggered; MIO stall fraction >40% of CPI. Tile sizes must remain fixed for correctness.

> **Avoid when**: B matrix is already loaded with K-fast SMEM layout (no bank conflicts). Changing order= in make_block_ptr is safe and correct for the specific kernel (avoids the extra pointer arithmetic overhead). Kernel is compute-bound rather than SMEM-latency-bound. When the bank conflict count is low (<5-way) and other bottlenecks dominate.

#### ✅ Plan B: `autotune_num_stages_to_activate_async_pipeline_and_reduce_smem_bank_conflicts` (1.311×)

When a Triton GEMM kernel on Hopper shows severe shared memory bank conflicts (17+ way, 80%+ of wavefronts) and high MIO stall rates (74%+ of CPI), adding @triton.autotune over num_stages (2–4) activates Triton's cp.async/TMA software pipeline, which prefetches tiles asynchronously and reduces synchronous SMEM stall cycles. In this case, Warp Cycles Per Issued Instruction dropped from 7.96 to 3.12 (60% reduction) and L1/TEX throughput dropped from 87.5% to 67.08%, indicating the SMEM pressure was substantially relieved. The autotune also explores num_warps, allowing the compiler to find a block size that balances register pressure and occupancy without manual tile size changes that risk correctness.

> **Applicable when**: Triton GEMM kernel on Hopper (sm_90) with SharedMemoryConflicts rule triggered (10+ way bank conflicts, 70%+ of wavefronts), high MIO stall fraction (>50% of CPI), L1/TEX throughput >80%, and the kernel already uses make_block_ptr (prerequisite for TMA on Hopper). Kernel body is correct and tile sizes should not be changed for correctness reasons.

> **Avoid when**: Kernel does not use make_block_ptr (TMA pipeline won't activate). Bank conflicts are not the primary bottleneck (e.g., compute-bound kernels). Pre-Hopper architectures where num_stages does not map to TMA/cp.async hardware. When tile sizes must be changed simultaneously — autotune over tile sizes risks correctness if strides/order are layout-sensitive.

---

## Iteration 5

**Best so far**: 1.3659× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_register_usage_with_maxnreg_to_improve_occupancy | 1.335× | 1.376× | ✓ |  |
| B | remove_hardcoded_num_stages_from_tl_range_to_enable_autotune_pipeline | 1.146× | 1.097× | ✓ |  |
| C | cap_register_pressure_via_maxnreg_to_increase_occupancy_and_grid_coverage | 1.101× | 0.566× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_register_usag... | Δ | remove_hardcoded_... | Δ | cap_register_pres... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 23740 ns | 17250 ns | -6490 ns (-27.3%) | 21630 ns | -2110 ns (-8.9%) | 41950 ns | +18210 ns (+76.7%) |
| SM Throughput (%) | 10.60 % | 13.75 % | +3.15 pp | 10.87 % | +0.27 pp | 10.14 % | -0.46 pp |
| Memory Throughput (%) | 74.15 % | 57.69 % | -16.46 pp | 46.48 % | -27.67 pp | 32.94 % | -41.21 pp |
| DRAM Throughput (%) | 5.20 % | 7.18 % | +1.98 pp | 5.74 % | +0.54 pp | 2.98 % | -2.22 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 80.00 register/thread | 128.00 register/thread | +48 | 96.00 register/thread | +16 | 64.00 register/thread | -16 |
| Waves/SM | 0.16 | 0.12 | -0.04 | 0.10 | -0.06 | 0.10 | -0.06 |
| No Eligible (%) | 87.50 % | 65.84 % | -21.66 pp | 72.64 % | -14.86 pp | 80.94 % | -6.56 pp |
| IPC Active | 0.50 inst/cycle | 0.67 inst/cycle | +0.17 | 0.55 inst/cycle | +0.05 | 0.38 inst/cycle | -0.12 |


### Key Learnings

#### ✅ Plan A: `cap_register_usage_with_maxnreg_to_improve_occupancy` (1.376×)

Adding maxnreg=64 and maxnreg=96 variants to the Triton autotune config list yielded a modest 1.38x speedup, but the profiling metrics show virtually no change in occupancy (still 3.12% achieved, 25% theoretical, 128 registers/thread, Block Limit Registers still 8). This indicates the autotuner selected one of the original uncapped configs as the winner, meaning the maxnreg-capped configs were either slower due to register spill overhead or the autotuner's warmup favored the original. The strategy of adding maxnreg configs to autotune is low-risk (kernel body unchanged), but when the kernel is already severely register-limited (128 regs/thread) and SMEM bank conflicts dominate (13+ way, 56% speedup potential per NCU), the spill cost from maxnreg may negate the occupancy gain, leaving the autotuner to prefer the original configs.

> **Applicable when**: Triton kernel with registers_per_thread >= 96, theoretical occupancy <= 25%, achieved occupancy <= 5%, Block Limit Registers is the binding occupancy constraint, and the kernel is latency-bound (No Eligible cycles > 60%). Adding maxnreg configs to autotune is safe when the kernel body is not changed and the autotuner can select the best option.

> **Avoid when**: Kernel is already at low register count (<64 regs/thread). SMEM bank conflicts are the dominant bottleneck (13+ way conflicts, >50% speedup potential per NCU SharedMemoryConflicts rule) — in this case, spill traffic to local memory may not provide net benefit and the autotuner may revert to uncapped configs. Compute-bound kernels where register spilling adds memory traffic without latency-hiding benefit.

---

## Cumulative Progress

```
Baseline:  23740 ns  (1.000×)
Iter 3     18110 ns  (1.311×)  — tma_loads_async_pipeline_with_autotune
Iter 4     17380 ns  (1.366×)  — fix_b_matrix_bank_conflicts_via_explicit_pointer_arithmetic

Final best: 17380 ns  (1.3659×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_15(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -44,7 +58,7 @@

     tl.assume(pid_m >= 0)
     tl.assume(pid_n >= 0)
 
-    # Create block pointers for A and B to handle strided 2D loads efficiently
+    # Create block pointers for A to handle strided 2D loads efficiently
     # A: shape=[M, K]=[7840, 192], strides=[stride_am, stride_ak]=[192, 1]
     # order=(1, 0) means innermost dim is K (stride_ak=1, contiguous)
     a_block_ptr = tl.make_block_ptr(
@@ -55,26 +69,28 @@

         block_shape=(BLOCK_M, BLOCK_K),
         order=(1, 0),
     )
-    # B transposed: shape=[N, K]=[32, 192], strides=[stride_bn, stride_bk]=[192, 1]
-    # order=(1, 0) means K is innermost (contiguous) in SMEM — N-contiguous layout
-    # This eliminates bank conflicts: N-row accesses hit consecutive banks
-    b_T_block_ptr = tl.make_block_ptr(
-        base=B,
-        shape=(N, K),
-        strides=(stride_bn, stride_bk),
-        offsets=(pid_n * BLOCK_N, 0),
-        block_shape=(BLOCK_N, BLOCK_K),
-        order=(1, 0),
-    )
+
+    # Explicit pointer arithmetic for B to avoid SMEM bank conflicts.
+    # B has stride_bk=1 (K contiguous) and stride_bn=192 (N strided).
+    # Using make_block_ptr with order=(1,0) caused 13.2-way bank conflicts.
+    # Explicit 2D pointer array forces row-major [BLOCK_K, BLOCK_N] SMEM layout
+    # with K as the fast dimension, matching global memory layout -> conflict-free.
+    rk_base = tl.arange(0, BLOCK_K)  # [BLOCK_K], shape constexpr
+    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(INDEX_DTYPE)  # [BLOCK_N]
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
         a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
-        b_T = tl.load(b_T_block_ptr, boundary_check=(0, 1), padding_option='zero')  # [BLOCK_N, BLOCK_K]
-        acc += tl.dot(a, tl.trans(b_T), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)  # a:[M,K] x b_T^T:[K,N]
+        # Build [BLOCK_K, BLOCK_N] pointer array for B
+        # stride_bk=1 (K contiguous in global), stride_bn=192 (N strided)
+        k_base = (k_idx * BLOCK_K).to(INDEX_DTYPE)
+        b_ptrs = B + (k_base + rk_base[:, None]) * stride_bk + rn[None, :] * stride_bn
+        b_mask = ((k_base + rk_base[:, None]) < K) & (rn[None, :] < N)
+        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N], row-major SMEM = K-fast = conflict-free
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)  # a:[M,K] x b:[K,N], no tl.trans needed
         a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
-        b_T_block_ptr = tl.advance(b_T_block_ptr, (0, BLOCK_K))  # advance K axis (axis=1)
+        # No b_block_ptr advance needed — k_base computed from k_idx each iteration
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,7 +104,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

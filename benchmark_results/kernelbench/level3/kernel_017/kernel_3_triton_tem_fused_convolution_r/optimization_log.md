# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2670000 ns — **Best latency**: 1130000 ns (**2.3628×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_shared_memory_bank_conflicts_via_b_matrix_order_and_num_warps_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_relu_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_017/kernel_3_triton_tem_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131072,1,1), Waves/SM: 496.48
- **Registers/thread**: 238
- **Duration**: 2670000 ns
- **SM Throughput**: 20.27 % | **Memory**: 80.58 % | **DRAM**: 17.70 %
- **Achieved Occupancy**: 12.13 %
- **IPC Active**: 0.81 inst/cycle | **No Eligible**: 79.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ | **✓ best** |
| B | reduce_register_pressure_and_shared_memory_bank_conflicts_via_autotune | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2670000 ns | 2670000 ns | +0 ns (+0.0%) | 2670000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 20.27 % | 20.27 % | +0.00 pp | 20.27 % | +0.00 pp |
| Memory Throughput (%) | 80.58 % | 80.55 % | -0.03 pp | 80.54 % | -0.04 pp |
| DRAM Throughput (%) | 17.70 % | 17.70 % | +0.00 pp | 17.70 % | +0.00 pp |
| Achieved Occupancy (%) | 12.13 % | 12.13 % | +0.00 pp | 12.13 % | +0.00 pp |
| Registers/Thread | 238.00 register/thread | 238.00 register/thread | +0 | 238.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 79.62 % | 79.63 % | +0.01 pp | 79.62 % | +0.00 pp |
| IPC Active | 0.81 inst/cycle | 0.81 inst/cycle | +0.00 | 0.81 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr | 0.914× | 0.914× | ✓ |  |
| D | tma_loads_with_block_ptr_for_mio_stall_reduction | 1.904× | 1.328× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ | tma_loads_with_bl... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2670000 ns | 2920000 ns | +250000 ns (+9.4%) | 2010000 ns | -660000 ns (-24.7%) |
| SM Throughput (%) | 20.27 % | 18.77 % | -1.50 pp | 25.52 % | +5.25 pp |
| Memory Throughput (%) | 80.58 % | 73.61 % | -6.97 pp | 94.86 % | +14.28 pp |
| DRAM Throughput (%) | 17.70 % | 16.16 % | -1.54 pp | 23.46 % | +5.76 pp |
| Achieved Occupancy (%) | 12.13 % | 12.31 % | +0.18 pp | 30.40 % | +18.27 pp |
| Registers/Thread | 238.00 register/thread | 238.00 register/thread | +0 | 96.00 register/thread | -142 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 198.59 | -297.89 |
| No Eligible (%) | 79.62 % | 80.90 % | +1.28 pp | 74.45 % | -5.17 pp |
| IPC Active | 0.81 inst/cycle | 0.76 inst/cycle | -0.05 | 1.02 inst/cycle | +0.21 |

---

## Iteration 3

**Best so far**: 1.3284× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_b_matrix_order_swap_and_autotune_num_stages | 1.903× | 1.328× | ✓ |  |
| B | fix_shared_memory_bank_conflicts_via_b_matrix_order_and_num_warps_autotune | 2.345× | 2.363× | ✓ | **✓ best** |
| C | replace_tl_dot_with_direct_k_loop_accumulation_for_tiny_k | 0.956× | 0.954× | ✓ |  |
| D | autotune_num_warps_and_num_stages_for_tma_pipeline_and_latency_hiding | 0.914× | 0.914× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | fix_shared_memory... | Δ | replace_tl_dot_wi... | Δ | autotune_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2670000 ns | 2010000 ns | -660000 ns (-24.7%) | 1130000 ns | -1540000 ns (-57.7%) | 2800000 ns | +130000 ns (+4.9%) | 2920000 ns | +250000 ns (+9.4%) |
| SM Throughput (%) | 20.27 % | 25.51 % | +5.24 pp | 43.06 % | +22.79 pp | 39.52 % | +19.25 pp | 18.75 % | -1.52 pp |
| Memory Throughput (%) | 80.58 % | 94.84 % | +14.26 pp | 64.63 % | -15.95 pp | 79.21 % | -1.37 pp | 73.51 % | -7.07 pp |
| DRAM Throughput (%) | 17.70 % | 23.47 % | +5.77 pp | 41.68 % | +23.98 pp | 16.90 % | -0.80 pp | 16.17 % | -1.53 pp |
| Achieved Occupancy (%) | 12.13 % | 30.41 % | +18.28 pp | 12.06 % | -0.07 pp | 12.15 % | +0.02 pp | 12.31 % | +0.18 pp |
| Registers/Thread | 238.00 register/thread | 96.00 register/thread | -142 | 202.00 register/thread | -36 | 189.00 register/thread | -49 | 238.00 register/thread | +0 |
| Waves/SM | 496.48 | 198.59 | -297.89 | 124.12 | -372.36 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 79.62 % | 74.45 % | -5.17 pp | 56.45 % | -23.17 pp | 60.42 % | -19.20 pp | 80.91 % | +1.29 pp |
| IPC Active | 0.81 inst/cycle | 1.02 inst/cycle | +0.21 | 1.72 inst/cycle | +0.91 | 1.58 inst/cycle | +0.77 | 0.76 inst/cycle | -0.05 |

---

## Iteration 4

**Best so far**: 2.3628× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_register_count_via_maxnreg_to_improve_occupancy | 1.898× | 1.328× | ✓ |  |
| D | fix_b_matrix_block_ptr_order_to_eliminate_smem_bank_conflicts | 0.914× | 0.914× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_register_coun... | Δ | fix_b_matrix_bloc... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2670000 ns | 2010000 ns | -660000 ns (-24.7%) | 2920000 ns | +250000 ns (+9.4%) |
| SM Throughput (%) | 20.27 % | 25.51 % | +5.24 pp | 18.73 % | -1.54 pp |
| Memory Throughput (%) | 80.58 % | 94.84 % | +14.26 pp | 73.45 % | -7.13 pp |
| DRAM Throughput (%) | 17.70 % | 23.47 % | +5.77 pp | 16.16 % | -1.54 pp |
| Achieved Occupancy (%) | 12.13 % | 30.40 % | +18.27 pp | 12.31 % | +0.18 pp |
| Registers/Thread | 238.00 register/thread | 96.00 register/thread | -142 | 238.00 register/thread | +0 |
| Waves/SM | 496.48 | 198.59 | -297.89 | 496.48 | +0.00 |
| No Eligible (%) | 79.62 % | 74.45 % | -5.17 pp | 80.89 % | +1.27 pp |
| IPC Active | 0.81 inst/cycle | 1.02 inst/cycle | +0.21 | 0.76 inst/cycle | -0.05 |


### Key Learnings

#### ✅ Plan A: `cap_register_count_via_maxnreg_to_improve_occupancy` (1.328×)

Adding maxnreg constraints (e.g., 64, 96, 128) to triton.Config entries in @triton.autotune forces the compiler to spill excess registers to L2, reducing registers/thread from 202 to 96 and lifting theoretical occupancy from 12.5% to 31.25% (active warps/scheduler from 1.94 to 4.86). However, the net effect on this kernel was a regression in elapsed cycles (1.70M → 3.02M, ~1.78x slower per-kernel), because the increased warp count amplified pre-existing shared memory bank conflicts (11-way, 74% excessive wavefronts) and MIO stalls (Warp Cycles Per Issued Instruction jumped from 4.46 to 19.01), overwhelming the latency-hiding benefit. The 1.33x end-to-end speedup reported reflects a different baseline comparison context, not a per-kernel improvement — the kernel itself ran slower after maxnreg was applied. When shared memory bank conflicts are the dominant bottleneck, increasing occupancy via maxnreg amplifies conflict pressure rather than hiding it.

> **Applicable when**: Kernel has very high registers/thread (>150), low theoretical occupancy (<20%), Block Limit Registers <= 3, high no-eligible cycles (>60%), AND shared memory bank conflicts are low (<3-way average). The L2 hit rate should be high (>85%) so register spills are served cheaply from L2 rather than DRAM.

> **Avoid when**: Kernel already has significant shared memory bank conflicts (>5-way average, >50% excessive wavefronts) — increasing occupancy via maxnreg will amplify MIO pipeline pressure and cause Warp Cycles Per Issued Instruction to spike (here from 4.46 to 19.01), making the kernel slower despite higher occupancy. Also avoid when the kernel is already at a good occupancy level (>25%) or when the primary bottleneck is compute rather than latency hiding.

---

## Iteration 5

**Best so far**: 2.3628× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_smem_order_to_column_major_to_eliminate_bank_conflicts | 1.907× | 1.328× | ✓ |  |
| B | tma_loads_via_tensor_descriptor_api | 1.911× | 1.328× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_smem... | Δ | tma_loads_via_ten... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2670000 ns | 2010000 ns | -660000 ns (-24.7%) | 2010000 ns | -660000 ns (-24.7%) |
| SM Throughput (%) | 20.27 % | 25.51 % | +5.24 pp | 25.51 % | +5.24 pp |
| Memory Throughput (%) | 80.58 % | 94.85 % | +14.27 pp | 94.86 % | +14.28 pp |
| DRAM Throughput (%) | 17.70 % | 23.47 % | +5.77 pp | 23.47 % | +5.77 pp |
| Achieved Occupancy (%) | 12.13 % | 30.40 % | +18.27 pp | 30.40 % | +18.27 pp |
| Registers/Thread | 238.00 register/thread | 96.00 register/thread | -142 | 96.00 register/thread | -142 |
| Waves/SM | 496.48 | 198.59 | -297.89 | 198.59 | -297.89 |
| No Eligible (%) | 79.62 % | 74.45 % | -5.17 pp | 74.45 % | -5.17 pp |
| IPC Active | 0.81 inst/cycle | 1.02 inst/cycle | +0.21 | 1.02 inst/cycle | +0.21 |


### Key Learnings

#### ❌ Plan A: `fix_b_matrix_smem_order_to_column_major_to_eliminate_bank_conflicts` (1.328×)

Changing the B matrix tl.make_block_ptr from order=(1,0) to order=(0,1) on a Hopper (sm_90) Triton kernel produced zero measurable change: all profiling metrics are statistically identical before and after (Duration 2.01ms, L1/TEX throughput 95%, bank conflict way-count 11.0, conflict wavefront pct 73.97%, No Eligible warps 74.45%, elapsed cycles 3.020M vs 3.020M). The SharedMemoryConflicts rule still fires with identical severity (70.27% speedup potential) and UncoalescedSharedAccess still reports 68% excessive wavefronts. This confirms that on Hopper, Triton's TMA/wgmma backend normalizes or ignores the user-specified order parameter for tl.make_block_ptr when generating wgmma shared memory layouts — the compiler internally selects the shared memory layout required by the wgmma instruction regardless of the order hint, so neither order=(0,1) nor order=(1,0) can fix the bank conflict pattern on this architecture.

> **Applicable when**: Never: this change has no measurable effect on Hopper TMA kernels in either direction.

> **Avoid when**: Any Triton kernel using tl.make_block_ptr on Hopper (sm_90+) where the kernel uses TMA/wgmma paths. Swapping the B matrix order parameter between (0,1) and (1,0) does not reduce shared memory bank conflicts — all metrics remain identical. The SharedMemoryConflicts rule (11-way, 70% speedup potential) persists unchanged after this modification in both directions. Do not waste optimization budget on order parameter swaps for B matrix block pointers on Hopper.

---

## Cumulative Progress

```
Baseline:  2670000 ns  (1.000×)
Iter 2     2010000 ns  (1.328×)  — tma_loads_with_block_ptr_for_mio_stall_reduction
Iter 3     1130000 ns  (2.363×)  — fix_shared_memory_bank_conflicts_via_b_matrix_order_and_num_warps_autotune

Final best: 1130000 ns  (2.3628×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,47 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Original configs without maxnreg (baseline)
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=5),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=5),
+        # maxnreg=128: moderate spill, ~2x occupancy
+        triton.Config({}, num_warps=1, num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=2, num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=4, num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=8, num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=1, num_stages=3, maxnreg=128),
+        triton.Config({}, num_warps=4, num_stages=3, maxnreg=128),
+        triton.Config({}, num_warps=8, num_stages=3, maxnreg=128),
+        # maxnreg=96: aggressive spill, ~3x occupancy
+        triton.Config({}, num_warps=1, num_stages=2, maxnreg=96),
+        triton.Config({}, num_warps=2, num_stages=2, maxnreg=96),
+        triton.Config({}, num_warps=4, num_stages=2, maxnreg=96),
+        triton.Config({}, num_warps=8, num_stages=2, maxnreg=96),
+        triton.Config({}, num_warps=1, num_stages=3, maxnreg=96),
+        triton.Config({}, num_warps=4, num_stages=3, maxnreg=96),
+        triton.Config({}, num_warps=8, num_stages=3, maxnreg=96),
+        # maxnreg=64: maximum spill, ~4x occupancy (may be spill-bound)
+        triton.Config({}, num_warps=1, num_stages=2, maxnreg=64),
+        triton.Config({}, num_warps=2, num_stages=2, maxnreg=64),
+        triton.Config({}, num_warps=4, num_stages=2, maxnreg=64),
+        triton.Config({}, num_warps=1, num_stages=3, maxnreg=64),
+        triton.Config({}, num_warps=4, num_stages=3, maxnreg=64),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_tem_fused_convolution_relu_3(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
@@ -44,54 +85,56 @@

     tl.assume(pid_m >= 0)
     tl.assume(pid_n >= 0)
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Create TMA block pointers for A and B before the loop
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    B_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),   # row-major: N is fast dim in smem -> no bank conflicts
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
+    # Load bias using block pointer (1D coalesced load, then broadcast)
+    bias_block_ptr = tl.make_block_ptr(
+        base=in_ptr0,
+        shape=(N,),
+        strides=(1,),
+        offsets=(pid_n * BLOCK_N,),
+        block_shape=(BLOCK_N,),
+        order=(0,),
+    )
+    bias = tl.load(bias_block_ptr, boundary_check=(0,), padding_option='zero')
+    tmp0 = tl.broadcast_to(bias[None, :], (BLOCK_M, BLOCK_N))
+    tmp1 = acc + tmp0
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 6*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 64*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 6*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
-
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
-
-    # inductor generates a suffix
-    xindex = idx_n + 64*idx_m
-    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp1 = acc + tmp0
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
+    # Store output using block pointer
+    out_block_ptr = tl.make_block_ptr(
+        base=out_ptr0,
+        shape=(M, N),
+        strides=(N, 1),
+        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
+        block_shape=(BLOCK_M, BLOCK_N),
+        order=(1, 0),
+    )
+    tl.store(out_block_ptr, tmp1, boundary_check=(0, 1))
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

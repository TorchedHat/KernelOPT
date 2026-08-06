# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_32` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 22850 ns — **Best latency**: 15330 ns (**1.4905×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`autotune_block_n_block_k_as_constexpr_kwargs_to_find_optimal_tile`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_32`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_32_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (44,1,1), Waves/SM: 0.04
- **Registers/thread**: 56
- **Duration**: 22850 ns
- **SM Throughput**: 2.61 % | **Memory**: 25.68 % | **DRAM**: 1.96 %
- **Achieved Occupancy**: 5.98 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 90.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_tile_size_right_sizing | 1.730× | 1.940× | ✓ | **✓ best** |
| C | memory_coalescing_tile_reshape_for_small_M_gemm | 0.943× | 0.836× | ✓ |  |
| D | autotune_tile_sizes_for_underutilized_small_gemm | 1.397× | 1.433× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_with_ti... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22850 ns | 11780 ns | -11070 ns (-48.4%) | 27330 ns | +4480 ns (+19.6%) | 15940 ns | -6910 ns (-30.2%) |
| SM Throughput (%) | 2.61 % | 1.89 % | -0.72 pp | 3.71 % | +1.10 pp | 3.41 % | +0.80 pp |
| Memory Throughput (%) | 25.68 % | 13.50 % | -12.18 pp | 22.74 % | -2.94 pp | 21.23 % | -4.45 pp |
| DRAM Throughput (%) | 1.96 % | 3.81 % | +1.85 pp | 1.65 % | -0.31 pp | 2.85 % | +0.89 pp |
| Achieved Occupancy (%) | 5.98 % | 3.12 % | -2.86 pp | 6.18 % | +0.20 pp | 6.25 % | +0.27 pp |
| Registers/Thread | 56.00 register/thread | 42.00 register/thread | -14 | 78.00 register/thread | +22 | 120.00 register/thread | +64 |
| Waves/SM | 0.04 | 0.03 | -0.01 | 0.06 | +0.02 | 0.33 | +0.29 |
| No Eligible (%) | 90.99 % | 85.69 % | -5.30 pp | 87.63 % | -3.36 pp | 87.71 % | -3.28 pp |
| IPC Active | 0.34 inst/cycle | 0.29 inst/cycle | -0.05 | 0.49 inst/cycle | +0.15 | 0.49 inst/cycle | +0.15 |

---

## Iteration 2

**Best so far**: 1.9397× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_shared_memory_bank_conflicts_via_larger_block_k_and_block_n | 1.482× | 1.347× | ✓ |  |
| B | block_ptr_loads_to_eliminate_shared_memory_bank_conflicts | 1.414× | 1.509× | ✓ |  |
| C | block_ptr_loads_with_block_m_right_sizing_to_eliminate_mio_stalls | 1.769× | 1.165× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ | block_ptr_loads_t... | Δ | block_ptr_loads_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22850 ns | 16960 ns | -5890 ns (-25.8%) | 15140 ns | -7710 ns (-33.7%) | 19620 ns | -3230 ns (-14.1%) |
| SM Throughput (%) | 2.61 % | 2.35 % | -0.26 pp | 3.12 % | +0.51 pp | 1.46 % | -1.15 pp |
| Memory Throughput (%) | 25.68 % | 18.67 % | -7.01 pp | 23.34 % | -2.34 pp | 15.48 % | -10.20 pp |
| DRAM Throughput (%) | 1.96 % | 2.64 % | +0.68 pp | 2.98 % | +1.02 pp | 2.28 % | +0.32 pp |
| Achieved Occupancy (%) | 5.98 % | 6.25 % | +0.27 pp | 6.25 % | +0.27 pp | 6.25 % | +0.27 pp |
| Registers/Thread | 56.00 register/thread | 41.00 register/thread | -15 | 108.00 register/thread | +52 | 90.00 register/thread | +34 |
| Waves/SM | 0.04 | 0.06 | +0.02 | 0.33 | +0.29 | 0.07 | +0.03 |
| No Eligible (%) | 90.99 % | 92.03 % | +1.04 pp | 88.40 % | -2.59 pp | 95.20 % | +4.21 pp |
| IPC Active | 0.34 inst/cycle | 0.32 inst/cycle | -0.02 | 0.47 inst/cycle | +0.13 | 0.19 inst/cycle | -0.15 |

---

## Iteration 3

**Best so far**: 1.9397× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_shared_memory_to_increase_occupancy_via_smaller_block_k_n | 1.817× | 1.077× | ✓ | **✓ best** |
| B | eliminate_k_loop_bank_conflicts_via_full_k_single_tile | 1.438× | 1.457× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ | eliminate_k_loop_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 22850 ns | 21220 ns | -1630 ns (-7.1%) | 15680 ns | -7170 ns (-31.4%) |
| SM Throughput (%) | 2.61 % | 0.88 % | -1.73 pp | 2.54 % | -0.07 pp |
| Memory Throughput (%) | 25.68 % | 7.53 % | -18.15 pp | 20.50 % | -5.18 pp |
| DRAM Throughput (%) | 1.96 % | 2.12 % | +0.16 pp | 2.87 % | +0.91 pp |
| Achieved Occupancy (%) | 5.98 % | 1.56 % | -4.42 pp | 6.25 % | +0.27 pp |
| Registers/Thread | 56.00 register/thread | 78.00 register/thread | +22 | 43.00 register/thread | -13 |
| Waves/SM | 0.04 | 0.01 | -0.03 | 0.11 | +0.07 |
| No Eligible (%) | 90.99 % | 86.61 % | -4.38 pp | 91.04 % | +0.05 pp |
| IPC Active | 0.34 inst/cycle | 0.13 inst/cycle | -0.21 | 0.36 inst/cycle | +0.02 |

---

## Iteration 4

**Best so far**: 1.9397× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_block_n_block_k_to_fill_sms_and_break_bank_conflict_pattern | 2.084× | 2.334× | ✓ | **✓ best** |
| B | tma_loads_warp_specialization_to_overlap_async_copy_with_mma | 1.438× | 1.467× | ✓ |  |
| C | simplify_indexing_to_cut_register_pressure_and_raise_occupancy | 1.746× | 0.976× | ✓ |  |
| D | simplify_indexing_to_cut_register_pressure_and_smem_usage | 1.747× | 0.972× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_block_n_... | Δ | tma_loads_warp_sp... | Δ | simplify_indexing... | Δ | simplify_indexing... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22850 ns | 9790 ns | -13060 ns (-57.2%) | 15580 ns | -7270 ns (-31.8%) | 23420 ns | +570 ns (+2.5%) | 23520 ns | +670 ns (+2.9%) |
| SM Throughput (%) | 2.61 % | 1.59 % | -1.02 pp | 2.56 % | -0.05 pp | 0.76 % | -1.85 pp | 0.76 % | -1.85 pp |
| Memory Throughput (%) | 25.68 % | 15.29 % | -10.39 pp | 20.68 % | -5.00 pp | 8.87 % | -16.81 pp | 8.85 % | -16.83 pp |
| DRAM Throughput (%) | 1.96 % | 4.58 % | +2.62 pp | 2.89 % | +0.93 pp | 1.91 % | -0.05 pp | 1.91 % | -0.05 pp |
| Achieved Occupancy (%) | 5.98 % | 3.12 % | -2.86 pp | 6.25 % | +0.27 pp | 1.56 % | -4.42 pp | 1.56 % | -4.42 pp |
| Registers/Thread | 56.00 register/thread | 32.00 register/thread | -24 | 43.00 register/thread | -13 | 61.00 register/thread | +5 | 61.00 register/thread | +5 |
| Waves/SM | 0.04 | 0.01 | -0.03 | 0.11 | +0.07 | 0.01 | -0.03 | 0.01 | -0.03 |
| No Eligible (%) | 90.99 % | 87.48 % | -3.51 pp | 91.11 % | +0.12 pp | 89.66 % | -1.33 pp | 89.73 % | -1.26 pp |
| IPC Active | 0.34 inst/cycle | 0.25 inst/cycle | -0.09 | 0.36 inst/cycle | +0.02 | 0.10 inst/cycle | -0.24 | 0.10 inst/cycle | -0.24 |

---

## Iteration 5

**Best so far**: 2.334× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | break_smem_bank_conflicts_via_block_k16_and_wider_block_n | 1.996× | 1.940× | ✓ |  |
| C | reduce_block_n_k_to_32_cut_smem_and_raise_occupancy | 2.108× | 2.310× | ✓ |  |
| D | autotune_block_n_block_k_as_constexpr_kwargs_to_find_optimal_tile | 2.113× | 1.490× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | break_smem_bank_c... | Δ | reduce_block_n_k_... | Δ | autotune_block_n_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22850 ns | 11780 ns | -11070 ns (-48.4%) | 9890 ns | -12960 ns (-56.7%) | 15330 ns | -7520 ns (-32.9%) |
| SM Throughput (%) | 2.61 % | 1.22 % | -1.39 pp | 1.62 % | -0.99 pp | 0.69 % | -1.92 pp |
| Memory Throughput (%) | 25.68 % | 4.22 % | -21.46 pp | 15.57 % | -10.11 pp | 5.25 % | -20.43 pp |
| DRAM Throughput (%) | 1.96 % | 3.80 % | +1.84 pp | 4.54 % | +2.58 pp | 2.92 % | +0.96 pp |
| Achieved Occupancy (%) | 5.98 % | 1.56 % | -4.42 pp | 3.12 % | -2.86 pp | 1.56 % | -4.42 pp |
| Registers/Thread | 56.00 register/thread | 51.00 register/thread | -5 | 32.00 register/thread | -24 | 62.00 register/thread | +6 |
| Waves/SM | 0.04 | 0.01 | -0.03 | 0.01 | -0.03 | 0.01 | -0.03 |
| No Eligible (%) | 90.99 % | 81.55 % | -9.44 pp | 87.49 % | -3.50 pp | 90.27 % | -0.72 pp |
| IPC Active | 0.34 inst/cycle | 0.18 inst/cycle | -0.16 | 0.25 inst/cycle | -0.09 | 0.10 inst/cycle | -0.24 |

---

## Cumulative Progress

```
Baseline:  22850 ns  (1.000×)
Iter 1     11780 ns  (1.940×)  — tma_loads_with_tile_size_right_sizing
Iter 4     9790 ns  (2.334×)  — autotune_block_n_block_k_to_fill_sms_and_break_bank_conflict_pattern

Final best: 15330 ns  (1.4905×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,15 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=1, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=1, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=1, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_32(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
     BLOCK_K : tl.constexpr = 32
-    GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
     A = arg_A
@@ -31,50 +48,39 @@

     stride_bn = 384
 
     # based on triton.ops.matmul
-    pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    pid = tl.program_id(0)
+    grid_n = tl.cdiv(N, BLOCK_N)
+    pid_m = pid // grid_n
+    pid_n = pid % grid_n
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    # Create block pointers BEFORE the loop
+    # A: shape (M=2, K=384), strides (stride_am=384, stride_ak=1)
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),  # row-major: K is fast dim
+    )
+    # B: shape (K=384, N=1408), strides (stride_bk=1, stride_bn=384)
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
 
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
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 384*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1408*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+    for _ in range(0, tl.cdiv(K, BLOCK_K)):
+        a = tl.load(a_block_ptr, boundary_check=(0, 1))
+        b = tl.load(b_block_ptr, boundary_check=(0, 1))
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -104,8 +110,6 @@

     tmp16 = tl.maximum(tmp15, tmp14)
     tmp17 = (tmp16 / tmp8)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1408*idx_m, [BLOCK_M, BLOCK_N])), tmp17, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

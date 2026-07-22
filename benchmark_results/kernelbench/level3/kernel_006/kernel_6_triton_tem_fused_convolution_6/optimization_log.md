# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5220000 ns — **Best latency**: 5570000 ns (**0.9372×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_6_triton_tem_fused_convolution_6/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (11760,1,1), Waves/SM: 29.70
- **Registers/thread**: 168
- **Duration**: 5220000 ns
- **SM Throughput**: 39.75 % | **Memory**: 89.66 % | **DRAM**: 5.22 %
- **Achieved Occupancy**: 18.61 %
- **IPC Active**: 1.61 inst/cycle | **No Eligible**: 59.83 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ |  |
| D | add_autotune_reduce_register_pressure_and_bank_conflicts | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5220000 ns | 5220000 ns | +0 ns (+0.0%) | 5220000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 39.75 % | 39.76 % | +0.01 pp | 39.77 % | +0.02 pp |
| Memory Throughput (%) | 89.66 % | 89.67 % | +0.01 pp | 89.69 % | +0.03 pp |
| DRAM Throughput (%) | 5.22 % | 5.23 % | +0.01 pp | 5.23 % | +0.01 pp |
| Achieved Occupancy (%) | 18.61 % | 18.61 % | +0.00 pp | 18.61 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 |
| No Eligible (%) | 59.83 % | 59.82 % | -0.01 pp | 59.82 % | -0.01 pp |
| IPC Active | 1.61 inst/cycle | 1.61 inst/cycle | +0.00 | 1.61 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_matrix_b_load_index_to_eliminate_shared_memory_bank_conflicts | 1.000× | 1.000× | ✓ |  |
| B | fix_b_matrix_index_coalesced_load_with_autotune | 1.000× | 1.000× | ✓ |  |
| C | fix_b_load_coalescing_via_block_ptr_transpose | 1.028× | 1.030× | ✓ | **✓ best** |
| D | fix_b_matrix_load_index_stride_and_eliminate_bank_conflicts | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_matrix_b_load... | Δ | fix_b_matrix_inde... | Δ | fix_b_load_coales... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5220000 ns | 5220000 ns | +0 ns (+0.0%) | 5220000 ns | +0 ns (+0.0%) | 5070000 ns | -150000 ns (-2.9%) | 5220000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 39.75 % | 39.78 % | +0.03 pp | 39.80 % | +0.05 pp | 40.85 % | +1.10 pp | 39.77 % | +0.02 pp |
| Memory Throughput (%) | 89.66 % | 89.71 % | +0.05 pp | 89.76 % | +0.10 pp | 92.27 % | +2.61 pp | 89.70 % | +0.04 pp |
| DRAM Throughput (%) | 5.22 % | 5.22 % | +0.00 pp | 5.22 % | +0.00 pp | 5.37 % | +0.15 pp | 5.23 % | +0.01 pp |
| Achieved Occupancy (%) | 18.61 % | 18.61 % | +0.00 pp | 18.61 % | +0.00 pp | 18.61 % | +0.00 pp | 18.61 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 |
| No Eligible (%) | 59.83 % | 59.82 % | -0.01 pp | 59.81 % | -0.02 pp | 58.74 % | -1.09 pp | 59.83 % | +0.00 pp |
| IPC Active | 1.61 inst/cycle | 1.61 inst/cycle | +0.00 | 1.61 inst/cycle | +0.00 | 1.65 inst/cycle | +0.04 | 1.61 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0296× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | transpose_b_load_eliminate_bank_conflicts_via_explicit_transpose | 1.010× | 1.012× | ✓ |  |
| C | block_ptr_swizzled_smem_to_eliminate_bank_conflicts | 1.025× | 1.026× | ✓ |  |
| D | fix_b_load_via_make_block_ptr_swizzle | 1.028× | 1.028× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | transpose_b_load_... | Δ | block_ptr_swizzle... | Δ | fix_b_load_via_ma... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5220000 ns | 5160000 ns | -60000 ns (-1.1%) | 5090000 ns | -130000 ns (-2.5%) | 5080000 ns | -140000 ns (-2.7%) |
| SM Throughput (%) | 39.75 % | 40.23 % | +0.48 pp | 40.70 % | +0.95 pp | 40.85 % | +1.10 pp |
| Memory Throughput (%) | 89.66 % | 90.60 % | +0.94 pp | 91.96 % | +2.30 pp | 92.27 % | +2.61 pp |
| DRAM Throughput (%) | 5.22 % | 5.28 % | +0.06 pp | 5.35 % | +0.13 pp | 5.37 % | +0.15 pp |
| Achieved Occupancy (%) | 18.61 % | 18.61 % | +0.00 pp | 18.61 % | +0.00 pp | 18.61 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 |
| No Eligible (%) | 59.83 % | 59.35 % | -0.48 pp | 58.90 % | -0.93 pp | 58.74 % | -1.09 pp |
| IPC Active | 1.61 inst/cycle | 1.63 inst/cycle | +0.02 | 1.64 inst/cycle | +0.03 | 1.65 inst/cycle | +0.04 |

---

## Iteration 4

**Best so far**: 1.0296× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_and_reduce_block_m_for_occupancy | 1.009× | 1.010× | ✓ |  |
| B | eliminate_bank_conflicts_via_shared_memory_padding | 1.024× | 1.024× | ✓ |  |
| C | convert_a_matrix_load_to_block_ptr_and_cap_registers | 1.028× | 0.937× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | eliminate_bank_co... | Δ | convert_a_matrix_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5220000 ns | 5170000 ns | -50000 ns (-1.0%) | 5100000 ns | -120000 ns (-2.3%) | 5570000 ns | +350000 ns (+6.7%) |
| SM Throughput (%) | 39.75 % | 40.29 % | +0.54 pp | 40.68 % | +0.93 pp | 41.92 % | +2.17 pp |
| Memory Throughput (%) | 89.66 % | 90.72 % | +1.06 pp | 91.92 % | +2.26 pp | 90.76 % | +1.10 pp |
| DRAM Throughput (%) | 5.22 % | 5.27 % | +0.05 pp | 5.35 % | +0.13 pp | 5.05 % | -0.17 pp |
| Achieved Occupancy (%) | 18.61 % | 18.61 % | +0.00 pp | 18.61 % | +0.00 pp | 24.75 % | +6.14 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 128.00 register/thread | -40 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 22.27 | -7.43 |
| No Eligible (%) | 59.83 % | 59.35 % | -0.48 pp | 58.90 % | -0.93 pp | 57.69 % | -2.14 pp |
| IPC Active | 1.61 inst/cycle | 1.63 inst/cycle | +0.02 | 1.64 inst/cycle | +0.03 | 1.69 inst/cycle | +0.08 |

---

## Iteration 5

**Best so far**: 1.0296× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | convert_a_matrix_load_to_block_ptr_for_swizzled_smem | 1.028× | 0.937× | ✓ | **✓ best** |
| C | increase_block_k_and_reduce_maxnreg_to_break_bank_conflict_pattern | 0.785× | 0.272× | ✓ |  |
| D | add_triton_autotune | 1.025× | 1.026× | ✓ |  |


### Metric Comparison

| Metric | Baseline | convert_a_matrix_... | Δ | increase_block_k_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5220000 ns | 5570000 ns | +350000 ns (+6.7%) | 19230000 ns | +14010000 ns (+268.4%) | 5090000 ns | -130000 ns (-2.5%) |
| SM Throughput (%) | 39.75 % | 41.91 % | +2.16 pp | 26.47 % | -13.28 pp | 40.77 % | +1.02 pp |
| Memory Throughput (%) | 89.66 % | 90.73 % | +1.07 pp | 70.58 % | -19.08 pp | 92.12 % | +2.46 pp |
| DRAM Throughput (%) | 5.22 % | 5.05 % | -0.17 pp | 25.82 % | +20.60 pp | 5.36 % | +0.14 pp |
| Achieved Occupancy (%) | 18.61 % | 24.75 % | +6.14 pp | 12.45 % | -6.16 pp | 18.61 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 128.00 register/thread | -40 | 64.00 register/thread | -104 | 168.00 register/thread | +0 |
| Waves/SM | 29.70 | 22.27 | -7.43 | 44.55 | +14.85 | 29.70 | +0.00 |
| No Eligible (%) | 59.83 % | 57.69 % | -2.14 pp | 82.41 % | +22.58 pp | 58.90 % | -0.93 pp |
| IPC Active | 1.61 inst/cycle | 1.69 inst/cycle | +0.08 | 0.70 inst/cycle | -0.91 | 1.64 inst/cycle | +0.03 |

---

## Cumulative Progress

```
Baseline:  5220000 ns  (1.000×)
Iter 2     5070000 ns  (1.030×)  — fix_b_load_coalescing_via_block_ptr_transpose

Final best: 5570000 ns  (0.9372×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4,  num_stages=3, maxnreg=128),
+        triton.Config({}, num_warps=8,  num_stages=3, maxnreg=128),
+        triton.Config({}, num_warps=4,  num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=8,  num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=4,  num_stages=4, maxnreg=128),
+        triton.Config({}, num_warps=8,  num_stages=4, maxnreg=128),
+        triton.Config({}, num_warps=4,  num_stages=1, maxnreg=128),
+        triton.Config({}, num_warps=8,  num_stages=1, maxnreg=128),
+        triton.Config({}, num_warps=2,  num_stages=3, maxnreg=128),
+        triton.Config({}, num_warps=2,  num_stages=4, maxnreg=128),
+        triton.Config({}, num_warps=4,  num_stages=3),
+        triton.Config({}, num_warps=8,  num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_6(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -44,36 +62,49 @@

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
+
+    # A_ptr: load A as (BLOCK_M, BLOCK_K) with order=(1,0) for swizzled shared memory
+    # A[m, k] = base + m*stride_am + k*stride_ak = base + m*480 + k*1
+    # order=(1,0) means dim-1 (K) is contiguous — enables hardware swizzling
+    # This eliminates 8-way shared memory bank conflicts from A tile loads
+    A_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+
+    # B_ptr: load B transposed as (BLOCK_N, BLOCK_K) for coalesced access
+    # B[k, n] = base + k*stride_bk + n*stride_bn = base + k*1 + n*480
+    # Load as B_T[n, k] with shape=(N,K), strides=(stride_bn, stride_bk)=(480,1)
+    # order=(1,0) means dim-1 (K) is contiguous — enables coalesced loads
+    # This eliminates 8-way shared memory bank conflicts from strided N access
+    B_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(N, K),
+        strides=(stride_bn, stride_bk),
+        offsets=(pid_n * BLOCK_N, 0),
+        block_shape=(BLOCK_N, BLOCK_K),
+        order=(1, 0),
+    )
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        a = tl.load(A_ptr, boundary_check=(0, 1))
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 480*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 192*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # Load B transposed: b_T shape is (BLOCK_N, BLOCK_K)
+        # tl.trans() is a zero-cost metadata op giving (BLOCK_K, BLOCK_N) for tl.dot
+        b_T = tl.load(B_ptr, boundary_check=(0, 1))
+        b = tl.trans(b_T)
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+
+        # Advance along K dimension
+        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
+        B_ptr = tl.advance(B_ptr, (0, BLOCK_K))
 
 
     # rematerialize rm and rn to save registers
@@ -88,8 +119,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

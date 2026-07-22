# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 49090 ns — **Best latency**: 48740 ns (**1.0072×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_b_block_ptr_to_load_transposed_tile_directly_eliminating_tl_trans`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_004/kernel_5_triton_tem_fused_addmm_convolu/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.24
- **Registers/thread**: 128
- **Duration**: 49090 ns
- **SM Throughput**: 20.92 % | **Memory**: 80.34 % | **DRAM**: 2.81 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.91 inst/cycle | **No Eligible**: 77.15 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.005× | 0.992× | ✓ | **✓ best** |
| B | add_autotune_reduce_register_pressure_fix_bank_conflicts | 0.999× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 49090 ns | 49470 ns | +380 ns (+0.8%) | 49180 ns | +90 ns (+0.2%) |
| SM Throughput (%) | 20.92 % | 21.00 % | +0.08 pp | 20.95 % | +0.03 pp |
| Memory Throughput (%) | 80.34 % | 80.64 % | +0.30 pp | 80.47 % | +0.13 pp |
| DRAM Throughput (%) | 2.81 % | 2.79 % | -0.02 pp | 2.80 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 77.15 % | 77.15 % | +0.00 pp | 77.15 % | +0.00 pp |
| IPC Active | 0.91 inst/cycle | 0.91 inst/cycle | +0.00 | 0.91 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_index_and_tma_loads | 1.005× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_inde... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 49090 ns | 48800 ns | -290 ns (-0.6%) |
| SM Throughput (%) | 20.92 % | 21.04 % | +0.12 pp |
| Memory Throughput (%) | 80.34 % | 80.85 % | +0.51 pp |
| DRAM Throughput (%) | 2.81 % | 2.83 % | +0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 |
| No Eligible (%) | 77.15 % | 77.15 % | +0.00 pp |
| IPC Active | 0.91 inst/cycle | 0.91 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_to_eliminate_shared_memory_bank_conflicts | 1.001× | 1.001× | ✓ |  |
| B | fix_b_block_ptr_smem_order_to_eliminate_bank_conflicts | 1.010× | 1.001× | ✓ | **✓ best** |
| C | fix_b_block_ptr_order_for_coalesced_shared_memory_access | 1.008× | 1.002× | ✓ |  |
| D | fix_b_block_ptr_order_to_eliminate_bank_conflicts | 1.004× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | fix_b_block_ptr_s... | Δ | fix_b_block_ptr_o... | Δ | fix_b_block_ptr_o... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 49090 ns | 49060 ns | -30 ns (-0.1%) | 49060 ns | -30 ns (-0.1%) | 48990 ns | -100 ns (-0.2%) | 48930 ns | -160 ns (-0.3%) |
| SM Throughput (%) | 20.92 % | 21.01 % | +0.09 pp | 21.12 % | +0.20 pp | 21.16 % | +0.24 pp | 21.08 % | +0.16 pp |
| Memory Throughput (%) | 80.34 % | 80.73 % | +0.39 pp | 81.15 % | +0.81 pp | 81.33 % | +0.99 pp | 81.00 % | +0.66 pp |
| DRAM Throughput (%) | 2.81 % | 2.81 % | +0.00 pp | 2.81 % | +0.00 pp | 2.82 % | +0.01 pp | 2.82 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 77.15 % | 77.14 % | -0.01 pp | 77.13 % | -0.02 pp | 77.12 % | -0.03 pp | 77.14 % | -0.01 pp |
| IPC Active | 0.91 inst/cycle | 0.91 inst/cycle | +0.00 | 0.91 inst/cycle | +0.00 | 0.91 inst/cycle | +0.00 | 0.91 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0006× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_smem_order_and_reduce_register_pressure | 1.005× | 1.003× | ✓ |  |
| B | fix_b_block_ptr_smem_order_for_bank_conflict_elimination | 1.006× | 1.005× | ✓ |  |
| C | fix_b_block_ptr_order_to_match_memory_layout_and_eliminate_bank_conflicts | 1.006× | 1.003× | ✓ |  |
| D | fix_b_block_ptr_transpose_load_to_eliminate_bank_conflicts | 1.005× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_s... | Δ | fix_b_block_ptr_s... | Δ | fix_b_block_ptr_o... | Δ | fix_b_block_ptr_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 49090 ns | 48960 ns | -130 ns (-0.3%) | 48830 ns | -260 ns (-0.5%) | 48930 ns | -160 ns (-0.3%) | 48930 ns | -160 ns (-0.3%) |
| SM Throughput (%) | 20.92 % | 21.09 % | +0.17 pp | 21.12 % | +0.20 pp | 21.08 % | +0.16 pp | 21.09 % | +0.17 pp |
| Memory Throughput (%) | 80.34 % | 81.06 % | +0.72 pp | 81.17 % | +0.83 pp | 80.99 % | +0.65 pp | 81.09 % | +0.75 pp |
| DRAM Throughput (%) | 2.81 % | 2.82 % | +0.01 pp | 2.82 % | +0.01 pp | 2.82 % | +0.01 pp | 2.82 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 77.15 % | 77.15 % | +0.00 pp | 77.10 % | -0.05 pp | 77.26 % | +0.11 pp | 77.20 % | +0.05 pp |
| IPC Active | 0.91 inst/cycle | 0.91 inst/cycle | +0.00 | 0.92 inst/cycle | +0.01 | 0.92 inst/cycle | +0.01 | 0.91 inst/cycle | +0.00 |

---

## Iteration 5

**Best so far**: 1.0006× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_smem_order_for_tl_dot_conflict_free_access | 1.004× | 0.996× | ✓ |  |
| B | fix_b_block_ptr_smem_order_and_reduce_tile_for_occupancy_v2 | 1.009× | 1.005× | ✓ |  |
| C | fix_b_block_ptr_to_load_transposed_tile_directly_eliminating_tl_trans | 1.010× | 1.007× | ✓ | **✓ best** |
| D | fix_b_block_ptr_smem_layout_order_only | 1.005× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_s... | Δ | fix_b_block_ptr_s... | Δ | fix_b_block_ptr_t... | Δ | fix_b_block_ptr_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 49090 ns | 49280 ns | +190 ns (+0.4%) | 48860 ns | -230 ns (-0.5%) | 48740 ns | -350 ns (-0.7%) | 48900 ns | -190 ns (-0.4%) |
| SM Throughput (%) | 20.92 % | 21.06 % | +0.14 pp | 20.98 % | +0.06 pp | 21.11 % | +0.19 pp | 21.11 % | +0.19 pp |
| Memory Throughput (%) | 80.34 % | 80.94 % | +0.60 pp | 80.64 % | +0.30 pp | 81.14 % | +0.80 pp | 81.12 % | +0.78 pp |
| DRAM Throughput (%) | 2.81 % | 2.80 % | -0.01 pp | 2.82 % | +0.01 pp | 2.83 % | +0.02 pp | 2.82 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 77.15 % | 77.08 % | -0.07 pp | 77.09 % | -0.06 pp | 77.10 % | -0.05 pp | 77.14 % | -0.01 pp |
| IPC Active | 0.91 inst/cycle | 0.91 inst/cycle | +0.00 | 0.92 inst/cycle | +0.01 | 0.92 inst/cycle | +0.01 | 0.92 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  49090 ns  (1.000×)
Iter 3     49060 ns  (1.001×)  — fix_b_block_ptr_smem_order_to_eliminate_bank_conflicts
Iter 5     48740 ns  (1.007×)  — fix_b_block_ptr_to_load_transposed_tile_directly_eliminating_tl_trans

Final best: 48740 ns  (1.0072×)
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
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5(arg_A, arg_B, in_ptr2, out_ptr1):
     EVEN_K : tl.constexpr = False
@@ -44,40 +58,36 @@

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
 
+    # Define block pointers for TMA-style loads before the k loop
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # Load B^T directly as (BLOCK_K, BLOCK_N) tile to avoid tl.trans() smem bank conflicts.
+    # B is stored as (120, 400) with strides (400, 1).
+    # B^T[k, n] = B_flat[n*400 + k*1], so shape=(K,N), strides=(stride_bk=1, stride_bn=400).
+    B_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 400*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 120*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 400*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        # b is already (BLOCK_K, BLOCK_N) = B^T tile, no tl.trans() needed
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))  # advance along K dim (axis=0)
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -93,8 +103,6 @@

     tmp2 = tl.full([1], 0, tl.int32)
     tmp3 = tl.maximum(tmp2, tmp1)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 120*idx_m, [BLOCK_M, BLOCK_N])), tmp3, mask)
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

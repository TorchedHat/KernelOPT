# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 153470 ns — **Best latency**: 150560 ns (**1.0193×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_b_block_ptr_order_to_match_memory_layout`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_3_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1470,1,1), Waves/SM: 3.71
- **Registers/thread**: 168
- **Duration**: 153470 ns
- **SM Throughput**: 35.83 % | **Memory**: 80.90 % | **DRAM**: 4.47 %
- **Achieved Occupancy**: 17.72 %
- **IPC Active**: 1.56 inst/cycle | **No Eligible**: 60.93 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.002× | 0.994× | ✓ | **✓ best** |
| B | add_autotune_to_fix_bank_conflicts_and_tile_sizes | 0.999× | 0.999× | ✓ |  |
| D | autotune_tile_sizes_to_reduce_register_pressure_and_bank_conflicts | 0.998× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_to_f... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 153470 ns | 154370 ns | +900 ns (+0.6%) | 153600 ns | +130 ns (+0.1%) | 153280 ns | -190 ns (-0.1%) |
| SM Throughput (%) | 35.83 % | 35.90 % | +0.07 pp | 35.78 % | -0.05 pp | 35.85 % | +0.02 pp |
| Memory Throughput (%) | 80.90 % | 81.05 % | +0.15 pp | 80.78 % | -0.12 pp | 80.96 % | +0.06 pp |
| DRAM Throughput (%) | 4.47 % | 4.45 % | -0.02 pp | 4.46 % | -0.01 pp | 4.49 % | +0.02 pp |
| Achieved Occupancy (%) | 17.72 % | 17.72 % | +0.00 pp | 17.72 % | +0.00 pp | 17.72 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 60.93 % | 60.95 % | +0.02 pp | 60.96 % | +0.03 pp | 60.91 % | -0.02 pp |
| IPC Active | 1.56 inst/cycle | 1.56 inst/cycle | +0.00 | 1.56 inst/cycle | +0.00 | 1.56 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_transposed_load_to_eliminate_bank_conflicts | 0.999× | 1.001× | ✓ |  |
| C | fix_transposed_b_load_to_eliminate_shared_memory_bank_conflicts | 1.022× | 1.016× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ | fix_transposed_b_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 153470 ns | 153380 ns | -90 ns (-0.1%) | 151070 ns | -2400 ns (-1.6%) |
| SM Throughput (%) | 35.83 % | 35.82 % | -0.01 pp | 35.59 % | -0.24 pp |
| Memory Throughput (%) | 80.90 % | 80.99 % | +0.09 pp | 82.75 % | +1.85 pp |
| DRAM Throughput (%) | 4.47 % | 4.47 % | +0.00 pp | 4.56 % | +0.09 pp |
| Achieved Occupancy (%) | 17.72 % | 17.72 % | +0.00 pp | 17.75 % | +0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 60.93 % | 60.96 % | +0.03 pp | 61.23 % | +0.30 pp |
| IPC Active | 1.56 inst/cycle | 1.56 inst/cycle | +0.00 | 1.55 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0159× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_eliminate_transposed_b_shared_memory_conflicts | 1.022× | 1.018× | ✓ |  |
| C | eliminate_tl_trans_bank_conflicts_by_loading_b_in_kn_layout | 0.998× | 0.995× | ✓ |  |
| D | tma_loads_eliminate_shared_memory_transpose | 1.026× | 1.024× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_elimina... | Δ | eliminate_tl_tran... | Δ | tma_loads_elimina... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 153470 ns | 150690 ns | -2780 ns (-1.8%) | 154180 ns | +710 ns (+0.5%) | 149920 ns | -3550 ns (-2.3%) |
| SM Throughput (%) | 35.83 % | 35.68 % | -0.15 pp | 35.88 % | +0.05 pp | 36.07 % | +0.24 pp |
| Memory Throughput (%) | 80.90 % | 82.85 % | +1.95 pp | 81.15 % | +0.25 pp | 83.37 % | +2.47 pp |
| DRAM Throughput (%) | 4.47 % | 4.55 % | +0.08 pp | 4.47 % | +0.00 pp | 4.57 % | +0.10 pp |
| Achieved Occupancy (%) | 17.72 % | 17.75 % | +0.03 pp | 17.72 % | +0.00 pp | 17.73 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 60.93 % | 61.10 % | +0.17 pp | 60.98 % | +0.05 pp | 60.76 % | -0.17 pp |
| IPC Active | 1.56 inst/cycle | 1.55 inst/cycle | -0.01 | 1.56 inst/cycle | +0.00 | 1.57 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.0237× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_block_ptr_order_to_match_memory_layout | 1.029× | 1.019× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 153470 ns | 150560 ns | -2910 ns (-1.9%) |
| SM Throughput (%) | 35.83 % | 35.74 % | -0.09 pp |
| Memory Throughput (%) | 80.90 % | 82.96 % | +2.06 pp |
| DRAM Throughput (%) | 4.47 % | 4.55 % | +0.08 pp |
| Achieved Occupancy (%) | 17.72 % | 17.74 % | +0.02 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 |
| No Eligible (%) | 60.93 % | 61.11 % | +0.18 pp |
| IPC Active | 1.56 inst/cycle | 1.55 inst/cycle | -0.01 |

---

## Iteration 5

**Best so far**: 1.0237× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_block_ptr_order_to_n_major_smem_without_tl_trans | 1.027× | 1.022× | ✓ |  |
| C | eviction_policies_for_l2_residency_control | 1.022× | 1.020× | ✓ |  |
| D | load_b_as_NK_then_transpose_for_coalesced_shared_memory | 1.028× | 1.022× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | eviction_policies... | Δ | load_b_as_NK_then... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 153470 ns | 150180 ns | -3290 ns (-2.1%) | 150460 ns | -3010 ns (-2.0%) | 150180 ns | -3290 ns (-2.1%) |
| SM Throughput (%) | 35.83 % | 35.46 % | -0.37 pp | 35.60 % | -0.23 pp | 35.65 % | -0.18 pp |
| Memory Throughput (%) | 80.90 % | 82.30 % | +1.40 pp | 82.58 % | +1.68 pp | 82.86 % | +1.96 pp |
| DRAM Throughput (%) | 4.47 % | 4.56 % | +0.09 pp | 4.56 % | +0.09 pp | 4.56 % | +0.09 pp |
| Achieved Occupancy (%) | 17.72 % | 17.74 % | +0.02 pp | 17.72 % | +0.00 pp | 17.75 % | +0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 60.93 % | 61.09 % | +0.16 pp | 61.03 % | +0.10 pp | 61.12 % | +0.19 pp |
| IPC Active | 1.56 inst/cycle | 1.55 inst/cycle | -0.01 | 1.56 inst/cycle | +0.00 | 1.55 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  153470 ns  (1.000×)
Iter 2     151070 ns  (1.016×)  — fix_transposed_b_load_to_eliminate_shared_memory_bank_conflicts
Iter 3     149920 ns  (1.024×)  — tma_loads_eliminate_shared_memory_transpose

Final best: 150560 ns  (1.0193×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=['M', 'N', 'K'],
+)
 @triton.jit
 def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -44,37 +56,37 @@

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
 
+    # Use block pointers for coalesced, bank-conflict-free loads
+    # A: shape (M, K) = (31360, 96), strides (96, 1), load [BLOCK_M, BLOCK_K] tiles
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)
+    )
+    # B: reinterpret as shape (K, N) = (96, 384), strides (1, 96)
+    # Load directly as [BLOCK_K, BLOCK_N] — no tl.trans needed, eliminates bank conflicts
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1)
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 96*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 384*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        a = tl.load(a_block_ptr)       # [BLOCK_M, BLOCK_K], coalesced row-major
+        b = tl.load(b_block_ptr)       # [BLOCK_K, BLOCK_N], no tl.trans needed
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))  # advance K dimension
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +98,6 @@

     # inductor generates a suffix
     xindex = idx_n + 384*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
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

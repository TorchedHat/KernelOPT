# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 48830 ns — **Best latency**: 48610 ns (**1.0045×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_b_matrix_strided_load_to_eliminate_shared_memory_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_29_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.19
- **Registers/thread**: 94
- **Duration**: 48830 ns
- **SM Throughput**: 12.48 % | **Memory**: 85.02 % | **DRAM**: 1.67 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.56 inst/cycle | **No Eligible**: 86.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_swizzled_shared_memory | 0.987× | 0.981× | ✓ |  |
| C | eviction_policies_streaming_loads | 0.998× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_with_sw... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 48830 ns | 49790 ns | +960 ns (+2.0%) | 48860 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 12.48 % | 12.50 % | +0.02 pp | 12.50 % | +0.02 pp |
| Memory Throughput (%) | 85.02 % | 83.41 % | -1.61 pp | 85.06 % | +0.04 pp |
| DRAM Throughput (%) | 1.67 % | 1.63 % | -0.04 pp | 1.67 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 92.00 register/thread | -2 | 94.00 register/thread | +0 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.19 | +0.00 |
| No Eligible (%) | 86.10 % | 86.05 % | -0.05 pp | 86.06 % | -0.04 pp |
| IPC Active | 0.56 inst/cycle | 0.56 inst/cycle | +0.00 | 0.56 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_strided_load_to_eliminate_shared_memory_bank_conflicts | 1.001× | 1.004× | ✓ | **✓ best** |
| B | tma_loads_async_descriptor | 0.990× | 0.989× | ✓ |  |
| C | fix_b_matrix_shared_memory_layout_to_eliminate_bank_conflicts | 0.983× | 0.985× | ✓ |  |
| D | fix_shared_memory_bank_conflicts_b_matrix_order | 0.987× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_stri... | Δ | tma_loads_async_d... | Δ | fix_b_matrix_shar... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48830 ns | 48610 ns | -220 ns (-0.5%) | 49380 ns | +550 ns (+1.1%) | 49570 ns | +740 ns (+1.5%) | 49630 ns | +800 ns (+1.6%) |
| SM Throughput (%) | 12.48 % | 12.46 % | -0.02 pp | 12.57 % | +0.09 pp | 12.52 % | +0.04 pp | 12.49 % | +0.01 pp |
| Memory Throughput (%) | 85.02 % | 84.79 % | -0.23 pp | 83.86 % | -1.16 pp | 83.57 % | -1.45 pp | 83.38 % | -1.64 pp |
| DRAM Throughput (%) | 1.67 % | 1.67 % | +0.00 pp | 1.65 % | -0.02 pp | 1.64 % | -0.03 pp | 1.64 % | -0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 94.00 register/thread | +0 | 92.00 register/thread | -2 | 92.00 register/thread | -2 | 92.00 register/thread | -2 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.19 | +0.00 | 0.19 | +0.00 | 0.19 | +0.00 |
| No Eligible (%) | 86.10 % | 86.09 % | -0.01 pp | 85.99 % | -0.11 pp | 86.06 % | -0.04 pp | 86.06 % | -0.04 pp |
| IPC Active | 0.56 inst/cycle | 0.56 inst/cycle | +0.00 | 0.56 inst/cycle | +0.00 | 0.56 inst/cycle | +0.00 | 0.56 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0045× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | b_matrix_transposed_smem_staging_via_block_ptr | 0.992× | 0.990× | ✓ |  |
| B | fix_b_matrix_smem_order_to_eliminate_bank_conflicts | 0.982× | 0.979× | ✓ |  |
| C | fix_b_matrix_load_order_to_eliminate_bank_conflicts | 0.989× | 0.987× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 0.962× | 0.963× | ✓ |  |


### Metric Comparison

| Metric | Baseline | b_matrix_transpos... | Δ | fix_b_matrix_smem... | Δ | fix_b_matrix_load... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48830 ns | 49310 ns | +480 ns (+1.0%) | 49860 ns | +1030 ns (+2.1%) | 49500 ns | +670 ns (+1.4%) | 50720 ns | +1890 ns (+3.9%) |
| SM Throughput (%) | 12.48 % | 12.58 % | +0.10 pp | 12.47 % | -0.01 pp | 12.56 % | +0.08 pp | 12.73 % | +0.25 pp |
| Memory Throughput (%) | 85.02 % | 84.06 % | -0.96 pp | 83.23 % | -1.79 pp | 83.84 % | -1.18 pp | 82.54 % | -2.48 pp |
| DRAM Throughput (%) | 1.67 % | 1.65 % | -0.02 pp | 1.63 % | -0.04 pp | 1.64 % | -0.03 pp | 1.60 % | -0.07 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 92.00 register/thread | -2 | 92.00 register/thread | -2 | 92.00 register/thread | -2 | 64.00 register/thread | -30 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.19 | +0.00 | 0.19 | +0.00 | 0.12 | -0.07 |
| No Eligible (%) | 86.10 % | 86.11 % | +0.01 pp | 86.07 % | -0.03 pp | 85.98 % | -0.12 pp | 85.77 % | -0.33 pp |
| IPC Active | 0.56 inst/cycle | 0.56 inst/cycle | +0.00 | 0.56 inst/cycle | +0.00 | 0.56 inst/cycle | +0.00 | 0.57 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.0045× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_smem_bank_conflicts_via_direct_b_matrix_load_without_transpose | 0.993× | 0.989× | ✓ |  |
| B | fix_b_matrix_block_ptr_order_to_eliminate_smem_bank_conflicts | 0.965× | 0.962× | ✓ |  |
| C | b_matrix_coalesced_global_load_via_transposed_smem_tile | 0.993× | 0.989× | ✓ |  |
| D | fix_b_matrix_block_ptr_order_for_coalesced_smem_access | 0.971× | 0.965× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_smem_ba... | Δ | fix_b_matrix_bloc... | Δ | b_matrix_coalesce... | Δ | fix_b_matrix_bloc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48830 ns | 49380 ns | +550 ns (+1.1%) | 50780 ns | +1950 ns (+4.0%) | 49380 ns | +550 ns (+1.1%) | 50590 ns | +1760 ns (+3.6%) |
| SM Throughput (%) | 12.48 % | 12.56 % | +0.08 pp | 12.72 % | +0.24 pp | 12.49 % | +0.01 pp | 12.75 % | +0.27 pp |
| Memory Throughput (%) | 85.02 % | 83.82 % | -1.20 pp | 82.53 % | -2.49 pp | 83.46 % | -1.56 pp | 82.67 % | -2.35 pp |
| DRAM Throughput (%) | 1.67 % | 1.65 % | -0.02 pp | 1.60 % | -0.07 pp | 1.65 % | -0.02 pp | 1.61 % | -0.06 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 92.00 register/thread | -2 | 64.00 register/thread | -30 | 92.00 register/thread | -2 | 64.00 register/thread | -30 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.12 | -0.07 | 0.19 | +0.00 | 0.12 | -0.07 |
| No Eligible (%) | 86.10 % | 86.01 % | -0.09 pp | 85.77 % | -0.33 pp | 85.99 % | -0.11 pp | 85.84 % | -0.26 pp |
| IPC Active | 0.56 inst/cycle | 0.56 inst/cycle | +0.00 | 0.57 inst/cycle | +0.01 | 0.56 inst/cycle | +0.00 | 0.57 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  48830 ns  (1.000×)
Iter 2     48610 ns  (1.004×)  — fix_b_matrix_strided_load_to_eliminate_shared_memory_bank_conflicts

Final best: 48610 ns  (1.0045×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_stages=3, num_warps=4),
+        triton.Config({}, num_stages=4, num_warps=4),
+        triton.Config({}, num_stages=3, num_warps=8),
+        triton.Config({}, num_stages=4, num_warps=8),
+        triton.Config({}, num_stages=2, num_warps=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -44,37 +55,38 @@

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
 
+    # Create block pointer for A
+    # A layout: shape=(M,K)=(1960,480), strides=(480,1) — K is fast dim
+    A_block_ptr = tl.make_block_ptr(
+        A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B layout transposed: load as [BLOCK_N, BLOCK_K] with strides (stride_bn=480, stride_bk=1)
+    # This gives unit stride along K (fast dim), eliminating bank conflicts in SMEM
+    # B_t[n, k] = B[k, n], so we load shape=(N,K) with strides=(stride_bn, stride_bk)
+    B_block_ptr = tl.make_block_ptr(
+        B,
+        shape=(N, K),
+        strides=(stride_bn, stride_bk),
+        offsets=(pid_n * BLOCK_N, 0),
+        block_shape=(BLOCK_N, BLOCK_K),
+        order=(1, 0),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), eviction_policy='evict_first')
+        b_t = tl.load(B_block_ptr, boundary_check=(0, 1), eviction_policy='evict_first')
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 480*idx_m
-        a = tl.load(A + (xindex))
+        acc += tl.dot(a, tl.trans(b_t), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 112*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_K))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -85,9 +97,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 112*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_last')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

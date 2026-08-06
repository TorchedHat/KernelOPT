# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 20990 ns — **Best latency**: 20740 ns (**1.0121×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`annotate_streaming_loads_with_evict_first_cache_policy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_33_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (120,1,1), Waves/SM: 0.23
- **Registers/thread**: 128
- **Duration**: 20990 ns
- **SM Throughput**: 18.35 % | **Memory**: 69.60 % | **DRAM**: 0.93 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.90 inst/cycle | **No Eligible**: 77.50 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_fix_b_matrix_pointer_arithmetic | 0.953× | 0.952× | ✓ |  |
| C | annotate_streaming_loads_with_evict_first_cache_policy | 1.011× | 1.012× | ✓ | **✓ best** |
| D | fix_b_matrix_load_and_autotune_tile_sizes | 1.000× | 0.622× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_fix_b_m... | Δ | annotate_streamin... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20990 ns | 22050 ns | +1060 ns (+5.1%) | 20740 ns | -250 ns (-1.2%) | 33730 ns | +12740 ns (+60.7%) |
| SM Throughput (%) | 18.35 % | 17.31 % | -1.04 pp | 18.35 % | +0.00 pp | 11.91 % | -6.44 pp |
| Memory Throughput (%) | 69.60 % | 65.67 % | -3.93 pp | 69.61 % | +0.01 pp | 79.70 % | +10.10 pp |
| DRAM Throughput (%) | 0.93 % | 0.88 % | -0.05 pp | 0.94 % | +0.01 pp | 0.57 % | -0.36 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 12.48 % | +6.24 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 80.00 register/thread | -48 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 | 0.30 | +0.07 |
| No Eligible (%) | 77.50 % | 78.80 % | +1.30 pp | 77.53 % | +0.03 pp | 86.12 % | +8.62 pp |
| IPC Active | 0.90 inst/cycle | 0.85 inst/cycle | -0.05 | 0.90 inst/cycle | +0.00 | 0.56 inst/cycle | -0.34 |

---

## Iteration 3

**Best so far**: 1.0121× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_block_ptr_order_to_eliminate_bank_conflicts | 0.963× | 0.962× | ✓ |  |
| C | transpose_b_matrix_load_for_coalesced_global_access_and_bank_conflict_elimination | 0.995× | 0.619× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bloc... | Δ | transpose_b_matri... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 20990 ns | 21820 ns | +830 ns (+4.0%) | 33890 ns | +12900 ns (+61.5%) |
| SM Throughput (%) | 18.35 % | 17.44 % | -0.91 pp | 11.89 % | -6.46 pp |
| Memory Throughput (%) | 69.60 % | 66.14 % | -3.46 pp | 79.57 % | +9.97 pp |
| DRAM Throughput (%) | 0.93 % | 0.89 % | -0.04 pp | 0.57 % | -0.36 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 12.48 % | +6.24 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 80.00 register/thread | -48 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.30 | +0.07 |
| No Eligible (%) | 77.50 % | 78.87 % | +1.37 pp | 86.11 % | +8.61 pp |
| IPC Active | 0.90 inst/cycle | 0.85 inst/cycle | -0.05 | 0.56 inst/cycle | -0.34 |

---

## Iteration 4

**Best so far**: 1.0121× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_and_autotune_tile_sizes_as_constexpr | 0.957× | 0.913× | ✓ |  |
| B | tma_pipeline_with_autotuned_tile_sizes_and_num_stages | 0.961× | 0.960× | ✓ |  |
| C | eliminate_tl_trans_bank_conflicts_by_direct_k_major_b_load | 0.997× | 0.621× | ✓ |  |
| D | eliminate_tl_trans_to_remove_shared_memory_bank_conflicts | 1.006× | 0.621× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | tma_pipeline_with... | Δ | eliminate_tl_tran... | Δ | eliminate_tl_tran... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20990 ns | 22980 ns | +1990 ns (+9.5%) | 21860 ns | +870 ns (+4.1%) | 33820 ns | +12830 ns (+61.1%) | 33820 ns | +12830 ns (+61.1%) |
| SM Throughput (%) | 18.35 % | 15.99 % | -2.36 pp | 17.46 % | -0.89 pp | 11.88 % | -6.47 pp | 11.87 % | -6.48 pp |
| Memory Throughput (%) | 69.60 % | 61.18 % | -8.42 pp | 66.20 % | -3.40 pp | 79.51 % | +9.91 pp | 79.48 % | +9.88 pp |
| DRAM Throughput (%) | 0.93 % | 0.85 % | -0.08 pp | 0.89 % | -0.04 pp | 0.57 % | -0.36 pp | 0.57 % | -0.36 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 12.48 % | +6.24 pp | 12.48 % | +6.24 pp |
| Registers/Thread | 128.00 register/thread | 108.00 register/thread | -20 | 128.00 register/thread | +0 | 80.00 register/thread | -48 | 80.00 register/thread | -48 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 | 0.30 | +0.07 | 0.30 | +0.07 |
| No Eligible (%) | 77.50 % | 80.17 % | +2.67 pp | 78.79 % | +1.29 pp | 86.04 % | +8.54 pp | 86.05 % | +8.55 pp |
| IPC Active | 0.90 inst/cycle | 0.79 inst/cycle | -0.11 | 0.85 inst/cycle | -0.05 | 0.55 inst/cycle | -0.35 | 0.56 inst/cycle | -0.34 |


### Key Learnings

#### ❌ Plan D: `eliminate_tl_trans_to_remove_shared_memory_bank_conflicts` (0.621×)

Replacing a transposed B-matrix load (tl.load of [BLOCK_N, BLOCK_K] + tl.trans()) with a natural [BLOCK_K, BLOCK_N] strided pointer load — while simultaneously adding @triton.autotune over num_warps/num_stages and doubling block size from 128 to 256 threads — caused a 1.6x regression. Despite registers/thread dropping from 128 to 80 and achieved occupancy doubling (6.24%→12.48%), shared memory bank conflicts worsened from 11.9-way to 16.8-way, MIO stall fraction rose from 60.5% to 83.1% of cycles, and Warp Cycles Per Issued Instruction tripled (4.77→14.29). The confounding factor is that doubling block size with fixed tile dimensions (BLOCK_M=64, BLOCK_N=64, BLOCK_K=32) amplified the bank conflict pattern — the same failure mode observed in iteration 2 (item b7c3f91e). Changing the B-load access pattern alone does not resolve bank conflicts when tile sizes and block size are simultaneously changed in ways that worsen the shared memory stride pattern.

> **Applicable when**: Never — this combination caused regression. The natural [BLOCK_K, BLOCK_N] B-load pointer arithmetic may be correct in isolation, but combined with doubling block size and autotuning only num_warps/num_stages (not tile sizes), it amplifies bank conflicts.

> **Avoid when**: Do not attempt to fix shared memory bank conflicts by changing B-matrix load layout (removing tl.trans or switching from block_ptr to manual pointer arithmetic) while simultaneously increasing block size with fixed tile dimensions. When BLOCK_M/BLOCK_N/BLOCK_K are not exposed as autotuned constexpr parameters, changing block size with fixed tiles amplifies bank conflict patterns rather than resolving them. This is the second iteration (items b7c3f91e and a9f2b3c1) confirming that autotuning only num_warps/num_stages without tile size variation causes regression in this kernel.

---

## Cumulative Progress

```
Baseline:  20990 ns  (1.000×)
Iter 2     20740 ns  (1.012×)  — annotate_streaming_loads_with_evict_first_cache_policy

Final best: 20740 ns  (1.0121×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33(arg_A, arg_B, out_ptr0):
@@ -44,37 +45,22 @@

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
 
+    # TMA block pointer loads — correct strides, hardware-coalesced 2D tile loads
+    A_block_ptr = tl.make_block_ptr(
+        base=A, shape=(M, K), strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
+    B_block_ptr = tl.make_block_ptr(
+        base=B, shape=(K, N), strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 160*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 960*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 160*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +72,6 @@

     # inductor generates a suffix
     xindex = idx_n + 960*idx_m
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

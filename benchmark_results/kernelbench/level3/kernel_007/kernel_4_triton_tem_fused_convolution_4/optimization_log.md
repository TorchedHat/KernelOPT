# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1700000 ns — **Best latency**: 1810000 ns (**0.9392×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_4_triton_tem_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 9.90
- **Registers/thread**: 168
- **Duration**: 1700000 ns
- **SM Throughput**: 40.45 % | **Memory**: 91.51 % | **DRAM**: 12.94 %
- **Achieved Occupancy**: 18.41 %
- **IPC Active**: 1.64 inst/cycle | **No Eligible**: 58.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.994× | ✓ |  |
| C | fix_b_matrix_load_coalescing_to_eliminate_shared_memory_bank_conflicts | 1.005× | 1.006× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1700000 ns | 1710000 ns | +10000 ns (+0.6%) | 1690000 ns | -10000 ns (-0.6%) |
| SM Throughput (%) | 40.45 % | 40.51 % | +0.06 pp | 40.73 % | +0.28 pp |
| Memory Throughput (%) | 91.51 % | 91.65 % | +0.14 pp | 92.23 % | +0.72 pp |
| DRAM Throughput (%) | 12.94 % | 12.93 % | -0.01 pp | 13.04 % | +0.10 pp |
| Achieved Occupancy (%) | 18.41 % | 18.41 % | +0.00 pp | 18.41 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 | 9.90 | +0.00 |
| No Eligible (%) | 58.96 % | 58.96 % | +0.00 pp | 58.64 % | -0.32 pp |
| IPC Active | 1.64 inst/cycle | 1.64 inst/cycle | +0.00 | 1.65 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0059× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_b_tile_shared_memory_bank_conflicts_via_block_ptr_swizzle | 0.990× | 0.988× | ✓ |  |
| B | tma_loads | 0.998× | 1.000× | ✓ |  |
| C | fix_b_matrix_load_via_coalesced_access_replacing_transposed_gather | 1.008× | 1.006× | ✓ |  |
| D | tma_loads_replace_scalar_pointer_arithmetic_with_tensor_descriptors | 1.004× | 1.000× | ✓ |  |
| E | add_triton_autotune | 1.009× | 0.939× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_b_tile_... | Δ | tma_loads | Δ | fix_b_matrix_load... | Δ | tma_loads_replace... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1700000 ns | 1720000 ns | +20000 ns (+1.2%) | 1700000 ns | +0 ns (+0.0%) | 1690000 ns | -10000 ns (-0.6%) | 1700000 ns | +0 ns (+0.0%) | 1810000 ns | +110000 ns (+6.5%) |
| SM Throughput (%) | 40.45 % | 39.84 % | -0.61 pp | 40.50 % | +0.05 pp | 40.76 % | +0.31 pp | 40.53 % | +0.08 pp | 37.85 % | -2.60 pp |
| Memory Throughput (%) | 91.51 % | 90.51 % | -1.00 pp | 91.46 % | -0.05 pp | 92.19 % | +0.68 pp | 91.85 % | +0.34 pp | 86.27 % | -5.24 pp |
| DRAM Throughput (%) | 12.94 % | 12.80 % | -0.14 pp | 12.94 % | +0.00 pp | 13.06 % | +0.12 pp | 12.99 % | +0.05 pp | 12.22 % | -0.72 pp |
| Achieved Occupancy (%) | 18.41 % | 18.40 % | -0.01 pp | 18.41 % | +0.00 pp | 18.41 % | +0.00 pp | 18.41 % | +0.00 pp | 18.38 % | -0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 | 9.90 | +0.00 | 9.90 | +0.00 | 9.90 | +0.00 | 9.90 | +0.00 |
| No Eligible (%) | 58.96 % | 59.57 % | +0.61 pp | 58.88 % | -0.08 pp | 58.60 % | -0.36 pp | 58.89 % | -0.07 pp | 61.58 % | +2.62 pp |
| IPC Active | 1.64 inst/cycle | 1.62 inst/cycle | -0.02 | 1.64 inst/cycle | +0.00 | 1.66 inst/cycle | +0.02 | 1.64 inst/cycle | +0.00 | 1.54 inst/cycle | -0.10 |

---

## Iteration 3

**Best so far**: 1.0059× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_shared_memory_bank_conflicts_by_changing_block_k | 0.908× | 0.899× | ✓ |  |
| B | eliminate_b_tile_bank_conflicts_via_explicit_transpose_in_registers | 0.998× | 0.994× | ✓ |  |
| C | eliminate_b_tile_bank_conflicts_via_register_transpose | 1.002× | 1.000× | ✓ |  |
| D | eliminate_b_tile_bank_conflicts_via_explicit_smem_transpose_with_padding | 1.006× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ | eliminate_b_tile_... | Δ | eliminate_b_tile_... | Δ | eliminate_b_tile_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1700000 ns | 1890000 ns | +190000 ns (+11.2%) | 1710000 ns | +10000 ns (+0.6%) | 1700000 ns | +0 ns (+0.0%) | 1690000 ns | -10000 ns (-0.6%) |
| SM Throughput (%) | 40.45 % | 36.87 % | -3.58 pp | 40.37 % | -0.08 pp | 40.50 % | +0.05 pp | 40.75 % | +0.30 pp |
| Memory Throughput (%) | 91.51 % | 82.62 % | -8.89 pp | 91.38 % | -0.13 pp | 91.59 % | +0.08 pp | 92.29 % | +0.78 pp |
| DRAM Throughput (%) | 12.94 % | 11.72 % | -1.22 pp | 12.90 % | -0.04 pp | 12.95 % | +0.01 pp | 13.07 % | +0.13 pp |
| Achieved Occupancy (%) | 18.41 % | 12.48 % | -5.93 pp | 18.41 % | +0.00 pp | 18.40 % | -0.01 pp | 18.40 % | -0.01 pp |
| Registers/Thread | 168.00 register/thread | 238.00 register/thread | +70 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 14.85 | +4.95 | 9.90 | +0.00 | 9.90 | +0.00 | 9.90 | +0.00 |
| No Eligible (%) | 58.96 % | 62.08 % | +3.12 pp | 59.08 % | +0.12 pp | 58.89 % | -0.07 pp | 58.64 % | -0.32 pp |
| IPC Active | 1.64 inst/cycle | 1.52 inst/cycle | -0.12 | 1.64 inst/cycle | +0.00 | 1.64 inst/cycle | +0.00 | 1.65 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.0059× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | load_b_transposed_to_eliminate_shared_memory_bank_conflicts | 0.987× | 0.983× | ✓ |  |
| C | eliminate_b_tile_bank_conflicts_via_k_major_direct_load_no_transpose | 0.993× | 0.994× | ✓ |  |
| D | load_b_in_natural_row_major_layout_and_transpose_dot | 0.913× | 0.904× | ✓ |  |
| E | add_triton_autotune | 0.998× | 0.961× | ✓ |  |


### Metric Comparison

| Metric | Baseline | load_b_transposed... | Δ | eliminate_b_tile_... | Δ | load_b_in_natural... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1700000 ns | 1730000 ns | +30000 ns (+1.8%) | 1710000 ns | +10000 ns (+0.6%) | 1880000 ns | +180000 ns (+10.6%) | 1770000 ns | +70000 ns (+4.1%) |
| SM Throughput (%) | 40.45 % | 40.01 % | -0.44 pp | 40.35 % | -0.10 pp | 37.00 % | -3.45 pp | 38.64 % | -1.81 pp |
| Memory Throughput (%) | 91.51 % | 90.28 % | -1.23 pp | 91.31 % | -0.20 pp | 82.95 % | -8.56 pp | 87.92 % | -3.59 pp |
| DRAM Throughput (%) | 12.94 % | 12.78 % | -0.16 pp | 12.90 % | -0.04 pp | 11.77 % | -1.17 pp | 12.46 % | -0.48 pp |
| Achieved Occupancy (%) | 18.41 % | 18.41 % | +0.00 pp | 18.41 % | +0.00 pp | 12.47 % | -5.94 pp | 18.46 % | +0.05 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 238.00 register/thread | +70 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 | 9.90 | +0.00 | 14.85 | +4.95 | 9.90 | +0.00 |
| No Eligible (%) | 58.96 % | 59.40 % | +0.44 pp | 59.08 % | +0.12 pp | 61.88 % | +2.92 pp | 60.55 % | +1.59 pp |
| IPC Active | 1.64 inst/cycle | 1.62 inst/cycle | -0.02 | 1.64 inst/cycle | +0.00 | 1.53 inst/cycle | -0.11 | 1.58 inst/cycle | -0.06 |

---

## Cumulative Progress

```
Baseline:  1700000 ns  (1.000×)
Iter 1     1690000 ns  (1.006×)  — fix_b_matrix_load_coalescing_to_eliminate_shared_memory_bank_conflicts

Final best: 1810000 ns  (0.9392×)
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
 def triton_tem_fused_convolution_4(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -54,27 +55,37 @@

         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # A_block_ptr remains unchanged — A is row-major (stride_ak=1), already coalesced
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)
+    )
+    # B_block_ptr removed — replaced with scalar pointer arithmetic inside k-loop
+    # to eliminate 8-way shared memory bank conflicts from column-major B layout
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        # Load A using block pointer (unchanged)
+        a = tl.load(A_block_ptr)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 480*idx_m
-        a = tl.load(A + (xindex))
+        # Load B using scalar pointer arithmetic to bypass shared memory staging
+        # stride_bk=1 means K is the contiguous dimension — fully coalesced global loads
+        rk = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)    # shape [BLOCK_K]
+        rn_b = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # shape [BLOCK_N]
+        b_ptrs = B + rk[:, None] * stride_bk + rn_b[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]
+        b_mask = (rk[:, None] < K) & (rn_b[None, :] < N)
+        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 64*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        # Advance A block pointer by BLOCK_K along the K dimension
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        # No B_block_ptr advance needed — scalar arithmetic recomputes offsets each iteration
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +99,6 @@

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

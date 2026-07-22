# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 42080 ns — **Best latency**: 41340 ns (**1.0179×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_b_matrix_smem_order_to_eliminate_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_13_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.62
- **Registers/thread**: 154
- **Duration**: 42080 ns
- **SM Throughput**: 18.69 % | **Memory**: 78.26 % | **DRAM**: 8.78 %
- **Achieved Occupancy**: 11.79 %
- **IPC Active**: 0.86 inst/cycle | **No Eligible**: 78.63 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.005× | 1.009× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 42080 ns | 41700 ns | -380 ns (-0.9%) |
| SM Throughput (%) | 18.69 % | 18.95 % | +0.26 pp |
| Memory Throughput (%) | 78.26 % | 79.34 % | +1.08 pp |
| DRAM Throughput (%) | 8.78 % | 8.85 % | +0.07 pp |
| Achieved Occupancy (%) | 11.79 % | 11.79 % | +0.00 pp |
| Registers/Thread | 154.00 register/thread | 154.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 |
| No Eligible (%) | 78.63 % | 78.60 % | -0.03 pp |
| IPC Active | 0.86 inst/cycle | 0.86 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0091× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_corrected_b_matrix_index | 1.010× | 1.018× | ✓ | **✓ best** |
| C | evict_first_on_streaming_A_matrix_load_to_preserve_L2_for_B | 1.005× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_with_co... | Δ | evict_first_on_st... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 42080 ns | 41340 ns | -740 ns (-1.8%) | 41730 ns | -350 ns (-0.8%) |
| SM Throughput (%) | 18.69 % | 18.73 % | +0.04 pp | 18.84 % | +0.15 pp |
| Memory Throughput (%) | 78.26 % | 80.38 % | +2.12 pp | 78.91 % | +0.65 pp |
| DRAM Throughput (%) | 8.78 % | 8.92 % | +0.14 pp | 8.84 % | +0.06 pp |
| Achieved Occupancy (%) | 11.79 % | 11.86 % | +0.07 pp | 11.78 % | -0.01 pp |
| Registers/Thread | 154.00 register/thread | 128.00 register/thread | -26 | 154.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.46 | -0.16 | 0.62 | +0.00 |
| No Eligible (%) | 78.63 % | 78.74 % | +0.11 pp | 78.52 % | -0.11 pp |
| IPC Active | 0.86 inst/cycle | 0.85 inst/cycle | -0.01 | 0.86 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0179× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_block_ptr_order_to_eliminate_shared_memory_bank_conflicts | 1.009× | 1.025× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bloc... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 42080 ns | 41060 ns | -1020 ns (-2.4%) |
| SM Throughput (%) | 18.69 % | 18.67 % | -0.02 pp |
| Memory Throughput (%) | 78.26 % | 80.11 % | +1.85 pp |
| DRAM Throughput (%) | 8.78 % | 8.98 % | +0.20 pp |
| Achieved Occupancy (%) | 11.79 % | 11.85 % | +0.06 pp |
| Registers/Thread | 154.00 register/thread | 128.00 register/thread | -26 |
| Waves/SM | 0.62 | 0.46 | -0.16 |
| No Eligible (%) | 78.63 % | 78.71 % | +0.08 pp |
| IPC Active | 0.86 inst/cycle | 0.85 inst/cycle | -0.01 |

---

## Iteration 4

**Best so far**: 1.0179× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | evict_first_on_streaming_A_loads_evict_last_on_reused_B_loads | 1.010× | 1.023× | ✓ |  |
| D | fix_b_matrix_smem_order_to_eliminate_bank_conflicts | 1.007× | 1.018× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_on_st... | Δ | fix_b_matrix_smem... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 42080 ns | 41120 ns | -960 ns (-2.3%) | 41340 ns | -740 ns (-1.8%) |
| SM Throughput (%) | 18.69 % | 18.69 % | +0.00 pp | 18.73 % | +0.04 pp |
| Memory Throughput (%) | 78.26 % | 80.19 % | +1.93 pp | 80.37 % | +2.11 pp |
| DRAM Throughput (%) | 8.78 % | 8.97 % | +0.19 pp | 8.92 % | +0.14 pp |
| Achieved Occupancy (%) | 11.79 % | 11.86 % | +0.07 pp | 11.86 % | +0.07 pp |
| Registers/Thread | 154.00 register/thread | 128.00 register/thread | -26 | 128.00 register/thread | -26 |
| Waves/SM | 0.62 | 0.46 | -0.16 | 0.46 | -0.16 |
| No Eligible (%) | 78.63 % | 79.00 % | +0.37 pp | 78.72 % | +0.09 pp |
| IPC Active | 0.86 inst/cycle | 0.85 inst/cycle | -0.01 | 0.85 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  42080 ns  (1.000×)
Iter 1     41700 ns  (1.009×)  — tf32_tensor_cores
Iter 2     41340 ns  (1.018×)  — tma_loads_with_corrected_b_matrix_index

Final best: 41340 ns  (1.0179×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13(arg_A, arg_B, out_ptr0):
@@ -44,40 +45,35 @@

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
 
+    # Use tl.make_block_ptr for TMA-style loads with correct strides
+    # A: shape (M=31360, K=144), strides (stride_am=144, stride_ak=1), row-major
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)
+    )
+    # B: shape (K=144, N=24), strides (stride_bk=1, stride_bn=144)
+    # N is fast dimension for dot access pattern, so order=(1, 0) aligns SMEM layout
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0)
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
-        xindex = idx_n + 144*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 24*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-
+        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -89,8 +85,6 @@

     # inductor generates a suffix
     xindex = idx_n + 24*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
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

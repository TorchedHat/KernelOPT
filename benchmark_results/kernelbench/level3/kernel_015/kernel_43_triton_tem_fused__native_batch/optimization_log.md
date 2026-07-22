# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_43` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 133600 ns — **Best latency**: 129500 ns (**1.0317×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`tma_loads_replace_scalar_pointer_arithmetic`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_43`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_43_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (246,1,1), Waves/SM: 0.62
- **Registers/thread**: 168
- **Duration**: 133600 ns
- **SM Throughput**: 34.44 % | **Memory**: 77.80 % | **DRAM**: 2.56 %
- **Achieved Occupancy**: 11.94 %
- **IPC Active**: 1.49 inst/cycle | **No Eligible**: 62.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 1.002× | ✓ |  |
| B | tma_loads_replace_scalar_pointer_arithmetic | 1.032× | 1.032× | ✓ | **✓ best** |
| C | autotune_tile_sizes_to_reduce_register_pressure_and_bank_conflicts | 1.004× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_replace... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 133600 ns | 133380 ns | -220 ns (-0.2%) | 129500 ns | -4100 ns (-3.1%) | 133150 ns | -450 ns (-0.3%) |
| SM Throughput (%) | 34.44 % | 34.64 % | +0.20 pp | 35.64 % | +1.20 pp | 34.44 % | +0.00 pp |
| Memory Throughput (%) | 77.80 % | 78.26 % | +0.46 pp | 80.23 % | +2.43 pp | 77.81 % | +0.01 pp |
| DRAM Throughput (%) | 2.56 % | 2.57 % | +0.01 pp | 2.64 % | +0.08 pp | 2.57 % | +0.01 pp |
| Achieved Occupancy (%) | 11.94 % | 11.94 % | +0.00 pp | 11.94 % | +0.00 pp | 11.94 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 | 0.62 | +0.00 |
| No Eligible (%) | 62.56 % | 62.58 % | +0.02 pp | 61.41 % | -1.15 pp | 62.54 % | -0.02 pp |
| IPC Active | 1.49 inst/cycle | 1.50 inst/cycle | +0.01 | 1.54 inst/cycle | +0.05 | 1.49 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0317× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_to_eliminate_shared_memory_bank_conflicts | 1.028× | 1.028× | ✓ |  |
| B | fix_b_matrix_transposed_load_to_eliminate_bank_conflicts | 1.003× | 0.999× | ✓ |  |
| C | fix_b_matrix_load_stride_to_eliminate_shared_memory_bank_conflicts | 1.003× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | fix_b_matrix_tran... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 133600 ns | 130020 ns | -3580 ns (-2.7%) | 133760 ns | +160 ns (+0.1%) | 133570 ns | -30 ns (-0.0%) |
| SM Throughput (%) | 34.44 % | 35.67 % | +1.23 pp | 34.61 % | +0.17 pp | 34.46 % | +0.02 pp |
| Memory Throughput (%) | 77.80 % | 80.30 % | +2.50 pp | 78.21 % | +0.41 pp | 77.89 % | +0.09 pp |
| DRAM Throughput (%) | 2.56 % | 2.63 % | +0.07 pp | 2.56 % | +0.00 pp | 2.56 % | +0.00 pp |
| Achieved Occupancy (%) | 11.94 % | 11.94 % | +0.00 pp | 11.94 % | +0.00 pp | 11.94 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 | 0.62 | +0.00 |
| No Eligible (%) | 62.56 % | 61.38 % | -1.18 pp | 62.68 % | +0.12 pp | 62.59 % | +0.03 pp |
| IPC Active | 1.49 inst/cycle | 1.54 inst/cycle | +0.05 | 1.50 inst/cycle | +0.01 | 1.49 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0317× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_and_increase_num_warps_for_occupancy | 1.030× | 1.031× | ✓ |  |
| C | transpose_b_tile_into_padded_shared_memory_to_eliminate_bank_conflicts | 1.029× | 1.029× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | transpose_b_tile_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 133600 ns | 129600 ns | -4000 ns (-3.0%) | 129790 ns | -3810 ns (-2.9%) |
| SM Throughput (%) | 34.44 % | 35.61 % | +1.17 pp | 35.43 % | +0.99 pp |
| Memory Throughput (%) | 77.80 % | 80.16 % | +2.36 pp | 80.35 % | +2.55 pp |
| DRAM Throughput (%) | 2.56 % | 2.64 % | +0.08 pp | 2.65 % | +0.09 pp |
| Achieved Occupancy (%) | 11.94 % | 11.94 % | +0.00 pp | 11.94 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 |
| No Eligible (%) | 62.56 % | 61.35 % | -1.21 pp | 61.64 % | -0.92 pp |
| IPC Active | 1.49 inst/cycle | 1.54 inst/cycle | +0.05 | 1.53 inst/cycle | +0.04 |

---

## Cumulative Progress

```
Baseline:  133600 ns  (1.000×)
Iter 1     129500 ns  (1.032×)  — tma_loads_replace_scalar_pointer_arithmetic

Final best: 129500 ns  (1.0317×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_43(arg_A, arg_B, out_ptr0):
@@ -44,37 +45,35 @@

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
 
+    # A has shape [M, K], row-major: stride_am=512, stride_ak=1
+    # order=(1, 0): last dim (K) is contiguous (stride_ak=1)
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)
+    )
+    # B has shape [K, N], strides (stride_bk=1, stride_bn=512)
+    # order=(1, 0): dim-1 (N) is contiguous in smem — eliminates bank conflicts
+    B_block_ptr = tl.make_block_ptr(
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
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 512*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 256*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +85,6 @@

     # inductor generates a suffix
     xindex = idx_n + 256*idx_m
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

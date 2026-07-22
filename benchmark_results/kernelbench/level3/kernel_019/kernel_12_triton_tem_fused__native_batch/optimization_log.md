# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 69790 ns — **Best latency**: 67650 ns (**1.0316×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`tma_loads_fix_b_matrix_order_for_bank_conflict_elimination`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_12_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (246,1,1), Waves/SM: 0.62
- **Registers/thread**: 168
- **Duration**: 69790 ns
- **SM Throughput**: 33.50 % | **Memory**: 75.85 % | **DRAM**: 2.43 %
- **Achieved Occupancy**: 11.92 %
- **IPC Active**: 1.48 inst/cycle | **No Eligible**: 63.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.984× | 0.996× | ✓ |  |
| C | autotune_tile_sizes_to_reduce_register_pressure_and_bank_conflicts | 1.002× | 0.720× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 1.001× | 0.933× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 69790 ns | 70080 ns | +290 ns (+0.4%) | 96990 ns | +27200 ns (+39.0%) | 74780 ns | +4990 ns (+7.2%) |
| SM Throughput (%) | 33.50 % | 33.70 % | +0.20 pp | 24.28 % | -9.22 pp | 30.89 % | -2.61 pp |
| Memory Throughput (%) | 75.85 % | 77.46 % | +1.61 pp | 35.95 % | -39.90 pp | 70.28 % | -5.57 pp |
| DRAM Throughput (%) | 2.43 % | 2.45 % | +0.02 pp | 1.78 % | -0.65 pp | 2.27 % | -0.16 pp |
| Achieved Occupancy (%) | 11.92 % | 11.78 % | -0.14 pp | 5.88 % | -6.04 pp | 11.93 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 | 255.00 register/thread | +87 | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.93 | +0.31 | 0.47 | -0.15 | 0.62 | +0.00 |
| No Eligible (%) | 63.07 % | 63.48 % | +0.41 pp | 69.79 % | +6.72 pp | 66.19 % | +3.12 pp |
| IPC Active | 1.48 inst/cycle | 1.45 inst/cycle | -0.03 | 1.13 inst/cycle | -0.35 | 1.35 inst/cycle | -0.13 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_transposed_load_with_block_ptr_to_eliminate_bank_conflicts | 1.002× | 1.003× | ✓ |  |
| B | fix_b_matrix_transposed_load_with_block_ptr_to_eliminate_bank_conflicts | 1.026× | 0.991× | ✓ | **✓ best** |
| C | fix_b_matrix_load_and_autotune_tile_sizes_for_register_pressure | 1.003× | 0.721× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ | fix_b_matrix_tran... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 69790 ns | 69570 ns | -220 ns (-0.3%) | 70430 ns | +640 ns (+0.9%) | 96770 ns | +26980 ns (+38.7%) |
| SM Throughput (%) | 33.50 % | 33.66 % | +0.16 pp | 32.78 % | -0.72 pp | 24.28 % | -9.22 pp |
| Memory Throughput (%) | 75.85 % | 77.01 % | +1.16 pp | 74.39 % | -1.46 pp | 35.96 % | -39.89 pp |
| DRAM Throughput (%) | 2.43 % | 2.45 % | +0.02 pp | 2.41 % | -0.02 pp | 1.78 % | -0.65 pp |
| Achieved Occupancy (%) | 11.92 % | 11.77 % | -0.15 pp | 11.94 % | +0.02 pp | 5.88 % | -6.04 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 | 168.00 register/thread | +0 | 255.00 register/thread | +87 |
| Waves/SM | 0.62 | 0.93 | +0.31 | 0.62 | +0.00 | 0.47 | -0.15 |
| No Eligible (%) | 63.07 % | 63.41 % | +0.34 pp | 63.86 % | +0.79 pp | 69.79 % | +6.72 pp |
| IPC Active | 1.48 inst/cycle | 1.46 inst/cycle | -0.02 | 1.44 inst/cycle | -0.04 | 1.14 inst/cycle | -0.34 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_smem_order_and_autotune_tile_sizes_with_correct_accumulation | 1.005× | 1.003× | ✓ |  |
| B | expose_tile_sizes_as_kernel_args_and_fix_b_matrix_order_for_bank_conflicts | 1.025× | 0.986× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_smem... | Δ | expose_tile_sizes... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 69790 ns | 69570 ns | -220 ns (-0.3%) | 70820 ns | +1030 ns (+1.5%) |
| SM Throughput (%) | 33.50 % | 33.73 % | +0.23 pp | 32.89 % | -0.61 pp |
| Memory Throughput (%) | 75.85 % | 77.17 % | +1.32 pp | 74.64 % | -1.21 pp |
| DRAM Throughput (%) | 2.43 % | 2.45 % | +0.02 pp | 2.40 % | -0.03 pp |
| Achieved Occupancy (%) | 11.92 % | 11.77 % | -0.15 pp | 11.94 % | +0.02 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.93 | +0.31 | 0.62 | +0.00 |
| No Eligible (%) | 63.07 % | 63.41 % | +0.34 pp | 63.75 % | +0.68 pp |
| IPC Active | 1.48 inst/cycle | 1.46 inst/cycle | -0.02 | 1.45 inst/cycle | -0.03 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_block_ptr_order_to_match_actual_memory_layout_and_correct_accumulation | 1.004× | 1.002× | ✓ |  |
| C | fix_b_matrix_block_ptr_order_to_eliminate_bank_conflicts | 1.030× | 0.991× | ✓ |  |
| D | tma_loads_fix_b_matrix_order_for_bank_conflict_elimination | 1.030× | 1.032× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bloc... | Δ | fix_b_matrix_bloc... | Δ | tma_loads_fix_b_m... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 69790 ns | 69660 ns | -130 ns (-0.2%) | 70400 ns | +610 ns (+0.9%) | 67650 ns | -2140 ns (-3.1%) |
| SM Throughput (%) | 33.50 % | 33.63 % | +0.13 pp | 32.58 % | -0.92 pp | 34.49 % | +0.99 pp |
| Memory Throughput (%) | 75.85 % | 76.94 % | +1.09 pp | 73.95 % | -1.90 pp | 77.89 % | +2.04 pp |
| DRAM Throughput (%) | 2.43 % | 2.45 % | +0.02 pp | 2.41 % | -0.02 pp | 2.51 % | +0.08 pp |
| Achieved Occupancy (%) | 11.92 % | 11.78 % | -0.14 pp | 11.94 % | +0.02 pp | 11.93 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.93 | +0.31 | 0.62 | +0.00 | 0.62 | +0.00 |
| No Eligible (%) | 63.07 % | 63.47 % | +0.40 pp | 63.90 % | +0.83 pp | 62.00 % | -1.07 pp |
| IPC Active | 1.48 inst/cycle | 1.46 inst/cycle | -0.02 | 1.45 inst/cycle | -0.03 | 1.52 inst/cycle | +0.04 |

---

## Iteration 5

**Best so far**: 1.0316× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_smem_order_single_line_to_eliminate_bank_conflicts | 1.026× | 0.989× | ✓ |  |
| C | cap_registers_with_maxnreg_to_improve_occupancy_and_hide_bank_conflict_latency | 0.838× | 0.750× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_smem... | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 69790 ns | 70590 ns | +800 ns (+1.1%) | 93060 ns | +23270 ns (+33.3%) |
| SM Throughput (%) | 33.50 % | 32.57 % | -0.93 pp | 32.06 % | -1.44 pp |
| Memory Throughput (%) | 75.85 % | 73.93 % | -1.92 pp | 64.65 % | -11.20 pp |
| DRAM Throughput (%) | 2.43 % | 2.40 % | -0.03 pp | 1.90 % | -0.53 pp |
| Achieved Occupancy (%) | 11.92 % | 11.94 % | +0.02 pp | 11.89 % | -0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 128.00 register/thread | -40 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.47 | -0.15 |
| No Eligible (%) | 63.07 % | 63.79 % | +0.72 pp | 64.93 % | +1.86 pp |
| IPC Active | 1.48 inst/cycle | 1.45 inst/cycle | -0.03 | 1.40 inst/cycle | -0.08 |

---

## Cumulative Progress

```
Baseline:  69790 ns  (1.000×)
Iter 4     67650 ns  (1.032×)  — tma_loads_fix_b_matrix_order_for_bank_conflict_elimination

Final best: 67650 ns  (1.0316×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_12(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -44,37 +54,36 @@

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
 
+    # A: shape (M=7840, K=256), strides (stride_am=256, stride_ak=1), row-major
+    # order=(1, 0): K-dim (index 1) is contiguous in memory
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B: shape (K=256, N=256), strides (stride_bk=1, stride_bn=256)
+    # B[k,n] = B_ptr + k*1 + n*256 => K is fast/contiguous dim in memory
+    # order=(0, 1): K-dim (index 0) is fast within the tile -> matches stride_bk=1, no bank conflicts
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 256*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 256*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
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
@@ -86,8 +95,6 @@

     # inductor generates a suffix
     xindex = idx_n + 256*idx_m
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

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1860000 ns — **Best latency**: 1810000 ns (**1.0276×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`shared_memory_padding_to_eliminate_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_4_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2940,1,1), Waves/SM: 7.42
- **Registers/thread**: 168
- **Duration**: 1860000 ns
- **SM Throughput**: 38.64 % | **Memory**: 87.23 % | **DRAM**: 4.72 %
- **Achieved Occupancy**: 18.22 %
- **IPC Active**: 1.60 inst/cycle | **No Eligible**: 60.09 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 0.842× | ✓ |  |
| B | tma_loads_with_autotuned_tile_sizes | 1.028× | 1.028× | ✓ | **✓ best** |
| C | eviction_policies_for_l2_residency | 1.000× | 1.000× | ✓ |  |
| D | add_autotune_reduce_register_pressure_and_bank_conflicts | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ | eviction_policies... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1860000 ns | 2210000 ns | +350000 ns (+18.8%) | 1810000 ns | -50000 ns (-2.7%) | 1860000 ns | +0 ns (+0.0%) | 1860000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 38.64 % | 32.29 % | -6.35 pp | 39.67 % | +1.03 pp | 38.57 % | -0.07 pp | 38.67 % | +0.03 pp |
| Memory Throughput (%) | 87.23 % | 73.37 % | -13.86 pp | 89.58 % | +2.35 pp | 87.07 % | -0.16 pp | 87.29 % | +0.06 pp |
| DRAM Throughput (%) | 4.72 % | 3.98 % | -0.74 pp | 4.85 % | +0.13 pp | 4.69 % | -0.03 pp | 4.72 % | +0.00 pp |
| Achieved Occupancy (%) | 18.22 % | 12.48 % | -5.74 pp | 18.21 % | -0.01 pp | 18.22 % | +0.00 pp | 18.22 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 242.00 register/thread | +74 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 7.42 | 11.14 | +3.72 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 60.09 % | 65.19 % | +5.10 pp | 58.97 % | -1.12 pp | 60.08 % | -0.01 pp | 60.09 % | +0.00 pp |
| IPC Active | 1.60 inst/cycle | 1.39 inst/cycle | -0.21 | 1.64 inst/cycle | +0.04 | 1.60 inst/cycle | +0.00 | 1.60 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0276× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_tile_sizes_and_fix_b_access_order_to_eliminate_bank_conflicts | 1.028× | 1.028× | ✓ | **✓ best** |
| C | reduce_block_k_to_eliminate_shared_memory_bank_conflicts | 0.954× | 0.881× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_tile_sizes... | Δ | reduce_block_k_to... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1860000 ns | 1810000 ns | -50000 ns (-2.7%) | 2110000 ns | +250000 ns (+13.4%) |
| SM Throughput (%) | 38.64 % | 39.68 % | +1.04 pp | 34.43 % | -4.21 pp |
| Memory Throughput (%) | 87.23 % | 89.60 % | +2.37 pp | 77.22 % | -10.01 pp |
| DRAM Throughput (%) | 4.72 % | 4.85 % | +0.13 pp | 4.15 % | -0.57 pp |
| Achieved Occupancy (%) | 18.22 % | 18.21 % | -0.01 pp | 12.42 % | -5.80 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 242.00 register/thread | +74 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 11.14 | +3.72 |
| No Eligible (%) | 60.09 % | 58.98 % | -1.11 pp | 63.11 % | +3.02 pp |
| IPC Active | 1.60 inst/cycle | 1.64 inst/cycle | +0.04 | 1.48 inst/cycle | -0.12 |

---

## Iteration 3

**Best so far**: 1.0276× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_padding_to_eliminate_bank_conflicts | 1.028× | 1.028× | ✓ |  |
| C | explicit_smem_staging_with_padding_to_eliminate_b_bank_conflicts_and_cap_registers | 0.781× | 0.762× | ✓ |  |
| D | tma_loads_via_block_pointers_to_eliminate_bank_conflicts | 0.949× | 0.949× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_pad... | Δ | explicit_smem_sta... | Δ | tma_loads_via_blo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1860000 ns | 1810000 ns | -50000 ns (-2.7%) | 2440000 ns | +580000 ns (+31.2%) | 1960000 ns | +100000 ns (+5.4%) |
| SM Throughput (%) | 38.64 % | 39.71 % | +1.07 pp | 35.41 % | -3.23 pp | 36.79 % | -1.85 pp |
| Memory Throughput (%) | 87.23 % | 89.66 % | +2.43 pp | 88.22 % | +0.99 pp | 82.91 % | -4.32 pp |
| DRAM Throughput (%) | 4.72 % | 4.84 % | +0.12 pp | 4.58 % | -0.14 pp | 4.51 % | -0.21 pp |
| Achieved Occupancy (%) | 18.22 % | 18.21 % | -0.01 pp | 24.11 % | +5.89 pp | 18.14 % | -0.08 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 128.00 register/thread | -40 | 168.00 register/thread | +0 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 5.57 | -1.85 | 7.42 | +0.00 |
| No Eligible (%) | 60.09 % | 58.98 % | -1.11 pp | 63.32 % | +3.23 pp | 62.14 % | +2.05 pp |
| IPC Active | 1.60 inst/cycle | 1.64 inst/cycle | +0.04 | 1.47 inst/cycle | -0.13 | 1.51 inst/cycle | -0.09 |

---

## Iteration 4

**Best so far**: 1.0276× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_block_ptr_memory_order_to_eliminate_bank_conflicts | 0.764× | 0.762× | ✓ |  |
| C | remove_maxnreg_to_eliminate_local_memory_spill_and_transpose_b_load | 0.954× | 0.881× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_m... | Δ | remove_maxnreg_to... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1860000 ns | 2440000 ns | +580000 ns (+31.2%) | 2110000 ns | +250000 ns (+13.4%) |
| SM Throughput (%) | 38.64 % | 35.64 % | -3.00 pp | 34.40 % | -4.24 pp |
| Memory Throughput (%) | 87.23 % | 88.25 % | +1.02 pp | 77.14 % | -10.09 pp |
| DRAM Throughput (%) | 4.72 % | 4.58 % | -0.14 pp | 4.16 % | -0.56 pp |
| Achieved Occupancy (%) | 18.22 % | 24.10 % | +5.88 pp | 12.43 % | -5.79 pp |
| Registers/Thread | 168.00 register/thread | 128.00 register/thread | -40 | 242.00 register/thread | +74 |
| Waves/SM | 7.42 | 5.57 | -1.85 | 11.14 | +3.72 |
| No Eligible (%) | 60.09 % | 63.08 % | +2.99 pp | 63.09 % | +3.00 pp |
| IPC Active | 1.60 inst/cycle | 1.48 inst/cycle | -0.12 | 1.48 inst/cycle | -0.12 |

---

## Cumulative Progress

```
Baseline:  1860000 ns  (1.000×)
Iter 1     1810000 ns  (1.028×)  — tma_loads_with_autotuned_tile_sizes

Final best: 1810000 ns  (1.0276×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,76 +5,69 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
-    ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 128
-    BLOCK_N : tl.constexpr = 64
-    BLOCK_K : tl.constexpr = 32
-    GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
-    INDEX_DTYPE : tl.constexpr = tl.int32
-    A = arg_A
-    B = arg_B
+    ACC_TYPE: tl.constexpr = tl.float32
+    BLOCK_M: tl.constexpr = 128
+    BLOCK_N: tl.constexpr = 64
+    BLOCK_K: tl.constexpr = 32
+    GROUP_M: tl.constexpr = 8
+    ALLOW_TF32: tl.constexpr = False
+    INDEX_DTYPE: tl.constexpr = tl.int32
 
     M = 125440
     N = 192
     K = 672
+
     if M * N == 0:
-        # early exit due to zero-size input(s)
         return
-    stride_am = 672
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 672
 
     # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    grid_m = tl.cdiv(M, BLOCK_M)
+    grid_n = tl.cdiv(N, BLOCK_N)
 
     # re-order program ID for better L2 performance
     width = GROUP_M * grid_n
     group_id = pid // width
     group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
     pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
+    pid_n = (pid % width) // group_size
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
+    # Create block pointers for A (shape: M x K, row-major strides: stride_am=672, stride_ak=1)
+    A_ptr = tl.make_block_ptr(
+        base=arg_A,
+        shape=(M, K),
+        strides=(672, 1),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+
+    # Create block pointers for B (shape: K x N, strides: stride_bk=1, stride_bn=672)
+    # order=(0, 1) means K-dimension is the fast axis in shared memory layout,
+    # so consecutive threads read consecutive K-elements landing on consecutive banks,
+    # eliminating the 8-way bank conflicts caused by stride-672 N-dimension
+    B_ptr = tl.make_block_ptr(
+        base=arg_B,
+        shape=(K, N),
+        strides=(1, 672),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),  # CHANGED from (1, 0): K is fast axis in smem -> consecutive banks
+    )
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 672*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 192*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 672*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+    for _ in range(tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_ptr, boundary_check=(0, 1))
+        b = tl.load(B_ptr, boundary_check=(0, 1))
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
+        B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -84,10 +77,8 @@

     mask = (idx_m < M) & (idx_n < N)
 
     # inductor generates a suffix
-    xindex = idx_n + 192*idx_m
+    xindex = idx_n + 192 * idx_m
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

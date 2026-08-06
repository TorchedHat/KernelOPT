# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 24510 ns — **Best latency**: 24450 ns (**1.0025×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_to_increase_waves_and_reduce_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_3_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (980,1,1), Waves/SM: 0.82
- **Registers/thread**: 56
- **Duration**: 24510 ns
- **SM Throughput**: 15.83 % | **Memory**: 72.93 % | **DRAM**: 13.39 %
- **Achieved Occupancy**: 44.56 %
- **IPC Active**: 0.75 inst/cycle | **No Eligible**: 81.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.991× | 1.004× | ✓ |  |
| B | tma_loads_via_block_ptr_and_tile_size_tuning | 1.065× | 1.055× | ✓ |  |
| D | autotune_tile_sizes_to_increase_waves_and_reduce_bank_conflicts | 1.263× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_via_blo... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24510 ns | 24420 ns | -90 ns (-0.4%) | 23230 ns | -1280 ns (-5.2%) | 24450 ns | -60 ns (-0.2%) |
| SM Throughput (%) | 15.83 % | 15.90 % | +0.07 pp | 16.57 % | +0.74 pp | 15.81 % | -0.02 pp |
| Memory Throughput (%) | 72.93 % | 73.23 % | +0.30 pp | 76.43 % | +3.50 pp | 72.82 % | -0.11 pp |
| DRAM Throughput (%) | 13.39 % | 13.46 % | +0.07 pp | 14.14 % | +0.75 pp | 13.43 % | +0.04 pp |
| Achieved Occupancy (%) | 44.56 % | 44.62 % | +0.06 pp | 35.72 % | -8.84 pp | 44.56 % | +0.00 pp |
| Registers/Thread | 56.00 register/thread | 56.00 register/thread | +0 | 65.00 register/thread | +9 | 56.00 register/thread | +0 |
| Waves/SM | 0.82 | 0.82 | +0.00 | 1.06 | +0.24 | 0.82 | +0.00 |
| No Eligible (%) | 81.12 % | 81.12 % | +0.00 pp | 80.40 % | -0.72 pp | 81.13 % | +0.01 pp |
| IPC Active | 0.75 inst/cycle | 0.75 inst/cycle | +0.00 | 0.78 inst/cycle | +0.03 | 0.75 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0025× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_to_improve_occupancy_and_fix_b_matrix_smem_layout | 1.054× | 1.055× | ✓ |  |
| B | reduce_register_pressure_to_increase_occupancy_and_hide_mio_stalls | 0.991× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 24510 ns | 23230 ns | -1280 ns (-5.2%) | 24420 ns | -90 ns (-0.4%) |
| SM Throughput (%) | 15.83 % | 16.43 % | +0.60 pp | 15.87 % | +0.04 pp |
| Memory Throughput (%) | 72.93 % | 75.82 % | +2.89 pp | 73.12 % | +0.19 pp |
| DRAM Throughput (%) | 13.39 % | 14.12 % | +0.73 pp | 13.45 % | +0.06 pp |
| Achieved Occupancy (%) | 44.56 % | 35.66 % | -8.90 pp | 44.71 % | +0.15 pp |
| Registers/Thread | 56.00 register/thread | 65.00 register/thread | +9 | 56.00 register/thread | +0 |
| Waves/SM | 0.82 | 1.06 | +0.24 | 0.82 | +0.00 |
| No Eligible (%) | 81.12 % | 80.40 % | -0.72 pp | 81.27 % | +0.15 pp |
| IPC Active | 0.75 inst/cycle | 0.78 inst/cycle | +0.03 | 0.75 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0025× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reshape_tile_geometry_to_reduce_smem_bank_conflicts_and_increase_arithmetic_intensity | 0.649× | 0.604× | ✓ |  |
| C | smem_padding_to_eliminate_structural_bank_conflicts | 1.000× | 1.004× | ✓ |  |
| D | smem_padding_to_eliminate_structural_bank_conflicts_on_b_matrix | 1.067× | 1.054× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reshape_tile_geom... | Δ | smem_padding_to_e... | Δ | smem_padding_to_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24510 ns | 40610 ns | +16100 ns (+65.7%) | 24420 ns | -90 ns (-0.4%) | 23260 ns | -1250 ns (-5.1%) |
| SM Throughput (%) | 15.83 % | 17.46 % | +1.63 pp | 15.94 % | +0.11 pp | 16.59 % | +0.76 pp |
| Memory Throughput (%) | 72.93 % | 83.12 % | +10.19 pp | 72.97 % | +0.04 pp | 76.54 % | +3.61 pp |
| DRAM Throughput (%) | 13.39 % | 8.08 % | -5.31 pp | 13.46 % | +0.07 pp | 14.13 % | +0.74 pp |
| Achieved Occupancy (%) | 44.56 % | 26.37 % | -18.19 pp | 44.68 % | +0.12 pp | 35.72 % | -8.84 pp |
| Registers/Thread | 56.00 register/thread | 96.00 register/thread | +40 | 56.00 register/thread | +0 | 65.00 register/thread | +9 |
| Waves/SM | 0.82 | 1.48 | +0.66 | 0.82 | +0.00 | 1.06 | +0.24 |
| No Eligible (%) | 81.12 % | 80.11 % | -1.01 pp | 80.95 % | -0.17 pp | 80.42 % | -0.70 pp |
| IPC Active | 0.75 inst/cycle | 0.79 inst/cycle | +0.04 | 0.76 inst/cycle | +0.01 | 0.78 inst/cycle | +0.03 |

---

## Cumulative Progress

```
Baseline:  24510 ns  (1.000×)
Iter 1     24450 ns  (1.002×)  — autotune_tile_sizes_to_increase_waves_and_reduce_bank_conflicts

Final best: 24450 ns  (1.0025×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3(arg_A, arg_B, out_ptr0):
@@ -16,8 +17,6 @@

     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
-    A = arg_A
-    B = arg_B
 
     M = 125440
     N = 16
@@ -25,10 +24,6 @@

     if M * N == 0:
         # early exit due to zero-size input(s)
         return
-    stride_am = 32
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 32
 
     # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
@@ -44,50 +39,46 @@

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
+    # A: shape (M, K), strides (32, 1) — row-major, TMA block_ptr
+    a_ptr = tl.make_block_ptr(
+        base=arg_A,
+        shape=(M, K),
+        strides=(32, 1),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B: shape (K, N), strides (1, 32) — original stride_bk=1, stride_bn=32
+    # order=(1, 0): N is the fast-varying dimension in shared memory,
+    # matching how tl.dot reads B and enabling TMA swizzle to eliminate bank conflicts
+    b_ptr = tl.make_block_ptr(
+        base=arg_B,
+        shape=(K, N),
+        strides=(1, 32),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+    for _ in range(0, tl.cdiv(K, BLOCK_K)):
+        a = tl.load(a_ptr, boundary_check=(0, 1))
+        b = tl.load(b_ptr, boundary_check=(0, 1))
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        a_ptr = tl.advance(a_ptr, (0, BLOCK_K))
+        b_ptr = tl.advance(b_ptr, (BLOCK_K, 0))
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 32*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 16*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 32*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
-
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
-
-    # inductor generates a suffix
-    xindex = idx_n + 16*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    # store result via block_ptr
+    out_ptr2 = tl.make_block_ptr(
+        base=out_ptr0,
+        shape=(M, N),
+        strides=(16, 1),
+        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
+        block_shape=(BLOCK_M, BLOCK_N),
+        order=(1, 0),
+    )
+    tl.store(out_ptr2, acc, boundary_check=(0, 1))
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

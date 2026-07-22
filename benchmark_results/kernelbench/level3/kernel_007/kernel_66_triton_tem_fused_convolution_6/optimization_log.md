# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_66` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 62880 ns — **Best latency**: 71230 ns (**0.8828×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_66`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_66_triton_tem_fused_convolution_6/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (93,1,1), Waves/SM: 0.18
- **Registers/thread**: 128
- **Duration**: 62880 ns
- **SM Throughput**: 15.51 % | **Memory**: 59.89 % | **DRAM**: 1.46 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 77.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 1.002× | 0.817× | ✓ | **✓ best** |
| C | autotune_tile_sizes_to_reduce_register_pressure_and_increase_occupancy | 0.998× | 0.813× | ✓ |  |
| D | autotune_tile_sizes_and_warps_to_reduce_register_pressure_and_improve_occupancy | 0.997× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 62880 ns | 76990 ns | +14110 ns (+22.4%) | 77310 ns | +14430 ns (+22.9%) | 62720 ns | -160 ns (-0.3%) |
| SM Throughput (%) | 15.51 % | 12.15 % | -3.36 pp | 12.18 % | -3.33 pp | 15.57 % | +0.06 pp |
| Memory Throughput (%) | 59.89 % | 27.16 % | -32.73 pp | 27.23 % | -32.66 pp | 60.14 % | +0.25 pp |
| DRAM Throughput (%) | 1.46 % | 1.20 % | -0.26 pp | 1.19 % | -0.27 pp | 1.46 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 214.00 register/thread | +86 | 214.00 register/thread | +86 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 77.08 % | 61.67 % | -15.41 pp | 61.75 % | -15.33 pp | 77.02 % | -0.06 pp |
| IPC Active | 0.92 inst/cycle | 0.76 inst/cycle | -0.16 | 0.77 inst/cycle | -0.15 | 0.92 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_via_smem_padding | 1.016× | 1.012× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 62880 ns | 62140 ns | -740 ns (-1.2%) |
| SM Throughput (%) | 15.51 % | 15.63 % | +0.12 pp |
| Memory Throughput (%) | 59.89 % | 60.09 % | +0.20 pp |
| DRAM Throughput (%) | 1.46 % | 1.48 % | +0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 |
| No Eligible (%) | 77.08 % | 76.86 % | -0.22 pp |
| IPC Active | 0.92 inst/cycle | 0.93 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0119× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_block_ptr_smem_layout_order_to_eliminate_bank_conflicts | 1.011× | 1.009× | ✓ |  |
| C | fix_shared_memory_access_order_for_coalesced_mma_reads | 1.014× | 1.007× | ✓ |  |
| D | increase_block_k_and_cap_registers_to_improve_occupancy_and_reduce_bank_conflicts | 0.899× | 0.526× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_block_ptr_sme... | Δ | fix_shared_memory... | Δ | increase_block_k_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 62880 ns | 62340 ns | -540 ns (-0.9%) | 62430 ns | -450 ns (-0.7%) | 119620 ns | +56740 ns (+90.2%) |
| SM Throughput (%) | 15.51 % | 15.67 % | +0.16 pp | 15.70 % | +0.19 pp | 13.29 % | -2.22 pp |
| Memory Throughput (%) | 59.89 % | 60.24 % | +0.35 pp | 60.34 % | +0.45 pp | 45.77 % | -14.12 pp |
| DRAM Throughput (%) | 1.46 % | 1.47 % | +0.01 pp | 1.47 % | +0.01 pp | 0.78 % | -0.68 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 64.00 register/thread | -64 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.23 | +0.05 |
| No Eligible (%) | 77.08 % | 76.86 % | -0.22 pp | 76.90 % | -0.18 pp | 83.20 % | +6.12 pp |
| IPC Active | 0.92 inst/cycle | 0.93 inst/cycle | +0.01 | 0.93 inst/cycle | +0.01 | 0.67 inst/cycle | -0.25 |

---

## Iteration 4

**Best so far**: 1.0119× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | tf32x3_tensor_cores_for_fp32_accurate_gemm | 1.849× | 0.880× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32x3_tensor_cor... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 62880 ns | 71420 ns | +8540 ns (+13.6%) |
| SM Throughput (%) | 15.51 % | 9.64 % | -5.87 pp |
| Memory Throughput (%) | 59.89 % | 25.17 % | -34.72 pp |
| DRAM Throughput (%) | 1.46 % | 1.30 % | -0.16 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 64.00 register/thread | -64 |
| Waves/SM | 0.18 | 0.35 | +0.17 |
| No Eligible (%) | 77.08 % | 88.83 % | +11.75 pp |
| IPC Active | 0.92 inst/cycle | 0.44 inst/cycle | -0.48 |

---

## Iteration 5

**Best so far**: 1.0119× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_single_pass_replace_tf32x3 | 1.852× | 0.883× | ✓ | **✓ best** |
| D | fix_tf32x3_correctness_bug_and_increase_tile_sizes_for_parallelism | 0.835× | 0.822× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ | fix_tf32x3_correc... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 62880 ns | 71230 ns | +8350 ns (+13.3%) | 76510 ns | +13630 ns (+21.7%) |
| SM Throughput (%) | 15.51 % | 9.57 % | -5.94 pp | 13.55 % | -1.96 pp |
| Memory Throughput (%) | 59.89 % | 25.23 % | -34.66 pp | 52.79 % | -7.10 pp |
| DRAM Throughput (%) | 1.46 % | 1.31 % | -0.15 pp | 1.20 % | -0.26 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 64.00 register/thread | -64 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.35 | +0.17 | 0.23 | +0.05 |
| No Eligible (%) | 77.08 % | 88.83 % | +11.75 pp | 79.31 % | +2.23 pp |
| IPC Active | 0.92 inst/cycle | 0.45 inst/cycle | -0.47 | 0.83 inst/cycle | -0.09 |

---

## Cumulative Progress

```
Baseline:  62880 ns  (1.000×)
Iter 2     62140 ns  (1.012×)  — fix_shared_memory_bank_conflicts_via_smem_padding

Final best: 71230 ns  (0.8828×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3, maxnreg=64),
+        triton.Config({}, num_warps=4, num_stages=4, maxnreg=64),
+        triton.Config({}, num_warps=4, num_stages=2, maxnreg=64),
+        triton.Config({}, num_warps=8, num_stages=3, maxnreg=64),
+        triton.Config({}, num_warps=8, num_stages=4, maxnreg=64),
+        triton.Config({}, num_warps=8, num_stages=2, maxnreg=64),
+        triton.Config({}, num_warps=2, num_stages=3, maxnreg=64),
+        triton.Config({}, num_warps=2, num_stages=4, maxnreg=64),
+        triton.Config({}, num_warps=4, num_stages=1, maxnreg=64),
+        triton.Config({}, num_warps=8, num_stages=1, maxnreg=64),
+        triton.Config({}, num_warps=4, num_stages=3, maxnreg=128),
+        triton.Config({}, num_warps=8, num_stages=3, maxnreg=128),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_66(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
@@ -12,9 +30,8 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 64
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
     A = arg_A
     B = arg_B
@@ -46,38 +63,38 @@

 
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Create block pointers BEFORE the k loop to use hardware-optimal smem layout
+    # order=(1,0) for A: K dimension is innermost in smem -> coalesced, no bank conflicts
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)
+    )
+    # order=(0,1) for B: K dimension is innermost in smem -> coalesced, no bank conflicts
+    # B global layout is column-major: B[k,n] stored at B_ptr[n*528 + k]
+    # shape=(K,N), strides=(stride_bk=1, stride_bn=528)
+    B_block_ptr = tl.make_block_ptr(
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
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 528*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 160*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 528*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        # boundary_check=(0,1) required because K=528 is not divisible by BLOCK_K=64
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc += tl.dot(a, b, input_precision='tf32x3', out_dtype=ACC_TYPE)
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -91,8 +108,6 @@

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

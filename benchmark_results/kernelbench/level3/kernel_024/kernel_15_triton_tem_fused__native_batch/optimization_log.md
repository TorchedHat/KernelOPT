# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 34240 ns — **Best latency**: 43650 ns (**0.7844×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_15`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_15_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5,1,1), Waves/SM: 0.01
- **Registers/thread**: 62
- **Duration**: 34240 ns
- **SM Throughput**: 0.31 % | **Memory**: 3.27 % | **DRAM**: 0.22 %
- **Achieved Occupancy**: 6.12 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 91.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | split_k_parallelism | 1.312× | 1.213× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | split_k_paralleli... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 34240 ns | 28220 ns | -6020 ns (-17.6%) |
| SM Throughput (%) | 0.31 % | 0.32 % | +0.01 pp |
| Memory Throughput (%) | 3.27 % | 2.16 % | -1.11 pp |
| DRAM Throughput (%) | 0.22 % | 0.27 % | +0.05 pp |
| Achieved Occupancy (%) | 6.12 % | 3.09 % | -3.03 pp |
| Registers/Thread | 62.00 register/thread | 96.00 register/thread | +34 |
| Waves/SM | 0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.27 % | 81.21 % | -10.06 pp |
| IPC Active | 0.34 inst/cycle | 0.37 inst/cycle | +0.03 |

---

## Iteration 2

**Best so far**: 1.2133× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.305× | 1.194× | ✓ |  |
| B | persistent_kernel_sm_tile_loop | 1.008× | 1.019× | ✓ |  |
| C | reduce_block_k_to_cut_smem_bank_conflicts_and_raise_occupancy | 1.443× | 0.956× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | persistent_kernel... | Δ | reduce_block_k_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 34240 ns | 28670 ns | -5570 ns (-16.3%) | 33600 ns | -640 ns (-1.9%) | 35810 ns | +1570 ns (+4.6%) |
| SM Throughput (%) | 0.31 % | 0.33 % | +0.02 pp | 0.29 % | -0.02 pp | 0.23 % | -0.08 pp |
| Memory Throughput (%) | 3.27 % | 2.18 % | -1.09 pp | 3.30 % | +0.03 pp | 1.61 % | -1.66 pp |
| DRAM Throughput (%) | 0.22 % | 0.27 % | +0.05 pp | 0.22 % | +0.00 pp | 0.20 % | -0.02 pp |
| Achieved Occupancy (%) | 6.12 % | 3.09 % | -3.03 pp | 6.15 % | +0.03 pp | 3.09 % | -3.03 pp |
| Registers/Thread | 62.00 register/thread | 96.00 register/thread | +34 | 56.00 register/thread | -6 | 64.00 register/thread | +2 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.00 | -0.01 |
| No Eligible (%) | 91.27 % | 81.54 % | -9.73 pp | 91.64 % | +0.37 pp | 87.14 % | -4.13 pp |
| IPC Active | 0.34 inst/cycle | 0.36 inst/cycle | +0.02 | 0.33 inst/cycle | -0.01 | 0.25 inst/cycle | -0.09 |

---

## Iteration 3

**Best so far**: 1.2133× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_transposed_load_to_eliminate_smem_bank_conflicts | 1.011× | 1.020× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 34240 ns | 33570 ns | -670 ns (-2.0%) |
| SM Throughput (%) | 0.31 % | 0.29 % | -0.02 pp |
| Memory Throughput (%) | 3.27 % | 3.31 % | +0.04 pp |
| DRAM Throughput (%) | 0.22 % | 0.22 % | +0.00 pp |
| Achieved Occupancy (%) | 6.12 % | 6.14 % | +0.02 pp |
| Registers/Thread | 62.00 register/thread | 56.00 register/thread | -6 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 91.27 % | 91.73 % | +0.46 pp |
| IPC Active | 0.34 inst/cycle | 0.32 inst/cycle | -0.02 |

---

## Iteration 4

**Best so far**: 1.2133× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broadcast_bug_and_autotune_block_n_with_num_warps_stages | 1.507× | 1.328× | ✓ |  |
| C | split_k_k_dimension_parallelism_with_atomic_reduction | 1.508× | 0.784× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_broadcast_bug... | Δ | split_k_k_dimensi... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 34240 ns | 25790 ns | -8450 ns (-24.7%) | 43650 ns | +9410 ns (+27.5%) |
| SM Throughput (%) | 0.31 % | 0.20 % | -0.11 pp | 0.19 % | -0.12 pp |
| Memory Throughput (%) | 3.27 % | 2.29 % | -0.98 pp | 1.42 % | -1.85 pp |
| DRAM Throughput (%) | 0.22 % | 0.28 % | +0.06 pp | 0.18 % | -0.04 pp |
| Achieved Occupancy (%) | 6.12 % | 3.05 % | -3.07 pp | 1.56 % | -4.56 pp |
| Registers/Thread | 62.00 register/thread | 78.00 register/thread | +16 | 128.00 register/thread | +66 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.27 % | 88.11 % | -3.16 pp | 79.17 % | -12.10 pp |
| IPC Active | 0.34 inst/cycle | 0.23 inst/cycle | -0.11 | 0.21 inst/cycle | -0.13 |

---

## Iteration 5

**Best so far**: 1.2133× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  34240 ns  (1.000×)
Iter 1     28220 ns  (1.213×)  — split_k_parallelism

Final best: 43650 ns  (0.7844×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({}, num_warps=1, num_stages=1),
+    triton.Config({}, num_warps=2, num_stages=1),
+    triton.Config({}, num_warps=4, num_stages=1),
+    triton.Config({}, num_warps=8, num_stages=1),
+    triton.Config({}, num_warps=1, num_stages=2),
+    triton.Config({}, num_warps=2, num_stages=2),
+    triton.Config({}, num_warps=4, num_stages=2),
+    triton.Config({}, num_warps=8, num_stages=2),
+    triton.Config({}, num_warps=2, num_stages=3),
+    triton.Config({}, num_warps=4, num_stages=3),
+], key=[])
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_15(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
     EVEN_K : tl.constexpr = False
@@ -16,6 +29,8 @@

     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
+    NUM_TILES : tl.constexpr = 5
+    NUM_SMS : tl.constexpr = 132
     A = arg_A
     B = arg_B
 
@@ -30,82 +45,85 @@

     stride_bk = 1
     stride_bn = 576
 
-    # based on triton.ops.matmul
-    pid = tl.program_id(0).to(INDEX_DTYPE)
+    # persistent kernel: one block per SM, loops over assigned tiles
+    sm_id = tl.program_id(0).to(INDEX_DTYPE)
     grid_m = (M + BLOCK_M - 1) // BLOCK_M
     grid_n = (N + BLOCK_N - 1) // BLOCK_N
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    for tile_id in range(sm_id, NUM_TILES, NUM_SMS):
+        pid = tile_id
 
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
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
+        # re-order program ID for better L2 performance
+        width = GROUP_M * grid_n
+        group_id = pid // width
+        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
+        pid_m = group_id * GROUP_M + (pid % group_size)
+        pid_n = (pid % width) // (group_size)
+        tl.assume(pid_m >= 0)
+        tl.assume(pid_n >= 0)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+        if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
+            offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
+        else:
+            offs_a_m = rm % M
+        if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
+            offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
+        else:
+            offs_b_n = rn % N
+        offs_k = tl.arange(0, BLOCK_K)
+        # CRITICAL: acc must be inside the tile loop to reset per tile
+        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 576*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+            
+            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
+            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
+            
+            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
+            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
+            idx_m = offs_a_m[:, None]
+            idx_n = a_k_idx_vals
+            xindex = idx_n + 576*idx_m
+            a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+
+            idx_m = b_k_idx_vals          # shape [BLOCK_K, 1], k-offsets
+            idx_n = offs_b_n[None, :]     # shape [1, BLOCK_N], n-offsets
+            # B layout: stride_bk=1, stride_bn=576 => B[k,n] at offset k + 576*n
+            xindex = idx_m + 576 * idx_n  # broadcasts to [BLOCK_K, BLOCK_N] automatically
+            b = tl.load(B + xindex, mask=b_mask, other=0.0)
+
+            
+            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+            
+
+        # rematerialize rm and rn to save registers
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+        idx_m = rm[:, None]
+        idx_n = rn[None, :]
+        mask = (idx_m < M) & (idx_n < N)
+
+        # inductor generates a suffix
         xindex = idx_n + 144*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 576*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
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
-    xindex = idx_n + 144*idx_m
-    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp2 = tl.load(in_ptr3 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp13 = tl.load(in_ptr5 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp1 = acc - tmp0
-    tmp3 = tl.full([1], 1e-05, tl.float32)
-    tmp4 = tmp2 + tmp3
-    tmp5 = tl.sqrt_rn(tmp4)
-    tmp6 = tl.full([1], 1, tl.int32)
-    tmp7 = (tmp6 / tmp5)
-    tmp8 = tl.full([1], 1.0, tl.float32)
-    tmp9 = tmp7 * tmp8
-    tmp10 = tmp1 * tmp9
-    tmp12 = tmp10 * tmp11
-    tmp14 = tmp12 + tmp13
-    tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 144*idx_m, [BLOCK_M, BLOCK_N])), tmp14, mask)
-
-
+        tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
+        tmp2 = tl.load(in_ptr3 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
+        tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
+        tmp13 = tl.load(in_ptr5 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
+        tmp1 = acc - tmp0
+        tmp3 = tl.full([1], 1e-05, tl.float32)
+        tmp4 = tmp2 + tmp3
+        tmp5 = tl.sqrt_rn(tmp4)
+        tmp6 = tl.full([1], 1, tl.int32)
+        tmp7 = (tmp6 / tmp5)
+        tmp8 = tl.full([1], 1.0, tl.float32)
+        tmp9 = tmp7 * tmp8
+        tmp10 = tmp1 * tmp9
+        tmp12 = tmp10 * tmp11
+        tmp14 = tmp12 + tmp13
+        tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 144*idx_m, [BLOCK_M, BLOCK_N])), tmp14, mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

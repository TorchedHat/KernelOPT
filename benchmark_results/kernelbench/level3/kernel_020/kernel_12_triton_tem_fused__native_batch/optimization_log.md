# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 41860 ns — **Best latency**: 41440 ns (**1.0101×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`tma_loads_via_launcher_tensor_descriptors`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_12_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.62
- **Registers/thread**: 154
- **Duration**: 41860 ns
- **SM Throughput**: 18.83 % | **Memory**: 78.87 % | **DRAM**: 8.83 %
- **Achieved Occupancy**: 11.79 %
- **IPC Active**: 0.86 inst/cycle | **No Eligible**: 78.51 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.005× | 1.006× | ✓ | **✓ best** |
| C | eviction_policy_streaming_loads | 1.004× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | eviction_policy_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 41860 ns | 41630 ns | -230 ns (-0.5%) | 41730 ns | -130 ns (-0.3%) |
| SM Throughput (%) | 18.83 % | 18.64 % | -0.19 pp | 18.90 % | +0.07 pp |
| Memory Throughput (%) | 78.87 % | 78.05 % | -0.82 pp | 79.06 % | +0.19 pp |
| DRAM Throughput (%) | 8.83 % | 8.88 % | +0.05 pp | 8.85 % | +0.02 pp |
| Achieved Occupancy (%) | 11.79 % | 11.79 % | +0.00 pp | 11.79 % | +0.00 pp |
| Registers/Thread | 154.00 register/thread | 154.00 register/thread | +0 | 154.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 |
| No Eligible (%) | 78.51 % | 78.53 % | +0.02 pp | 78.65 % | +0.14 pp |
| IPC Active | 0.86 inst/cycle | 0.86 inst/cycle | +0.00 | 0.85 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0055× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_indexing_to_eliminate_bank_conflicts | 1.000× | 1.003× | ✓ |  |
| C | fix_b_matrix_load_stride_to_eliminate_bank_conflicts | 1.005× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 41860 ns | 41730 ns | -130 ns (-0.3%) | 41950 ns | +90 ns (+0.2%) |
| SM Throughput (%) | 18.83 % | 18.86 % | +0.03 pp | 18.90 % | +0.07 pp |
| Memory Throughput (%) | 78.87 % | 79.05 % | +0.18 pp | 79.06 % | +0.19 pp |
| DRAM Throughput (%) | 8.83 % | 8.85 % | +0.02 pp | 8.80 % | -0.03 pp |
| Achieved Occupancy (%) | 11.79 % | 11.79 % | +0.00 pp | 11.79 % | +0.00 pp |
| Registers/Thread | 154.00 register/thread | 154.00 register/thread | +0 | 154.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 |
| No Eligible (%) | 78.51 % | 78.55 % | +0.04 pp | 78.67 % | +0.16 pp |
| IPC Active | 0.86 inst/cycle | 0.86 inst/cycle | +0.00 | 0.86 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0055× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | transpose_b_load_to_eliminate_smem_bank_conflicts | 1.005× | 1.002× | ✓ |  |
| B | tma_loads_correct_api_inside_jit_kernel | 1.010× | 1.013× | ✓ | **✓ best** |
| D | tma_loads_inside_kernel_for_b_matrix | 0.990× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | transpose_b_load_... | Δ | tma_loads_correct... | Δ | tma_loads_inside_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41860 ns | 41790 ns | -70 ns (-0.2%) | 41310 ns | -550 ns (-1.3%) | 42820 ns | +960 ns (+2.3%) |
| SM Throughput (%) | 18.83 % | 18.99 % | +0.16 pp | 18.66 % | -0.17 pp | 18.86 % | +0.03 pp |
| Memory Throughput (%) | 78.87 % | 79.61 % | +0.74 pp | 80.09 % | +1.22 pp | 78.65 % | -0.22 pp |
| DRAM Throughput (%) | 8.83 % | 8.83 % | +0.00 pp | 8.94 % | +0.11 pp | 8.63 % | -0.20 pp |
| Achieved Occupancy (%) | 11.79 % | 11.79 % | +0.00 pp | 11.86 % | +0.07 pp | 11.83 % | +0.04 pp |
| Registers/Thread | 154.00 register/thread | 154.00 register/thread | +0 | 128.00 register/thread | -26 | 128.00 register/thread | -26 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.46 | -0.16 | 0.46 | -0.16 |
| No Eligible (%) | 78.51 % | 78.54 % | +0.03 pp | 78.69 % | +0.18 pp | 78.72 % | +0.21 pp |
| IPC Active | 0.86 inst/cycle | 0.86 inst/cycle | +0.00 | 0.85 inst/cycle | -0.01 | 0.86 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0133× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_to_eliminate_smem_bank_conflicts | 0.985× | 0.990× | ✓ |  |
| B | eliminate_smem_bank_conflicts_via_b_matrix_transposed_staging | 0.993× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | eliminate_smem_ba... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 41860 ns | 42270 ns | +410 ns (+1.0%) | 42560 ns | +700 ns (+1.7%) |
| SM Throughput (%) | 18.83 % | 18.83 % | +0.00 pp | 18.72 % | -0.11 pp |
| Memory Throughput (%) | 78.87 % | 78.56 % | -0.31 pp | 78.06 % | -0.81 pp |
| DRAM Throughput (%) | 8.83 % | 8.74 % | -0.09 pp | 8.68 % | -0.15 pp |
| Achieved Occupancy (%) | 11.79 % | 11.81 % | +0.02 pp | 11.82 % | +0.03 pp |
| Registers/Thread | 154.00 register/thread | 128.00 register/thread | -26 | 128.00 register/thread | -26 |
| Waves/SM | 0.62 | 0.46 | -0.16 | 0.46 | -0.16 |
| No Eligible (%) | 78.51 % | 78.63 % | +0.12 pp | 78.56 % | +0.05 pp |
| IPC Active | 0.86 inst/cycle | 0.86 inst/cycle | +0.00 | 0.86 inst/cycle | +0.00 |

---

## Iteration 5

**Best so far**: 1.0133× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_launcher_tensor_descriptors | 1.010× | 1.010× | ✓ | **✓ best** |
| D | replace_a_matrix_scalar_loads_with_block_ptr_to_eliminate_local_memory_spills | 1.006× | 1.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_lau... | Δ | replace_a_matrix_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 41860 ns | 41440 ns | -420 ns (-1.0%) | 41280 ns | -580 ns (-1.4%) |
| SM Throughput (%) | 18.83 % | 18.72 % | -0.11 pp | 18.76 % | -0.07 pp |
| Memory Throughput (%) | 78.87 % | 80.36 % | +1.49 pp | 80.13 % | +1.26 pp |
| DRAM Throughput (%) | 8.83 % | 8.91 % | +0.08 pp | 8.95 % | +0.12 pp |
| Achieved Occupancy (%) | 11.79 % | 11.86 % | +0.07 pp | 11.80 % | +0.01 pp |
| Registers/Thread | 154.00 register/thread | 128.00 register/thread | -26 | 128.00 register/thread | -26 |
| Waves/SM | 0.62 | 0.46 | -0.16 | 0.46 | -0.16 |
| No Eligible (%) | 78.51 % | 78.86 % | +0.35 pp | 78.88 % | +0.37 pp |
| IPC Active | 0.86 inst/cycle | 0.85 inst/cycle | -0.01 | 0.85 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  41860 ns  (1.000×)
Iter 1     41630 ns  (1.006×)  — tf32_tensor_cores
Iter 3     41310 ns  (1.013×)  — tma_loads_correct_api_inside_jit_kernel

Final best: 41440 ns  (1.0101×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,10 +5,11 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = False
-    USE_FAST_ACCUM : tl.constexpr = False
+    EVEN_K : tl.constexpr = True
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 32
@@ -44,40 +45,38 @@

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
 
+    # A is row-major (M, K) with stride_am=144, stride_ak=1
+    # Use make_block_ptr for coalesced loads (fixes uncoalesced scalar gather)
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+
+    # B is column-major (K, N) with stride_bk=1, stride_bn=144
+    # order=(0, 1) means K is the contiguous dim, reducing bank conflicts
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
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 144*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 24*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -89,8 +88,6 @@

     # inductor generates a suffix
     xindex = idx_n + 24*idx_m
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

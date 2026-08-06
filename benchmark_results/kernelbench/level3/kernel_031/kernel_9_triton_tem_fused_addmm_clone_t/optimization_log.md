# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_clone_t_transpose_view_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 68640 ns — **Best latency**: 68830 ns (**0.9972×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_clone_t_transpose_view_9`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_9_triton_tem_fused_addmm_clone_t/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 255
- **Duration**: 68640 ns
- **SM Throughput**: 36.10 % | **Memory**: 54.53 % | **DRAM**: 5.70 %
- **Achieved Occupancy**: 12.14 %
- **IPC Active**: 1.54 inst/cycle | **No Eligible**: 61.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.007× | 1.000× | ✓ |  |
| B | fix_b_load_and_reduce_register_pressure_via_autotune | 1.010× | 0.997× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fix_b_load_and_re... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 68640 ns | 68640 ns | +0 ns (+0.0%) | 68830 ns | +190 ns (+0.3%) |
| SM Throughput (%) | 36.10 % | 35.89 % | -0.21 pp | 35.87 % | -0.23 pp |
| Memory Throughput (%) | 54.53 % | 54.22 % | -0.31 pp | 54.15 % | -0.38 pp |
| DRAM Throughput (%) | 5.70 % | 5.69 % | -0.01 pp | 5.69 % | -0.01 pp |
| Achieved Occupancy (%) | 12.14 % | 12.15 % | +0.01 pp | 12.15 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 61.42 % | 61.54 % | +0.12 pp | 61.45 % | +0.03 pp |
| IPC Active | 1.54 inst/cycle | 1.54 inst/cycle | +0.00 | 1.54 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr_with_correct_b_transpose | 1.007× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 68640 ns | 68260 ns | -380 ns (-0.6%) |
| SM Throughput (%) | 36.10 % | 35.73 % | -0.37 pp |
| Memory Throughput (%) | 54.53 % | 53.42 % | -1.11 pp |
| DRAM Throughput (%) | 5.70 % | 5.57 % | -0.13 pp |
| Achieved Occupancy (%) | 12.14 % | 12.09 % | -0.05 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 |
| No Eligible (%) | 61.42 % | 61.26 % | -0.16 pp |
| IPC Active | 1.54 inst/cycle | 1.54 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | evict_first_on_streaming_gemm_tile_loads | 0.986× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_on_st... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 68640 ns | 70500 ns | +1860 ns (+2.7%) |
| SM Throughput (%) | 36.10 % | 35.61 % | -0.49 pp |
| Memory Throughput (%) | 54.53 % | 53.41 % | -1.12 pp |
| DRAM Throughput (%) | 5.70 % | 5.71 % | +0.01 pp |
| Achieved Occupancy (%) | 12.14 % | 12.17 % | +0.03 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 |
| No Eligible (%) | 61.42 % | 61.87 % | +0.45 pp |
| IPC Active | 1.54 inst/cycle | 1.53 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  68640 ns  (1.000×)

Final best: 68830 ns  (0.9972×)
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
 def triton_tem_fused_addmm_clone_t_transpose_view_9(arg_A, arg_B, out_ptr0):
@@ -44,50 +45,45 @@

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
 
+    # Create block pointers BEFORE the loop
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),  # 128, 1 — row-major
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)  # row-major: K is inner dim
+    )
+    # B stored as (K, N): B[k,n] = B_ptr + k*stride_bk + n*stride_bn
+    # stride_bk=1, stride_bn=128 => K dimension is contiguous
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),  # 1, 128
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1)  # K is contiguous (stride=1)
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 128*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 128*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
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
-    xindex = idx_n + 128*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    # Output store with block_ptr
+    out_block_ptr = tl.make_block_ptr(
+        base=out_ptr0,
+        shape=(M, N),
+        strides=(128, 1),
+        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
+        block_shape=(BLOCK_M, BLOCK_N),
+        order=(1, 0)
+    )
+    tl.store(out_block_ptr, acc, boundary_check=(0, 1))
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

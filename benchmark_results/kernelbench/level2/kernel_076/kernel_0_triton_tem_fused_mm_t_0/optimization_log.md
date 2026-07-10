# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_mm_t_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7440000 ns — **Best latency**: 7330000 ns (**1.0150×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_mm_t_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_076/kernel_0_triton_tem_fused_mm_t_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 7440000 ns
- **SM Throughput**: 40.58 % | **Memory**: 58.80 % | **DRAM**: 1.01 %
- **Achieved Occupancy**: 12.42 %
- **IPC Active**: 1.68 inst/cycle | **No Eligible**: 57.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7440000 ns | 7440000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 40.58 % | 40.61 % | +0.03 pp |
| Memory Throughput (%) | 58.80 % | 58.83 % | +0.03 pp |
| DRAM Throughput (%) | 1.01 % | 1.01 % | +0.00 pp |
| Achieved Occupancy (%) | 12.42 % | 12.42 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 57.98 % | 58.07 % | +0.09 pp |
| IPC Active | 1.68 inst/cycle | 1.68 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_transposed_load_and_autotune_tiles | 1.001× | 1.000× | ✓ |  |
| C | eviction_policies_streaming_l2_residency | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7440000 ns | 7440000 ns | +0 ns (+0.0%) | 7440000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 40.58 % | 40.67 % | +0.09 pp | 40.64 % | +0.06 pp |
| Memory Throughput (%) | 58.80 % | 58.92 % | +0.12 pp | 58.88 % | +0.08 pp |
| DRAM Throughput (%) | 1.01 % | 1.02 % | +0.01 pp | 1.01 % | +0.00 pp |
| Achieved Occupancy (%) | 12.42 % | 12.45 % | +0.03 pp | 12.50 % | +0.08 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 57.98 % | 57.91 % | -0.07 pp | 58.00 % | +0.02 pp |
| IPC Active | 1.68 inst/cycle | 1.68 inst/cycle | +0.00 | 1.68 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| E | add_triton_autotune | 1.016× | 1.015× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7440000 ns | 7330000 ns | -110000 ns (-1.5%) |
| SM Throughput (%) | 40.58 % | 41.30 % | +0.72 pp |
| Memory Throughput (%) | 58.80 % | 59.59 % | +0.79 pp |
| DRAM Throughput (%) | 1.01 % | 1.01 % | +0.00 pp |
| Achieved Occupancy (%) | 12.42 % | 12.50 % | +0.08 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 57.98 % | 57.43 % | -0.55 pp |
| IPC Active | 1.68 inst/cycle | 1.71 inst/cycle | +0.03 |

---

## Iteration 4

**Best so far**: 1.015× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 5

**Best so far**: 1.015× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  7440000 ns  (1.000×)
Iter 3     7330000 ns  (1.015×)  — add_triton_autotune

Final best: 7330000 ns  (1.0150×)
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
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_mm_t_0(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -60,18 +70,18 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 8192*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 8192*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 8192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # B tile: shape [BLOCK_K, BLOCK_N]
+        # B stored row-major: stride_bk=1, stride_bn=8192
+        # B[k, n] = B_ptr[k * 1 + n * 8192]
+        b_rk = offs_k[:, None] + (k_idx * BLOCK_K)   # shape [BLOCK_K, 1]
+        b_rn = offs_b_n[None, :]                       # shape [1, BLOCK_N]
+        b = tl.load(B + b_rk * 1 + b_rn * 8192)       # shape [BLOCK_K, BLOCK_N]
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -86,8 +96,6 @@

     # inductor generates a suffix
     xindex = idx_n + 8192*idx_m
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

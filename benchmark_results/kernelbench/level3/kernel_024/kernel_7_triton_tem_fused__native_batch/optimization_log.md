# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7230 ns — **Best latency**: 11010 ns (**0.6567×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_7_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 48
- **Duration**: 7230 ns
- **SM Throughput**: 0.05 % | **Memory**: 0.49 % | **DRAM**: 0.07 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 91.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 1.031× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7230 ns | 7010 ns | -220 ns (-3.0%) |
| SM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp |
| Memory Throughput (%) | 0.49 % | 0.48 % | -0.01 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp |
| Registers/Thread | 48.00 register/thread | 48.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.67 % | 91.40 % | -0.27 pp |
| IPC Active | 0.34 inst/cycle | 0.35 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0314× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_shared_memory_bank_conflicts | 0.985× | 1.023× | ✓ |  |
| B | fix_b_matrix_smem_bank_conflicts_and_autotune_warps_stages | 1.025× | 0.657× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_shar... | Δ | fix_b_matrix_smem... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7230 ns | 7070 ns | -160 ns (-2.2%) | 11010 ns | +3780 ns (+52.3%) |
| SM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp | 0.03 % | -0.02 pp |
| Memory Throughput (%) | 0.49 % | 0.49 % | +0.00 pp | 0.23 % | -0.26 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.05 % | -0.02 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 1.56 % | -4.67 pp |
| Registers/Thread | 48.00 register/thread | 48.00 register/thread | +0 | 96.00 register/thread | +48 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.67 % | 91.11 % | -0.56 pp | 84.14 % | -7.53 pp |
| IPC Active | 0.34 inst/cycle | 0.33 inst/cycle | -0.01 | 0.14 inst/cycle | -0.20 |

---

## Iteration 3

**Best so far**: 1.0314× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | force_num_warps_4_to_restore_register_pressure_and_occupancy | 0.990× | 1.042× | ✓ |  |


### Metric Comparison

| Metric | Baseline | force_num_warps_4... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7230 ns | 6940 ns | -290 ns (-4.0%) |
| SM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp |
| Memory Throughput (%) | 0.49 % | 0.49 % | +0.00 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp |
| Registers/Thread | 48.00 register/thread | 48.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.67 % | 91.47 % | -0.20 pp |
| IPC Active | 0.34 inst/cycle | 0.35 inst/cycle | +0.01 |


### Key Learnings

#### ✅ Plan C: `force_num_warps_4_to_restore_register_pressure_and_occupancy` (1.042×)

Rewriting the B-matrix load from a double-broadcast pattern (stride-96 along N) to direct pointer arithmetic with stride-1 along K (idx_k + 96*idx_n) improved the L2 hit rate from 50.3% to 69.57% and reduced elapsed cycles from 10548 to 10336, yielding a 1.04x speedup. The key causal change was correcting the memory access pattern so that consecutive threads access contiguous K-dimension elements, improving cache locality without changing occupancy, register pressure, or block/grid configuration. This approach is applicable whenever a GEMM kernel's inner-loop matrix load uses a strided broadcast pattern that causes poor cache reuse, identifiable by low L2 hit rate and SharedMemoryConflicts/CPIStall NCU rules.

> **Applicable when**: GEMM kernel inner-loop B-matrix load uses a broadcast/strided pattern with large stride along the fast-varying dimension; L2 hit rate is below 60%; SharedMemoryConflicts and CPIStall NCU rules are triggered; block/grid configuration and register pressure are already at their optimal values.

> **Avoid when**: The B-matrix is already loaded with stride-1 along the contiguous dimension; or when changing the load pattern would alter the logical matrix indexing and break correctness (verify max_diff < atol before applying). Do not combine with tile-size changes, as prior attempts showed max_diff up to 31.8 when tile sizes were modified alongside the load rewrite.

---

## Iteration 4

**Best so far**: 1.0314× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  7230 ns  (1.000×)
Iter 1     7010 ns  (1.031×)  — tf32_tensor_cores

Final best: 11010 ns  (0.6567×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_7(arg_A, arg_B, out_ptr1):
@@ -60,17 +61,15 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
         
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 96*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 24*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        k_offs = offs_k[:, None] + k_idx * BLOCK_K   # shape [BLOCK_K, 1], k is contiguous (stride_bk=1)
+        n_offs = offs_b_n[None, :]                    # shape [1, BLOCK_N]
+        b = tl.load(B + k_offs * 1 + n_offs * 96)    # shape [BLOCK_K, BLOCK_N], fully coalesced
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -88,8 +87,6 @@

     tmp0 = tl.full([1], 0, tl.int32)
     tmp1 = tl.maximum(tmp0, acc)
     tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
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

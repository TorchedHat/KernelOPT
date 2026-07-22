# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_gelu_t_view_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 194270 ns — **Best latency**: 193180 ns (**1.0056×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_gelu_t_view_21`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_21_triton_tem_fused_addmm_gelu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (246,1,1), Waves/SM: 0.93
- **Registers/thread**: 255
- **Duration**: 194270 ns
- **SM Throughput**: 35.82 % | **Memory**: 81.48 % | **DRAM**: 2.75 %
- **Achieved Occupancy**: 11.87 %
- **IPC Active**: 1.53 inst/cycle | **No Eligible**: 61.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.998× | 1.004× | ✓ |  |
| B | tma_loads_replace_scalar_pointer_arithmetic | 1.003× | 0.989× | ✓ | **✓ best** |
| C | eviction_policy_streaming_loads | 1.002× | 1.005× | ✓ |  |
| D | add_autotune_reduce_register_pressure_increase_occupancy | 1.002× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_replace... | Δ | eviction_policy_s... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 194270 ns | 193570 ns | -700 ns (-0.4%) | 196450 ns | +2180 ns (+1.1%) | 193340 ns | -930 ns (-0.5%) | 193790 ns | -480 ns (-0.2%) |
| SM Throughput (%) | 35.82 % | 35.89 % | +0.07 pp | 35.17 % | -0.65 pp | 35.83 % | +0.01 pp | 35.60 % | -0.22 pp |
| Memory Throughput (%) | 81.48 % | 81.65 % | +0.17 pp | 80.87 % | -0.61 pp | 81.49 % | +0.01 pp | 80.97 % | -0.51 pp |
| DRAM Throughput (%) | 2.75 % | 2.76 % | +0.01 pp | 2.66 % | -0.09 pp | 2.76 % | +0.01 pp | 2.75 % | +0.00 pp |
| Achieved Occupancy (%) | 11.87 % | 11.87 % | +0.00 pp | 11.88 % | +0.01 pp | 11.87 % | +0.00 pp | 11.87 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.93 | +0.00 | 0.93 | +0.00 | 0.93 | +0.00 |
| No Eligible (%) | 61.72 % | 61.73 % | +0.01 pp | 62.30 % | +0.58 pp | 61.91 % | +0.19 pp | 61.73 % | +0.01 pp |
| IPC Active | 1.53 inst/cycle | 1.53 inst/cycle | +0.00 | 1.50 inst/cycle | -0.03 | 1.53 inst/cycle | +0.00 | 1.53 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_b_matrix_load_with_block_ptr_for_coalesced_access | 0.999× | 0.997× | ✓ |  |
| E | add_triton_autotune | 1.009× | 1.006× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 194270 ns | 194850 ns | +580 ns (+0.3%) | 193180 ns | -1090 ns (-0.6%) |
| SM Throughput (%) | 35.82 % | 35.23 % | -0.59 pp | 35.82 % | +0.00 pp |
| Memory Throughput (%) | 81.48 % | 81.36 % | -0.12 pp | 81.47 % | -0.01 pp |
| DRAM Throughput (%) | 2.75 % | 2.71 % | -0.04 pp | 2.74 % | -0.01 pp |
| Achieved Occupancy (%) | 11.87 % | 11.88 % | +0.01 pp | 11.87 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.93 | +0.00 |
| No Eligible (%) | 61.72 % | 61.90 % | +0.18 pp | 61.79 % | +0.07 pp |
| IPC Active | 1.53 inst/cycle | 1.52 inst/cycle | -0.01 | 1.53 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0056× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | cap_registers_with_maxnreg_and_fix_a_matrix_block_ptr_load | 1.002× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 194270 ns | 195040 ns | +770 ns (+0.4%) |
| SM Throughput (%) | 35.82 % | 35.27 % | -0.55 pp |
| Memory Throughput (%) | 81.48 % | 81.07 % | -0.41 pp |
| DRAM Throughput (%) | 2.75 % | 2.64 % | -0.11 pp |
| Achieved Occupancy (%) | 11.87 % | 11.88 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 |
| No Eligible (%) | 61.72 % | 62.31 % | +0.59 pp |
| IPC Active | 1.53 inst/cycle | 1.51 inst/cycle | -0.02 |

---

## Iteration 5

**Best so far**: 1.0056× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  194270 ns  (1.000×)
Iter 3     193180 ns  (1.006×)  — add_triton_autotune

Final best: 193180 ns  (1.0056×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=5),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=1),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=5),
+        triton.Config({}, num_warps=16, num_stages=3),
+        triton.Config({}, num_warps=16, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_addmm_gelu_t_view_21(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -58,7 +80,7 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
         b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
@@ -72,9 +94,7 @@

         xindex = idx_n + 192*idx_m
         b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +106,6 @@

     # inductor generates a suffix
     xindex = idx_n + 192*idx_m
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

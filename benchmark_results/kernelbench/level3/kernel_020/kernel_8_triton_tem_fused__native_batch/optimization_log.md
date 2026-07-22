# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 27740 ns — **Best latency**: 27550 ns (**1.0069×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`streaming_eviction_policy_for_a_matrix_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_8_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.46
- **Registers/thread**: 128
- **Duration**: 27740 ns
- **SM Throughput**: 17.73 % | **Memory**: 72.79 % | **DRAM**: 8.88 %
- **Achieved Occupancy**: 11.81 %
- **IPC Active**: 0.83 inst/cycle | **No Eligible**: 79.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.007× | ✓ |  |
| B | add_autotune_reduce_register_pressure_fix_bank_conflicts | 1.145× | 1.004× | ✓ | **✓ best** |
| C | annotate_loads_with_cache_eviction_policies | 1.002× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_redu... | Δ | annotate_loads_wi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27740 ns | 27550 ns | -190 ns (-0.7%) | 27620 ns | -120 ns (-0.4%) | 27900 ns | +160 ns (+0.6%) |
| SM Throughput (%) | 17.73 % | 17.69 % | -0.04 pp | 17.68 % | -0.05 pp | 17.58 % | -0.15 pp |
| Memory Throughput (%) | 72.79 % | 72.64 % | -0.15 pp | 72.52 % | -0.27 pp | 72.07 % | -0.72 pp |
| DRAM Throughput (%) | 8.88 % | 8.93 % | +0.05 pp | 8.90 % | +0.02 pp | 8.83 % | -0.05 pp |
| Achieved Occupancy (%) | 11.81 % | 11.83 % | +0.02 pp | 11.84 % | +0.03 pp | 11.82 % | +0.01 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 |
| No Eligible (%) | 79.25 % | 79.16 % | -0.09 pp | 79.17 % | -0.08 pp | 79.03 % | -0.22 pp |
| IPC Active | 0.83 inst/cycle | 0.83 inst/cycle | +0.00 | 0.83 inst/cycle | +0.00 | 0.83 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0043× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_coalescing_and_add_software_pipelining | 0.997× | 1.003× | ✓ |  |
| B | fix_shared_memory_bank_conflicts_via_transposed_b_load | 1.149× | 1.002× | ✓ | **✓ best** |
| C | increase_num_warps_and_pipeline_stages_to_hide_mio_bank_conflict_stalls | 1.007× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_shared_memory... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27740 ns | 27650 ns | -90 ns (-0.3%) | 27680 ns | -60 ns (-0.2%) | 27740 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 17.73 % | 17.24 % | -0.49 pp | 17.62 % | -0.11 pp | 17.83 % | +0.10 pp |
| Memory Throughput (%) | 72.79 % | 71.85 % | -0.94 pp | 72.30 % | -0.49 pp | 73.09 % | +0.30 pp |
| DRAM Throughput (%) | 8.88 % | 8.90 % | +0.02 pp | 8.89 % | +0.01 pp | 8.88 % | +0.00 pp |
| Achieved Occupancy (%) | 11.81 % | 11.91 % | +0.10 pp | 11.81 % | +0.00 pp | 11.82 % | +0.01 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 |
| No Eligible (%) | 79.25 % | 79.75 % | +0.50 pp | 79.13 % | -0.12 pp | 79.19 % | -0.06 pp |
| IPC Active | 0.83 inst/cycle | 0.80 inst/cycle | -0.03 | 0.84 inst/cycle | +0.01 | 0.83 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0043× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_load_eliminate_tl_trans_to_remove_bank_conflicts | 1.146× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_load_elimin... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 27740 ns | 27940 ns | +200 ns (+0.7%) |
| SM Throughput (%) | 17.73 % | 17.80 % | +0.07 pp |
| Memory Throughput (%) | 72.79 % | 73.03 % | +0.24 pp |
| DRAM Throughput (%) | 8.88 % | 8.81 % | -0.07 pp |
| Achieved Occupancy (%) | 11.81 % | 11.83 % | +0.02 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.46 | 0.46 | +0.00 |
| No Eligible (%) | 79.25 % | 79.12 % | -0.13 pp |
| IPC Active | 0.83 inst/cycle | 0.83 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0043× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_load_coalescing_via_transposed_tile_load | 1.149× | 0.993× | ✓ |  |
| C | streaming_eviction_policy_for_a_matrix_loads | 1.152× | 1.007× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_load_coales... | Δ | streaming_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 27740 ns | 27940 ns | +200 ns (+0.7%) | 27550 ns | -190 ns (-0.7%) |
| SM Throughput (%) | 17.73 % | 17.71 % | -0.02 pp | 17.58 % | -0.15 pp |
| Memory Throughput (%) | 72.79 % | 72.65 % | -0.14 pp | 72.12 % | -0.67 pp |
| DRAM Throughput (%) | 8.88 % | 8.82 % | -0.06 pp | 8.93 % | +0.05 pp |
| Achieved Occupancy (%) | 11.81 % | 11.79 % | -0.02 pp | 11.83 % | +0.02 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 |
| No Eligible (%) | 79.25 % | 79.07 % | -0.18 pp | 79.14 % | -0.11 pp |
| IPC Active | 0.83 inst/cycle | 0.83 inst/cycle | +0.00 | 0.84 inst/cycle | +0.01 |

---

## Iteration 5

**Best so far**: 1.0069× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  27740 ns  (1.000×)
Iter 1     27620 ns  (1.004×)  — add_autotune_reduce_register_pressure_fix_bank_conflicts
Iter 4     27550 ns  (1.007×)  — streaming_eviction_policy_for_a_matrix_loads

Final best: 27550 ns  (1.0069×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -58,23 +70,22 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 96*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 24*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # Load B directly as [BLOCK_K, BLOCK_N]: consecutive threads access consecutive N columns (stride=1)
+        # stride_bk=1, stride_bn=96 — B is stored as [K, N] row-major
+        rk = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
+        b_ptrs = B + rk[:, None] * 1 + offs_b_n[None, :] * 96
+        b = tl.load(b_ptrs, mask=(rk[:, None] < K) & (offs_b_n[None, :] < N), other=0.0, eviction_policy='evict_last')
+        # b shape is [BLOCK_K, BLOCK_N] = [32, 32] -- correct for tl.dot(a, b), no tl.trans needed
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +97,6 @@

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

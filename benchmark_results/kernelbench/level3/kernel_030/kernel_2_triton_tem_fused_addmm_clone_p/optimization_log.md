# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_clone_permute_t_view_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 131010 ns — **Best latency**: 131010 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_clone_permute_t_view_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_2_triton_tem_fused_addmm_clone_p/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1225,1,1), Waves/SM: 3.09
- **Registers/thread**: 168
- **Duration**: 131010 ns
- **SM Throughput**: 35.02 % | **Memory**: 78.90 % | **DRAM**: 3.43 %
- **Achieved Occupancy**: 17.92 %
- **IPC Active**: 1.56 inst/cycle | **No Eligible**: 60.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.004× | 0.996× | ✓ | **✓ best** |
| C | autotune_tile_sizes_to_reduce_register_pressure_and_bank_conflicts | 1.003× | 0.986× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 131010 ns | 131580 ns | +570 ns (+0.4%) | 132860 ns | +1850 ns (+1.4%) |
| SM Throughput (%) | 35.02 % | 35.12 % | +0.10 pp | 35.03 % | +0.01 pp |
| Memory Throughput (%) | 78.90 % | 79.14 % | +0.24 pp | 78.94 % | +0.04 pp |
| DRAM Throughput (%) | 3.43 % | 3.42 % | -0.01 pp | 3.39 % | -0.04 pp |
| Achieved Occupancy (%) | 17.92 % | 17.92 % | +0.00 pp | 17.91 % | -0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 3.09 | 3.09 | +0.00 | 3.09 | +0.00 |
| No Eligible (%) | 60.95 % | 60.91 % | -0.04 pp | 60.97 % | +0.02 pp |
| IPC Active | 1.56 inst/cycle | 1.56 inst/cycle | +0.00 | 1.56 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_smem_bank_conflicts_via_staged_transposed_load | 1.003× | 1.000× | ✓ |  |
| B | tma_loads_correct_b_transposition | 1.014× | 1.023× | ✓ | **✓ best** |
| C | fix_b_matrix_load_address_to_eliminate_shared_memory_bank_conflicts | 1.000× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_smem... | Δ | tma_loads_correct... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 131010 ns | 131010 ns | +0 ns (+0.0%) | 128100 ns | -2910 ns (-2.2%) | 131360 ns | +350 ns (+0.3%) |
| SM Throughput (%) | 35.02 % | 34.85 % | -0.17 pp | 35.13 % | +0.11 pp | 34.97 % | -0.05 pp |
| Memory Throughput (%) | 78.90 % | 78.65 % | -0.25 pp | 80.85 % | +1.95 pp | 78.80 % | -0.10 pp |
| DRAM Throughput (%) | 3.43 % | 3.47 % | +0.04 pp | 3.50 % | +0.07 pp | 3.44 % | +0.01 pp |
| Achieved Occupancy (%) | 17.92 % | 17.91 % | -0.01 pp | 17.95 % | +0.03 pp | 17.90 % | -0.02 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 3.09 | 3.09 | +0.00 | 3.09 | +0.00 | 3.09 | +0.00 |
| No Eligible (%) | 60.95 % | 61.03 % | +0.08 pp | 60.81 % | -0.14 pp | 60.94 % | -0.01 pp |
| IPC Active | 1.56 inst/cycle | 1.56 inst/cycle | +0.00 | 1.57 inst/cycle | +0.01 | 1.56 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0227× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 4

**Best so far**: 1.0227× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipeline_smem_via_num_stages_autotune | 1.002× | 0.934× | ✓ |  |
| B | explicit_smem_b_transpose_to_eliminate_bank_conflicts | 1.000× | 0.998× | ✓ |  |
| C | eviction_policies_for_l2_residency_control | 0.997× | 0.995× | ✓ |  |
| D | software_pipeline_k_loop_to_hide_smem_bank_conflict_latency | 1.001× | 0.934× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipeline... | Δ | explicit_smem_b_t... | Δ | eviction_policies... | Δ | software_pipeline... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 131010 ns | 140320 ns | +9310 ns (+7.1%) | 131330 ns | +320 ns (+0.2%) | 131680 ns | +670 ns (+0.5%) | 140190 ns | +9180 ns (+7.0%) |
| SM Throughput (%) | 35.02 % | 31.73 % | -3.29 pp | 34.89 % | -0.13 pp | 34.80 % | -0.22 pp | 32.07 % | -2.95 pp |
| Memory Throughput (%) | 78.90 % | 76.67 % | -2.23 pp | 78.72 % | -0.18 pp | 78.44 % | -0.46 pp | 77.47 % | -1.43 pp |
| DRAM Throughput (%) | 3.43 % | 3.22 % | -0.21 pp | 3.42 % | -0.01 pp | 3.37 % | -0.06 pp | 3.22 % | -0.21 pp |
| Achieved Occupancy (%) | 17.92 % | 12.09 % | -5.83 pp | 17.94 % | +0.02 pp | 17.91 % | -0.01 pp | 12.13 % | -5.79 pp |
| Registers/Thread | 168.00 register/thread | 232.00 register/thread | +64 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 232.00 register/thread | +64 |
| Waves/SM | 3.09 | 4.64 | +1.55 | 3.09 | +0.00 | 3.09 | +0.00 | 4.64 | +1.55 |
| No Eligible (%) | 60.95 % | 65.20 % | +4.25 pp | 60.99 % | +0.04 pp | 61.08 % | +0.13 pp | 65.23 % | +4.28 pp |
| IPC Active | 1.56 inst/cycle | 1.39 inst/cycle | -0.17 | 1.56 inst/cycle | +0.00 | 1.56 inst/cycle | +0.00 | 1.39 inst/cycle | -0.17 |

---

## Cumulative Progress

```
Baseline:  131010 ns  (1.000×)
Iter 2     128100 ns  (1.023×)  — tma_loads_correct_b_transposition

Final best: 131010 ns  (1.0000×)
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
 def triton_tem_fused_addmm_clone_permute_t_view_2(arg_A, arg_B, out_ptr0):
@@ -58,23 +59,18 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        k_base = k_idx * BLOCK_K
 
+        # A load unchanged: A[m, k] = A_ptr + m*96 + k*1
         idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
+        idx_n = offs_k[None, :] + k_base
         xindex = idx_n + 96*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 288*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # B load FIXED: B[k, n] = B_ptr + k*1 + n*96, direct [BLOCK_K, BLOCK_N] load, no broadcast_to
+        b = tl.load(B + (offs_k[:, None] + k_base) + offs_b_n[None, :] * 96)
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +82,6 @@

     # inductor generates a suffix
     xindex = idx_n + 288*idx_m
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

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3030000 ns — **Best latency**: 3040000 ns (**0.9967×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_013/kernel_1_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 248.24
- **Registers/thread**: 214
- **Duration**: 3030000 ns
- **SM Throughput**: 27.33 % | **Memory**: 69.19 % | **DRAM**: 21.47 %
- **Achieved Occupancy**: 12.43 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 72.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ | **✓ best** |
| C | eviction_policies_streaming_A_reused_B | 0.986× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3030000 ns | 3030000 ns | +0 ns (+0.0%) | 3040000 ns | +10000 ns (+0.3%) |
| SM Throughput (%) | 27.33 % | 27.39 % | +0.06 pp | 27.37 % | +0.04 pp |
| Memory Throughput (%) | 69.19 % | 69.34 % | +0.15 pp | 69.08 % | -0.11 pp |
| DRAM Throughput (%) | 21.47 % | 21.49 % | +0.02 pp | 21.40 % | -0.07 pp |
| Achieved Occupancy (%) | 12.43 % | 12.43 % | +0.00 pp | 12.43 % | +0.00 pp |
| Registers/Thread | 214.00 register/thread | 214.00 register/thread | +0 | 214.00 register/thread | +0 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 248.24 | +0.00 |
| No Eligible (%) | 72.19 % | 72.26 % | +0.07 pp | 72.32 % | +0.13 pp |
| IPC Active | 1.11 inst/cycle | 1.10 inst/cycle | -0.01 | 1.11 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_transposed_load_to_eliminate_bank_conflicts | 1.000× | 0.997× | ✓ | **✓ best** |
| B | tma_loads_via_block_ptr | 0.997× | 0.993× | ✓ |  |
| C | fix_b_matrix_transposed_load_to_eliminate_bank_conflicts | 0.987× | 0.993× | ✓ |  |
| D | tma_loads_shared_memory_padding_eliminate_bank_conflicts | 0.994× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ | tma_loads_via_blo... | Δ | fix_b_matrix_tran... | Δ | tma_loads_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3030000 ns | 3040000 ns | +10000 ns (+0.3%) | 3050000 ns | +20000 ns (+0.7%) | 3050000 ns | +20000 ns (+0.7%) | 3030000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 27.33 % | 27.29 % | -0.04 pp | 27.29 % | -0.04 pp | 27.43 % | +0.10 pp | 27.40 % | +0.07 pp |
| Memory Throughput (%) | 69.19 % | 69.10 % | -0.09 pp | 68.97 % | -0.22 pp | 69.20 % | +0.01 pp | 69.05 % | -0.14 pp |
| DRAM Throughput (%) | 21.47 % | 21.40 % | -0.07 pp | 21.31 % | -0.16 pp | 21.35 % | -0.12 pp | 21.47 % | +0.00 pp |
| Achieved Occupancy (%) | 12.43 % | 12.43 % | +0.00 pp | 12.43 % | +0.00 pp | 12.43 % | +0.00 pp | 12.43 % | +0.00 pp |
| Registers/Thread | 214.00 register/thread | 214.00 register/thread | +0 | 214.00 register/thread | +0 | 214.00 register/thread | +0 | 214.00 register/thread | +0 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 248.24 | +0.00 | 248.24 | +0.00 | 248.24 | +0.00 |
| No Eligible (%) | 72.19 % | 72.33 % | +0.14 pp | 72.38 % | +0.19 pp | 72.19 % | +0.00 pp | 72.33 % | +0.14 pp |
| IPC Active | 1.11 inst/cycle | 1.11 inst/cycle | +0.00 | 1.10 inst/cycle | -0.01 | 1.11 inst/cycle | +0.00 | 1.10 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_to_improve_occupancy | 0.987× | 1.000× | ✓ |  |
| B | smem_transpose_b_conflict_free_load | 0.996× | 0.997× | ✓ |  |
| C | fix_b_matrix_load_order_and_reduce_register_pressure_via_smaller_block_m | 0.996× | 0.997× | ✓ |  |
| D | shared_memory_padding_to_eliminate_b_matrix_bank_conflicts | 0.984× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | smem_transpose_b_... | Δ | fix_b_matrix_load... | Δ | shared_memory_pad... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3030000 ns | 3030000 ns | +0 ns (+0.0%) | 3040000 ns | +10000 ns (+0.3%) | 3040000 ns | +10000 ns (+0.3%) | 3030000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 27.33 % | 27.40 % | +0.07 pp | 27.33 % | +0.00 pp | 27.29 % | -0.04 pp | 27.30 % | -0.03 pp |
| Memory Throughput (%) | 69.19 % | 69.12 % | -0.07 pp | 69.06 % | -0.13 pp | 68.96 % | -0.23 pp | 68.81 % | -0.38 pp |
| DRAM Throughput (%) | 21.47 % | 21.44 % | -0.03 pp | 21.39 % | -0.08 pp | 21.38 % | -0.09 pp | 21.45 % | -0.02 pp |
| Achieved Occupancy (%) | 12.43 % | 12.43 % | +0.00 pp | 12.43 % | +0.00 pp | 12.43 % | +0.00 pp | 12.43 % | +0.00 pp |
| Registers/Thread | 214.00 register/thread | 214.00 register/thread | +0 | 214.00 register/thread | +0 | 214.00 register/thread | +0 | 214.00 register/thread | +0 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 248.24 | +0.00 | 248.24 | +0.00 | 248.24 | +0.00 |
| No Eligible (%) | 72.19 % | 72.26 % | +0.07 pp | 72.34 % | +0.15 pp | 72.37 % | +0.18 pp | 72.20 % | +0.01 pp |
| IPC Active | 1.11 inst/cycle | 1.11 inst/cycle | +0.00 | 1.11 inst/cycle | +0.00 | 1.11 inst/cycle | +0.00 | 1.11 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_coalescing_via_transposed_smem_load | 0.985× | 1.000× | ✓ |  |
| B | fix_b_smem_layout_order_for_conflict_free_loads | 0.997× | 0.990× | ✓ |  |
| D | fix_b_block_ptr_order_for_coalesced_tma_load | 0.997× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_b_smem_layout... | Δ | fix_b_block_ptr_o... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3030000 ns | 3030000 ns | +0 ns (+0.0%) | 3060000 ns | +30000 ns (+1.0%) | 3040000 ns | +10000 ns (+0.3%) |
| SM Throughput (%) | 27.33 % | 27.38 % | +0.05 pp | 27.32 % | -0.01 pp | 27.38 % | +0.05 pp |
| Memory Throughput (%) | 69.19 % | 69.04 % | -0.15 pp | 69.03 % | -0.16 pp | 69.20 % | +0.01 pp |
| DRAM Throughput (%) | 21.47 % | 21.46 % | -0.01 pp | 21.26 % | -0.21 pp | 21.41 % | -0.06 pp |
| Achieved Occupancy (%) | 12.43 % | 12.43 % | +0.00 pp | 12.43 % | +0.00 pp | 12.43 % | +0.00 pp |
| Registers/Thread | 214.00 register/thread | 214.00 register/thread | +0 | 214.00 register/thread | +0 | 214.00 register/thread | +0 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 248.24 | +0.00 | 248.24 | +0.00 |
| No Eligible (%) | 72.19 % | 72.18 % | -0.01 pp | 72.27 % | +0.08 pp | 72.41 % | +0.22 pp |
| IPC Active | 1.11 inst/cycle | 1.11 inst/cycle | +0.00 | 1.10 inst/cycle | -0.01 | 1.10 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  3030000 ns  (1.000×)

Final best: 3040000 ns  (0.9967×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_1(arg_A, arg_B, out_ptr0):
@@ -60,18 +61,15 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 32*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 64*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 32*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # B layout: stride_bk=1, stride_bn=32 => B[k, n] = B_ptr + k*1 + n*32
+        rk = offs_k[:, None] + k_idx * BLOCK_K   # shape [BLOCK_K, 1]
+        b = tl.load(B + rk * 1 + offs_b_n[None, :] * 32)  # shape [BLOCK_K, BLOCK_N]
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -85,9 +83,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 64*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

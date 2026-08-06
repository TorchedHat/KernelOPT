# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_43` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 58500 ns — **Best latency**: 58620 ns (**0.9980×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_43`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_43_triton_tem_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (93,1,1), Waves/SM: 0.18
- **Registers/thread**: 128
- **Duration**: 58500 ns
- **SM Throughput**: 15.51 % | **Memory**: 60.26 % | **DRAM**: 1.52 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 76.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.997× | 0.998× | ✓ |  |
| B | tma_loads_with_autotune_tile_sizes | 0.890× | 0.889× | ✓ |  |
| D | autotune_tile_sizes_reduce_register_pressure | 0.997× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 58500 ns | 58620 ns | +120 ns (+0.2%) | 65820 ns | +7320 ns (+12.5%) | 58620 ns | +120 ns (+0.2%) |
| SM Throughput (%) | 15.51 % | 15.45 % | -0.06 pp | 13.94 % | -1.57 pp | 15.46 % | -0.05 pp |
| Memory Throughput (%) | 60.26 % | 60.04 % | -0.22 pp | 54.03 % | -6.23 pp | 60.09 % | -0.17 pp |
| DRAM Throughput (%) | 1.52 % | 1.52 % | +0.00 pp | 1.36 % | -0.16 pp | 1.52 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.23 | +0.05 | 0.18 | +0.00 |
| No Eligible (%) | 76.96 % | 76.95 % | -0.01 pp | 79.32 % | +2.36 pp | 76.94 % | -0.02 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 | 0.83 inst/cycle | -0.09 | 0.92 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_bank_conflicts_via_smem_transpose_and_padding | 0.994× | 1.002× | ✓ |  |
| B | fix_b_matrix_strided_load_to_eliminate_shared_memory_bank_conflicts | 1.001× | 0.998× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bank... | Δ | fix_b_matrix_stri... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 58500 ns | 58400 ns | -100 ns (-0.2%) | 58620 ns | +120 ns (+0.2%) |
| SM Throughput (%) | 15.51 % | 15.46 % | -0.05 pp | 15.49 % | -0.02 pp |
| Memory Throughput (%) | 60.26 % | 60.07 % | -0.19 pp | 60.19 % | -0.07 pp |
| DRAM Throughput (%) | 1.52 % | 1.52 % | +0.00 pp | 1.52 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 76.96 % | 76.92 % | -0.04 pp | 76.97 % | +0.01 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 | 0.92 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_bank_conflicts_via_transposed_load_and_tl_trans | 0.995× | 0.999× | ✓ |  |
| B | tma_loads_via_block_ptr_with_software_pipelining | 0.942× | 0.944× | ✓ |  |
| D | tma_loads_via_tensor_descriptor | 0.947× | 0.944× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bank... | Δ | tma_loads_via_blo... | Δ | tma_loads_via_ten... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 58500 ns | 58560 ns | +60 ns (+0.1%) | 61980 ns | +3480 ns (+5.9%) | 61980 ns | +3480 ns (+5.9%) |
| SM Throughput (%) | 15.51 % | 15.52 % | +0.01 pp | 14.81 % | -0.70 pp | 14.82 % | -0.69 pp |
| Memory Throughput (%) | 60.26 % | 60.30 % | +0.04 pp | 56.80 % | -3.46 pp | 56.87 % | -3.39 pp |
| DRAM Throughput (%) | 1.52 % | 1.52 % | +0.00 pp | 1.44 % | -0.08 pp | 1.44 % | -0.08 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 76.96 % | 76.94 % | -0.02 pp | 78.13 % | +1.17 pp | 78.13 % | +1.17 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 | 0.87 inst/cycle | -0.05 | 0.87 inst/cycle | -0.05 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_smem_staging_with_padding_to_eliminate_b_matrix_bank_conflicts | 0.941× | 0.943× | ✓ |  |
| B | eliminate_b_matrix_bank_conflicts_via_raw_pointer_arithmetic_transposed_load | 0.908× | 0.903× | ✓ |  |
| C | fix_b_matrix_bank_conflicts_via_explicit_ptr_transposed_load | 0.908× | 0.905× | ✓ |  |
| D | add_triton_autotune | 0.939× | 0.940× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_smem_sta... | Δ | eliminate_b_matri... | Δ | fix_b_matrix_bank... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 58500 ns | 62050 ns | +3550 ns (+6.1%) | 64770 ns | +6270 ns (+10.7%) | 64610 ns | +6110 ns (+10.4%) | 62210 ns | +3710 ns (+6.3%) |
| SM Throughput (%) | 15.51 % | 14.79 % | -0.72 pp | 14.26 % | -1.25 pp | 14.26 % | -1.25 pp | 14.86 % | -0.65 pp |
| Memory Throughput (%) | 60.26 % | 56.70 % | -3.56 pp | 54.67 % | -5.59 pp | 54.62 % | -5.64 pp | 56.98 % | -3.28 pp |
| DRAM Throughput (%) | 1.52 % | 1.43 % | -0.09 pp | 1.37 % | -0.15 pp | 1.38 % | -0.14 pp | 1.43 % | -0.09 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 140.00 register/thread | +12 | 140.00 register/thread | +12 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.23 | +0.05 | 0.23 | +0.05 | 0.18 | +0.00 |
| No Eligible (%) | 76.96 % | 78.11 % | +1.15 pp | 78.87 % | +1.91 pp | 78.86 % | +1.90 pp | 78.12 % | +1.16 pp |
| IPC Active | 0.92 inst/cycle | 0.87 inst/cycle | -0.05 | 0.84 inst/cycle | -0.08 | 0.84 inst/cycle | -0.08 | 0.87 inst/cycle | -0.05 |

---

## Cumulative Progress

```
Baseline:  58500 ns  (1.000×)

Final best: 58620 ns  (0.9980×)
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
 def triton_tem_fused_convolution_43(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -60,20 +61,24 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 512*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 160*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # B[n, k] is at offset n*512 + k  (stride_bn=512, stride_bk=1)
+        # Load b as [BLOCK_N, BLOCK_K] to make K (stride=1) the fast-varying dimension
+        # This eliminates shared memory bank conflicts (stride_bn=512 caused 12-way conflicts)
+        k_base = k_idx * BLOCK_K
+        ni = offs_b_n[:, None]              # [BLOCK_N, 1]  <- swapped to make K fast dim
+        ki = offs_k[None, :]               # [1, BLOCK_K]  <- swapped to make K fast dim
+        b_ptr = B + ni * 512 + (ki + k_base)  # [BLOCK_N, BLOCK_K], K is fast-varying
+        mask_b = (ni < N) & (ki + k_base < K)
+        b = tl.load(b_ptr, mask=mask_b, other=0.0)  # shape [BLOCK_N, BLOCK_K]
 
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        # tl.trans(b) gives [BLOCK_K, BLOCK_N], mathematically identical to original
+        acc += tl.dot(a, tl.trans(b), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
 
     # rematerialize rm and rn to save registers
@@ -88,8 +93,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
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

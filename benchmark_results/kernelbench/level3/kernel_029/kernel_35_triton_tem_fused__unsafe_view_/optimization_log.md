# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 143140 ns — **Best latency**: 143520 ns (**0.9974×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_35_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (372,1,1), Waves/SM: 0.94
- **Registers/thread**: 168
- **Duration**: 143140 ns
- **SM Throughput**: 36.82 % | **Memory**: 82.76 % | **DRAM**: 0.77 %
- **Achieved Occupancy**: 17.42 %
- **IPC Active**: 1.59 inst/cycle | **No Eligible**: 60.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_eliminate_bank_conflicts_and_improve_occupancy | 0.738× | 0.715× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 143140 ns | 200320 ns | +57180 ns (+39.9%) |
| SM Throughput (%) | 36.82 % | 26.79 % | -10.03 pp |
| Memory Throughput (%) | 82.76 % | 60.80 % | -21.96 pp |
| DRAM Throughput (%) | 0.77 % | 0.57 % | -0.20 pp |
| Achieved Occupancy (%) | 17.42 % | 10.79 % | -6.63 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 |
| Waves/SM | 0.94 | 1.41 | +0.47 |
| No Eligible (%) | 60.19 % | 64.84 % | +4.65 pp |
| IPC Active | 1.59 inst/cycle | 1.42 inst/cycle | -0.17 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_transposed_load_to_eliminate_bank_conflicts | 0.739× | 0.717× | ✓ |  |
| B | tma_loads_via_block_ptr_to_reduce_register_pressure_and_bank_conflicts | 0.695× | 0.675× | ✓ |  |
| C | annotate_streaming_loads_with_evict_first_to_preserve_L2_residency | 0.738× | 0.716× | ✓ |  |
| D | revert_block_k_and_fix_b_matrix_coalesced_load | 1.002× | 0.997× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ | tma_loads_via_blo... | Δ | annotate_streamin... | Δ | revert_block_k_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 143140 ns | 199710 ns | +56570 ns (+39.5%) | 212030 ns | +68890 ns (+48.1%) | 199870 ns | +56730 ns (+39.6%) | 143520 ns | +380 ns (+0.3%) |
| SM Throughput (%) | 36.82 % | 26.77 % | -10.05 pp | 25.20 % | -11.62 pp | 26.90 % | -9.92 pp | 36.76 % | -0.06 pp |
| Memory Throughput (%) | 82.76 % | 60.75 % | -22.01 pp | 56.91 % | -25.85 pp | 61.02 % | -21.74 pp | 82.62 % | -0.14 pp |
| DRAM Throughput (%) | 0.77 % | 0.58 % | -0.19 pp | 0.53 % | -0.24 pp | 0.57 % | -0.20 pp | 0.77 % | +0.00 pp |
| Achieved Occupancy (%) | 17.42 % | 10.58 % | -6.84 pp | 11.30 % | -6.12 pp | 10.73 % | -6.69 pp | 17.42 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 | 255.00 register/thread | +87 | 255.00 register/thread | +87 | 168.00 register/thread | +0 |
| Waves/SM | 0.94 | 1.41 | +0.47 | 1.41 | +0.47 | 1.41 | +0.47 | 0.94 | +0.00 |
| No Eligible (%) | 60.19 % | 64.55 % | +4.36 pp | 65.74 % | +5.55 pp | 64.72 % | +4.53 pp | 60.10 % | -0.09 pp |
| IPC Active | 1.59 inst/cycle | 1.42 inst/cycle | -0.17 | 1.37 inst/cycle | -0.22 | 1.42 inst/cycle | -0.17 | 1.59 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan A: `fix_b_matrix_transposed_load_to_eliminate_bank_conflicts` (0.717×)

Correcting the B-matrix load index from a column-major address (idx_m + 384*idx_n) to the intended row-major address (idx_m*1 + idx_n*384) produced virtually no change in any profiling metric: duration dropped by only 0.3% (200.32us → 199.71us), registers per thread remained at 255, occupancy stayed at 12.5% theoretical / ~10.6% achieved, shared memory bank conflicts remained at 8-way on ~70% of wavefronts, and MIO stalls remained at ~49% of cycles. This indicates the two index expressions are mathematically equivalent for this kernel's access pattern, or the Inductor-generated double broadcast_to wrapper was already computing the correct addresses despite its unusual form. When all key metrics are unchanged after a load-index fix, the original code was not actually buggy — the apparent 'transposition' was either intentional or already resolved by the compiler.

> **Applicable when**: Never: this specific index rewrite produced no measurable improvement.

> **Avoid when**: When the before and after index expressions are mathematically equivalent for the actual tensor strides in use, or when the compiler's broadcast_to wrapper already resolves to the same memory addresses. Do not attempt to 'fix' Inductor-generated load indices based on visual inspection alone without verifying that the addresses actually differ at runtime. When registers_per_thread=255 and block limits are already at 2 blocks/SM, index-only changes cannot improve occupancy or bank conflicts.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipeline_async_loads_to_hide_mio_stall_latency | 1.000× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipeline... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 143140 ns | 143780 ns | +640 ns (+0.4%) |
| SM Throughput (%) | 36.82 % | 36.75 % | -0.07 pp |
| Memory Throughput (%) | 82.76 % | 82.60 % | -0.16 pp |
| DRAM Throughput (%) | 0.77 % | 0.77 % | +0.00 pp |
| Achieved Occupancy (%) | 17.42 % | 17.43 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.94 | 0.94 | +0.00 |
| No Eligible (%) | 60.19 % | 60.16 % | -0.03 pp |
| IPC Active | 1.59 inst/cycle | 1.59 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  143140 ns  (1.000×)

Final best: 143520 ns  (0.9974×)
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
 def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35(arg_A, arg_B, out_ptr0):
@@ -67,11 +68,12 @@

         xindex = idx_n + 384*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1536*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # B has shape (N=1536, K=384), strides (384, 1)
+        # B[n_col, k_row] is at offset n_col * 384 + k_row
+        # offs_b_n[None, :] = n_col, shape [1, BLOCK_N]
+        # b_k_idx_vals = k_row, shape [BLOCK_K, 1]
+        # result shape [BLOCK_K, BLOCK_N] with contiguous n-dim => coalesced loads
+        b = tl.load(B + (offs_b_n[None, :] * 384 + b_k_idx_vals))
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -86,8 +88,6 @@

     # inductor generates a suffix
     xindex = idx_n + 1536*idx_m
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

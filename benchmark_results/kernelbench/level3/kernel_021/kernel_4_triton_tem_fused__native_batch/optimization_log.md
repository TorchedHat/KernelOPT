# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1860000 ns — **Best latency**: 1860000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_4_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2940,1,1), Waves/SM: 7.42
- **Registers/thread**: 168
- **Duration**: 1860000 ns
- **SM Throughput**: 38.71 % | **Memory**: 87.38 % | **DRAM**: 4.71 %
- **Achieved Occupancy**: 18.22 %
- **IPC Active**: 1.60 inst/cycle | **No Eligible**: 60.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.995× | ✓ |  |
| B | tma_loads_with_autotuned_tile_sizes | 1.028× | 1.022× | ✓ | **✓ best** |
| C | reduce_register_pressure_and_add_eviction_policies | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1860000 ns | 1870000 ns | +10000 ns (+0.5%) | 1820000 ns | -40000 ns (-2.2%) | 1860000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 38.71 % | 38.66 % | -0.05 pp | 39.74 % | +1.03 pp | 38.68 % | -0.03 pp |
| Memory Throughput (%) | 87.38 % | 87.27 % | -0.11 pp | 89.75 % | +2.37 pp | 87.32 % | -0.06 pp |
| DRAM Throughput (%) | 4.71 % | 4.71 % | +0.00 pp | 4.85 % | +0.14 pp | 4.72 % | +0.01 pp |
| Achieved Occupancy (%) | 18.22 % | 18.22 % | +0.00 pp | 18.21 % | -0.01 pp | 18.21 % | -0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 60.10 % | 60.09 % | -0.01 pp | 58.98 % | -1.12 pp | 60.10 % | +0.00 pp |
| IPC Active | 1.60 inst/cycle | 1.60 inst/cycle | +0.00 | 1.64 inst/cycle | +0.04 | 1.60 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.022× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.022× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_transpose_to_eliminate_bank_conflicts | 1.000× | 1.000× | ✓ |  |
| B | tma_loads_fix_b_matrix_indexing | 1.000× | 0.995× | ✓ |  |
| C | eviction_policies_b_matrix_l2_resident_output_streaming | 0.998× | 1.000× | ✓ |  |
| D | fix_b_matrix_load_indexing_and_autotune_tile_sizes | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | tma_loads_fix_b_m... | Δ | eviction_policies... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1860000 ns | 1860000 ns | +0 ns (+0.0%) | 1870000 ns | +10000 ns (+0.5%) | 1860000 ns | +0 ns (+0.0%) | 1860000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 38.71 % | 38.68 % | -0.03 pp | 38.71 % | +0.00 pp | 38.63 % | -0.08 pp | 38.61 % | -0.10 pp |
| Memory Throughput (%) | 87.38 % | 87.33 % | -0.05 pp | 87.40 % | +0.02 pp | 87.20 % | -0.18 pp | 87.17 % | -0.21 pp |
| DRAM Throughput (%) | 4.71 % | 4.72 % | +0.01 pp | 4.71 % | +0.00 pp | 4.69 % | -0.02 pp | 4.72 % | +0.01 pp |
| Achieved Occupancy (%) | 18.22 % | 18.21 % | -0.01 pp | 18.22 % | +0.00 pp | 18.22 % | +0.00 pp | 18.22 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 60.10 % | 60.09 % | -0.01 pp | 60.09 % | -0.01 pp | 60.09 % | -0.01 pp | 60.09 % | -0.01 pp |
| IPC Active | 1.60 inst/cycle | 1.60 inst/cycle | +0.00 | 1.60 inst/cycle | +0.00 | 1.60 inst/cycle | +0.00 | 1.60 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  1860000 ns  (1.000×)
Iter 1     1820000 ns  (1.022×)  — tma_loads_with_autotuned_tile_sizes

Final best: 1860000 ns  (1.0000×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4(arg_A, arg_B, out_ptr0):
@@ -65,12 +66,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 672*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 192*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 672*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 672*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_first')
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -86,8 +87,6 @@

     # inductor generates a suffix
     xindex = idx_n + 192*idx_m
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

# Kernel Optimization Log

## Summary

- **Kernel**: `elu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3010000 ns — **Best latency**: 3020000 ns (**0.9967×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `elu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_031/synthesis/kernel_0_triton_poi_fused_elu_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 29
- **Duration**: 3010000 ns
- **SM Throughput**: 65.02 % | **Memory**: 86.82 % | **DRAM**: 86.82 %
- **Achieved Occupancy**: 84.41 %
- **IPC Active**: 2.61 inst/cycle | **No Eligible**: 35.03 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_and_larger_blocks_to_autotune | 1.000× | 0.923× | ✓ |  |
| B | add_software_pipelining_via_num_stages_autotune | 1.000× | 1.000× | ✓ |  |
| C | add_software_pipelining_via_num_stages_and_vectorized_loads | 1.001× | 1.000× | ✓ | **✓ best** |
| D | add_software_pipelining_and_expand_autotune_configs | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_an... | Δ | add_software_pipe... | Δ | add_software_pipe... | Δ | add_software_pipe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010000 ns | 3260000 ns | +250000 ns (+8.3%) | 3010000 ns | +0 ns (+0.0%) | 3010000 ns | +0 ns (+0.0%) | 3010000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 65.02 % | 63.55 % | -1.47 pp | 65.15 % | +0.13 pp | 64.83 % | -0.19 pp | 65.61 % | +0.59 pp |
| Memory Throughput (%) | 86.82 % | 80.37 % | -6.45 pp | 86.84 % | +0.02 pp | 86.89 % | +0.07 pp | 86.93 % | +0.11 pp |
| DRAM Throughput (%) | 86.82 % | 80.37 % | -6.45 pp | 86.84 % | +0.02 pp | 86.89 % | +0.07 pp | 86.93 % | +0.11 pp |
| Achieved Occupancy (%) | 84.41 % | 81.30 % | -3.11 pp | 84.58 % | +0.17 pp | 84.65 % | +0.24 pp | 84.28 % | -0.13 pp |
| Registers/Thread | 29.00 register/thread | 23.00 register/thread | -6 | 29.00 register/thread | +0 | 29.00 register/thread | +0 | 29.00 register/thread | +0 |
| Waves/SM | 744.73 | 1489.45 | +744.72 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 35.03 % | 36.65 % | +1.62 pp | 35.09 % | +0.06 pp | 35.05 % | +0.02 pp | 35.14 % | +0.11 pp |
| IPC Active | 2.61 inst/cycle | 2.53 inst/cycle | -0.08 | 2.60 inst/cycle | -0.01 | 2.63 inst/cycle | +0.02 | 2.59 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_streaming_eviction_policy | 0.976× | 0.391× | ✓ |  |
| B | fix_num_warps_block_size_mismatch_and_add_streaming_cache_hint | 0.942× | 0.450× | ✓ |  |
| C | streaming_cache_eviction_policy_for_l2_pressure | 0.977× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_num_warps_blo... | Δ | streaming_cache_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010000 ns | 7700000 ns | +4690000 ns (+155.8%) | 6690000 ns | +3680000 ns (+122.3%) | 3030000 ns | +20000 ns (+0.7%) |
| SM Throughput (%) | 65.02 % | 32.67 % | -32.35 pp | 47.65 % | -17.37 pp | 67.52 % | +2.50 pp |
| Memory Throughput (%) | 86.82 % | 33.99 % | -52.83 pp | 39.14 % | -47.68 pp | 86.42 % | -0.40 pp |
| DRAM Throughput (%) | 86.82 % | 33.99 % | -52.83 pp | 39.14 % | -47.68 pp | 86.42 % | -0.40 pp |
| Achieved Occupancy (%) | 84.41 % | 27.51 % | -56.90 pp | 68.24 % | -16.17 pp | 84.06 % | -0.35 pp |
| Registers/Thread | 29.00 register/thread | 17.00 register/thread | -12 | 16.00 register/thread | -13 | 29.00 register/thread | +0 |
| Waves/SM | 744.73 | 2978.91 | +2234.18 | 5957.82 | +5213.09 | 744.73 | +0.00 |
| No Eligible (%) | 35.03 % | 67.38 % | +32.35 pp | 51.12 % | +16.09 pp | 31.87 % | -3.16 pp |
| IPC Active | 2.61 inst/cycle | 1.31 inst/cycle | -1.30 | 1.93 inst/cycle | -0.68 | 2.72 inst/cycle | +0.11 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | alpha_constexpr_fma_fusion_for_elu_negative_branch | 0.975× | 0.981× | ✓ |  |
| B | restore_matched_num_warps_to_block_size_for_streaming_kernel | 1.001× | 0.920× | ✓ |  |
| C | prune_low_occupancy_autotune_configs_and_fuse_fp32_elu_compute | 1.003× | 0.997× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | alpha_constexpr_f... | Δ | restore_matched_n... | Δ | prune_low_occupan... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010000 ns | 3070000 ns | +60000 ns (+2.0%) | 3270000 ns | +260000 ns (+8.6%) | 3020000 ns | +10000 ns (+0.3%) |
| SM Throughput (%) | 65.02 % | 66.44 % | +1.42 pp | 63.51 % | -1.51 pp | 23.72 % | -41.30 pp |
| Memory Throughput (%) | 86.82 % | 85.26 % | -1.56 pp | 80.13 % | -6.69 pp | 86.53 % | -0.29 pp |
| DRAM Throughput (%) | 86.82 % | 85.26 % | -1.56 pp | 80.13 % | -6.69 pp | 86.53 % | -0.29 pp |
| Achieved Occupancy (%) | 84.41 % | 85.12 % | +0.71 pp | 81.34 % | -3.07 pp | 85.71 % | +1.30 pp |
| Registers/Thread | 29.00 register/thread | 27.00 register/thread | -2 | 23.00 register/thread | -6 | 21.00 register/thread | -8 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 1489.45 | +744.72 | 744.73 | +0.00 |
| No Eligible (%) | 35.03 % | 33.45 % | -1.58 pp | 36.26 % | +1.23 pp | 76.13 % | +41.10 pp |
| IPC Active | 2.61 inst/cycle | 2.63 inst/cycle | +0.02 | 2.53 inst/cycle | -0.08 | 0.96 inst/cycle | -1.65 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expm1_fma_fusion_with_constexpr_alpha | 0.998× | 0.990× | ✓ |  |
| B | fma_fusion_via_expm1_to_exp_minus_one_restructure_for_non_fused_fp32_reduction | 0.974× | 0.977× | ✓ |  |
| C | alpha_constexpr_fma_fusion_and_register_pressure_reduction | 0.997× | 0.915× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expm1_fma_fusion_... | Δ | fma_fusion_via_ex... | Δ | alpha_constexpr_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010000 ns | 3040000 ns | +30000 ns (+1.0%) | 3080000 ns | +70000 ns (+2.3%) | 3290000 ns | +280000 ns (+9.3%) |
| SM Throughput (%) | 65.02 % | 64.48 % | -0.54 pp | 66.47 % | +1.45 pp | 61.91 % | -3.11 pp |
| Memory Throughput (%) | 86.82 % | 86.09 % | -0.73 pp | 84.93 % | -1.89 pp | 79.57 % | -7.25 pp |
| DRAM Throughput (%) | 86.82 % | 86.09 % | -0.73 pp | 84.93 % | -1.89 pp | 79.57 % | -7.25 pp |
| Achieved Occupancy (%) | 84.41 % | 85.43 % | +1.02 pp | 85.08 % | +0.67 pp | 81.14 % | -3.27 pp |
| Registers/Thread | 29.00 register/thread | 27.00 register/thread | -2 | 27.00 register/thread | -2 | 26.00 register/thread | -3 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 1489.45 | +744.72 |
| No Eligible (%) | 35.03 % | 35.39 % | +0.36 pp | 33.52 % | -1.51 pp | 37.27 % | +2.24 pp |
| IPC Active | 2.61 inst/cycle | 2.58 inst/cycle | -0.03 | 2.64 inst/cycle | +0.03 | 2.51 inst/cycle | -0.10 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_firing_and_sweep_warp_counts_for_large_blocks | 0.999× | 0.990× | ✓ |  |
| B | fix_autotune_key_to_include_alpha_constexpr_and_use_tl_exp_fma_fusion | 0.981× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_firi... | Δ | fix_autotune_key_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3010000 ns | 3040000 ns | +30000 ns (+1.0%) | 3040000 ns | +30000 ns (+1.0%) |
| SM Throughput (%) | 65.02 % | 64.51 % | -0.51 pp | 23.99 % | -41.03 pp |
| Memory Throughput (%) | 86.82 % | 86.20 % | -0.62 pp | 86.04 % | -0.78 pp |
| DRAM Throughput (%) | 86.82 % | 86.20 % | -0.62 pp | 86.04 % | -0.78 pp |
| Achieved Occupancy (%) | 84.41 % | 85.74 % | +1.33 pp | 85.75 % | +1.34 pp |
| Registers/Thread | 29.00 register/thread | 27.00 register/thread | -2 | 20.00 register/thread | -9 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 35.03 % | 35.38 % | +0.35 pp | 75.95 % | +40.92 pp |
| IPC Active | 2.61 inst/cycle | 2.60 inst/cycle | -0.01 | 0.97 inst/cycle | -1.64 |

---

## Cumulative Progress

```
Baseline:  3010000 ns  (1.000×)

Final best: 3020000 ns  (0.9967×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,14 +4,15 @@

 from triton.language.extra import libdevice
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=1),
     ],
     key=['numel'],
 )
@@ -20,7 +21,7 @@

     in_ptr,
     out_ptr,
     numel,
-    alpha,
+    alpha: tl.constexpr,
     BLOCK_SIZE: tl.constexpr,
 ):
     pid = tl.program_id(0)
@@ -31,12 +32,12 @@

 
     # ELU: x if x > 0 else alpha * (exp(x) - 1)
     is_positive = x > 0.0
-    # For negative branch: alpha * expm1(x)
-    neg_val = alpha * libdevice.expm1(x)
+    # libdevice.expm1 is numerically accurate near x=0 and enables FMA fusion.
+    # With alpha=1.0 as constexpr, compiler eliminates the multiply entirely.
+    neg_val = alpha * libdevice.expm1(x)   # 1 intrinsic; multiply folded away when alpha=1.0
     result = tl.where(is_positive, x, neg_val)
 
     tl.store(out_ptr + offsets, result, mask=mask)
-
 
 def kernel_function(*inputs):
     x = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

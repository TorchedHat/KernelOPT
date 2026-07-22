# Kernel Optimization Log

## Summary

- **Kernel**: `softplus_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3140000 ns — **Best latency**: 3100000 ns (**1.0129×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fuse_fp32_ops_via_libdevice_log1p_low_warp_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `softplus_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_029/synthesis/kernel_0_triton_poi_fused_softplus_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 32
- **Duration**: 3140000 ns
- **SM Throughput**: 79.74 % | **Memory**: 83.27 % | **DRAM**: 83.27 %
- **Achieved Occupancy**: 87.50 %
- **IPC Active**: 3.19 inst/cycle | **No Eligible**: 20.46 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_size_and_add_vectorized_loads | 1.056× | 1.013× | ✓ | **✓ best** |
| B | fuse_softplus_computation_eliminate_branch_divergence | 1.021× | 0.997× | ✓ |  |
| C | vectorized_loads_stores_with_wider_blocks | 1.018× | 1.013× | ✓ |  |
| D | fuse_softplus_computation_eliminate_redundant_exp_and_increase_block_size | 1.019× | 0.887× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_si... | Δ | fuse_softplus_com... | Δ | vectorized_loads_... | Δ | fuse_softplus_com... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3140000 ns | 3100000 ns | -40000 ns (-1.3%) | 3150000 ns | +10000 ns (+0.3%) | 3100000 ns | -40000 ns (-1.3%) | 3540000 ns | +400000 ns (+12.7%) |
| SM Throughput (%) | 79.74 % | 71.12 % | -8.62 pp | 80.93 % | +1.19 pp | 78.62 % | -1.12 pp | 73.19 % | -6.55 pp |
| Memory Throughput (%) | 83.27 % | 84.30 % | +1.03 pp | 83.05 % | -0.22 pp | 84.44 % | +1.17 pp | 73.91 % | -9.36 pp |
| DRAM Throughput (%) | 83.27 % | 84.30 % | +1.03 pp | 83.05 % | -0.22 pp | 84.44 % | +1.17 pp | 73.91 % | -9.36 pp |
| Achieved Occupancy (%) | 87.50 % | 83.12 % | -4.38 pp | 87.50 % | +0.00 pp | 52.08 % | -35.42 pp | 59.37 % | -28.13 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 48.00 register/thread | +16 | 40.00 register/thread | +8 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 595.78 | -148.95 | 992.97 | +248.24 |
| No Eligible (%) | 20.46 % | 28.68 % | +8.22 pp | 18.89 % | -1.57 pp | 21.52 % | +1.06 pp | 26.72 % | +6.26 pp |
| IPC Active | 3.19 inst/cycle | 2.86 inst/cycle | -0.33 | 3.26 inst/cycle | +0.07 | 3.15 inst/cycle | -0.04 | 2.93 inst/cycle | -0.26 |

---

## Iteration 2

**Best so far**: 1.0129× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_eviction_policy_and_use_log1p_with_expanded_autotune | 1.016× | 0.997× | ✓ |  |
| B | reduce_register_pressure_via_fused_softplus_math_and_eviction_policy | 1.032× | 0.966× | ✓ |  |
| C | streaming_cache_hints_with_larger_block_autotune | 0.961× | 0.963× | ✓ |  |
| D | reduce_register_pressure_via_native_ops_and_low_warp_autotune_configs | 1.032× | 1.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_eviction_poli... | Δ | reduce_register_p... | Δ | streaming_cache_h... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3140000 ns | 3150000 ns | +10000 ns (+0.3%) | 3250000 ns | +110000 ns (+3.5%) | 3260000 ns | +120000 ns (+3.8%) | 3090000 ns | -50000 ns (-1.6%) |
| SM Throughput (%) | 79.74 % | 76.00 % | -3.74 pp | 62.23 % | -17.51 pp | 81.53 % | +1.79 pp | 72.79 % | -6.95 pp |
| Memory Throughput (%) | 83.27 % | 83.16 % | -0.11 pp | 80.54 % | -2.73 pp | 80.35 % | -2.92 pp | 84.63 % | +1.36 pp |
| DRAM Throughput (%) | 83.27 % | 83.16 % | -0.11 pp | 80.54 % | -2.73 pp | 80.35 % | -2.92 pp | 84.63 % | +1.36 pp |
| Achieved Occupancy (%) | 87.50 % | 80.66 % | -6.84 pp | 44.92 % | -42.58 pp | 88.30 % | +0.80 pp | 80.91 % | -6.59 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 54.00 register/thread | +22 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 372.36 | -372.37 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 20.46 % | 23.34 % | +2.88 pp | 37.73 % | +17.27 pp | 18.48 % | -1.98 pp | 26.26 % | +5.80 pp |
| IPC Active | 3.19 inst/cycle | 3.06 inst/cycle | -0.13 | 2.50 inst/cycle | -0.69 | 3.28 inst/cycle | +0.09 | 2.93 inst/cycle | -0.26 |

---

## Iteration 3

**Best so far**: 1.0129× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_ops_via_libdevice_log1p_low_warp_configs | 1.046× | 1.013× | ✓ |  |
| B | enable_l2_compression_and_hide_l1tex_latency_via_pipelining | 1.009× | 1.000× | ✓ |  |
| C | reduce_register_pressure_via_log1p_and_high_warp_autotune | 1.042× | 0.997× | ✓ |  |
| D | fuse_fp32_instructions_via_native_tl_math_and_staged_pipeline_autotune | 0.999× | 0.997× | ✓ |  |
| E | add_triton_autotune | 1.032× | 1.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_ops_via... | Δ | enable_l2_compres... | Δ | reduce_register_p... | Δ | fuse_fp32_instruc... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3140000 ns | 3100000 ns | -40000 ns (-1.3%) | 3140000 ns | +0 ns (+0.0%) | 3150000 ns | +10000 ns (+0.3%) | 3150000 ns | +10000 ns (+0.3%) | 3090000 ns | -50000 ns (-1.6%) |
| SM Throughput (%) | 79.74 % | 72.18 % | -7.56 pp | 69.51 % | -10.23 pp | 69.68 % | -10.06 pp | 73.28 % | -6.46 pp | 74.18 % | -5.56 pp |
| Memory Throughput (%) | 83.27 % | 84.46 % | +1.19 pp | 83.34 % | +0.07 pp | 83.02 % | -0.25 pp | 83.02 % | -0.25 pp | 84.57 % | +1.30 pp |
| DRAM Throughput (%) | 83.27 % | 84.46 % | +1.19 pp | 83.34 % | +0.07 pp | 83.02 % | -0.25 pp | 83.02 % | -0.25 pp | 84.57 % | +1.30 pp |
| Achieved Occupancy (%) | 87.50 % | 51.06 % | -36.44 pp | 56.72 % | -30.78 pp | 42.27 % | -45.23 pp | 85.51 % | -1.99 pp | 80.24 % | -7.26 pp |
| Registers/Thread | 32.00 register/thread | 48.00 register/thread | +16 | 40.00 register/thread | +8 | 48.00 register/thread | +16 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 744.73 | 595.78 | -148.95 | 496.48 | -248.25 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 20.46 % | 28.03 % | +7.57 pp | 31.12 % | +10.66 pp | 30.44 % | +9.98 pp | 26.10 % | +5.64 pp | 26.97 % | +6.51 pp |
| IPC Active | 3.19 inst/cycle | 2.87 inst/cycle | -0.32 | 2.72 inst/cycle | -0.47 | 2.78 inst/cycle | -0.41 | 2.92 inst/cycle | -0.27 | 2.91 inst/cycle | -0.28 |

---

## Cumulative Progress

```
Baseline:  3140000 ns  (1.000×)
Iter 1     3100000 ns  (1.013×)  — increase_block_size_and_add_vectorized_loads

Final best: 3100000 ns  (1.0129×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,14 +4,19 @@

 from triton.language.extra import libdevice
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=2,  num_stages=4),
     ],
     key=['n_elements'],
 )
@@ -29,17 +34,12 @@

 
     x = tl.load(input_ptr + offsets, mask=mask)
 
-    # softplus(x) = log(1 + exp(x)), but for x > 20, use x directly for numerical stability
-    threshold = tl.full([1], 20.0, dtype=tl.float32)
-    is_large = x > threshold
-
-    exp_x = libdevice.exp(x)
-    log1p_exp_x = libdevice.log1p(exp_x)
-
-    result = tl.where(is_large, x, log1p_exp_x)
+    # Numerically stable: max(x,0) + log1p(exp(-|x|))
+    # exp(-|x|) in [0,1] always; for large |x| it underflows to 0 safely
+    abs_x = tl.abs(x)
+    result = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
 
     tl.store(output_ptr + offsets, result, mask=mask)
-
 
 def kernel_function(x):
     output = torch.empty_like(x)
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

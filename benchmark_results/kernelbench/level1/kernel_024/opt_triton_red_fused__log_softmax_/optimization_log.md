# Kernel Optimization Log

## Summary

- **Kernel**: `log_softmax_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 14070000 ns — **Best latency**: 4750000 ns (**2.9621×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`ultra_large_block_n_with_pass2_num_stages_1_for_log_softmax`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `log_softmax_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_024/synthesis/kernel_0_triton_red_fused__log_softmax_prepare_so.py`
- **Framework**: triton
- **Launch**: Block (1024,1,1), Grid (4096,1,1), Waves/SM: 15.52
- **Registers/thread**: 32
- **Duration**: 14070000 ns
- **SM Throughput**: 74.17 % | **Memory**: 75.97 % | **DRAM**: 27.91 %
- **Achieved Occupancy**: 98.70 %
- **IPC Active**: 2.55 inst/cycle | **No Eligible**: 36.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_for_large_N_log_softmax | 1.042× | 3.000× | ✓ |  |
| B | fuse_two_pass_logsoftmax_eviction_policy_and_fma | 1.000× | 1.056× | ✓ |  |
| C | vectorized_loads_stores_with_cache_hints | 1.037× | 1.000× | ✓ |  |
| D | fuse_two_pass_log_softmax_into_single_pass_with_sram_buffering | 1.047× | 2.866× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | fuse_two_pass_log... | Δ | vectorized_loads_... | Δ | fuse_two_pass_log... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14070000 ns | 4690000 ns | -9380000 ns (-66.7%) | 13320000 ns | -750000 ns (-5.3%) | 14070000 ns | +0 ns (+0.0%) | 4910000 ns | -9160000 ns (-65.1%) |
| SM Throughput (%) | 74.17 % | 37.01 % | -37.16 pp | 66.83 % | -7.34 pp | 74.17 % | +0.00 pp | 48.65 % | -25.52 pp |
| Memory Throughput (%) | 75.97 % | 83.81 % | +7.84 pp | 75.03 % | -0.94 pp | 75.99 % | +0.02 pp | 80.09 % | +4.12 pp |
| DRAM Throughput (%) | 27.91 % | 83.81 % | +55.90 pp | 29.49 % | +1.58 pp | 27.91 % | +0.00 pp | 80.09 % | +52.18 pp |
| Achieved Occupancy (%) | 98.70 % | 49.99 % | -48.71 pp | 86.62 % | -12.08 pp | 98.70 % | +0.00 pp | 96.78 % | -1.92 pp |
| Registers/Thread | 32.00 register/thread | 63.00 register/thread | +31 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 15.52 | 31.03 | +15.51 | 1.94 | -13.58 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 36.40 % | 62.17 % | +25.77 pp | 39.37 % | +2.97 pp | 36.32 % | -0.08 pp | 50.97 % | +14.57 pp |
| IPC Active | 2.55 inst/cycle | 1.51 inst/cycle | -1.04 | 2.43 inst/cycle | -0.12 | 2.54 inst/cycle | -0.01 | 1.95 inst/cycle | -0.60 |

---

## Iteration 2

**Best so far**: 2.8656× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | pass2_num_stages_1_with_fma_scalar_hoist | 1.047× | 2.907× | ✓ |  |
| B | fma_hoist_loop_invariant_scalar_and_eviction_policy_hints | 1.019× | 2.814× | ✓ |  |
| C | increase_block_n_to_8192_16384_and_set_pass2_num_stages_1 | 1.019× | 2.831× | ✓ |  |
| D | ultra_large_block_n_with_pass2_num_stages_1_for_log_softmax | 1.050× | 2.962× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | pass2_num_stages_... | Δ | fma_hoist_loop_in... | Δ | increase_block_n_... | Δ | ultra_large_block... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14070000 ns | 4840000 ns | -9230000 ns (-65.6%) | 5000000 ns | -9070000 ns (-64.5%) | 4970000 ns | -9100000 ns (-64.7%) | 4750000 ns | -9320000 ns (-66.2%) |
| SM Throughput (%) | 74.17 % | 29.48 % | -44.69 pp | 47.41 % | -26.76 pp | 29.44 % | -44.73 pp | 29.88 % | -44.29 pp |
| Memory Throughput (%) | 75.97 % | 81.15 % | +5.18 pp | 78.53 % | +2.56 pp | 78.98 % | +3.01 pp | 82.74 % | +6.77 pp |
| DRAM Throughput (%) | 27.91 % | 81.15 % | +53.24 pp | 78.53 % | +50.62 pp | 78.98 % | +51.07 pp | 82.74 % | +54.83 pp |
| Achieved Occupancy (%) | 98.70 % | 48.90 % | -49.80 pp | 96.60 % | -2.10 pp | 48.84 % | -49.86 pp | 48.77 % | -49.93 pp |
| Registers/Thread | 32.00 register/thread | 37.00 register/thread | +5 | 32.00 register/thread | +0 | 37.00 register/thread | +5 | 37.00 register/thread | +5 |
| Waves/SM | 15.52 | 31.03 | +15.51 | 15.52 | +0.00 | 31.03 | +15.51 | 31.03 | +15.51 |
| No Eligible (%) | 36.40 % | 70.09 % | +33.69 pp | 52.55 % | +16.15 pp | 69.90 % | +33.50 pp | 69.29 % | +32.89 pp |
| IPC Active | 2.55 inst/cycle | 1.19 inst/cycle | -1.36 | 1.90 inst/cycle | -0.65 | 1.20 inst/cycle | -1.35 | 1.22 inst/cycle | -1.33 |

---

## Iteration 3

**Best so far**: 2.9621× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restore_pass2_autotuned_num_stages_to_recover_register_pipeline_depth | 1.036× | 2.987× | ✓ |  |
| B | restore_pass2_autotuned_num_stages_to_fix_compiler_scheduling_regression | 1.041× | 3.000× | ✓ |  |
| C | restore_high_register_large_block_n_with_deeper_pass1_pipelining | 1.044× | 3.000× | ✓ |  |
| D | restore_large_block_n_autotune_configs_with_deep_pipelining | 1.018× | 2.831× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restore_pass2_aut... | Δ | restore_pass2_aut... | Δ | restore_high_regi... | Δ | restore_large_blo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14070000 ns | 4710000 ns | -9360000 ns (-66.5%) | 4690000 ns | -9380000 ns (-66.7%) | 4690000 ns | -9380000 ns (-66.7%) | 4970000 ns | -9100000 ns (-64.7%) |
| SM Throughput (%) | 74.17 % | 35.49 % | -38.68 pp | 35.79 % | -38.38 pp | 36.81 % | -37.36 pp | 29.43 % | -44.74 pp |
| Memory Throughput (%) | 75.97 % | 83.33 % | +7.36 pp | 83.85 % | +7.88 pp | 83.85 % | +7.88 pp | 79.01 % | +3.04 pp |
| DRAM Throughput (%) | 27.91 % | 83.33 % | +55.42 pp | 83.85 % | +55.94 pp | 83.85 % | +55.94 pp | 79.01 % | +51.10 pp |
| Achieved Occupancy (%) | 98.70 % | 49.99 % | -48.71 pp | 49.99 % | -48.71 pp | 49.99 % | -48.71 pp | 48.86 % | -49.84 pp |
| Registers/Thread | 32.00 register/thread | 64.00 register/thread | +32 | 64.00 register/thread | +32 | 63.00 register/thread | +31 | 37.00 register/thread | +5 |
| Waves/SM | 15.52 | 31.03 | +15.51 | 31.03 | +15.51 | 31.03 | +15.51 | 31.03 | +15.51 |
| No Eligible (%) | 36.40 % | 64.00 % | +27.60 pp | 63.54 % | +27.14 pp | 62.85 % | +26.45 pp | 69.99 % | +33.59 pp |
| IPC Active | 2.55 inst/cycle | 1.44 inst/cycle | -1.11 | 1.43 inst/cycle | -1.12 | 1.48 inst/cycle | -1.07 | 1.20 inst/cycle | -1.35 |

---

## Iteration 4

**Best so far**: 2.9621× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | split_pass1_pass2_independent_num_stages_autotune | 1.044× | 2.695× | ✓ |  |
| B | fast_expf_in_pass1_online_softmax_for_fma_fusion | 1.048× | 3.013× | ✓ |  |
| C | split_pass1_pass2_num_stages_for_independent_pipeline_depth | 1.039× | 2.860× | ✓ |  |
| D | restore_pass2_tl_range_num_stages_to_autotuned_constexpr | 1.043× | 3.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | split_pass1_pass2... | Δ | fast_expf_in_pass... | Δ | split_pass1_pass2... | Δ | restore_pass2_tl_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14070000 ns | 5220000 ns | -8850000 ns (-62.9%) | 4670000 ns | -9400000 ns (-66.8%) | 4920000 ns | -9150000 ns (-65.0%) | 4690000 ns | -9380000 ns (-66.7%) |
| SM Throughput (%) | 74.17 % | 30.59 % | -43.58 pp | 32.39 % | -41.78 pp | 31.16 % | -43.01 pp | 35.81 % | -38.36 pp |
| Memory Throughput (%) | 75.97 % | 75.24 % | -0.73 pp | 84.07 % | +8.10 pp | 79.90 % | +3.93 pp | 83.78 % | +7.81 pp |
| DRAM Throughput (%) | 27.91 % | 75.24 % | +47.33 pp | 84.07 % | +56.16 pp | 79.90 % | +51.99 pp | 83.78 % | +55.87 pp |
| Achieved Occupancy (%) | 98.70 % | 49.99 % | -48.71 pp | 49.99 % | -48.71 pp | 49.99 % | -48.71 pp | 49.99 % | -48.71 pp |
| Registers/Thread | 32.00 register/thread | 61.00 register/thread | +29 | 63.00 register/thread | +31 | 64.00 register/thread | +32 | 64.00 register/thread | +32 |
| Waves/SM | 15.52 | 31.03 | +15.51 | 31.03 | +15.51 | 31.03 | +15.51 | 31.03 | +15.51 |
| No Eligible (%) | 36.40 % | 69.78 % | +33.38 pp | 67.11 % | +30.71 pp | 68.49 % | +32.09 pp | 63.67 % | +27.27 pp |
| IPC Active | 2.55 inst/cycle | 1.20 inst/cycle | -1.35 | 1.32 inst/cycle | -1.23 | 1.26 inst/cycle | -1.29 | 1.45 inst/cycle | -1.10 |


### Key Learnings

#### ✅ Plan B: `fast_expf_in_pass1_online_softmax_for_fma_fusion` (3.013×)

In Triton online softmax kernels, replacing `tl.exp()` with `libdevice.fast_expf()` in the reduction pass reduces non-fused FP32 instruction count (343M → 237M, ~31% reduction) by enabling FMA fusion around the multiply-add in the online softmax update. The fast_expf call maps to PTX ex2.approx.f32 (single-cycle SFU op) rather than the multi-cycle ex2.f32, allowing the compiler to fuse surrounding multiply-add pairs into FMA instructions. The benefit is modest (~0.4% wall-clock improvement) when the kernel is already DRAM-saturated (>83% DRAM SOL), because compute latency in Pass 1 is largely hidden by memory pipelining; the FPInstructions rule overstates the potential gain in this regime.

> **Applicable when**: Triton kernel where FPInstructions NCU rule is triggered with a high ratio of non-fused to fused FP32 instructions (>50:1), and tl.exp() calls appear inside a multiply-add chain (e.g. online softmax: `sum * exp(old_max - new_max) + sum(exp(x - new_max))`). Most effective when the kernel is NOT already fully DRAM-saturated (DRAM SOL < 80%), so that compute latency reduction translates to wall-clock improvement.

> **Avoid when**: When the kernel is already DRAM-bandwidth-saturated (DRAM SOL > 83%), because the compute latency reduction from fast_expf is hidden by memory pipelining and the wall-clock gain is negligible (<1%). Also avoid when numerical precision is critical — fast_expf has ~1 ULP approximation error vs tl.exp's full precision. Do not apply to tl.log() or other transcendentals that lack a fast_* equivalent in libdevice.

---

## Cumulative Progress

```
Baseline:  14070000 ns  (1.000×)
Iter 1     4910000 ns  (2.866×)  — fuse_two_pass_log_softmax_into_single_pass_with_sram_buffering
Iter 2     4750000 ns  (2.962×)  — ultra_large_block_n_with_pass2_num_stages_1_for_log_softmax

Final best: 4750000 ns  (2.9621×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,16 +3,25 @@

 import triton.language as tl
 from triton.language.extra import libdevice
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_N': 1024, 'num_stages': 2}, num_warps=32),
-        triton.Config({'BLOCK_N': 2048, 'num_stages': 2}, num_warps=32),
+        triton.Config({'BLOCK_N': 16384, 'num_stages': 2}, num_warps=32),
+        triton.Config({'BLOCK_N': 16384, 'num_stages': 3}, num_warps=32),
+        triton.Config({'BLOCK_N': 16384, 'num_stages': 4}, num_warps=32),
+        triton.Config({'BLOCK_N': 8192, 'num_stages': 2}, num_warps=32),
+        triton.Config({'BLOCK_N': 8192, 'num_stages': 3}, num_warps=32),
+        triton.Config({'BLOCK_N': 8192, 'num_stages': 4}, num_warps=32),
+        triton.Config({'BLOCK_N': 8192, 'num_stages': 5}, num_warps=16),
+        triton.Config({'BLOCK_N': 4096, 'num_stages': 3}, num_warps=16),
+        triton.Config({'BLOCK_N': 4096, 'num_stages': 4}, num_warps=16),
+        triton.Config({'BLOCK_N': 4096, 'num_stages': 5}, num_warps=16),
         triton.Config({'BLOCK_N': 4096, 'num_stages': 2}, num_warps=32),
-        triton.Config({'BLOCK_N': 512, 'num_stages': 3}, num_warps=16),
-        triton.Config({'BLOCK_N': 1024, 'num_stages': 3}, num_warps=16),
         triton.Config({'BLOCK_N': 2048, 'num_stages': 3}, num_warps=16),
-        triton.Config({'BLOCK_N': 256, 'num_stages': 4}, num_warps=8),
-        triton.Config({'BLOCK_N': 512, 'num_stages': 4}, num_warps=8),
+        triton.Config({'BLOCK_N': 2048, 'num_stages': 4}, num_warps=8),
+        triton.Config({'BLOCK_N': 1024, 'num_stages': 4}, num_warps=8),
+        triton.Config({'BLOCK_N': 1024, 'num_stages': 5}, num_warps=8),
+        triton.Config({'BLOCK_N': 512,  'num_stages': 5}, num_warps=4),
     ],
     key=['N'],
 )
@@ -52,20 +61,24 @@

         row_max = new_max
     
     log_sum = tl.log(row_sum)
+    # Hoist loop-invariant scalar: offset = -(row_max + log_sum)
+    # Equivalent to subtracting row_max and log_sum separately, but enables FMA
+    offset = -(row_max + log_sum)
     
     # Pass 2: compute log_softmax = x - max - log(sum(exp(x - max)))
     out_start = output_ptr + row_idx * N
     
+    # Use autotuned num_stages to allow compiler to select optimal pipeline depth
+    # (2-5 stages) and maintain the high-register schedule (~63 regs/thread)
     for col_off in tl.range(0, N, BLOCK_N, num_stages=num_stages):
         cols = col_off + tl.arange(0, BLOCK_N)
         mask = cols < N
         x = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
         
-        # log_softmax = x - max - log_sum
-        out = x - row_max - log_sum
+        # Single addition with hoisted offset (FMA-friendly)
+        out = x + offset
         
         tl.store(out_start + cols, out.to(tl.float32), mask=mask)
-
 
 def kernel_function(*inputs):
     x, = inputs
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

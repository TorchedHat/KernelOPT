# Kernel Optimization Log

## Summary

- **Kernel**: `avg_pool1d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2680000 ns — **Best latency**: 2710000 ns (**0.9889×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `avg_pool1d_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_044/synthesis/kernel_0_triton_poi_fused_avg_pool2d_unsqueeze_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (524296,1,1), Waves/SM: 441.33
- **Registers/thread**: 56
- **Duration**: 2680000 ns
- **SM Throughput**: 86.96 % | **Memory**: 32.78 % | **DRAM**: 32.47 %
- **Achieved Occupancy**: 52.75 %
- **IPC Active**: 3.48 inst/cycle | **No Eligible**: 12.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | vectorized_loads_with_precomputed_base_pointer | 0.840× | 0.802× | ✓ |  |
| E | add_triton_autotune | 1.022× | 0.934× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2680000 ns | 3340000 ns | +660000 ns (+24.6%) | 2870000 ns | +190000 ns (+7.1%) |
| SM Throughput (%) | 86.96 % | 80.94 % | -6.02 pp | 92.60 % | +5.64 pp |
| Memory Throughput (%) | 32.78 % | 26.08 % | -6.70 pp | 33.74 % | +0.96 pp |
| DRAM Throughput (%) | 32.47 % | 26.08 % | -6.39 pp | 30.28 % | -2.19 pp |
| Achieved Occupancy (%) | 52.75 % | 35.48 % | -17.27 pp | 85.23 % | +32.48 pp |
| Registers/Thread | 56.00 register/thread | 80.00 register/thread | +24 | 25.00 register/thread | -31 |
| Waves/SM | 441.33 | 661.99 | +220.66 | 1985.97 | +1544.64 |
| No Eligible (%) | 12.96 % | 19.00 % | +6.04 pp | 7.32 % | -5.64 pp |
| IPC Active | 3.48 inst/cycle | 3.24 inst/cycle | -0.24 | 3.71 inst/cycle | +0.23 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_accumulation_into_fma_instructions | 1.070× | 0.989× | ✓ | **✓ best** |
| B | fuse_multiply_into_accumulation_loop_for_fma_generation | 1.025× | 0.893× | ✓ |  |
| C | persistent_tile_scheduling_with_register_pressure_fix | 0.807× | 0.865× | ✓ |  |
| D | revert_base_pointer_hoisting_and_remove_num_stages | 0.744× | 0.749× | ✓ |  |
| E | add_triton_autotune | 1.017× | 0.937× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_accumul... | Δ | fuse_multiply_int... | Δ | persistent_tile_s... | Δ | revert_base_point... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2680000 ns | 2710000 ns | +30000 ns (+1.1%) | 3000000 ns | +320000 ns (+11.9%) | 3100000 ns | +420000 ns (+15.7%) | 3580000 ns | +900000 ns (+33.6%) | 2860000 ns | +180000 ns (+6.7%) |
| SM Throughput (%) | 86.96 % | 90.99 % | +4.03 pp | 90.33 % | +3.37 pp | 90.41 % | +3.45 pp | 78.21 % | -8.75 pp | 92.58 % | +5.62 pp |
| Memory Throughput (%) | 32.78 % | 35.88 % | +3.10 pp | 32.03 % | -0.75 pp | 28.12 % | -4.66 pp | 24.32 % | -8.46 pp | 33.73 % | +0.95 pp |
| DRAM Throughput (%) | 32.47 % | 32.09 % | -0.38 pp | 28.97 % | -3.50 pp | 28.12 % | -4.35 pp | 24.32 % | -8.15 pp | 30.40 % | -2.07 pp |
| Achieved Occupancy (%) | 52.75 % | 83.80 % | +31.05 pp | 85.95 % | +33.20 pp | 47.69 % | -5.06 pp | 29.37 % | -23.38 pp | 85.18 % | +32.43 pp |
| Registers/Thread | 56.00 register/thread | 25.00 register/thread | -31 | 30.00 register/thread | -26 | 64.00 register/thread | +8 | 82.00 register/thread | +26 | 25.00 register/thread | -31 |
| Waves/SM | 441.33 | 1985.97 | +1544.64 | 1985.97 | +1544.64 | 496.49 | +55.16 | 794.39 | +353.06 | 1985.97 | +1544.64 |
| No Eligible (%) | 12.96 % | 8.92 % | -4.04 pp | 9.60 % | -3.36 pp | 9.51 % | -3.45 pp | 21.74 % | +8.78 pp | 7.34 % | -5.62 pp |
| IPC Active | 3.48 inst/cycle | 3.64 inst/cycle | +0.16 | 3.62 inst/cycle | +0.14 | 3.62 inst/cycle | +0.14 | 3.13 inst/cycle | -0.35 | 3.71 inst/cycle | +0.23 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fma_prescaled_accumulation_to_eliminate_nonfused_fp32 | 0.817× | 0.882× | ✓ |  |
| C | fma_accumulation_via_reciprocal_weight | 1.030× | 0.893× | ✓ |  |
| E | add_triton_autotune | 1.029× | 0.893× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_prescaled_acc... | Δ | fma_accumulation_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2680000 ns | 3040000 ns | +360000 ns (+13.4%) | 3000000 ns | +320000 ns (+11.9%) | 3000000 ns | +320000 ns (+11.9%) |
| SM Throughput (%) | 86.96 % | 90.10 % | +3.14 pp | 90.33 % | +3.37 pp | 90.32 % | +3.36 pp |
| Memory Throughput (%) | 32.78 % | 28.68 % | -4.10 pp | 32.04 % | -0.74 pp | 32.03 % | -0.75 pp |
| DRAM Throughput (%) | 32.47 % | 28.68 % | -3.79 pp | 28.98 % | -3.49 pp | 28.97 % | -3.50 pp |
| Achieved Occupancy (%) | 52.75 % | 47.68 % | -5.07 pp | 85.94 % | +33.19 pp | 85.97 % | +33.22 pp |
| Registers/Thread | 56.00 register/thread | 64.00 register/thread | +8 | 30.00 register/thread | -26 | 30.00 register/thread | -26 |
| Waves/SM | 441.33 | 496.49 | +55.16 | 1985.97 | +1544.64 | 1985.97 | +1544.64 |
| No Eligible (%) | 12.96 % | 9.83 % | -3.13 pp | 9.58 % | -3.38 pp | 9.58 % | -3.38 pp |
| IPC Active | 3.48 inst/cycle | 3.61 inst/cycle | +0.13 | 3.62 inst/cycle | +0.14 | 3.62 inst/cycle | +0.14 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | scalar_cta_index_decomposition_to_reduce_alu_and_register_pressure | 0.727× | 0.781× | ✓ |  |
| D | add_triton_autotune | 1.031× | 0.893× | ✓ |  |


### Metric Comparison

| Metric | Baseline | scalar_cta_index_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2680000 ns | 3430000 ns | +750000 ns (+28.0%) | 3000000 ns | +320000 ns (+11.9%) |
| SM Throughput (%) | 86.96 % | 87.63 % | +0.67 pp | 90.34 % | +3.38 pp |
| Memory Throughput (%) | 32.78 % | 25.34 % | -7.44 pp | 32.04 % | -0.74 pp |
| DRAM Throughput (%) | 32.47 % | 25.34 % | -7.13 pp | 28.99 % | -3.48 pp |
| Achieved Occupancy (%) | 52.75 % | 29.34 % | -23.41 pp | 85.98 % | +33.23 pp |
| Registers/Thread | 56.00 register/thread | 96.00 register/thread | +40 | 30.00 register/thread | -26 |
| Waves/SM | 441.33 | 794.39 | +353.06 | 1985.97 | +1544.64 |
| No Eligible (%) | 12.96 % | 12.30 % | -0.66 pp | 9.59 % | -3.37 pp |
| IPC Active | 3.48 inst/cycle | 3.51 inst/cycle | +0.03 | 3.62 inst/cycle | +0.14 |


### Key Learnings

#### ❌ Plan B: `scalar_cta_index_decomposition_to_reduce_alu_and_register_pressure` (0.781×)

Replacing per-element vector integer division chains (4 vector div/mod ops) with scalar CTA-level decomposition plus a persistent while loop was intended to reduce register pressure, but instead caused registers per thread to jump from 30 to 96. This collapsed theoretical occupancy from 100% to 31.25% (Block Limit Registers dropped from 16 to 5 blocks/SM) and achieved occupancy from 85.94% to 29.34%, increasing kernel duration from 3.0ms to 3.43ms. The persistent while loop pattern (tile_id += num_programs) forces the compiler to keep many more values live across loop iterations — including the scalar decomposition intermediates, the vector out_pos, channel_v, batch_v, win_start, and accumulator — dramatically inflating the register file. The warp cycles per issued instruction actually improved (15.07 → 5.34) but the occupancy collapse more than offset any latency-hiding benefit.

> **Applicable when**: Never — this specific pattern caused regression. Avoid combining persistent while loops with scalar CTA-level decomposition in Triton when the kernel is already register-pressure sensitive.

> **Avoid when**: When the kernel is already at or near 100% theoretical occupancy with moderate register counts (30 registers/thread). Persistent while loops in Triton force the compiler to keep all live values across iterations, dramatically inflating register usage. The scalar decomposition intermediates (tile_start, out_start_s, tmp_s, channel_s, batch_s) plus the resulting vectors (out_pos, channel_v, batch_v, win_start, acc) all remain live simultaneously, causing register blowup. This pattern is especially harmful when the inner loop is a tl.static_range (fully unrolled), as all unrolled iterations' values must be live concurrently.

---

## Cumulative Progress

```
Baseline:  2680000 ns  (1.000×)

Final best: 2710000 ns  (0.9889×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,13 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
+        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
         triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
     ],
     key=['output_length'],
 )
@@ -35,18 +34,16 @@

 
     acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
 
+    inv_ks = 1.0 / kernel_size  # 1/8 = 0.125, exact in FP32
     for k in tl.static_range(0, 8):
         in_pos = win_start + k
         valid = (in_pos >= 0) & (in_pos < input_length) & mask
         in_idx = batch * channels * input_length + channel * input_length + in_pos
         val = tl.load(input_ptr + in_idx, mask=valid, other=0.0)
-        acc += val
-
-    result = acc / kernel_size
+        acc = tl.math.fma(val, inv_ks, acc)  # FFMA: acc = val*(1/8) + acc
 
     out_idx = batch * channels * output_length + channel * output_length + out_pos
-    tl.store(output_ptr + out_idx, result, mask=mask)
-
+    tl.store(output_ptr + out_idx, acc, mask=mask)
 
 def kernel_function(x):
     batch_size, channels, input_length = x.shape
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_53` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5150 ns — **Best latency**: 5150 ns (**1.0000×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_53`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_53_triton_poi_fused_convolution_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 28
- **Duration**: 5150 ns
- **SM Throughput**: 40.33 % | **Memory**: 39.56 % | **DRAM**: 4.70 %
- **Achieved Occupancy**: 70.37 %
- **IPC Active**: 2.79 inst/cycle | **No Eligible**: 28.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.001× | 0.001× | ✓ |  |
| B | fix_static_tile_sizes_eliminate_partial_wave | 1.024× | 1.000× | ✓ | **✓ best** |
| C | fix_tile_sizes_manual_no_autotune | 1.023× | 1.000× | ✓ |  |
| D | fix_tile_sizes_static_vectorized_stores | 1.022× | 0.970× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_static_tile_s... | Δ | fix_tile_sizes_ma... | Δ | fix_tile_sizes_st... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5150 ns | 9360000 ns | +9354850 ns (+181647.6%) | 5150 ns | +0 ns (+0.0%) | 5150 ns | +0 ns (+0.0%) | 5310 ns | +160 ns (+3.1%) |
| SM Throughput (%) | 40.33 % | 69.77 % | +29.44 pp | 40.29 % | -0.04 pp | 40.19 % | -0.14 pp | 40.17 % | -0.16 pp |
| Memory Throughput (%) | 39.56 % | 88.13 % | +48.57 pp | 39.73 % | +0.17 pp | 39.59 % | +0.03 pp | 39.67 % | +0.11 pp |
| DRAM Throughput (%) | 4.70 % | 0.01 % | -4.69 pp | 4.71 % | +0.01 pp | 4.71 % | +0.01 pp | 4.56 % | -0.14 pp |
| Achieved Occupancy (%) | 70.37 % | 85.04 % | +14.67 pp | 70.95 % | +0.58 pp | 70.51 % | +0.14 pp | 70.75 % | +0.38 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 28.99 % | 28.74 % | -0.25 pp | 29.82 % | +0.83 pp | 30.72 % | +1.73 pp | 30.67 % | +1.68 pp |
| IPC Active | 2.79 inst/cycle | 2.85 inst/cycle | +0.06 | 2.73 inst/cycle | -0.06 | 2.78 inst/cycle | -0.01 | 2.80 inst/cycle | +0.01 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.001×)

Converting a 2D grid kernel into a persistent 1D-grid kernel with an internal `range(sm_id, num_y_tiles, num_sms)` loop caused a ~3,155x increase in executed instructions (1.6M → 5.17B), inflating kernel duration from 5.15µs to 9.36ms — a ~1,800x regression. The root cause is that Triton's `range(start, stop, step)` with all three arguments being runtime (non-constexpr) values does not compile to an efficient strided loop; `tl.num_programs(0)` and `tl.program_id(0)` are runtime values, making start and step non-constexpr and triggering pathological instruction generation. Although achieved occupancy improved (71% → 85%) and L2 hit rate rose to 99.83% due to repeated cache access, the instruction count explosion completely dominates all other gains.

> **Applicable when**: Never — this pattern caused a ~1800x regression in both iteration 2 and iteration 3. Do not apply persistent kernel loops with runtime non-constexpr start/step in Triton range().

> **Avoid when**: Always avoid: using `range(runtime_start, runtime_stop, runtime_step)` in Triton JIT kernels where start and step are non-constexpr runtime values (e.g., `tl.program_id(0)` and `tl.num_programs(0)`). This causes catastrophic instruction explosion (~3155x more instructions observed across two independent iterations). Even when occupancy and L2 hit rate improve, the instruction count blowup makes the kernel orders of magnitude slower.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_static_2d_grid_no_autotune | 1.029× | 1.000× | ✓ | **✓ best** |
| B | fix_eviction_policy_streaming_transpose | 0.958× | 0.970× | ✓ |  |
| C | fix_tile_sizes_eliminate_partial_wave_no_autotune_yblock256 | 1.024× | 0.975× | ✓ |  |
| D | fix_persistent_kernel_grid_launch_and_xblock_size | 0.001× | 0.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_st... | Δ | fix_eviction_poli... | Δ | fix_tile_sizes_el... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5150 ns | 5150 ns | +0 ns (+0.0%) | 5310 ns | +160 ns (+3.1%) | 5280 ns | +130 ns (+2.5%) | 9360000 ns | +9354850 ns (+181647.6%) |
| SM Throughput (%) | 40.33 % | 40.33 % | +0.00 pp | 41.84 % | +1.51 pp | 40.10 % | -0.23 pp | 69.57 % | +29.24 pp |
| Memory Throughput (%) | 39.56 % | 39.75 % | +0.19 pp | 39.75 % | +0.19 pp | 39.55 % | -0.01 pp | 87.87 % | +48.31 pp |
| DRAM Throughput (%) | 4.70 % | 4.69 % | -0.01 pp | 4.57 % | -0.13 pp | 4.59 % | -0.11 pp | 0.01 % | -4.69 pp |
| Achieved Occupancy (%) | 70.37 % | 71.13 % | +0.76 pp | 71.77 % | +1.40 pp | 70.65 % | +0.28 pp | 85.02 % | +14.65 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 32.00 register/thread | +4 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 28.99 % | 33.76 % | +4.77 pp | 27.03 % | -1.96 pp | 27.90 % | -1.09 pp | 28.60 % | -0.39 pp |
| IPC Active | 2.79 inst/cycle | 2.81 inst/cycle | +0.02 | 2.79 inst/cycle | +0.00 | 2.76 inst/cycle | -0.03 | 2.86 inst/cycle | +0.07 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_xblock_static_no_autotune_xblock16_yblock64 | 0.994× | 1.000× | ✓ |  |
| B | fix_xblock16_yblock64_static_no_autotune | 1.001× | 0.994× | ✓ |  |
| C | fix_xblock_thread_utilization_manual_no_autotune | 0.979× | 0.952× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_xblock_static... | Δ | fix_xblock16_yblo... | Δ | fix_xblock_thread... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5150 ns | 5150 ns | +0 ns (+0.0%) | 5180 ns | +30 ns (+0.6%) | 5410 ns | +260 ns (+5.0%) |
| SM Throughput (%) | 40.33 % | 38.36 % | -1.97 pp | 39.51 % | -0.82 pp | 41.40 % | +1.07 pp |
| Memory Throughput (%) | 39.56 % | 37.83 % | -1.73 pp | 39.02 % | -0.54 pp | 39.20 % | -0.36 pp |
| DRAM Throughput (%) | 4.70 % | 4.71 % | +0.01 pp | 4.70 % | +0.00 pp | 4.47 % | -0.23 pp |
| Achieved Occupancy (%) | 70.37 % | 70.01 % | -0.36 pp | 70.74 % | +0.37 pp | 71.08 % | +0.71 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 28.99 % | 32.66 % | +3.67 pp | 28.93 % | -0.06 pp | 31.39 % | +2.40 pp |
| IPC Active | 2.79 inst/cycle | 2.71 inst/cycle | -0.08 | 2.81 inst/cycle | +0.02 | 2.85 inst/cycle | +0.06 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_yblock_reduce_partial_wave_no_xblock_change | 0.995× | 0.994× | ✓ |  |
| B | fix_thread_waste_reduce_grid_static_num_warps4 | 1.021× | 0.994× | ✓ |  |
| C | fix_tile_sizes_manual_yblock256_no_autotune_single_wave | 0.961× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_yblock_r... | Δ | fix_thread_waste_... | Δ | fix_tile_sizes_ma... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5150 ns | 5180 ns | +30 ns (+0.6%) | 5180 ns | +30 ns (+0.6%) | 5150 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 40.33 % | 38.27 % | -2.06 pp | 40.38 % | +0.05 pp | 41.96 % | +1.63 pp |
| Memory Throughput (%) | 39.56 % | 37.72 % | -1.84 pp | 39.83 % | +0.27 pp | 39.72 % | +0.16 pp |
| DRAM Throughput (%) | 4.70 % | 4.66 % | -0.04 pp | 4.68 % | -0.02 pp | 4.69 % | -0.01 pp |
| Achieved Occupancy (%) | 70.37 % | 71.79 % | +1.42 pp | 70.61 % | +0.24 pp | 71.69 % | +1.32 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 28.99 % | 26.77 % | -2.22 pp | 27.87 % | -1.12 pp | 25.78 % | -3.21 pp |
| IPC Active | 2.79 inst/cycle | 2.64 inst/cycle | -0.15 | 2.80 inst/cycle | +0.01 | 2.84 inst/cycle | +0.05 |

---

## Cumulative Progress

```
Baseline:  5150 ns  (1.000×)

Final best: 5150 ns  (1.0000×)
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
 def triton_poi_fused_convolution_53(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -19,10 +20,8 @@

     y3 = yindex
     y0 = (yindex % 128)
     y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=ymask & xmask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, mask=ymask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

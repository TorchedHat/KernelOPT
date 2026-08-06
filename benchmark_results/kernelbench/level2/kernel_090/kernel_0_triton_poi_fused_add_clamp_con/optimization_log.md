# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_clamp_convolution_gelu_leaky_relu_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2120000 ns — **Best latency**: 2120000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_clamp_convolution_gelu_leaky_relu_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_090/kernel_0_triton_poi_fused_add_clamp_con/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3444224,1,1), Waves/SM: 1630.79
- **Registers/thread**: 16
- **Duration**: 2120000 ns
- **SM Throughput**: 56.64 % | **Memory**: 33.60 % | **DRAM**: 33.60 %
- **Achieved Occupancy**: 53.11 %
- **IPC Active**: 2.27 inst/cycle | **No Eligible**: 43.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_fixed_manual_no_autotune | 1.000× | 1.000× | ✓ | **✓ best** |
| B | increase_xblock_static_to_hide_l1tex_latency | 1.000× | 1.000× | ✓ |  |
| C | vectorized_loads_via_manual_xblock_and_num_warps_increase | 1.000× | 1.000× | ✓ |  |
| D | increase_num_warps_via_fixed_xblock_1024 | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_f... | Δ | increase_xblock_s... | Δ | vectorized_loads_... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2120000 ns | 2120000 ns | +0 ns (+0.0%) | 2120000 ns | +0 ns (+0.0%) | 2120000 ns | +0 ns (+0.0%) | 2120000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 56.64 % | 56.74 % | +0.10 pp | 56.65 % | +0.01 pp | 56.74 % | +0.10 pp | 56.64 % | +0.00 pp |
| Memory Throughput (%) | 33.60 % | 33.61 % | +0.01 pp | 33.60 % | +0.00 pp | 33.60 % | +0.00 pp | 33.60 % | +0.00 pp |
| DRAM Throughput (%) | 33.60 % | 33.61 % | +0.01 pp | 33.60 % | +0.00 pp | 33.60 % | +0.00 pp | 33.60 % | +0.00 pp |
| Achieved Occupancy (%) | 53.11 % | 53.11 % | +0.00 pp | 53.17 % | +0.06 pp | 53.11 % | +0.00 pp | 53.18 % | +0.07 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1630.79 | 1630.79 | +0.00 | 1630.79 | +0.00 | 1630.79 | +0.00 | 1630.79 | +0.00 |
| No Eligible (%) | 43.19 % | 43.29 % | +0.10 pp | 43.20 % | +0.01 pp | 43.16 % | -0.03 pp | 43.18 % | -0.01 pp |
| IPC Active | 2.27 inst/cycle | 2.27 inst/cycle | +0.00 | 2.27 inst/cycle | +0.00 | 2.27 inst/cycle | +0.00 | 2.27 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_256_num_warps_8_fixed_launch | 1.000× | 1.000× | ✓ |  |
| C | increase_warps_per_block_via_fixed_xblock_512_num_warps_16 | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_2... | Δ | increase_warps_pe... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2120000 ns | 2120000 ns | +0 ns (+0.0%) | 2120000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 56.64 % | 56.65 % | +0.01 pp | 56.65 % | +0.01 pp |
| Memory Throughput (%) | 33.60 % | 33.60 % | +0.00 pp | 33.61 % | +0.01 pp |
| DRAM Throughput (%) | 33.60 % | 33.60 % | +0.00 pp | 33.61 % | +0.01 pp |
| Achieved Occupancy (%) | 53.11 % | 53.17 % | +0.06 pp | 53.18 % | +0.07 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1630.79 | 1630.79 | +0.00 | 1630.79 | +0.00 |
| No Eligible (%) | 43.19 % | 43.30 % | +0.11 pp | 43.19 % | +0.00 pp |
| IPC Active | 2.27 inst/cycle | 2.27 inst/cycle | +0.00 | 2.27 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | static_large_xblock_with_corrected_grid_no_autotune | 1.000× | 1.000× | ✓ |  |
| B | persistent_kernel_grid_stride_loop_to_hide_l1tex_latency | 0.001× | 0.001× | ✓ |  |
| C | persistent_kernel_multi_element_loop_to_hide_l1tex_latency | 1.000× | 1.000× | ✓ |  |
| D | fuse_fp32_instructions_via_explicit_fma_rewrite | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | static_large_xblo... | Δ | persistent_kernel... | Δ | persistent_kernel... | Δ | fuse_fp32_instruc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2120000 ns | 2120000 ns | +0 ns (+0.0%) | 2610000000 ns | +2607880000 ns (+123013.2%) | 2120000 ns | +0 ns (+0.0%) | 2120000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 56.64 % | 56.65 % | +0.01 pp | 85.00 % | +28.36 pp | 67.24 % | +10.60 pp | 55.82 % | -0.82 pp |
| Memory Throughput (%) | 33.60 % | 33.60 % | +0.00 pp | 44.91 % | +11.31 pp | 33.58 % | -0.02 pp | 33.61 % | +0.01 pp |
| DRAM Throughput (%) | 33.60 % | 33.60 % | +0.00 pp | 44.91 % | +11.31 pp | 33.58 % | -0.02 pp | 33.61 % | +0.01 pp |
| Achieved Occupancy (%) | 53.11 % | 53.17 % | +0.06 pp | 99.57 % | +46.46 pp | 58.84 % | +5.73 pp | 52.75 % | -0.36 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 22.00 register/thread | +6 | 21.00 register/thread | +5 | 16.00 register/thread | +0 |
| Waves/SM | 1630.79 | 1630.79 | +0.00 | 1630.79 | +0.00 | 1630.79 | +0.00 | 1630.79 | +0.00 |
| No Eligible (%) | 43.19 % | 43.31 % | +0.12 pp | 15.00 % | -28.19 pp | 32.57 % | -10.62 pp | 44.03 % | +0.84 pp |
| IPC Active | 2.27 inst/cycle | 2.27 inst/cycle | +0.00 | 3.40 inst/cycle | +1.13 | 2.69 inst/cycle | +0.42 | 2.24 inst/cycle | -0.03 |


### Key Learnings

#### ❌ Plan B: `persistent_kernel_grid_stride_loop_to_hide_l1tex_latency` (0.001×)

Converting a one-shot elementwise kernel to a persistent grid-stride while-loop (1056 CTAs, each iterating ~3263 tiles) caused a catastrophic ~1230x slowdown: duration went from 2.12ms to 2.61 seconds. Although occupancy improved from 53% to 99.6% and compute throughput rose from 56.6% to 85%, the executed instruction count exploded from ~951M to ~1753T — a ~1.8 million-fold increase — indicating the while-loop was not bounded correctly or the Triton compiler unrolled/replicated the loop body pathologically. The persistent kernel pattern is fundamentally incompatible with this Triton kernel as implemented: the grid size in the launch remained 3,444,224 (unchanged from the original) rather than the intended 1,056, meaning every original CTA also ran the full grid-stride loop, multiplying work by ~3263x.

> **Applicable when**: NEVER: this specific pattern (persistent loop with hardcoded stride but unchanged grid size) always causes catastrophic regression by multiplying total work by grid_size/intended_persistent_grid_size.

> **Avoid when**: Any Triton elementwise kernel where the grid launch size is not simultaneously reduced to match the persistent CTA count. If the grid remains at the original size (3.44M) while the stride is set for 1056 CTAs, each CTA processes the entire dataset, multiplying total work by ~3263x. Also avoid when the while-loop condition references xnumel as a runtime value that Triton cannot bound at compile time — this prevents loop unrolling optimizations and may cause compiler pathologies.

---

## Cumulative Progress

```
Baseline:  2120000 ns  (1.000×)

Final best: 2120000 ns  (1.0000×)
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
 def triton_poi_fused_add_clamp_convolution_gelu_leaky_relu_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
@@ -36,8 +37,6 @@

     tmp20 = tmp15 * tmp19
     tl.store(in_out_ptr0 + (x3), tmp20, None)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

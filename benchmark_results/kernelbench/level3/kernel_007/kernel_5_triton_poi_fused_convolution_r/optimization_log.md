# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 918720 ns — **Best latency**: 918940 ns (**0.9998×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_5_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1492992,1,1), Waves/SM: 706.91
- **Registers/thread**: 16
- **Duration**: 918720 ns
- **SM Throughput**: 23.81 % | **Memory**: 33.28 % | **DRAM**: 33.28 %
- **Achieved Occupancy**: 41.22 %
- **IPC Active**: 0.96 inst/cycle | **No Eligible**: 76.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_fixed_larger_xblock | 1.000× | 1.000× | ✓ |  |
| B | increase_num_warps_via_larger_xblock_fixed_config | 1.000× | 1.000× | ✓ |  |
| C | software_pipelining_increase_xblock_manual | 1.000× | 1.000× | ✓ |  |
| D | increase_xblock_and_num_warps_fixed_tile | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | increase_num_warp... | Δ | software_pipelini... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 918720 ns | 918530 ns | -190 ns (-0.0%) | 918850 ns | +130 ns (+0.0%) | 918910 ns | +190 ns (+0.0%) | 918880 ns | +160 ns (+0.0%) |
| SM Throughput (%) | 23.81 % | 23.90 % | +0.09 pp | 27.08 % | +3.27 pp | 23.80 % | -0.01 pp | 23.80 % | -0.01 pp |
| Memory Throughput (%) | 33.28 % | 33.28 % | +0.00 pp | 33.26 % | -0.02 pp | 33.26 % | -0.02 pp | 33.27 % | -0.01 pp |
| DRAM Throughput (%) | 33.28 % | 33.28 % | +0.00 pp | 33.26 % | -0.02 pp | 33.26 % | -0.02 pp | 33.27 % | -0.01 pp |
| Achieved Occupancy (%) | 41.22 % | 41.17 % | -0.05 pp | 41.54 % | +0.32 pp | 41.19 % | -0.03 pp | 41.25 % | +0.03 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 706.91 | 706.91 | +0.00 | 706.91 | +0.00 | 706.91 | +0.00 | 706.91 | +0.00 |
| No Eligible (%) | 76.01 % | 76.14 % | +0.13 pp | 72.73 % | -3.28 pp | 76.14 % | +0.13 pp | 76.14 % | +0.13 pp |
| IPC Active | 0.96 inst/cycle | 0.95 inst/cycle | -0.01 | 1.09 inst/cycle | +0.13 | 0.96 inst/cycle | +0.00 | 0.96 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | manual_fixed_larger_xblock_num_warps_no_autotune | 1.000× | 1.000× | ✓ |  |
| B | vectorized_loads_fixed_config_no_autotune | 1.000× | 1.000× | ✓ |  |
| C | vectorized_loads_inline_hints_fixed_xblock | 1.000× | 1.000× | ✓ |  |
| D | vectorized_loads_fixed_xblock_no_autotune | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | manual_fixed_larg... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 918720 ns | 918980 ns | +260 ns (+0.0%) | 918660 ns | -60 ns (-0.0%) | 918910 ns | +190 ns (+0.0%) | 918720 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 23.81 % | 23.80 % | -0.01 pp | 27.08 % | +3.27 pp | 23.90 % | +0.09 pp | 23.81 % | +0.00 pp |
| Memory Throughput (%) | 33.28 % | 33.26 % | -0.02 pp | 33.27 % | -0.01 pp | 33.26 % | -0.02 pp | 33.27 % | -0.01 pp |
| DRAM Throughput (%) | 33.28 % | 33.26 % | -0.02 pp | 33.27 % | -0.01 pp | 33.26 % | -0.02 pp | 33.27 % | -0.01 pp |
| Achieved Occupancy (%) | 41.22 % | 41.20 % | -0.02 pp | 41.54 % | +0.32 pp | 41.18 % | -0.04 pp | 41.23 % | +0.01 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 706.91 | 706.91 | +0.00 | 706.91 | +0.00 | 706.91 | +0.00 | 706.91 | +0.00 |
| No Eligible (%) | 76.01 % | 76.04 % | +0.03 pp | 72.74 % | -3.27 pp | 76.14 % | +0.13 pp | 76.13 % | +0.12 pp |
| IPC Active | 0.96 inst/cycle | 0.96 inst/cycle | +0.00 | 1.09 inst/cycle | +0.13 | 0.95 inst/cycle | -0.01 | 0.96 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fixed_xblock256_aligned_bias_reuse_num_warps8 | 1.000× | 1.000× | ✓ |  |
| B | increase_warps_per_block_via_num_warps_jit_hint_larger_xblock | 1.000× | 1.000× | ✓ | **✓ best** |
| C | increase_num_warps_launch_parameter_only | 1.000× | 1.000× | ✓ |  |
| D | cache_eviction_policy_streaming_load_bypass_l1 | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fixed_xblock256_a... | Δ | increase_warps_pe... | Δ | increase_num_warp... | Δ | cache_eviction_po... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 918720 ns | 918850 ns | +130 ns (+0.0%) | 918940 ns | +220 ns (+0.0%) | 918780 ns | +60 ns (+0.0%) | 918660 ns | -60 ns (-0.0%) |
| SM Throughput (%) | 23.81 % | 27.19 % | +3.38 pp | 23.80 % | -0.01 pp | 23.80 % | -0.01 pp | 27.08 % | +3.27 pp |
| Memory Throughput (%) | 33.28 % | 33.26 % | -0.02 pp | 33.26 % | -0.02 pp | 33.27 % | -0.01 pp | 33.28 % | +0.00 pp |
| DRAM Throughput (%) | 33.28 % | 33.26 % | -0.02 pp | 33.26 % | -0.02 pp | 33.27 % | -0.01 pp | 33.28 % | +0.00 pp |
| Achieved Occupancy (%) | 41.22 % | 41.46 % | +0.24 pp | 41.20 % | -0.02 pp | 41.22 % | +0.00 pp | 41.31 % | +0.09 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 706.91 | 706.91 | +0.00 | 706.91 | +0.00 | 706.91 | +0.00 | 706.91 | +0.00 |
| No Eligible (%) | 76.01 % | 72.84 % | -3.17 pp | 76.03 % | +0.02 pp | 76.11 % | +0.10 pp | 72.74 % | -3.27 pp |
| IPC Active | 0.96 inst/cycle | 1.09 inst/cycle | +0.13 | 0.96 inst/cycle | +0.00 | 0.96 inst/cycle | +0.00 | 1.09 inst/cycle | +0.13 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | 2d_grid_channel_spatial_bias_reuse_loop | 0.882× | 0.892× | ✓ |  |
| B | persistent_kernel_loop_over_tiles | 0.999× | 0.999× | ✓ |  |
| C | reorder_access_pattern_channel_innermost_for_coalescing | 1.000× | 1.000× | ✓ |  |
| D | shared_memory_bias_broadcast_to_reduce_l1tex_stalls | 1.000× | 1.000× | ✓ |  |
| E | add_triton_autotune | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | 2d_grid_channel_s... | Δ | persistent_kernel... | Δ | reorder_access_pa... | Δ | shared_memory_bia... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 918720 ns | 1030000 ns | +111280 ns (+12.1%) | 919330 ns | +610 ns (+0.1%) | 918620 ns | -100 ns (-0.0%) | 919100 ns | +380 ns (+0.0%) | 918500 ns | -220 ns (-0.0%) |
| SM Throughput (%) | 23.81 % | 54.88 % | +31.07 pp | 35.41 % | +11.60 pp | 27.18 % | +3.37 pp | 26.36 % | +2.55 pp | 30.48 % | +6.67 pp |
| Memory Throughput (%) | 33.28 % | 20.04 % | -13.24 pp | 33.25 % | -0.03 pp | 33.27 % | -0.01 pp | 33.26 % | -0.02 pp | 33.28 % | +0.00 pp |
| DRAM Throughput (%) | 33.28 % | 14.53 % | -18.75 pp | 33.25 % | -0.03 pp | 33.27 % | -0.01 pp | 33.26 % | -0.02 pp | 33.28 % | +0.00 pp |
| Achieved Occupancy (%) | 41.22 % | 29.71 % | -11.51 pp | 45.29 % | +4.07 pp | 41.49 % | +0.27 pp | 41.02 % | -0.20 pp | 41.53 % | +0.31 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 706.91 | 706.91 | +0.00 | 706.91 | +0.00 | 706.91 | +0.00 | 706.91 | +0.00 | 706.91 | +0.00 |
| No Eligible (%) | 76.01 % | 45.24 % | -30.77 pp | 64.64 % | -11.37 pp | 72.84 % | -3.17 pp | 73.53 % | -2.48 pp | 69.44 % | -6.57 pp |
| IPC Active | 0.96 inst/cycle | 2.17 inst/cycle | +1.21 | 1.41 inst/cycle | +0.45 | 1.09 inst/cycle | +0.13 | 1.05 inst/cycle | +0.09 | 1.22 inst/cycle | +0.26 |

---

## Cumulative Progress

```
Baseline:  918720 ns  (1.000×)

Final best: 918940 ns  (0.9998×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,22 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_convolution_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 191102976
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x2 = xindex
+    x2 = tl.max_contiguous(tl.multiple_of(xindex, XBLOCK), XBLOCK)
     x0 = (xindex % 256)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
+    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.full([1], 0, tl.int32)
     tmp4 = tl.maximum(tmp3, tmp2)
-    tl.store(in_out_ptr0 + (x2), tmp4, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp4, None, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

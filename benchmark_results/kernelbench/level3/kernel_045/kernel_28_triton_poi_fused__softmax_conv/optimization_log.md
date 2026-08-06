# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_28` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2210 ns — **Best latency**: 2370 ns (**0.9325×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_28`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_28_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 27
- **Duration**: 2210 ns
- **SM Throughput**: 0.02 % | **Memory**: 0.43 % | **DRAM**: 0.04 %
- **Achieved Occupancy**: 6.20 %
- **IPC Active**: 0.28 inst/cycle | **No Eligible**: 93.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_configuration_and_tile_sizes | — | 0.932× | ✓ | **✓ best** |
| B | fix_grid_and_tile_sizes_for_full_utilization | — | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_c... | Δ | fix_grid_and_tile... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2210 ns | 2370 ns | +160 ns (+7.2%) | 2210 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.02 % | 0.02 % | +0.00 pp | 0.02 % | +0.00 pp |
| Memory Throughput (%) | 0.43 % | 0.40 % | -0.03 pp | 0.44 % | +0.01 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.16 % | -0.04 pp | 6.21 % | +0.01 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 | 27.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.07 % | 92.65 % | -0.42 pp | 93.69 % | +0.62 pp |
| IPC Active | 0.28 inst/cycle | 0.26 inst/cycle | -0.02 | 0.26 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_to_3d_with_correct_index_arithmetic | — | 0.987× | ✓ |  |
| D | fix_grid_to_2d_preserving_kernel_index_arithmetic | — | 1.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_to_3d_wi... | Δ | fix_grid_to_2d_pr... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2210 ns | 2240 ns | +30 ns (+1.4%) | 2180 ns | -30 ns (-1.4%) |
| SM Throughput (%) | 0.02 % | 0.02 % | +0.00 pp | 0.02 % | +0.00 pp |
| Memory Throughput (%) | 0.43 % | 0.47 % | +0.04 pp | 0.43 % | +0.00 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.14 % | -0.06 pp | 6.19 % | -0.01 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 | 27.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.07 % | 92.98 % | -0.09 pp | 90.34 % | -2.73 pp |
| IPC Active | 0.28 inst/cycle | 0.36 inst/cycle | +0.08 | 0.37 inst/cycle | +0.09 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_to_3d_matching_kernel_pid_usage_with_correct_xblock | — | 1.000× | ✓ |  |
| B | fix_grid_to_3d_matching_kernel_pid_arithmetic_with_correct_xblock | — | 1.000× | ✓ |  |
| C | fix_grid_to_2d_and_correct_xblock_to_match_xnumel | — | 1.000× | ✓ |  |
| D | fix_grid_to_correct_3d_tuple_preserving_exact_index_arithmetic | — | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_to_3d_ma... | Δ | fix_grid_to_3d_ma... | Δ | fix_grid_to_2d_an... | Δ | fix_grid_to_corre... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2210 ns | 2210 ns | +0 ns (+0.0%) | 2210 ns | +0 ns (+0.0%) | 2210 ns | +0 ns (+0.0%) | 2210 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.02 % | 0.02 % | +0.00 pp | 0.02 % | +0.00 pp | 0.02 % | +0.00 pp | 0.02 % | +0.00 pp |
| Memory Throughput (%) | 0.43 % | 0.56 % | +0.13 pp | 0.45 % | +0.02 pp | 0.43 % | +0.00 pp | 0.44 % | +0.01 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.14 % | -0.06 pp | 6.15 % | -0.05 pp | 6.12 % | -0.08 pp | 6.20 % | +0.00 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 | 27.00 register/thread | +0 | 27.00 register/thread | +0 | 27.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.07 % | 90.54 % | -2.53 pp | 90.76 % | -2.31 pp | 94.07 % | +1.00 pp | 91.43 % | -1.64 pp |
| IPC Active | 0.28 inst/cycle | 0.31 inst/cycle | +0.03 | 0.32 inst/cycle | +0.04 | 0.36 inst/cycle | +0.08 | 0.33 inst/cycle | +0.05 |

---

## Cumulative Progress

```
Baseline:  2210 ns  (1.000×)

Final best: 2370 ns  (0.9325×)
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
 def triton_poi_fused__softmax_convolution_exp_sub_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -21,8 +22,6 @@

     y1 = yindex // 256
     tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 256*x2 + 1024*y1), tmp0, xmask & ymask)
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

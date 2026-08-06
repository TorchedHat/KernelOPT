# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2240 ns — **Best latency**: 2240 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_13`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_13_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2,1,1), Waves/SM: 0.00
- **Registers/thread**: 30
- **Duration**: 2240 ns
- **SM Throughput**: 0.05 % | **Memory**: 0.57 % | **DRAM**: 0.04 %
- **Achieved Occupancy**: 6.09 %
- **IPC Active**: 0.51 inst/cycle | **No Eligible**: 86.38 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_to_parallelize_y_dimension | — | 1.000× | ✓ | **✓ best** |
| B | fix_grid_launch_and_increase_tile_sizes | — | 0.933× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ | fix_grid_launch_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2240 ns | 2240 ns | +0 ns (+0.0%) | 2400 ns | +160 ns (+7.1%) |
| SM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp | 0.05 % | +0.00 pp |
| Memory Throughput (%) | 0.57 % | 0.46 % | -0.11 pp | 0.41 % | -0.16 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.09 % | 6.09 % | +0.00 pp | 6.15 % | +0.06 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.38 % | 86.29 % | -0.09 pp | 86.25 % | -0.13 pp |
| IPC Active | 0.51 inst/cycle | 0.51 inst/cycle | +0.00 | 0.33 inst/cycle | -0.18 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fix_grid_to_3d_and_expand_yblock | — | 0.922× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_to_3d_an... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2240 ns | 2430 ns | +190 ns (+8.5%) |
| SM Throughput (%) | 0.05 % | 0.04 % | -0.01 pp |
| Memory Throughput (%) | 0.57 % | 0.39 % | -0.18 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.09 % | 6.10 % | +0.01 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.38 % | 86.17 % | -0.21 pp |
| IPC Active | 0.51 inst/cycle | 0.50 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_rewrite_clean_2d_indexing | — | 0.896× | ✓ |  |
| E | add_triton_autotune | — | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_r... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2240 ns | 2500 ns | +260 ns (+11.6%) | 2240 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp | 0.04 % | -0.01 pp |
| Memory Throughput (%) | 0.57 % | 0.39 % | -0.18 pp | 0.57 % | +0.00 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.09 % | 6.09 % | +0.00 pp | 6.09 % | +0.00 pp |
| Registers/Thread | 30.00 register/thread | 28.00 register/thread | -2 | 30.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.38 % | 86.51 % | +0.13 pp | 89.44 % | +3.06 pp |
| IPC Active | 0.51 inst/cycle | 0.48 inst/cycle | -0.03 | 0.52 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  2240 ns  (1.000×)

Final best: 2240 ns  (1.0000×)
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
 def triton_poi_fused__softmax_convolution_exp_sub_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -21,7 +22,6 @@

     y1 = yindex // 256
     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

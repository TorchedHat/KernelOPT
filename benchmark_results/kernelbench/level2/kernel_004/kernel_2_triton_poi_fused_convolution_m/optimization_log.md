# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_mish_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5570 ns — **Best latency**: 5920 ns (**0.9409×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_mish_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_004/kernel_2_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.01
- **Registers/thread**: 32
- **Duration**: 5570 ns
- **SM Throughput**: 1.24 % | **Memory**: 0.45 % | **DRAM**: 0.09 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.83 inst/cycle | **No Eligible**: 78.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launch_grid_correct_2d_parallelism | 1.029× | 1.005× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.001× | 0.977× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launch_grid_c... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5570 ns | 5540 ns | -30 ns (-0.5%) | 5700 ns | +130 ns (+2.3%) |
| SM Throughput (%) | 1.24 % | 1.24 % | +0.00 pp | 1.23 % | -0.01 pp |
| Memory Throughput (%) | 0.45 % | 0.41 % | -0.04 pp | 0.40 % | -0.05 pp |
| DRAM Throughput (%) | 0.09 % | 0.09 % | +0.00 pp | 0.09 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 78.98 % | 79.47 % | +0.49 pp | 78.38 % | -0.60 pp |
| IPC Active | 0.83 inst/cycle | 0.80 inst/cycle | -0.03 | 0.82 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0054× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_launch_grid_correct_flat_yindex_with_safe_xblock | 1.002× | 0.995× | ✓ |  |
| C | fix_kernel_launch_grid_with_correct_lambda_and_2d_parallelism | 1.036× | 0.995× | ✓ | **✓ best** |
| D | fix_kernel_function_launch_grid_to_3d | 1.002× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launch_grid_c... | Δ | fix_kernel_launch... | Δ | fix_kernel_functi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5570 ns | 5600 ns | +30 ns (+0.5%) | 5600 ns | +30 ns (+0.5%) | 5630 ns | +60 ns (+1.1%) |
| SM Throughput (%) | 1.24 % | 1.25 % | +0.01 pp | 1.22 % | -0.02 pp | 1.19 % | -0.05 pp |
| Memory Throughput (%) | 0.45 % | 0.46 % | +0.01 pp | 0.37 % | -0.08 pp | 0.42 % | -0.03 pp |
| DRAM Throughput (%) | 0.09 % | 0.09 % | +0.00 pp | 0.09 % | +0.00 pp | 0.09 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 78.98 % | 79.03 % | +0.05 pp | 79.02 % | +0.04 pp | 79.43 % | +0.45 pp |
| IPC Active | 0.83 inst/cycle | 0.83 inst/cycle | +0.00 | 0.83 inst/cycle | +0.00 | 0.83 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0054× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launch_grid_pure_2d_no_pid2_wrapping | 0.998× | 1.000× | ✓ |  |
| C | fix_launch_grid_with_correct_2d_meta_lambda | 1.005× | 0.962× | ✓ |  |
| D | add_triton_autotune | 1.036× | 0.941× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_launch_grid_p... | Δ | fix_launch_grid_w... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5570 ns | 5570 ns | +0 ns (+0.0%) | 5790 ns | +220 ns (+3.9%) | 5920 ns | +350 ns (+6.3%) |
| SM Throughput (%) | 1.24 % | 1.23 % | -0.01 pp | 1.21 % | -0.03 pp | 1.24 % | +0.00 pp |
| Memory Throughput (%) | 0.45 % | 0.36 % | -0.09 pp | 0.35 % | -0.10 pp | 0.35 % | -0.10 pp |
| DRAM Throughput (%) | 0.09 % | 0.09 % | +0.00 pp | 0.08 % | -0.01 pp | 0.08 % | -0.01 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 78.98 % | 78.59 % | -0.39 pp | 78.97 % | -0.01 pp | 78.80 % | -0.18 pp |
| IPC Active | 0.83 inst/cycle | 0.84 inst/cycle | +0.01 | 0.83 inst/cycle | +0.00 | 0.83 inst/cycle | +0.00 |

---

## Iteration 5

**Best so far**: 1.0054× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | correct_2d_grid_launch_preserving_exact_kernel_indexing | 1.008× | 0.972× | ✓ |  |
| B | fix_launch_grid_correct_3d_ynumel_tiling | 1.005× | 0.946× | ✓ |  |
| D | add_triton_autotune | 0.245× | 0.946× | ✓ |  |


### Metric Comparison

| Metric | Baseline | correct_2d_grid_l... | Δ | fix_launch_grid_c... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5570 ns | 5730 ns | +160 ns (+2.9%) | 5890 ns | +320 ns (+5.7%) | 5890 ns | +320 ns (+5.7%) |
| SM Throughput (%) | 1.24 % | 1.23 % | -0.01 pp | 1.25 % | +0.01 pp | 1.23 % | -0.01 pp |
| Memory Throughput (%) | 0.45 % | 0.45 % | +0.00 pp | 0.42 % | -0.03 pp | 0.43 % | -0.02 pp |
| DRAM Throughput (%) | 0.09 % | 0.09 % | +0.00 pp | 0.08 % | -0.01 pp | 0.08 % | -0.01 pp |
| Achieved Occupancy (%) | 6.24 % | 6.23 % | -0.01 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 78.98 % | 78.94 % | -0.04 pp | 79.61 % | +0.63 pp | 79.53 % | +0.55 pp |
| IPC Active | 0.83 inst/cycle | 0.83 inst/cycle | +0.00 | 0.82 inst/cycle | -0.01 | 0.83 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  5570 ns  (1.000×)
Iter 2     5540 ns  (1.005×)  — fix_launch_grid_correct_2d_parallelism

Final best: 5920 ns  (0.9409×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,11 +5,12 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_convolution_mish_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4129024
     xnumel = 128
-    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
+    yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
@@ -37,8 +38,6 @@

     tmp15 = tmp9 * tmp14
     tl.store(out_ptr0 + (y0 + 64516*x2 + 8258048*y1), tmp15, xmask & ymask)
 
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

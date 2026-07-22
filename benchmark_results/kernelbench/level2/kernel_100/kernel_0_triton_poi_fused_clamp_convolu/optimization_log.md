# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clamp_convolution_div_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4170000 ns — **Best latency**: 4170000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clamp_convolution_div_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_100/kernel_0_triton_poi_fused_clamp_convolu/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6786800,1,1), Waves/SM: 3213.45
- **Registers/thread**: 16
- **Duration**: 4170000 ns
- **SM Throughput**: 33.81 % | **Memory**: 33.79 % | **DRAM**: 33.79 %
- **Achieved Occupancy**: 44.47 %
- **IPC Active**: 1.35 inst/cycle | **No Eligible**: 66.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_hardcoded_no_autotune | 1.000× | 1.000× | ✓ |  |
| B | increase_xblock_statically_to_hide_l1tex_latency | 1.000× | 1.000× | ✓ |  |
| C | increase_xblock_and_num_warps_for_vectorized_loads | 1.000× | 1.000× | ✓ | **✓ best** |
| D | hardcode_larger_xblock_and_num_warps_without_autotune | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_h... | Δ | increase_xblock_s... | Δ | increase_xblock_a... | Δ | hardcode_larger_x... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4170000 ns | 4170000 ns | +0 ns (+0.0%) | 4170000 ns | +0 ns (+0.0%) | 4170000 ns | +0 ns (+0.0%) | 4170000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 33.81 % | 33.71 % | -0.10 pp | 33.80 % | -0.01 pp | 33.71 % | -0.10 pp | 33.80 % | -0.01 pp |
| Memory Throughput (%) | 33.79 % | 33.79 % | +0.00 pp | 33.79 % | +0.00 pp | 33.78 % | -0.01 pp | 33.78 % | -0.01 pp |
| DRAM Throughput (%) | 33.79 % | 33.79 % | +0.00 pp | 33.79 % | +0.00 pp | 33.78 % | -0.01 pp | 33.78 % | -0.01 pp |
| Achieved Occupancy (%) | 44.47 % | 44.54 % | +0.07 pp | 44.53 % | +0.06 pp | 44.54 % | +0.07 pp | 44.48 % | +0.01 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 3213.45 | 3213.45 | +0.00 | 3213.45 | +0.00 | 3213.45 | +0.00 | 3213.45 | +0.00 |
| No Eligible (%) | 66.28 % | 66.18 % | -0.10 pp | 66.18 % | -0.10 pp | 66.18 % | -0.10 pp | 66.27 % | -0.01 pp |
| IPC Active | 1.35 inst/cycle | 1.35 inst/cycle | +0.00 | 1.35 inst/cycle | +0.00 | 1.35 inst/cycle | +0.00 | 1.35 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_grid_stride_loop_to_hide_l1tex_latency | 1.000× | 1.000× | ✓ |  |
| B | multi_xblock_dispatch_without_autotune_decorator | 1.000× | 1.000× | ✓ |  |
| D | persistent_kernel_scheduling_to_hide_l1tex_latency | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_grid_s... | Δ | multi_xblock_disp... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4170000 ns | 4170000 ns | +0 ns (+0.0%) | 4170000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 33.81 % | 41.92 % | +8.11 pp | 33.71 % | -0.10 pp |
| Memory Throughput (%) | 33.79 % | 33.78 % | -0.01 pp | 33.79 % | +0.00 pp |
| DRAM Throughput (%) | 33.79 % | 33.78 % | -0.01 pp | 33.79 % | +0.00 pp |
| Achieved Occupancy (%) | 44.47 % | 45.91 % | +1.44 pp | 44.47 % | +0.00 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 3213.45 | 3213.45 | +0.00 | 3213.45 | +0.00 |
| No Eligible (%) | 66.28 % | 57.94 % | -8.34 pp | 66.27 % | -0.01 pp |
| IPC Active | 1.35 inst/cycle | 1.68 inst/cycle | +0.33 | 1.35 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  4170000 ns  (1.000×)

Final best: 4170000 ns  (1.0000×)
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
 def triton_poi_fused_clamp_convolution_div_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
@@ -21,8 +22,6 @@

     tmp5 = tl.full([1], 0.5, tl.float32)
     tmp6 = tmp4 * tmp5
     tl.store(in_out_ptr0 + (x3), tmp6, xmask)
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

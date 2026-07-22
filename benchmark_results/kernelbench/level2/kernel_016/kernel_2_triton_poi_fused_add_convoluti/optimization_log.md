# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_convolution_hardtanh_mish_mul_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4060 ns — **Best latency**: 4060 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_convolution_hardtanh_mish_mul_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_016/kernel_2_triton_poi_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.00
- **Registers/thread**: 32
- **Duration**: 4060 ns
- **SM Throughput**: 0.55 % | **Memory**: 0.37 % | **DRAM**: 0.07 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.83 inst/cycle | **No Eligible**: 79.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_tile_sizes_and_fix_grid_for_full_sm_utilization | 0.951× | 1.057× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4060 ns | 3840 ns | -220 ns (-5.4%) |
| SM Throughput (%) | 0.55 % | 0.57 % | +0.02 pp |
| Memory Throughput (%) | 0.37 % | 0.39 % | +0.02 pp |
| DRAM Throughput (%) | 0.07 % | 0.08 % | +0.01 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 79.56 % | 77.84 % | -1.72 pp |
| IPC Active | 0.83 inst/cycle | 0.80 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_correctness_with_proper_3d_grid_and_xblock_64 | 0.957× | 1.015× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_c... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4060 ns | 4000 ns | -60 ns (-1.5%) |
| SM Throughput (%) | 0.55 % | 0.57 % | +0.02 pp |
| Memory Throughput (%) | 0.37 % | 0.38 % | +0.01 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 79.56 % | 77.50 % | -2.06 pp |
| IPC Active | 0.83 inst/cycle | 0.82 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  4060 ns  (1.000×)

Final best: 4060 ns  (1.0000×)
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
 def triton_poi_fused_add_convolution_hardtanh_mish_mul_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -39,8 +40,6 @@

     tmp17 = tmp15 * tmp16
     tl.store(out_ptr0 + (y0 + 65536*x2 + 4194304*y1), tmp17, xmask & ymask)
 
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

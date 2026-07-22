# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 94590 ns — **Best latency**: 94590 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_004/kernel_0_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (150528,1,1), Waves/SM: 71.27
- **Registers/thread**: 16
- **Duration**: 94590 ns
- **SM Throughput**: 24.28 % | **Memory**: 27.50 % | **DRAM**: 27.50 %
- **Achieved Occupancy**: 40.43 %
- **IPC Active**: 0.99 inst/cycle | **No Eligible**: 75.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_larger_xblock | 0.998× | 0.997× | ✓ |  |
| B | memory_coalescing_via_autotune_xblock | 0.997× | 0.996× | ✓ |  |
| C | software_pipelining_increase_xblock_and_num_stages | 0.888× | 0.825× | ✓ |  |
| D | increase_xblock_with_autotune | 0.888× | 0.903× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ | software_pipelini... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 94590 ns | 94880 ns | +290 ns (+0.3%) | 94940 ns | +350 ns (+0.4%) | 114690 ns | +20100 ns (+21.2%) | 104800 ns | +10210 ns (+10.8%) |
| SM Throughput (%) | 24.28 % | 24.08 % | -0.20 pp | 28.04 % | +3.76 pp | 60.12 % | +35.84 pp | 36.91 % | +12.63 pp |
| Memory Throughput (%) | 27.50 % | 27.38 % | -0.12 pp | 27.40 % | -0.10 pp | 38.21 % | +10.71 pp | 25.01 % | -2.49 pp |
| DRAM Throughput (%) | 27.50 % | 27.38 % | -0.12 pp | 27.40 % | -0.10 pp | 22.84 % | -4.66 pp | 25.01 % | -2.49 pp |
| Achieved Occupancy (%) | 40.43 % | 41.01 % | +0.58 pp | 41.70 % | +1.27 pp | 42.63 % | +2.20 pp | 36.13 % | -4.30 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 24.00 register/thread | +8 | 16.00 register/thread | +0 |
| Waves/SM | 71.27 | 71.27 | +0.00 | 71.27 | +0.00 | 71.27 | +0.00 | 71.27 | +0.00 |
| No Eligible (%) | 75.07 % | 75.19 % | +0.12 pp | 71.18 % | -3.89 pp | 38.78 % | -36.29 pp | 60.71 % | -14.36 pp |
| IPC Active | 0.99 inst/cycle | 0.99 inst/cycle | +0.00 | 1.16 inst/cycle | +0.17 | 2.44 inst/cycle | +1.45 | 1.56 inst/cycle | +0.57 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_for_l1tex_latency_hiding | 0.999× | 0.997× | ✓ |  |
| B | vectorized_loads_via_contiguity_hints | 0.995× | 1.001× | ✓ |  |
| C | vectorized_loads_with_fixed_autotune_grid | 0.887× | 0.905× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 94590 ns | 94880 ns | +290 ns (+0.3%) | 94500 ns | -90 ns (-0.1%) | 104480 ns | +9890 ns (+10.5%) |
| SM Throughput (%) | 24.28 % | 24.07 % | -0.21 pp | 28.18 % | +3.90 pp | 37.00 % | +12.72 pp |
| Memory Throughput (%) | 27.50 % | 27.37 % | -0.13 pp | 27.51 % | +0.01 pp | 25.09 % | -2.41 pp |
| DRAM Throughput (%) | 27.50 % | 27.37 % | -0.13 pp | 27.51 % | +0.01 pp | 25.09 % | -2.41 pp |
| Achieved Occupancy (%) | 40.43 % | 40.51 % | +0.08 pp | 41.59 % | +1.16 pp | 36.27 % | -4.16 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 71.27 | 71.27 | +0.00 | 71.27 | +0.00 | 71.27 | +0.00 |
| No Eligible (%) | 75.07 % | 75.18 % | +0.11 pp | 71.10 % | -3.97 pp | 60.26 % | -14.81 pp |
| IPC Active | 0.99 inst/cycle | 0.99 inst/cycle | +0.00 | 1.16 inst/cycle | +0.17 | 1.58 inst/cycle | +0.59 |

---

## Cumulative Progress

```
Baseline:  94590 ns  (1.000×)

Final best: 94590 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,11 +5,14 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 19267584
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
+    xindex = tl.max_contiguous(
+                 tl.multiple_of(xoffset + tl.arange(0, XBLOCK)[:], XBLOCK),
+                 XBLOCK)
     xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x3 = xindex
     x1 = ((xindex // 784) % 6)
@@ -19,8 +22,6 @@

     tmp3 = tl.full([1], 0, tl.int32)
     tmp4 = tl.maximum(tmp3, tmp2)
     tl.store(in_out_ptr0 + (x3), tmp4, None)
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

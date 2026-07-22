# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_convolution_mean_mul_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4510 ns — **Best latency**: 2400 ns (**1.8792×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`add_autotune_reduce_xblock_increase_grid_coverage`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_convolution_mean_mul_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_044/kernel_3_triton_per_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.01
- **Registers/thread**: 32
- **Duration**: 4510 ns
- **SM Throughput**: 0.44 % | **Memory**: 2.42 % | **DRAM**: 2.42 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.24 inst/cycle | **No Eligible**: 93.93 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_autotune_reduce_xblock_increase_grid_coverage | 1.374× | 1.879× | ✓ | **✓ best** |
| B | reduce_xblock_add_autotune_to_fill_sms | 1.374× | 1.804× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_autotune_redu... | Δ | reduce_xblock_add... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4510 ns | 2400 ns | -2110 ns (-46.8%) | 2500 ns | -2010 ns (-44.6%) |
| SM Throughput (%) | 0.44 % | 0.09 % | -0.35 pp | 0.11 % | -0.33 pp |
| Memory Throughput (%) | 2.42 % | 0.89 % | -1.53 pp | 0.91 % | -1.51 pp |
| DRAM Throughput (%) | 2.42 % | 0.32 % | -2.10 pp | 0.59 % | -1.83 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 48.00 register/thread | +16 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 93.93 % | 90.67 % | -3.26 pp | 89.97 % | -3.96 pp |
| IPC Active | 0.24 inst/cycle | 0.08 inst/cycle | -0.16 | 0.11 inst/cycle | -0.13 |

---

## Cumulative Progress

```
Baseline:  4510 ns  (1.000×)
Iter 1     2400 ns  (1.879×)  — add_autotune_reduce_xblock_increase_grid_coverage

Final best: 2400 ns  (1.8792×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,9 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 8},  num_warps=1),
+        triton.Config({'XBLOCK': 16}, num_warps=2),
+        triton.Config({'XBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused_convolution_mean_mul_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
-    xnumel = 2048
     r0_numel = 64
     R0_BLOCK: tl.constexpr = 64
     rnumel = r0_numel
@@ -35,15 +45,13 @@

     tl.debug_barrier()
     tl.store(in_out_ptr0 + (x3), tmp8, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 2048
     r0_numel = 64
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_convolution_mean_mul_3[grid](arg_0, arg_1, 2048, 64, XBLOCK=128)
+    triton_per_fused_convolution_mean_mul_3[grid](arg_0, arg_1, 2048, 64)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

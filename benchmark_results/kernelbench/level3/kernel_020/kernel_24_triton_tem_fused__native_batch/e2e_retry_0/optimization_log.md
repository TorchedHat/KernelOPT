# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 54430 ns — **Best latency**: 54590 ns (**0.9971×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_24_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (72,1,1), Waves/SM: 0.14
- **Registers/thread**: 128
- **Duration**: 54430 ns
- **SM Throughput**: 12.05 % | **Memory**: 46.83 % | **DRAM**: 1.71 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 76.93 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.005× | 0.997× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 54430 ns | 54590 ns | +160 ns (+0.3%) |
| SM Throughput (%) | 12.05 % | 12.06 % | +0.01 pp |
| Memory Throughput (%) | 46.83 % | 46.85 % | +0.02 pp |
| DRAM Throughput (%) | 1.71 % | 1.70 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.14 | 0.14 | +0.00 |
| No Eligible (%) | 76.93 % | 76.92 % | -0.01 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  54430 ns  (1.000×)

Final best: 54590 ns  (0.9971×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24(arg_A, arg_B, out_ptr0):
@@ -87,8 +88,6 @@

     xindex = idx_n + 112*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

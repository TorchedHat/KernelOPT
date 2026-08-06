# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_18` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 58110 ns — **Best latency**: 58140 ns (**0.9995×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_18`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_18_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.24
- **Registers/thread**: 128
- **Duration**: 58110 ns
- **SM Throughput**: 21.37 % | **Memory**: 83.54 % | **DRAM**: 1.10 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 77.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 58110 ns | 58140 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 21.37 % | 21.26 % | -0.11 pp |
| Memory Throughput (%) | 83.54 % | 83.12 % | -0.42 pp |
| DRAM Throughput (%) | 1.10 % | 1.10 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 |
| No Eligible (%) | 77.00 % | 77.03 % | +0.03 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  58110 ns  (1.000×)

Final best: 58140 ns  (0.9995×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,10 +5,11 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_18(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 64
@@ -87,8 +88,6 @@

     xindex = idx_n + 1024*idx_m
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

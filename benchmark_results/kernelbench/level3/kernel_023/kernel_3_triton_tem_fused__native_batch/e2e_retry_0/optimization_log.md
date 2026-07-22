# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 52900 ns — **Best latency**: 53090 ns (**0.9964×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_3_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (980,1,1), Waves/SM: 3.71
- **Registers/thread**: 214
- **Duration**: 52900 ns
- **SM Throughput**: 23.60 % | **Memory**: 59.68 % | **DRAM**: 8.82 %
- **Achieved Occupancy**: 12.40 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 72.92 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.005× | 0.996× | ✓ | **✓ best** |
| B | tma_loads_block_ptr_reduce_register_pressure | 0.993× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_block_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 52900 ns | 53090 ns | +190 ns (+0.4%) | 53440 ns | +540 ns (+1.0%) |
| SM Throughput (%) | 23.60 % | 23.46 % | -0.14 pp | 23.55 % | -0.05 pp |
| Memory Throughput (%) | 59.68 % | 59.33 % | -0.35 pp | 59.18 % | -0.50 pp |
| DRAM Throughput (%) | 8.82 % | 8.72 % | -0.10 pp | 8.71 % | -0.11 pp |
| Achieved Occupancy (%) | 12.40 % | 12.41 % | +0.01 pp | 12.30 % | -0.10 pp |
| Registers/Thread | 214.00 register/thread | 214.00 register/thread | +0 | 246.00 register/thread | +32 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 72.92 % | 72.92 % | +0.00 pp | 73.17 % | +0.25 pp |
| IPC Active | 1.08 inst/cycle | 1.08 inst/cycle | +0.00 | 1.07 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  52900 ns  (1.000×)

Final best: 53090 ns  (0.9964×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 64
@@ -87,8 +88,6 @@

     xindex = idx_n + 64*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

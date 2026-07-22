# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 37310 ns — **Best latency**: 37310 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_11_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.31
- **Registers/thread**: 168
- **Duration**: 37310 ns
- **SM Throughput**: 24.04 % | **Memory**: 53.42 % | **DRAM**: 3.35 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.19 inst/cycle | **No Eligible**: 70.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.994× | 1.001× | ✓ |  |
| C | fix_b_matrix_access_pattern_and_reduce_register_pressure_via_autotune | 0.990× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fix_b_matrix_acce... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 37310 ns | 37280 ns | -30 ns (-0.1%) | 37700 ns | +390 ns (+1.0%) |
| SM Throughput (%) | 24.04 % | 24.06 % | +0.02 pp | 24.09 % | +0.05 pp |
| Memory Throughput (%) | 53.42 % | 53.46 % | +0.04 pp | 53.61 % | +0.19 pp |
| DRAM Throughput (%) | 3.35 % | 3.36 % | +0.01 pp | 3.32 % | -0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 70.41 % | 70.36 % | -0.05 pp | 70.48 % | +0.07 pp |
| IPC Active | 1.19 inst/cycle | 1.18 inst/cycle | -0.01 | 1.18 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | cap_registers_with_maxnreg_to_improve_occupancy | 0.995× | 0.983× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 37310 ns | 37950 ns | +640 ns (+1.7%) |
| SM Throughput (%) | 24.04 % | 24.20 % | +0.16 pp |
| Memory Throughput (%) | 53.42 % | 53.86 % | +0.44 pp |
| DRAM Throughput (%) | 3.35 % | 3.30 % | -0.05 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 |
| No Eligible (%) | 70.41 % | 70.51 % | +0.10 pp |
| IPC Active | 1.19 inst/cycle | 1.17 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  37310 ns  (1.000×)

Final best: 37310 ns  (1.0000×)
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
 def triton_tem_fused_convolution_11(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 128
@@ -89,8 +90,6 @@

     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
 
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

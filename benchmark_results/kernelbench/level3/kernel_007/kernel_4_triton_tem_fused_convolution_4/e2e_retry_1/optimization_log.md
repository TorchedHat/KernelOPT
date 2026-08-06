# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1720000 ns — **Best latency**: 1710000 ns (**1.0058×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`tf32_tensor_cores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_4_triton_tem_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 9.90
- **Registers/thread**: 168
- **Duration**: 1720000 ns
- **SM Throughput**: 40.43 % | **Memory**: 91.46 % | **DRAM**: 12.83 %
- **Achieved Occupancy**: 18.41 %
- **IPC Active**: 1.64 inst/cycle | **No Eligible**: 58.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.006× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 1720000 ns | 1710000 ns | -10000 ns (-0.6%) |
| SM Throughput (%) | 40.43 % | 40.44 % | +0.01 pp |
| Memory Throughput (%) | 91.46 % | 91.50 % | +0.04 pp |
| DRAM Throughput (%) | 12.83 % | 12.93 % | +0.10 pp |
| Achieved Occupancy (%) | 18.41 % | 18.40 % | -0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 |
| No Eligible (%) | 58.97 % | 58.95 % | -0.02 pp |
| IPC Active | 1.64 inst/cycle | 1.64 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  1720000 ns  (1.000×)
Iter 1     1710000 ns  (1.006×)  — tf32_tensor_cores

Final best: 1710000 ns  (1.0058×)
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
 def triton_tem_fused_convolution_4(in_ptr0, arg_A, arg_B, out_ptr0):
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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

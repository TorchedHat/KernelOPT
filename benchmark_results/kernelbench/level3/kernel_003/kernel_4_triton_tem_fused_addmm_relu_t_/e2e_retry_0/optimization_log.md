# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 942850 ns — **Best latency**: 942880 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_4_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 942850 ns
- **SM Throughput**: 40.36 % | **Memory**: 58.16 % | **DRAM**: 1.33 %
- **Achieved Occupancy**: 12.29 %
- **IPC Active**: 1.66 inst/cycle | **No Eligible**: 58.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 942850 ns | 942880 ns | +30 ns (+0.0%) |
| SM Throughput (%) | 40.36 % | 40.34 % | -0.02 pp |
| Memory Throughput (%) | 58.16 % | 58.12 % | -0.04 pp |
| DRAM Throughput (%) | 1.33 % | 1.33 % | +0.00 pp |
| Achieved Occupancy (%) | 12.29 % | 12.29 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 58.40 % | 58.42 % | +0.02 pp |
| IPC Active | 1.66 inst/cycle | 1.66 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  942850 ns  (1.000×)

Final best: 942880 ns  (1.0000×)
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
 def triton_tem_fused_addmm_relu_t_4(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -91,7 +92,6 @@

     )
     tl.store(out_block_ptr, acc, boundary_check=(0, 1))
 
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

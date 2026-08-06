# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 825700 ns — **Best latency**: 826430 ns (**0.9991×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_1_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 165.49
- **Registers/thread**: 160
- **Duration**: 825700 ns
- **SM Throughput**: 25.61 % | **Memory**: 83.27 % | **DRAM**: 6.87 %
- **Achieved Occupancy**: 18.03 %
- **IPC Active**: 1.03 inst/cycle | **No Eligible**: 74.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.999× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 825700 ns | 826430 ns | +730 ns (+0.1%) |
| SM Throughput (%) | 25.61 % | 25.62 % | +0.01 pp |
| Memory Throughput (%) | 83.27 % | 83.29 % | +0.02 pp |
| DRAM Throughput (%) | 6.87 % | 6.87 % | +0.00 pp |
| Achieved Occupancy (%) | 18.03 % | 18.04 % | +0.01 pp |
| Registers/Thread | 160.00 register/thread | 160.00 register/thread | +0 |
| Waves/SM | 165.49 | 165.49 | +0.00 |
| No Eligible (%) | 74.28 % | 74.28 % | +0.00 pp |
| IPC Active | 1.03 inst/cycle | 1.03 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  825700 ns  (1.000×)

Final best: 826430 ns  (0.9991×)
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
 def triton_tem_fused_convolution_1(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -87,7 +88,6 @@

     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
 
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

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 176990 ns — **Best latency**: 177020 ns (**0.9998×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_1_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.32
- **Registers/thread**: 168
- **Duration**: 176990 ns
- **SM Throughput**: 26.97 % | **Memory**: 60.77 % | **DRAM**: 0.97 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.22 inst/cycle | **No Eligible**: 69.54 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 176990 ns | 177020 ns | +30 ns (+0.0%) |
| SM Throughput (%) | 26.97 % | 27.01 % | +0.04 pp |
| Memory Throughput (%) | 60.77 % | 60.86 % | +0.09 pp |
| DRAM Throughput (%) | 0.97 % | 0.97 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.32 | 0.32 | +0.00 |
| No Eligible (%) | 69.54 % | 69.53 % | -0.01 pp |
| IPC Active | 1.22 inst/cycle | 1.22 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  176990 ns  (1.000×)

Final best: 177020 ns  (0.9998×)
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
 def triton_tem_fused_addmm_relu_t_1(arg_A, arg_B, in_ptr2, out_ptr1):
@@ -91,8 +92,6 @@

     tmp3 = tl.maximum(tmp2, tmp1)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1024*idx_m, [BLOCK_M, BLOCK_N])), tmp3, mask)
 
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

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_native_layer_norm_t_view_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12370000 ns — **Best latency**: 12370000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_native_layer_norm_t_view_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_1_triton_tem_fused_addmm_native_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (9216,1,1), Waves/SM: 34.91
- **Registers/thread**: 255
- **Duration**: 12370000 ns
- **SM Throughput**: 41.52 % | **Memory**: 60.61 % | **DRAM**: 1.52 %
- **Achieved Occupancy**: 12.46 %
- **IPC Active**: 1.68 inst/cycle | **No Eligible**: 58.05 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 12370000 ns | 12370000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 41.52 % | 41.51 % | -0.01 pp |
| Memory Throughput (%) | 60.61 % | 60.60 % | -0.01 pp |
| DRAM Throughput (%) | 1.52 % | 1.55 % | +0.03 pp |
| Achieved Occupancy (%) | 12.46 % | 12.46 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 58.05 % | 58.10 % | +0.05 pp |
| IPC Active | 1.68 inst/cycle | 1.68 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  12370000 ns  (1.000×)

Final best: 12370000 ns  (1.0000×)
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
 def triton_tem_fused_addmm_native_layer_norm_t_view_1(arg_A, arg_B, out_ptr0):
@@ -87,7 +88,6 @@

     xindex = idx_n + 2304*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
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

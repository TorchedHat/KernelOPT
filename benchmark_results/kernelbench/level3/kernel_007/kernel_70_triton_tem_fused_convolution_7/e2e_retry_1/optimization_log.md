# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_70` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 55550 ns — **Best latency**: 55550 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_70`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_70_triton_tem_fused_convolution_7/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.16
- **Registers/thread**: 80
- **Duration**: 55550 ns
- **SM Throughput**: 12.41 % | **Memory**: 84.48 % | **DRAM**: 1.62 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.005× | 1.000× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_warps_to_fix_occupancy_and_bank_conflicts | 1.001× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 55550 ns | 55550 ns | +0 ns (+0.0%) | 55580 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 12.41 % | 12.42 % | +0.01 pp | 12.42 % | +0.01 pp |
| Memory Throughput (%) | 84.48 % | 84.59 % | +0.11 pp | 84.55 % | +0.07 pp |
| DRAM Throughput (%) | 1.62 % | 1.62 % | +0.00 pp | 1.62 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.16 | 0.16 | +0.00 | 0.16 | +0.00 |
| No Eligible (%) | 86.25 % | 86.25 % | +0.00 pp | 86.26 % | +0.01 pp |
| IPC Active | 0.55 inst/cycle | 0.55 inst/cycle | +0.00 | 0.55 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  55550 ns  (1.000×)

Final best: 55550 ns  (1.0000×)
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
 def triton_tem_fused_convolution_70(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -92,8 +93,6 @@

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

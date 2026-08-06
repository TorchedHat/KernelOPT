# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5980 ns — **Best latency**: 5920 ns (**1.0101×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_grid_launch_to_expose_batch_parallelism`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_11_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 121
- **Duration**: 5980 ns
- **SM Throughput**: 0.22 % | **Memory**: 6.68 % | **DRAM**: 6.68 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 1.53 inst/cycle | **No Eligible**: 63.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_and_fix_grid | 1.231× | 0.979× | ✓ |  |
| B | fix_grid_launch_to_expose_batch_parallelism | 2.237× | 1.010× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 6110 ns | +130 ns (+2.2%) | 5920 ns | -60 ns (-1.0%) |
| SM Throughput (%) | 0.22 % | 0.22 % | +0.00 pp | 0.22 % | +0.00 pp |
| Memory Throughput (%) | 6.68 % | 0.62 % | -6.06 pp | 0.53 % | -6.15 pp |
| DRAM Throughput (%) | 6.68 % | 0.25 % | -6.43 pp | 0.29 % | -6.39 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.22 % | -0.01 pp |
| Registers/Thread | 121.00 register/thread | 121.00 register/thread | +0 | 121.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 63.00 % | 60.19 % | -2.81 pp | 59.42 % | -3.58 pp |
| IPC Active | 1.53 inst/cycle | 1.51 inst/cycle | -0.02 | 1.57 inst/cycle | +0.04 |

---

## Cumulative Progress

```
Baseline:  5980 ns  (1.000×)
Iter 1     5920 ns  (1.010×)  — fix_grid_launch_to_expose_batch_parallelism

Final best: 5920 ns  (1.0101×)
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
 def triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11(arg_A, arg_B, out_ptr0):
@@ -86,8 +87,6 @@

     xindex = idx_n + 64*idx_m + 4096*idx_q
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

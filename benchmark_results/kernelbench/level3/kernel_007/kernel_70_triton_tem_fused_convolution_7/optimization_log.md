# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_70` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 55550 ns — **Best latency**: 55550 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_70`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_70_triton_tem_fused_convolution_7/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.16
- **Registers/thread**: 80
- **Duration**: 55550 ns
- **SM Throughput**: 12.44 % | **Memory**: 84.74 % | **DRAM**: 1.62 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_autotune_tile_sizes | 0.502× | 0.453× | ✓ |  |
| D | autotune_tile_sizes_to_increase_parallelism_and_fix_bank_conflicts | 0.995× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_with_au... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 55550 ns | 122660 ns | +67110 ns (+120.8%) | 55710 ns | +160 ns (+0.3%) |
| SM Throughput (%) | 12.44 % | 21.83 % | +9.39 pp | 12.43 % | -0.01 pp |
| Memory Throughput (%) | 84.74 % | 86.68 % | +1.94 pp | 84.63 % | -0.11 pp |
| DRAM Throughput (%) | 1.62 % | 0.74 % | -0.88 pp | 1.62 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 12.49 % | +6.24 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 138.00 register/thread | +58 | 80.00 register/thread | +0 |
| Waves/SM | 0.16 | 0.94 | +0.78 | 0.16 | +0.00 |
| No Eligible (%) | 86.19 % | 76.12 % | -10.07 pp | 86.18 % | -0.01 pp |
| IPC Active | 0.55 inst/cycle | 0.95 inst/cycle | +0.40 | 0.55 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_strided_broadcast_to_eliminate_bank_conflicts | 0.999× | 1.000× | ✓ |  |
| C | revert_to_scalar_ptrs_small_tiles_with_maxnreg_and_pipelining | 0.957× | 0.850× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tiles_and_maxnreg | 0.963× | 0.968× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_stri... | Δ | revert_to_scalar_... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 55550 ns | 55580 ns | +30 ns (+0.1%) | 65380 ns | +9830 ns (+17.7%) | 57380 ns | +1830 ns (+3.3%) |
| SM Throughput (%) | 12.44 % | 12.52 % | +0.08 pp | 10.50 % | -1.94 pp | 12.51 % | +0.07 pp |
| Memory Throughput (%) | 84.74 % | 84.71 % | -0.03 pp | 72.85 % | -11.89 pp | 83.05 % | -1.69 pp |
| DRAM Throughput (%) | 1.62 % | 1.62 % | +0.00 pp | 1.38 % | -0.24 pp | 1.57 % | -0.05 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 64.00 register/thread | -16 | 64.00 register/thread | -16 |
| Waves/SM | 0.16 | 0.16 | +0.00 | 0.12 | -0.04 | 0.12 | -0.04 |
| No Eligible (%) | 86.19 % | 86.11 % | -0.08 pp | 88.32 % | +2.13 pp | 86.17 % | -0.02 pp |
| IPC Active | 0.55 inst/cycle | 0.56 inst/cycle | +0.01 | 0.46 inst/cycle | -0.09 | 0.55 inst/cycle | +0.00 |

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

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=5),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_70(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
@@ -92,8 +106,6 @@

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

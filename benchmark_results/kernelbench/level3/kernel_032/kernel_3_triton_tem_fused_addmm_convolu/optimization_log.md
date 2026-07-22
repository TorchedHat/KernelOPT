# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_convolution_t_view_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 384260 ns — **Best latency**: 384260 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_convolution_t_view_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_032/kernel_3_triton_tem_fused_addmm_convolu/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4,1,1), Waves/SM: 0.01
- **Registers/thread**: 66
- **Duration**: 384260 ns
- **SM Throughput**: 0.24 % | **Memory**: 2.94 % | **DRAM**: 0.24 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.32 inst/cycle | **No Eligible**: 92.03 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.959× | 0.958× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 384260 ns | 400930 ns | +16670 ns (+4.3%) |
| SM Throughput (%) | 0.24 % | 0.26 % | +0.02 pp |
| Memory Throughput (%) | 2.94 % | 2.84 % | -0.10 pp |
| DRAM Throughput (%) | 0.24 % | 0.23 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 66.00 register/thread | 52.00 register/thread | -14 |
| Waves/SM | 0.01 | 0.00 | -0.01 |
| No Eligible (%) | 92.03 % | 91.46 % | -0.57 pp |
| IPC Active | 0.32 inst/cycle | 0.34 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_via_smem_staging_with_padding | 0.959× | 0.958× | ✓ |  |
| B | persistent_kernel_k_parallel_atomic_reduction | 0.958× | 0.956× | ✓ |  |
| D | fix_b_matrix_load_indexing_and_increase_grid_coverage | 0.951× | 0.950× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | persistent_kernel... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 384260 ns | 400930 ns | +16670 ns (+4.3%) | 402110 ns | +17850 ns (+4.6%) | 404510 ns | +20250 ns (+5.3%) |
| SM Throughput (%) | 0.24 % | 0.26 % | +0.02 pp | 0.26 % | +0.02 pp | 0.26 % | +0.02 pp |
| Memory Throughput (%) | 2.94 % | 2.84 % | -0.10 pp | 2.83 % | -0.11 pp | 2.81 % | -0.13 pp |
| DRAM Throughput (%) | 0.24 % | 0.23 % | -0.01 pp | 0.23 % | -0.01 pp | 0.23 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 66.00 register/thread | 52.00 register/thread | -14 | 52.00 register/thread | -14 | 53.00 register/thread | -13 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 92.03 % | 91.46 % | -0.57 pp | 91.46 % | -0.57 pp | 91.45 % | -0.58 pp |
| IPC Active | 0.32 inst/cycle | 0.34 inst/cycle | +0.02 | 0.34 inst/cycle | +0.02 | 0.34 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  384260 ns  (1.000×)

Final best: 384260 ns  (1.0000×)
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
 def triton_tem_fused_addmm_convolution_t_view_3(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -12,7 +13,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 16
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
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

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

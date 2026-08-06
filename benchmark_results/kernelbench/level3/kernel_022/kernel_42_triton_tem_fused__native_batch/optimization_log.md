# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 118850 ns — **Best latency**: 118850 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_42_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (80,1,1), Waves/SM: 0.30
- **Registers/thread**: 128
- **Duration**: 118850 ns
- **SM Throughput**: 7.62 % | **Memory**: 56.72 % | **DRAM**: 0.65 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.51 inst/cycle | **No Eligible**: 87.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_autotune_tile_sizes | 0.891× | 0.887× | ✓ |  |
| C | reduce_shared_memory_pressure_via_autotune_tile_sizes | 0.994× | 0.958× | ✓ |  |
| D | autotune_tile_sizes_reduce_shared_memory_pressure | 0.944× | 0.940× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_with_au... | Δ | reduce_shared_mem... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 118850 ns | 133950 ns | +15100 ns (+12.7%) | 124030 ns | +5180 ns (+4.4%) | 126430 ns | +7580 ns (+6.4%) |
| SM Throughput (%) | 7.62 % | 13.08 % | +5.46 pp | 7.40 % | -0.22 pp | 13.61 % | +5.99 pp |
| Memory Throughput (%) | 56.72 % | 50.96 % | -5.76 pp | 54.45 % | -2.27 pp | 53.39 % | -3.33 pp |
| DRAM Throughput (%) | 0.65 % | 0.57 % | -0.08 pp | 0.62 % | -0.03 pp | 0.61 % | -0.04 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 92.00 register/thread | -36 | 128.00 register/thread | +0 |
| Waves/SM | 0.30 | 0.15 | -0.15 | 0.12 | -0.18 | 0.15 | -0.15 |
| No Eligible (%) | 87.19 % | 78.04 % | -9.15 pp | 87.55 % | +0.36 pp | 76.90 % | -10.29 pp |
| IPC Active | 0.51 inst/cycle | 0.88 inst/cycle | +0.37 | 0.50 inst/cycle | -0.01 | 0.92 inst/cycle | +0.41 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_via_larger_block_n | 0.819× | 0.781× | ✓ |  |
| C | fix_shared_memory_bank_conflicts_b_matrix_order | 0.889× | 0.889× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 118850 ns | 152160 ns | +33310 ns (+28.0%) | 133660 ns | +14810 ns (+12.5%) |
| SM Throughput (%) | 7.62 % | 11.18 % | +3.56 pp | 13.07 % | +5.45 pp |
| Memory Throughput (%) | 56.72 % | 44.18 % | -12.54 pp | 50.94 % | -5.78 pp |
| DRAM Throughput (%) | 0.65 % | 0.51 % | -0.14 pp | 0.57 % | -0.08 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 164.00 register/thread | +36 | 128.00 register/thread | +0 |
| Waves/SM | 0.30 | 0.20 | -0.10 | 0.15 | -0.15 |
| No Eligible (%) | 87.19 % | 80.69 % | -6.50 pp | 78.05 % | -9.14 pp |
| IPC Active | 0.51 inst/cycle | 0.77 inst/cycle | +0.26 | 0.88 inst/cycle | +0.37 |

---

## Cumulative Progress

```
Baseline:  118850 ns  (1.000×)

Final best: 118850 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -12,7 +22,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -87,8 +97,6 @@

     xindex = idx_n + 320*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
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

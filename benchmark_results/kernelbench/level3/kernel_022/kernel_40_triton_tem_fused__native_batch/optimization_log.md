# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 62430 ns — **Best latency**: 62430 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_40_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (96,1,1), Waves/SM: 0.24
- **Registers/thread**: 80
- **Duration**: 62430 ns
- **SM Throughput**: 9.48 % | **Memory**: 64.94 % | **DRAM**: 1.04 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.54 inst/cycle | **No Eligible**: 86.52 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 0.507× | 0.493× | ✓ |  |
| C | reduce_shared_memory_bank_conflicts_via_autotune_tile_sizes | 0.979× | 0.978× | ✓ |  |
| D | autotune_tile_sizes_to_fix_occupancy_and_bank_conflicts | 0.418× | 0.404× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | reduce_shared_mem... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 62430 ns | 126500 ns | +64070 ns (+102.6%) | 63840 ns | +1410 ns (+2.3%) | 154620 ns | +92190 ns (+147.7%) |
| SM Throughput (%) | 9.48 % | 16.34 % | +6.86 pp | 9.52 % | +0.04 pp | 13.29 % | +3.81 pp |
| Memory Throughput (%) | 64.94 % | 64.07 % | -0.87 pp | 63.51 % | -1.43 pp | 52.51 % | -12.43 pp |
| DRAM Throughput (%) | 1.04 % | 0.51 % | -0.53 pp | 1.01 % | -0.03 pp | 0.42 % | -0.62 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 128.00 register/thread | +48 | 64.00 register/thread | -16 | 164.00 register/thread | +84 |
| Waves/SM | 0.24 | 0.18 | -0.06 | 0.09 | -0.15 | 0.24 | +0.00 |
| No Eligible (%) | 86.52 % | 76.91 % | -9.61 pp | 86.46 % | -0.06 pp | 80.70 % | -5.82 pp |
| IPC Active | 0.54 inst/cycle | 0.92 inst/cycle | +0.38 | 0.54 inst/cycle | +0.00 | 0.77 inst/cycle | +0.23 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_with_grid_lambda_to_fix_bank_conflicts_and_occupancy | 0.980× | 0.571× | ✓ |  |
| B | fix_register_explosion_via_autotune_with_correct_grid_lambda | 0.997× | 1.000× | ✓ |  |
| C | fix_tile_grid_mismatch_revert_implementation_flaw | 0.996× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_with_gri... | Δ | fix_register_expl... | Δ | fix_tile_grid_mis... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 62430 ns | 109340 ns | +46910 ns (+75.1%) | 62400 ns | -30 ns (-0.0%) | 62400 ns | -30 ns (-0.0%) |
| SM Throughput (%) | 9.48 % | 4.97 % | -4.51 pp | 9.50 % | +0.02 pp | 9.49 % | +0.01 pp |
| Memory Throughput (%) | 64.94 % | 38.65 % | -26.29 pp | 65.09 % | +0.15 pp | 65.03 % | +0.09 pp |
| DRAM Throughput (%) | 1.04 % | 0.59 % | -0.45 pp | 1.04 % | +0.00 pp | 1.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 78.00 register/thread | -2 | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.06 | -0.18 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 86.52 % | 86.12 % | -0.40 pp | 86.50 % | -0.02 pp | 86.48 % | -0.04 pp |
| IPC Active | 0.54 inst/cycle | 0.28 inst/cycle | -0.26 | 0.54 inst/cycle | +0.00 | 0.54 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  62430 ns  (1.000×)

Final best: 62430 ns  (1.0000×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40(arg_A, arg_B, out_ptr0):
@@ -12,7 +13,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 32
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -65,16 +66,14 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 1152*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 192*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 1152*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 1152*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_first')
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +85,6 @@

     # inductor generates a suffix
     xindex = idx_n + 192*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

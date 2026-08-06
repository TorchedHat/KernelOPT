# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 88380 ns — **Best latency**: 38590 ns (**2.2902×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`split_k_parallelism`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_31`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_31_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (12,1,1), Waves/SM: 0.01
- **Registers/thread**: 53
- **Duration**: 88380 ns
- **SM Throughput**: 0.77 % | **Memory**: 8.14 % | **DRAM**: 0.62 %
- **Achieved Occupancy**: 6.21 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 91.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.040× | 1.043× | ✓ |  |
| C | split_k_parallelism | 2.860× | 2.290× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | split_k_paralleli... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 88380 ns | 84740 ns | -3640 ns (-4.1%) | 38590 ns | -49790 ns (-56.3%) |
| SM Throughput (%) | 0.77 % | 0.73 % | -0.04 pp | 0.39 % | -0.38 pp |
| Memory Throughput (%) | 8.14 % | 8.44 % | +0.30 pp | 4.62 % | -3.52 pp |
| DRAM Throughput (%) | 0.62 % | 0.65 % | +0.03 pp | 1.42 % | +0.80 pp |
| Achieved Occupancy (%) | 6.21 % | 6.21 % | +0.00 pp | 3.12 % | -3.09 pp |
| Registers/Thread | 53.00 register/thread | 56.00 register/thread | +3 | 56.00 register/thread | +3 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 91.32 % | 91.73 % | +0.41 pp | 92.47 % | +1.15 pp |
| IPC Active | 0.34 inst/cycle | 0.33 inst/cycle | -0.01 | 0.15 inst/cycle | -0.19 |

---

## Iteration 2

**Best so far**: 2.2902× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_n_to_fill_sm_grid | 1.722× | 1.014× | ✓ |  |
| B | reduce_block_m_to_match_actual_m_with_autotune | 2.847× | 1.728× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_n_... | Δ | reduce_block_m_to... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 88380 ns | 87140 ns | -1240 ns (-1.4%) | 51140 ns | -37240 ns (-42.1%) |
| SM Throughput (%) | 0.77 % | 0.47 % | -0.30 pp | 0.23 % | -0.54 pp |
| Memory Throughput (%) | 8.14 % | 6.55 % | -1.59 pp | 1.88 % | -6.26 pp |
| DRAM Throughput (%) | 0.62 % | 0.63 % | +0.01 pp | 1.07 % | +0.45 pp |
| Achieved Occupancy (%) | 6.21 % | 3.12 % | -3.09 pp | 1.56 % | -4.65 pp |
| Registers/Thread | 53.00 register/thread | 125.00 register/thread | +72 | 72.00 register/thread | +19 |
| Waves/SM | 0.01 | 0.02 | +0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.32 % | 89.39 % | -1.93 pp | 90.13 % | -1.19 pp |
| IPC Active | 0.34 inst/cycle | 0.21 inst/cycle | -0.13 | 0.10 inst/cycle | -0.24 |

---

## Iteration 3

**Best so far**: 2.2902× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | reduce_block_n_to_cut_smem_bank_conflicts_and_register_pressure | 2.784× | 1.373× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_n_to... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 88380 ns | 64380 ns | -24000 ns (-27.2%) |
| SM Throughput (%) | 0.77 % | 0.31 % | -0.46 pp |
| Memory Throughput (%) | 8.14 % | 2.80 % | -5.34 pp |
| DRAM Throughput (%) | 0.62 % | 0.85 % | +0.23 pp |
| Achieved Occupancy (%) | 6.21 % | 1.56 % | -4.65 pp |
| Registers/Thread | 53.00 register/thread | 126.00 register/thread | +73 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 91.32 % | 85.86 % | -5.46 pp |
| IPC Active | 0.34 inst/cycle | 0.14 inst/cycle | -0.20 |


### Key Learnings

#### ✅ Plan C: `reduce_block_n_to_cut_smem_bank_conflicts_and_register_pressure` (1.373×)

Doubling BLOCK_N from 32 to 64 in a tiny Triton matmul (BLOCK_M=2, BLOCK_K=64) increased dynamic shared memory per block from 8.7 KB to 16.9 KB, which raised the bank conflict way from 12 to 17 and increased bank-conflicting wavefronts from 77.78% to 82.35% of all shared loads. Simultaneously, registers per thread nearly doubled from 72 to 126, reducing theoretical occupancy from 37.5% to 20.31% and increasing total executed instructions by ~80%. The net effect was a 26% longer kernel duration despite a modest improvement in Warp Cycles Per Issued Instruction (10.12 → 7.09), because the larger tile created more total work and more severe memory pressure rather than reducing it.

> **Applicable when**: Never: this direction (doubling BLOCK_N when smem bank conflicts already dominate) consistently worsens performance by increasing smem footprint, bank conflict severity, and register pressure simultaneously.

> **Avoid when**: When NCU reports SharedMemoryConflicts with high bank conflict way (>=8x) and MIO stalls dominate CPI: doubling BLOCK_N increases the [BLOCK_K x BLOCK_N] shared memory tile, which amplifies rather than reduces bank conflicts. Also avoid when registers per thread are already high (>60) — a larger accumulator tile [BLOCK_M, BLOCK_N] will push register usage toward the hardware limit (255), reducing theoretical occupancy further.

---

## Cumulative Progress

```
Baseline:  88380 ns  (1.000×)
Iter 1     38590 ns  (2.290×)  — split_k_parallelism

Final best: 38590 ns  (2.2902×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=1, num_stages=2),
+        triton.Config({}, num_warps=1, num_stages=3),
+        triton.Config({}, num_warps=1, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_31(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -102,8 +117,6 @@

     tmp14 = tmp12 + tmp13
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 384*idx_m, [BLOCK_M, BLOCK_N])), tmp14, mask)
 
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

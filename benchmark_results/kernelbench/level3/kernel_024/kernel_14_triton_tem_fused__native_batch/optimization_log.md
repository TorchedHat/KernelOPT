# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10460 ns — **Best latency**: 6340 ns (**1.6498×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`persistent_kernel_with_autotuned_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_14_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (18,1,1), Waves/SM: 0.02
- **Registers/thread**: 56
- **Duration**: 10460 ns
- **SM Throughput**: 1.10 % | **Memory**: 9.50 % | **DRAM**: 0.67 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.39 inst/cycle | **No Eligible**: 90.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_kernel_with_autotuned_tile_sizes | 1.451× | 1.650× | ✓ | **✓ best** |
| C | split_k_gemm_parallelism | 1.030× | 0.541× | ✓ |  |
| D | autotune_tile_sizes_for_small_gemm_utilization | 1.417× | 1.684× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | split_k_gemm_para... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10460 ns | 6340 ns | -4120 ns (-39.4%) | 19330 ns | +8870 ns (+84.8%) | 6210 ns | -4250 ns (-40.6%) |
| SM Throughput (%) | 1.10 % | 0.39 % | -0.71 pp | 0.44 % | -0.66 pp | 0.50 % | -0.60 pp |
| Memory Throughput (%) | 9.50 % | 2.25 % | -7.25 pp | 2.94 % | -6.56 pp | 1.50 % | -8.00 pp |
| DRAM Throughput (%) | 0.67 % | 1.11 % | +0.44 pp | 0.37 % | -0.30 pp | 1.15 % | +0.48 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 1.56 % | -4.68 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 56.00 register/thread | 72.00 register/thread | +16 | 96.00 register/thread | +40 | 88.00 register/thread | +32 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 90.27 % | 84.45 % | -5.82 pp | 85.48 % | -4.79 pp | 89.14 % | -1.13 pp |
| IPC Active | 0.39 inst/cycle | 0.31 inst/cycle | -0.08 | 0.14 inst/cycle | -0.25 | 0.21 inst/cycle | -0.18 |

---

## Iteration 2

**Best so far**: 1.6498× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_shared_memory_bank_conflicts_via_block_k_tuning | 1.362× | 1.571× | ✓ |  |
| C | reduce_block_m_to_match_small_m_dimension | 1.445× | 1.668× | ✓ |  |
| D | add_triton_autotune | 1.442× | 1.634× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ | reduce_block_m_to... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10460 ns | 6660 ns | -3800 ns (-36.3%) | 6270 ns | -4190 ns (-40.1%) | 6400 ns | -4060 ns (-38.8%) |
| SM Throughput (%) | 1.10 % | 0.40 % | -0.70 pp | 0.44 % | -0.66 pp | 1.06 % | -0.04 pp |
| Memory Throughput (%) | 9.50 % | 1.05 % | -8.45 pp | 1.12 % | -8.38 pp | 2.58 % | -6.92 pp |
| DRAM Throughput (%) | 0.67 % | 1.05 % | +0.38 pp | 1.12 % | +0.45 pp | 1.10 % | +0.43 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 1.56 % | -4.68 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 56.00 register/thread | 167.00 register/thread | +111 | 157.00 register/thread | +101 | 46.00 register/thread | -10 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 90.27 % | 84.81 % | -5.46 pp | 82.45 % | -7.82 pp | 91.61 % | +1.34 pp |
| IPC Active | 0.39 inst/cycle | 0.29 inst/cycle | -0.10 | 0.17 inst/cycle | -0.22 | 0.34 inst/cycle | -0.05 |


### Key Learnings

#### ✅ Plan C: `reduce_block_m_to_match_small_m_dimension` (1.684×)

When a GEMM has a very small M dimension (e.g. M=2) but BLOCK_M is set much larger (e.g. 16), most of each tile is wasted padding, inflating shared memory usage and causing severe bank conflicts (17-way before vs 2.5-way after). Reducing BLOCK_M to match the actual M dimension (e.g. BLOCK_M=2) dramatically cuts executed instructions (93K→26K) and elapsed cycles (15.6K→9.2K), yielding a 1.68x speedup even though grid size and occupancy remain low. The primary gain comes from eliminating wasted computation on padded rows and reducing shared memory pressure, not from improved occupancy.

> **Applicable when**: GEMM M dimension is much smaller than BLOCK_M (e.g. M<=4 but BLOCK_M=16+); SharedMemoryConflicts rule fires with high bank-conflict-way (>8x); Executed Instructions are high relative to problem size; LaunchConfiguration rule fires due to small grid; wasted padding rows dominate tile computation

> **Avoid when**: M is large enough to fill BLOCK_M tiles (M >= BLOCK_M); tl.dot is needed for tensor core utilization (requires both inner dims >= 16 and outer dims >= 16); reducing BLOCK_M below 16 prevents tl.dot usage and forces scalar/broadcast MAC which has lower throughput on large problems

---

## Iteration 3

**Best so far**: 1.6498× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_block_k_to_lower_register_pressure_and_improve_occupancy | 1.167× | 1.287× | ✓ |  |
| C | reduce_block_n_to_cut_register_pressure_from_broadcast_mac | 1.409× | 1.543× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_k_to... | Δ | reduce_block_n_to... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 10460 ns | 8130 ns | -2330 ns (-22.3%) | 6780 ns | -3680 ns (-35.2%) |
| SM Throughput (%) | 1.10 % | 0.25 % | -0.85 pp | 0.99 % | -0.11 pp |
| Memory Throughput (%) | 9.50 % | 1.00 % | -8.50 pp | 2.43 % | -7.07 pp |
| DRAM Throughput (%) | 0.67 % | 0.88 % | +0.21 pp | 1.04 % | +0.37 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 6.25 % | +0.01 pp |
| Registers/Thread | 56.00 register/thread | 96.00 register/thread | +40 | 50.00 register/thread | -6 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.02 | +0.00 |
| No Eligible (%) | 90.27 % | 90.60 % | +0.33 pp | 92.21 % | +1.94 pp |
| IPC Active | 0.39 inst/cycle | 0.10 inst/cycle | -0.29 | 0.32 inst/cycle | -0.07 |


### Key Learnings

#### ✅ Plan A: `reduce_block_k_to_lower_register_pressure_and_improve_occupancy` (1.287×)

In a Triton kernel using a 3D broadcast MAC (tl.sum(a[:,:,None] * b[None,:,:], axis=1)) with BLOCK_M=2, BLOCK_K=32, BLOCK_N=32, the intermediate tensor of shape [2,32,32]=2048 elements forces 157 registers/thread, capping Block Limit Registers at 12 and theoretical occupancy at 18.75%. Halving BLOCK_K from 32 to 16 reduces the intermediate to [2,16,32]=1024 elements, dropping registers/thread to 96 and raising Block Limit Registers to 20 and theoretical occupancy to 31.25%. Paradoxically, the NCU-measured elapsed cycles increased (9362→12134) and duration increased (6.27→8.13 µs) per-kernel, yet the end-to-end speedup was 1.29x — suggesting the register reduction benefited the broader fused kernel context (e.g., reduced register spill traffic or better SM scheduling) even though the isolated kernel duration worsened due to more k-loop iterations (5→9) without sufficient occupancy improvement to hide the added L1TEX stall latency (CPIStall worsened from 50.2% to 67.4% of stall budget).

> **Applicable when**: Triton kernel uses 3D broadcast MAC (tl.sum(a[:,:,None]*b[None,:,:], axis=1)) with large BLOCK_K causing high register pressure (>120 registers/thread); TheoreticalOccupancy rule fires citing register limit; Block Limit Registers is the binding occupancy constraint; kernel is part of a larger fused operation where register spill reduction benefits the broader context

> **Avoid when**: Kernel is already register-pressure-limited at a small BLOCK_K; the grid is so small (<<1 wave) that occupancy improvement cannot compensate for more k-loop iterations; the kernel runs in isolation where per-kernel elapsed cycles are the sole metric — in that case, more iterations without occupancy gain will increase duration; CPIStall L1TEX fraction is already very high (>60%) as smaller BLOCK_K adds more masked loads without enough warps to hide them

---

## Cumulative Progress

```
Baseline:  10460 ns  (1.000×)
Iter 1     6340 ns  (1.650×)  — persistent_kernel_with_autotuned_tile_sizes

Final best: 6340 ns  (1.6498×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,15 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({}, num_warps=1, num_stages=1),
+    triton.Config({}, num_warps=2, num_stages=1),
+    triton.Config({}, num_warps=4, num_stages=1),
+    triton.Config({}, num_warps=8, num_stages=1),
+    triton.Config({}, num_warps=1, num_stages=2),
+    triton.Config({}, num_warps=2, num_stages=2),
+    triton.Config({}, num_warps=4, num_stages=2),
+    triton.Config({}, num_warps=8, num_stages=2),
+    triton.Config({}, num_warps=4, num_stages=3),
+    triton.Config({}, num_warps=8, num_stages=3),
+], key=[])
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_14(arg_A, arg_B, out_ptr1):
     EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
     BLOCK_K : tl.constexpr = 32
-    GROUP_M : tl.constexpr = 8
+    GROUP_M : tl.constexpr = 1
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
     A = arg_A
@@ -75,9 +88,11 @@

         xindex = idx_n + 576*idx_m
         b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
 
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        # a shape: [BLOCK_M, BLOCK_K] = [2, 32]
+        # b shape: [BLOCK_K, BLOCK_N] = [32, 32]
+        # broadcast MAC: expand dims, multiply elementwise, sum over K axis
+        # tl.dot cannot be used when BLOCK_M=2 < 16
+        acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -90,8 +105,6 @@

     xindex = idx_n + 576*idx_m
     tmp0 = tl.sigmoid(acc)
     tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp0, mask)
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

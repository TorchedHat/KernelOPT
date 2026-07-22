# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_addmm_avg_pool2d_convolution_relu_t_view_22` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 51520 ns — **Best latency**: 39940 ns (**1.2899×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`restore_register_pressure_via_smaller_tiles_with_proper_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_addmm_avg_pool2d_convolution_relu_t_view_22`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_22_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32,1,1), Waves/SM: 0.06
- **Registers/thread**: 66
- **Duration**: 51520 ns
- **SM Throughput**: 1.94 % | **Memory**: 22.05 % | **DRAM**: 1.65 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.33 inst/cycle | **No Eligible**: 91.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr_with_autotuned_tile_sizes | 0.586× | 0.577× | ✓ |  |
| C | fix_b_matrix_load_coalescing_and_add_autotune | 1.241× | 1.282× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_fix_underutilization_and_bank_conflicts | 0.405× | 0.347× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ | fix_b_matrix_load... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 51520 ns | 89310 ns | +37790 ns (+73.4%) | 40190 ns | -11330 ns (-22.0%) | 148640 ns | +97120 ns (+188.5%) |
| SM Throughput (%) | 1.94 % | 3.67 % | +1.73 pp | 2.27 % | +0.33 pp | 4.15 % | +2.21 pp |
| Memory Throughput (%) | 22.05 % | 14.18 % | -7.87 pp | 14.98 % | -7.07 pp | 9.02 % | -13.03 pp |
| DRAM Throughput (%) | 1.65 % | 0.96 % | -0.69 pp | 2.12 % | +0.47 pp | 0.58 % | -1.07 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 66.00 register/thread | 211.00 register/thread | +145 | 110.00 register/thread | +44 | 254.00 register/thread | +188 |
| Waves/SM | 0.06 | 0.08 | +0.02 | 0.03 | -0.03 | 0.06 | +0.00 |
| No Eligible (%) | 91.67 % | 67.66 % | -24.01 pp | 80.23 % | -11.44 pp | 64.89 % | -26.78 pp |
| IPC Active | 0.33 inst/cycle | 0.65 inst/cycle | +0.32 | 0.39 inst/cycle | +0.06 | 0.70 inst/cycle | +0.37 |

---

## Iteration 2

**Best so far**: 1.2819× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_smaller_block_k_autotune | 0.877× | 0.669× | ✓ |  |
| B | revert_block_ptr_to_scalar_loads_reduce_register_pressure | 0.789× | 0.779× | ✓ |  |
| C | restore_register_pressure_via_smaller_tiles_with_proper_autotune | 1.418× | 1.290× | ✓ | **✓ best** |
| D | add_triton_autotune | 1.236× | 1.287× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | revert_block_ptr_... | Δ | restore_register_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 51520 ns | 77060 ns | +25540 ns (+49.6%) | 66110 ns | +14590 ns (+28.3%) | 39940 ns | -11580 ns (-22.5%) | 40030 ns | -11490 ns (-22.3%) |
| SM Throughput (%) | 1.94 % | 2.13 % | +0.19 pp | 2.93 % | +0.99 pp | 2.27 % | +0.33 pp | 2.27 % | +0.33 pp |
| Memory Throughput (%) | 22.05 % | 14.73 % | -7.32 pp | 17.73 % | -4.32 pp | 15.01 % | -7.04 pp | 15.00 % | -7.05 pp |
| DRAM Throughput (%) | 1.65 % | 1.10 % | -0.55 pp | 1.28 % | -0.37 pp | 2.13 % | +0.48 pp | 2.13 % | +0.48 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 66.00 register/thread | 96.00 register/thread | +30 | 126.00 register/thread | +60 | 110.00 register/thread | +44 | 110.00 register/thread | +44 |
| Waves/SM | 0.06 | 0.02 | -0.04 | 0.04 | -0.02 | 0.03 | -0.03 | 0.03 | -0.03 |
| No Eligible (%) | 91.67 % | 81.73 % | -9.94 pp | 74.93 % | -16.74 pp | 80.24 % | -11.43 pp | 80.26 % | -11.41 pp |
| IPC Active | 0.33 inst/cycle | 0.36 inst/cycle | +0.03 | 0.50 inst/cycle | +0.17 | 0.39 inst/cycle | +0.06 | 0.39 inst/cycle | +0.06 |


### Key Learnings

#### ❌ Plan B: `revert_block_ptr_to_scalar_loads_reduce_register_pressure` (0.779×)

Reverting from block_ptr to scalar pointer arithmetic while simultaneously changing tile sizes (BLOCK_N 32→128, BLOCK_K 128→32, BLOCK_M 16→8) caused a 1.28x slowdown despite reducing Warp Cycles Per Issued Instruction from 5.07 to 4.01. The dominant effect was a 2.13x increase in executed instructions (720K→1536K), driven by the larger BLOCK_N=128 tile requiring more work per block and more shared memory (24.6KB→34.8KB). Registers per thread increased from 110 to 126, and theoretical occupancy dropped from 25% to 18.75% due to the larger shared memory footprint. The bank conflict severity also worsened (12-way→17-way), indicating the new tile layout created a less favorable access pattern for shared memory. The starting kernel (before=40us) was already in a partially-recovered state from a prior regression, and the tile size changes made things worse rather than better.

> **Applicable when**: Never apply this specific combination: increasing BLOCK_N dramatically (32→128) while decreasing BLOCK_K (128→32) and BLOCK_M (16→8) in a tiny GEMM (M=10, N=1000, K=1024). The larger BLOCK_N tile increases total instruction count and shared memory pressure without proportional benefit.

> **Avoid when**: When the kernel is already using scalar pointer arithmetic and the tile sizes are reasonable (BLOCK_N=32, BLOCK_K=128). Changing tile sizes in a small GEMM without verifying the total instruction count and shared memory footprint will likely cause regression. Specifically, increasing BLOCK_N from 32 to 128 for N=1000 increases the number of output elements computed per block by 4x, which inflates instruction count and smem usage. Also avoid when bank conflict severity is already moderate (12-way) — changing tile layout can worsen it to 17-way.

---

## Iteration 3

**Best so far**: 1.2899× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_b_matrix_transpose_shared_memory_bank_conflicts | 0.877× | 0.669× | ✓ |  |
| C | eliminate_b_matrix_smem_bank_conflicts_via_explicit_transpose | 0.789× | 0.778× | ✓ |  |
| E | add_triton_autotune | 0.999× | 0.924× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_b_matri... | Δ | eliminate_b_matri... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 51520 ns | 76990 ns | +25470 ns (+49.4%) | 66210 ns | +14690 ns (+28.5%) | 55740 ns | +4220 ns (+8.2%) |
| SM Throughput (%) | 1.94 % | 2.13 % | +0.19 pp | 2.92 % | +0.98 pp | 1.74 % | -0.20 pp |
| Memory Throughput (%) | 22.05 % | 14.70 % | -7.35 pp | 17.65 % | -4.40 pp | 20.30 % | -1.75 pp |
| DRAM Throughput (%) | 1.65 % | 1.10 % | -0.55 pp | 1.28 % | -0.37 pp | 1.52 % | -0.13 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 66.00 register/thread | 96.00 register/thread | +30 | 126.00 register/thread | +60 | 66.00 register/thread | +0 |
| Waves/SM | 0.06 | 0.02 | -0.04 | 0.04 | -0.02 | 0.03 | -0.03 |
| No Eligible (%) | 91.67 % | 81.81 % | -9.86 pp | 75.04 % | -16.63 pp | 92.51 % | +0.84 pp |
| IPC Active | 0.33 inst/cycle | 0.36 inst/cycle | +0.03 | 0.50 inst/cycle | +0.17 | 0.30 inst/cycle | -0.03 |

---

## Iteration 4

**Best so far**: 1.2899× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.995× | 0.924× | ✓ |  |
| C | tf32_tensor_cores_with_grid_expansion | 1.398× | 1.153× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tf32_tensor_cores... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 51520 ns | 55780 ns | +4260 ns (+8.3%) | 44670 ns | -6850 ns (-13.3%) |
| SM Throughput (%) | 1.94 % | 1.75 % | -0.19 pp | 1.99 % | +0.05 pp |
| Memory Throughput (%) | 22.05 % | 20.42 % | -1.63 pp | 13.43 % | -8.62 pp |
| DRAM Throughput (%) | 1.65 % | 1.52 % | -0.13 pp | 1.90 % | +0.25 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 66.00 register/thread | 66.00 register/thread | +0 | 80.00 register/thread | +14 |
| Waves/SM | 0.06 | 0.03 | -0.03 | 0.02 | -0.04 |
| No Eligible (%) | 91.67 % | 92.54 % | +0.87 pp | 82.67 % | -9.00 pp |
| IPC Active | 0.33 inst/cycle | 0.30 inst/cycle | -0.03 | 0.35 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  51520 ns  (1.000×)
Iter 1     40190 ns  (1.282×)  — fix_b_matrix_load_coalescing_and_add_autotune
Iter 2     39940 ns  (1.290×)  — restore_register_pressure_via_smaller_tiles_with_proper_autotune

Final best: 39940 ns  (1.2899×)
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
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_addmm_avg_pool2d_convolution_relu_t_view_22(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -72,9 +82,7 @@

         xindex = idx_n + 1000*idx_m
         b = tl.load(B + ((tl.broadcast_to(idx_m + 1024*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
 
-
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +96,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
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

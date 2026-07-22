# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_19` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 15140 ns — **Best latency**: 9470 ns (**1.5987×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_block_n_to_halve_smem_and_fix_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_19`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_19_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (27,1,1), Waves/SM: 0.05
- **Registers/thread**: 64
- **Duration**: 15140 ns
- **SM Throughput**: 1.55 % | **Memory**: 15.93 % | **DRAM**: 1.03 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.35 inst/cycle | **No Eligible**: 91.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_tiny_gemm | 0.639× | 0.578× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 15140 ns | 26180 ns | +11040 ns (+72.9%) |
| SM Throughput (%) | 1.55 % | 2.86 % | +1.31 pp |
| Memory Throughput (%) | 15.93 % | 10.06 % | -5.87 pp |
| DRAM Throughput (%) | 1.03 % | 0.63 % | -0.40 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp |
| Registers/Thread | 64.00 register/thread | 168.00 register/thread | +104 |
| Waves/SM | 0.05 | 0.03 | -0.02 |
| No Eligible (%) | 91.27 % | 69.45 % | -21.82 pp |
| IPC Active | 0.35 inst/cycle | 0.61 inst/cycle | +0.26 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reshape_tile_sizes_with_dynamic_grid | 1.954× | 1.924× | ✓ | **✓ best** |
| B | persistent_kernel_sm_tiling | 1.377× | 1.452× | ✓ |  |
| D | autotune_tile_sizes_with_dynamic_grid_and_block_m_matching_m | 1.776× | 1.470× | ✓ |  |
| E | add_triton_autotune | 0.573× | 0.510× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reshape_tile_size... | Δ | persistent_kernel... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15140 ns | 7870 ns | -7270 ns (-48.0%) | 10430 ns | -4710 ns (-31.1%) | 10300 ns | -4840 ns (-32.0%) | 29700 ns | +14560 ns (+96.2%) |
| SM Throughput (%) | 1.55 % | 0.75 % | -0.80 pp | 0.49 % | -1.06 pp | 0.90 % | -0.65 pp | 2.65 % | +1.10 pp |
| Memory Throughput (%) | 15.93 % | 7.82 % | -8.11 pp | 3.51 % | -12.42 pp | 5.61 % | -10.32 pp | 16.43 % | +0.50 pp |
| DRAM Throughput (%) | 1.03 % | 1.97 % | +0.94 pp | 1.49 % | +0.46 pp | 1.50 % | +0.47 pp | 0.54 % | -0.49 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 6.24 % | +0.00 pp | 3.12 % | -3.12 pp | 6.25 % | +0.01 pp |
| Registers/Thread | 64.00 register/thread | 60.00 register/thread | -4 | 72.00 register/thread | +8 | 64.00 register/thread | +0 | 96.00 register/thread | +32 |
| Waves/SM | 0.05 | 0.01 | -0.04 | 0.07 | +0.02 | 0.01 | -0.04 | 0.04 | -0.01 |
| No Eligible (%) | 91.27 % | 90.26 % | -1.01 pp | 89.35 % | -1.92 pp | 89.05 % | -2.22 pp | 86.23 % | -5.04 pp |
| IPC Active | 0.35 inst/cycle | 0.20 inst/cycle | -0.15 | 0.41 inst/cycle | +0.06 | 0.22 inst/cycle | -0.13 | 0.55 inst/cycle | +0.20 |

---

## Iteration 3

**Best so far**: 1.9238× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_shared_memory_bank_conflicts_via_block_k_reduction | 1.806× | 1.502× | ✓ |  |
| B | reduce_block_n_to_halve_smem_and_fix_bank_conflicts | 1.966× | 1.599× | ✓ | **✓ best** |
| C | eliminate_shared_memory_bank_conflicts_via_smem_padding | 1.430× | 1.572× | ✓ |  |
| D | reduce_shared_memory_bank_conflicts_via_block_k_and_block_m_reshape | 1.936× | 1.609× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ | reduce_block_n_to... | Δ | eliminate_shared_... | Δ | reduce_shared_mem... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15140 ns | 10080 ns | -5060 ns (-33.4%) | 9470 ns | -5670 ns (-37.5%) | 9630 ns | -5510 ns (-36.4%) | 9410 ns | -5730 ns (-37.8%) |
| SM Throughput (%) | 1.55 % | 0.90 % | -0.65 pp | 0.46 % | -1.09 pp | 0.51 % | -1.04 pp | 0.56 % | -0.99 pp |
| Memory Throughput (%) | 15.93 % | 5.64 % | -10.29 pp | 3.12 % | -12.81 pp | 3.19 % | -12.74 pp | 5.82 % | -10.11 pp |
| DRAM Throughput (%) | 1.03 % | 1.54 % | +0.51 pp | 1.63 % | +0.60 pp | 1.61 % | +0.58 pp | 1.64 % | +0.61 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 1.56 % | -4.68 pp | 6.24 % | +0.00 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 68.00 register/thread | +4 | 44.00 register/thread | -20 | 56.00 register/thread | -8 |
| Waves/SM | 0.05 | 0.01 | -0.04 | 0.01 | -0.04 | 0.03 | -0.02 | 0.01 | -0.04 |
| No Eligible (%) | 91.27 % | 89.40 % | -1.87 pp | 88.49 % | -2.78 pp | 89.13 % | -2.14 pp | 92.93 % | +1.66 pp |
| IPC Active | 0.35 inst/cycle | 0.21 inst/cycle | -0.14 | 0.11 inst/cycle | -0.24 | 0.43 inst/cycle | +0.08 | 0.14 inst/cycle | -0.21 |

---

## Iteration 4

**Best so far**: 1.9238× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.964× | 1.582× | ✓ |  |
| B | tf32_tensor_cores | 1.932× | 1.572× | ✓ |  |
| C | tf32_tensor_cores | 1.438× | 1.572× | ✓ |  |
| D | tf32_tensor_cores | 1.774× | 1.460× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tf32_tensor_cores | Δ | tf32_tensor_cores | Δ | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15140 ns | 9570 ns | -5570 ns (-36.8%) | 9630 ns | -5510 ns (-36.4%) | 9630 ns | -5510 ns (-36.4%) | 10370 ns | -4770 ns (-31.5%) |
| SM Throughput (%) | 1.55 % | 0.56 % | -0.99 pp | 0.46 % | -1.09 pp | 0.50 % | -1.05 pp | 0.90 % | -0.65 pp |
| Memory Throughput (%) | 15.93 % | 5.74 % | -10.19 pp | 3.12 % | -12.81 pp | 3.15 % | -12.78 pp | 5.60 % | -10.33 pp |
| DRAM Throughput (%) | 1.03 % | 1.62 % | +0.59 pp | 1.61 % | +0.58 pp | 1.61 % | +0.58 pp | 1.50 % | +0.47 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 1.56 % | -4.68 pp | 6.24 % | +0.00 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 64.00 register/thread | 56.00 register/thread | -8 | 68.00 register/thread | +4 | 44.00 register/thread | -20 | 64.00 register/thread | +0 |
| Waves/SM | 0.05 | 0.01 | -0.04 | 0.01 | -0.04 | 0.03 | -0.02 | 0.01 | -0.04 |
| No Eligible (%) | 91.27 % | 93.21 % | +1.94 pp | 88.89 % | -2.38 pp | 89.13 % | -2.14 pp | 89.05 % | -2.22 pp |
| IPC Active | 0.35 inst/cycle | 0.14 inst/cycle | -0.21 | 0.11 inst/cycle | -0.24 | 0.44 inst/cycle | +0.09 | 0.22 inst/cycle | -0.13 |


### Key Learnings

#### ❌ Plan A: `tf32_tensor_cores` (1.582×)

Enabling TF32 (ALLOW_TF32=True) in tl.dot for a severely launch-bound GEMM (27 blocks on 132 SMs, 0.01 waves/SM, 3.12% achieved occupancy) produced no measurable improvement: all key metrics—duration (9.41→9.57 µs), compute SM throughput (0.56%→0.56%), executed instructions (41094→41094), and achieved occupancy (3.12%→3.12%)—remained identical before and after. The profiling data shows the speedup of 1.582x was not caused by this kernel change; the NCU-measured kernel duration actually regressed slightly (9.41→9.57 µs). When a kernel is in the extreme latency-dominated regime (<<1 wave/SM), Tensor Core throughput improvements are irrelevant because the bottleneck is grid underutilization and scheduling overhead, not arithmetic throughput. TF32 enabling only helps when compute throughput (SM SOL%) is the actual bottleneck, not when the kernel is idle >96% of the time due to insufficient parallelism.

> **Applicable when**: Never applicable as a standalone optimization when the kernel has <0.1 waves/SM and Compute (SM) Throughput < 5%. TF32 enabling is only beneficial when compute throughput is the measured bottleneck (Compute SOL% > 50%) and the grid is large enough to keep SMs busy.

> **Avoid when**: Kernel has grid_size << sm_count (waves/SM < 0.1), achieved occupancy < 10%, and Compute (SM) Throughput < 5%. In this regime, the LaunchConfiguration and SOLBottleneck NCU rules dominate; switching from FP32 FFMA to TF32 Tensor Cores cannot help because the hardware is idle, not compute-saturated. Also avoid when BLOCK_M is very small (e.g., 2) since the tl.dot tile is too small to efficiently utilize Tensor Core pipelines regardless of TF32 setting.

#### ❌ Plan B: `tf32_tensor_cores` (1.582×)

Enabling TF32 (ALLOW_TF32=True) in tl.dot for a severely launch-bound GEMM (27 blocks on 132 SMs, 0.01 waves/SM, 3.12% achieved occupancy) produced no measurable improvement: all key metrics—duration (9.41→9.57 µs), compute SM throughput (0.56%→0.56%), executed instructions (41094→41094), and achieved occupancy (3.12%→3.12%)—remained identical before and after. The profiling data shows the speedup of 1.582x was not caused by this kernel change; the NCU-measured kernel duration actually regressed slightly (9.41→9.57 µs). When a kernel is in the extreme latency-dominated regime (<<1 wave/SM), Tensor Core throughput improvements are irrelevant because the bottleneck is grid underutilization and scheduling overhead, not arithmetic throughput. TF32 enabling only helps when compute throughput (SM SOL%) is the actual bottleneck, not when the kernel is idle >96% of the time due to insufficient parallelism.

> **Applicable when**: Never applicable as a standalone optimization when the kernel has <0.1 waves/SM and Compute (SM) Throughput < 5%. TF32 enabling is only beneficial when compute throughput is the measured bottleneck (Compute SOL% > 50%) and the grid is large enough to keep SMs busy.

> **Avoid when**: Kernel has grid_size << sm_count (waves/SM < 0.1), achieved occupancy < 10%, and Compute (SM) Throughput < 5%. In this regime, the LaunchConfiguration and SOLBottleneck NCU rules dominate; switching from FP32 FFMA to TF32 Tensor Cores cannot help because the hardware is idle, not compute-saturated. Also avoid when BLOCK_M is very small (e.g., 2) since the tl.dot tile is too small to efficiently utilize Tensor Core pipelines regardless of TF32 setting.

#### ❌ Plan C: `tf32_tensor_cores` (1.582×)

Enabling TF32 (ALLOW_TF32=True) in tl.dot for a severely launch-bound GEMM (27 blocks on 132 SMs, 0.01 waves/SM, 3.12% achieved occupancy) produced no measurable improvement: all key metrics—duration (9.41→9.57 µs), compute SM throughput (0.56%→0.56%), executed instructions (41094→41094), and achieved occupancy (3.12%→3.12%)—remained identical before and after. The profiling data shows the speedup of 1.582x was not caused by this kernel change; the NCU-measured kernel duration actually regressed slightly (9.41→9.57 µs). When a kernel is in the extreme latency-dominated regime (<<1 wave/SM), Tensor Core throughput improvements are irrelevant because the bottleneck is grid underutilization and scheduling overhead, not arithmetic throughput. TF32 enabling only helps when compute throughput (SM SOL%) is the actual bottleneck, not when the kernel is idle >96% of the time due to insufficient parallelism.

> **Applicable when**: Never applicable as a standalone optimization when the kernel has <0.1 waves/SM and Compute (SM) Throughput < 5%. TF32 enabling is only beneficial when compute throughput is the measured bottleneck (Compute SOL% > 50%) and the grid is large enough to keep SMs busy.

> **Avoid when**: Kernel has grid_size << sm_count (waves/SM < 0.1), achieved occupancy < 10%, and Compute (SM) Throughput < 5%. In this regime, the LaunchConfiguration and SOLBottleneck NCU rules dominate; switching from FP32 FFMA to TF32 Tensor Cores cannot help because the hardware is idle, not compute-saturated. Also avoid when BLOCK_M is very small (e.g., 2) since the tl.dot tile is too small to efficiently utilize Tensor Core pipelines regardless of TF32 setting.

#### ❌ Plan D: `tf32_tensor_cores` (1.582×)

Enabling TF32 (ALLOW_TF32=True) in tl.dot for a severely launch-bound GEMM (27 blocks on 132 SMs, 0.01 waves/SM, 3.12% achieved occupancy) produced no measurable improvement: all key metrics—duration (9.41→9.57 µs), compute SM throughput (0.56%→0.56%), executed instructions (41094→41094), and achieved occupancy (3.12%→3.12%)—remained identical before and after. The profiling data shows the speedup of 1.582x was not caused by this kernel change; the NCU-measured kernel duration actually regressed slightly (9.41→9.57 µs). When a kernel is in the extreme latency-dominated regime (<<1 wave/SM), Tensor Core throughput improvements are irrelevant because the bottleneck is grid underutilization and scheduling overhead, not arithmetic throughput. TF32 enabling only helps when compute throughput (SM SOL%) is the actual bottleneck, not when the kernel is idle >96% of the time due to insufficient parallelism.

> **Applicable when**: Never applicable as a standalone optimization when the kernel has <0.1 waves/SM and Compute (SM) Throughput < 5%. TF32 enabling is only beneficial when compute throughput is the measured bottleneck (Compute SOL% > 50%) and the grid is large enough to keep SMs busy.

> **Avoid when**: Kernel has grid_size << sm_count (waves/SM < 0.1), achieved occupancy < 10%, and Compute (SM) Throughput < 5%. In this regime, the LaunchConfiguration and SOLBottleneck NCU rules dominate; switching from FP32 FFMA to TF32 Tensor Cores cannot help because the hardware is idle, not compute-saturated. Also avoid when BLOCK_M is very small (e.g., 2) since the tl.dot tile is too small to efficiently utilize Tensor Core pipelines regardless of TF32 setting.

---

## Iteration 5

**Best so far**: 1.9238× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | maximize_block_n_to_minimize_tile_count_and_eliminate_launch_overhead | 0.990× | 0.497× | ✓ |  |
| C | reduce_block_k_to_halve_smem_and_improve_occupancy | 1.499× | 1.660× | ✓ |  |
| D | expand_block_n_with_persistent_sm_stride_loop | 1.450× | 1.551× | ✓ |  |


### Metric Comparison

| Metric | Baseline | maximize_block_n_... | Δ | reduce_block_k_to... | Δ | expand_block_n_wi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15140 ns | 30430 ns | +15290 ns (+101.0%) | 9120 ns | -6020 ns (-39.8%) | 9760 ns | -5380 ns (-35.5%) |
| SM Throughput (%) | 1.55 % | 0.90 % | -0.65 pp | 0.64 % | -0.91 pp | 0.50 % | -1.05 pp |
| Memory Throughput (%) | 15.93 % | 6.77 % | -9.16 pp | 1.98 % | -13.95 pp | 3.14 % | -12.79 pp |
| DRAM Throughput (%) | 1.03 % | 0.54 % | -0.49 pp | 1.69 % | +0.66 pp | 1.58 % | +0.55 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 228.00 register/thread | +164 | 44.00 register/thread | -20 | 44.00 register/thread | -20 |
| Waves/SM | 0.05 | 0.03 | -0.02 | 0.02 | -0.03 | 0.03 | -0.02 |
| No Eligible (%) | 91.27 % | 81.26 % | -10.01 pp | 85.81 % | -5.46 pp | 89.16 % | -2.11 pp |
| IPC Active | 0.35 inst/cycle | 0.19 inst/cycle | -0.16 | 0.56 inst/cycle | +0.21 | 0.42 inst/cycle | +0.07 |


### Key Learnings

#### ❌ Plan B: `maximize_block_n_to_minimize_tile_count_and_eliminate_launch_overhead` (0.497×)

Increasing BLOCK_N from 32 to 256 in a tiny GEMM (M=2, N=864, K=216) caused registers per thread to explode from 56 to 228, dynamic shared memory per block to balloon from 4.35 KB to 33 KB, and theoretical occupancy to collapse from 56.25% to 9.38%. Despite the grid remaining at 27 blocks (the BLOCK_N change did not reduce tile count as expected because the grid lambda was not correctly applied), executed instructions grew 5.3x (41K to 218K) and kernel duration increased 3.2x (9.57 µs to 30.43 µs), yielding a 0.50x regression. The root cause is that a BLOCK_N=256 tile with BLOCK_M=2 and BLOCK_K=32 requires the compiler to hold a 2×256 accumulator in registers (512 float32 values = 2048 bytes), far exceeding the register budget and causing severe register pressure that limits occupancy and increases per-instruction latency. For tiny GEMM kernels where M is very small (e.g., M=2), BLOCK_N must remain small (≤64) to keep the accumulator register footprint manageable.

> **Applicable when**: Never: this is a negative result. Do not increase BLOCK_N beyond 64 when BLOCK_M is very small (≤4) in a Triton GEMM kernel.

> **Avoid when**: Any Triton GEMM kernel where BLOCK_M ≤ 4. The accumulator register footprint scales as BLOCK_M × BLOCK_N float32 values. With BLOCK_M=2 and BLOCK_N=256, the accumulator alone requires 512 float32 registers per thread (2048 bytes), which exceeds the practical register budget (~128-255 registers per thread on H100/H200) and causes severe register spilling, occupancy collapse, and instruction count explosion. The LaunchConfiguration NCU rule suggesting 'increase grid size' should NOT be addressed by increasing BLOCK_N when BLOCK_M is tiny — the register cost is prohibitive.

#### ✅ Plan D: `expand_block_n_with_persistent_sm_stride_loop` (1.551×)

For severely launch-bound GEMM kernels (grid_size << sm_count, waves/SM < 0.05), removing @triton.autotune, expanding BLOCK_N (e.g., 32→128) to reduce total tile count, and replacing the single-tile pid dispatch with a persistent SM-stride while loop (grid=NUM_SMS, tile_id += NUM_SMS) doubles achieved occupancy (3.12%→6.24%), reduces registers per thread (56→44), cuts warp cycles per issued instruction (14.5→9.45), and improves issue slot utilization (3.47%→10.57%). The NCU-measured kernel duration may appear similar or slightly longer due to the larger shared memory footprint per block, but the end-to-end wall-time speedup is ~1.55x because the persistent loop eliminates scheduling overhead for the many idle CTAs. This strategy is most effective when LaunchConfiguration and AchievedOccupancy NCU rules dominate with high speedup_pct (>70%), and the total tile count is much smaller than the SM count.

> **Applicable when**: Triton GEMM kernel where grid_size << sm_count (waves/SM < 0.05), LaunchConfiguration NCU rule fires with >70% speedup_pct, AchievedOccupancy is <5%, and the total tile count (ceil(M/BLOCK_M)*ceil(N/BLOCK_N)) is much smaller than the SM count. Removing autotune and hardcoding a persistent SM-stride loop is safe when problem dimensions are fixed and known at kernel-write time.

> **Avoid when**: Problem dimensions vary across calls (autotune is needed to adapt). Total tile count is already >= sm_count (grid is well-utilized). BLOCK_N expansion causes shared memory per block to exceed SM limits (Block Limit Shared Mem constraint). The kernel is memory-bandwidth-bound (DRAM throughput > 50%) rather than launch-bound. ALLOW_TF32 must remain False for correctness (do not change alongside this optimization).

---

## Cumulative Progress

```
Baseline:  15140 ns  (1.000×)
Iter 2     7870 ns  (1.924×)  — reshape_tile_sizes_with_dynamic_grid

Final best: 9470 ns  (1.5987×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_19(arg_A, arg_B, out_ptr1):
     EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -91,8 +101,6 @@

     tmp0 = tl.sigmoid(acc)
     tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp0, mask)
 
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

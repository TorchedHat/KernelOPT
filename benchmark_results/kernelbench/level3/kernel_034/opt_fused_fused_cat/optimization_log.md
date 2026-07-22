# Kernel Optimization Log

## Summary

- **Kernel**: `rnn_i2h_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 13310 ns — **Best latency**: 23300 ns (**0.5712×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `rnn_i2h_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_034/synthesis/kernel_4_fused_fused_cat.py`
- **Framework**: triton
- **Launch**: Block (64,1,1), Grid (4,1,1), Waves/SM: 0.02
- **Registers/thread**: 168
- **Duration**: 13310 ns
- **SM Throughput**: 0.45 % | **Memory**: 2.11 % | **DRAM**: 2.11 %
- **Achieved Occupancy**: 3.12 %
- **IPC Active**: 0.73 inst/cycle | **No Eligible**: 63.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_stages_to_lower_register_pressure_and_smem_usage | 0.891× | 0.202× | ✓ |  |
| D | eliminate_hidden_clone_pingpong_buffer_plus_low_register_autotune | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_stages... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 13310 ns | 65860 ns | +52550 ns (+394.8%) |
| SM Throughput (%) | 0.45 % | 0.10 % | -0.35 pp |
| Memory Throughput (%) | 2.11 % | 0.42 % | -1.69 pp |
| DRAM Throughput (%) | 2.11 % | 0.42 % | -1.69 pp |
| Achieved Occupancy (%) | 3.12 % | 3.12 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 96.00 register/thread | -72 |
| Waves/SM | 0.02 | 0.00 | -0.02 |
| No Eligible (%) | 63.07 % | 92.58 % | +29.51 pp |
| IPC Active | 0.73 inst/cycle | 0.15 inst/cycle | -0.58 |


### Key Learnings

#### ❌ Plan A: `reduce_num_stages_to_lower_register_pressure_and_smem_usage` (0.204×)

Replacing num_stages=3-4 autotune configs with num_stages=1-2 variants caused the autotuner to select a low-stage config, dropping dynamic shared memory from 43 KB to 18 KB and eliminating async prefetch overlap. Warp Cycles Per Issued Instruction jumped from 4.47 to 13.69 (3x worse), the no-eligible-warp rate rose from 77.66% to 92.68%, and the CPIStall rule shifted from fixed-latency dependency to L1TEX scoreboard dependency (69.4% of stall cycles), indicating global memory loads are fully exposed without pipelining. Duration inflated from 23.26 µs to 65.22 µs (~2.8x regression). The grid remained at 4 blocks (unchanged), so the intended batching of h2o timesteps was never implemented — only the autotune config list was modified. This is the fourth consecutive iteration where adding num_stages=1 configs to the autotune list caused the same ~3x regression on this tiny-grid kernel.

> **Applicable when**: Never applicable when grid_size << sm_count (e.g., 4 blocks vs 132 SMs) and the kernel is already latency-bound with Warp Cycles Per Issued Instruction > 4. Reducing num_stages only helps when Block Limit Shared Mem is the binding occupancy constraint AND there are enough blocks to fill the additional SM slots unlocked.

> **Avoid when**: Do not add num_stages=1 or num_stages=2 configs to autotune search spaces for any kernel where: (1) grid_size is far below sm_count (e.g., 4 blocks vs 132 SMs); (2) the kernel is already latency-bound (Warp Cycles Per Issued Instruction > 4, No Eligible > 70%); (3) the intended optimization (batched launcher) was not actually implemented in the diff. This regression has been confirmed four times independently on this kernel class across iterations 2, 4, 4, and 5.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | batch_h2o_over_all_timesteps_and_eliminate_hidden_clone | 1.004× | 0.204× | ✓ | **✓ best** |
| B | restore_pipelined_num_stages_plus_eliminate_clone_pingpong | 0.947× | 0.568× | ✓ |  |


### Metric Comparison

| Metric | Baseline | batch_h2o_over_al... | Δ | restore_pipelined... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 13310 ns | 65310 ns | +52000 ns (+390.7%) | 23420 ns | +10110 ns (+76.0%) |
| SM Throughput (%) | 0.45 % | 0.10 % | -0.35 pp | 0.29 % | -0.16 pp |
| Memory Throughput (%) | 2.11 % | 0.43 % | -1.68 pp | 1.19 % | -0.92 pp |
| DRAM Throughput (%) | 2.11 % | 0.43 % | -1.68 pp | 1.19 % | -0.92 pp |
| Achieved Occupancy (%) | 3.12 % | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 96.00 register/thread | -72 | 95.00 register/thread | -73 |
| Waves/SM | 0.02 | 0.00 | -0.02 | 0.01 | -0.01 |
| No Eligible (%) | 63.07 % | 92.52 % | +29.45 pp | 77.69 % | +14.62 pp |
| IPC Active | 0.73 inst/cycle | 0.15 inst/cycle | -0.58 | 0.44 inst/cycle | -0.29 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.014× | 0.572× | ✓ |  |
| B | batch_all_h2o_timesteps_into_single_kernel_launch | 0.984× | 0.571× | ✓ |  |
| C | tf32_tensor_cores | 0.967× | 0.575× | ✓ |  |
| D | batch_h2o_single_launch_over_all_timesteps_with_flat_grid | 1.039× | 0.205× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | batch_all_h2o_tim... | Δ | tf32_tensor_cores | Δ | batch_h2o_single_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13310 ns | 23260 ns | +9950 ns (+74.8%) | 23300 ns | +9990 ns (+75.1%) | 23140 ns | +9830 ns (+73.9%) | 64900 ns | +51590 ns (+387.6%) |
| SM Throughput (%) | 0.45 % | 0.29 % | -0.16 pp | 0.29 % | -0.16 pp | 0.29 % | -0.16 pp | 0.10 % | -0.35 pp |
| Memory Throughput (%) | 2.11 % | 1.20 % | -0.91 pp | 1.20 % | -0.91 pp | 1.21 % | -0.90 pp | 0.43 % | -1.68 pp |
| DRAM Throughput (%) | 2.11 % | 1.20 % | -0.91 pp | 1.20 % | -0.91 pp | 1.21 % | -0.90 pp | 0.43 % | -1.68 pp |
| Achieved Occupancy (%) | 3.12 % | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 95.00 register/thread | -73 | 95.00 register/thread | -73 | 95.00 register/thread | -73 | 96.00 register/thread | -72 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 | 0.00 | -0.02 |
| No Eligible (%) | 63.07 % | 77.67 % | +14.60 pp | 77.63 % | +14.56 pp | 77.67 % | +14.60 pp | 92.58 % | +29.51 pp |
| IPC Active | 0.73 inst/cycle | 0.45 inst/cycle | -0.28 | 0.44 inst/cycle | -0.29 | 0.44 inst/cycle | -0.29 | 0.15 inst/cycle | -0.58 |


### Key Learnings

#### ❌ Plan D: `batch_h2o_single_launch_over_all_timesteps_with_flat_grid` (0.206×)

The plan intended to batch all h2o timestep launches into a single flat-grid kernel call, but the actual diff only replaced autotune configs — introducing num_stages=1 and num_stages=2 entries for both rnn_i2h_kernel and rnn_h2o_kernel — without modifying the Python launcher loop. The grid remained at 4 blocks (unchanged from before), so no parallelism benefit was realized. The num_stages reduction dropped dynamic shared memory from 43 KB to 18 KB, eliminating async prefetch overlap: Warp Cycles Per Issued Instruction jumped from 4.48 to 13.47 (3x worse), the no-eligible-warp rate rose from 77.67% to 92.44%, and the CPIStall rule shifted from fixed-latency dependency (47% of stall cycles) to L1TEX scoreboard dependency (71% of stall cycles), indicating global memory loads are now fully exposed without pipelining. Duration inflated from 23.14 µs to 64.58 µs (~2.8x regression). This is the third consecutive iteration where adding num_stages=1 configs to the autotune list caused the autotuner to select the low-stage config and destroy latency hiding on this tiny-grid kernel.

> **Applicable when**: Never: this is a confirmed anti-pattern. Adding num_stages=1 or num_stages=2 configs to autotune lists for tiny-grid kernels (grid_size << sm_count) causes the autotuner to select the low-stage config, destroying async prefetch latency hiding. The grid remains unchanged so no occupancy benefit compensates for the lost pipelining.

> **Avoid when**: Do not add num_stages=1 or num_stages=2 configs to autotune search spaces for any kernel where: (1) grid_size << sm_count (e.g., 4 blocks vs 132 SMs); (2) the kernel is already latency-bound (Warp Cycles Per Issued Instruction > 4, No Eligible > 70%); (3) the intended optimization (batched launcher) was not actually implemented in the diff. This is the third consecutive iteration (items exp_batched_h2o_flat_grid_003, exp_batch_h2o_timesteps_empty_diff_003, and now iteration 4) where this exact anti-pattern caused a 2-3x regression. If the plan requires a Python launcher change, verify the diff contains the launcher change before submitting.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_clone_pingpong_and_batch_h2o_single_launch | 1.028× | 0.204× | ✓ |  |
| B | double_buffer_hidden_state_eliminate_clone_per_timestep | 1.045× | 0.571× | ✓ | **✓ best** |
| C | fuse_h2o_batch_all_timesteps_single_launch_with_clone_elimination | 1.029× | 0.572× | ✓ |  |
| D | fuse_h2o_batched_launch_with_hidden_buffer | 1.010× | 0.206× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_clone_p... | Δ | double_buffer_hid... | Δ | fuse_h2o_batch_al... | Δ | fuse_h2o_batched_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13310 ns | 65180 ns | +51870 ns (+389.7%) | 23300 ns | +9990 ns (+75.1%) | 23260 ns | +9950 ns (+74.8%) | 64580 ns | +51270 ns (+385.2%) |
| SM Throughput (%) | 0.45 % | 0.10 % | -0.35 pp | 0.29 % | -0.16 pp | 0.29 % | -0.16 pp | 0.10 % | -0.35 pp |
| Memory Throughput (%) | 2.11 % | 0.43 % | -1.68 pp | 1.20 % | -0.91 pp | 1.20 % | -0.91 pp | 0.43 % | -1.68 pp |
| DRAM Throughput (%) | 2.11 % | 0.43 % | -1.68 pp | 1.20 % | -0.91 pp | 1.20 % | -0.91 pp | 0.43 % | -1.68 pp |
| Achieved Occupancy (%) | 3.12 % | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 96.00 register/thread | -72 | 95.00 register/thread | -73 | 95.00 register/thread | -73 | 96.00 register/thread | -72 |
| Waves/SM | 0.02 | 0.00 | -0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.00 | -0.02 |
| No Eligible (%) | 63.07 % | 92.58 % | +29.51 pp | 77.64 % | +14.57 pp | 77.66 % | +14.59 pp | 92.44 % | +29.37 pp |
| IPC Active | 0.73 inst/cycle | 0.15 inst/cycle | -0.58 | 0.45 inst/cycle | -0.28 | 0.44 inst/cycle | -0.29 | 0.15 inst/cycle | -0.58 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | batch_h2o_single_launch_eliminate_clone_structural_wrapper_change | 0.945× | 0.574× | ✓ |  |
| B | restore_high_num_stages_and_eliminate_clone_with_pingpong_buffers | 0.992× | 0.573× | ✓ |  |
| C | fuse_i2h_and_h2o_into_single_rnn_step_kernel | 1.002× | 0.570× | ✓ |  |
| D | restructure_launcher_batch_h2o_single_launch_with_all_hidden_buffer | 1.020× | 0.204× | ✓ |  |
| E | add_triton_autotune | 0.976× | 0.570× | ✓ |  |


### Metric Comparison

| Metric | Baseline | batch_h2o_single_... | Δ | restore_high_num_... | Δ | fuse_i2h_and_h2o_... | Δ | restructure_launc... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13310 ns | 23200 ns | +9890 ns (+74.3%) | 23230 ns | +9920 ns (+74.5%) | 23360 ns | +10050 ns (+75.5%) | 65220 ns | +51910 ns (+390.0%) | 23360 ns | +10050 ns (+75.5%) |
| SM Throughput (%) | 0.45 % | 0.29 % | -0.16 pp | 0.29 % | -0.16 pp | 0.29 % | -0.16 pp | 0.10 % | -0.35 pp | 0.29 % | -0.16 pp |
| Memory Throughput (%) | 2.11 % | 1.20 % | -0.91 pp | 1.20 % | -0.91 pp | 1.20 % | -0.91 pp | 0.43 % | -1.68 pp | 1.20 % | -0.91 pp |
| DRAM Throughput (%) | 2.11 % | 1.20 % | -0.91 pp | 1.20 % | -0.91 pp | 1.20 % | -0.91 pp | 0.43 % | -1.68 pp | 1.20 % | -0.91 pp |
| Achieved Occupancy (%) | 3.12 % | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 95.00 register/thread | -73 | 95.00 register/thread | -73 | 95.00 register/thread | -73 | 96.00 register/thread | -72 | 95.00 register/thread | -73 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 | 0.00 | -0.02 | 0.01 | -0.01 |
| No Eligible (%) | 63.07 % | 77.60 % | +14.53 pp | 77.56 % | +14.49 pp | 77.90 % | +14.83 pp | 92.68 % | +29.61 pp | 77.90 % | +14.83 pp |
| IPC Active | 0.73 inst/cycle | 0.45 inst/cycle | -0.28 | 0.45 inst/cycle | -0.28 | 0.44 inst/cycle | -0.29 | 0.15 inst/cycle | -0.58 | 0.45 inst/cycle | -0.28 |

---

## Cumulative Progress

```
Baseline:  13310 ns  (1.000×)

Final best: 23300 ns  (0.5712×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -8,12 +8,15 @@

 # b_i2h: [hidden] = [256]
 # output hidden: [batch, hidden] = [8, 256]
 
+
 @triton.autotune(
     configs=[
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=2, num_stages=3),
         triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=2, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=2, num_stages=2),
     ],
     key=['HIDDEN_SIZE', 'COMBINED_SIZE'],
 )
@@ -104,14 +107,17 @@

 
 @triton.autotune(
     configs=[
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=2, num_stages=3),
         triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=2, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=4),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=2, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 64},  num_warps=2, num_stages=3),
     ],
-    key=['HIDDEN_SIZE', 'OUTPUT_SIZE'],
+    key=['HIDDEN_SIZE', 'OUTPUT_SIZE', 'BATCH'],
 )
 @triton.jit
+
 def rnn_h2o_kernel(
     hidden_ptr,
     w_h2o_ptr, b_h2o_ptr,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`

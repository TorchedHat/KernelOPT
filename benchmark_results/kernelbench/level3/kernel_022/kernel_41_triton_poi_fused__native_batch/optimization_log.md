# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3100 ns — **Best latency**: 3100 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_41_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (735,1,1), Waves/SM: 0.35
- **Registers/thread**: 22
- **Duration**: 3100 ns
- **SM Throughput**: 10.18 % | **Memory**: 5.02 % | **DRAM**: 5.02 %
- **Achieved Occupancy**: 30.07 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.38 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_grid_and_improve_occupancy | 0.952× | 0.896× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_occupancy | 0.994× | 0.951× | ✓ |  |
| C | increase_xblock_tile_size_with_autotune | 0.953× | 1.000× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 0.998× | 0.896× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 3460 ns | +360 ns (+11.6%) | 3260 ns | +160 ns (+5.2%) | 3100 ns | +0 ns (+0.0%) | 3460 ns | +360 ns (+11.6%) |
| SM Throughput (%) | 10.18 % | 16.56 % | +6.38 pp | 24.06 % | +13.88 pp | 13.25 % | +3.07 pp | 16.44 % | +6.26 pp |
| Memory Throughput (%) | 5.02 % | 5.00 % | -0.02 pp | 8.32 % | +3.30 pp | 5.13 % | +0.11 pp | 4.96 % | -0.06 pp |
| DRAM Throughput (%) | 5.02 % | 4.52 % | -0.50 pp | 4.79 % | -0.23 pp | 5.06 % | +0.04 pp | 4.51 % | -0.51 pp |
| Achieved Occupancy (%) | 30.07 % | 21.43 % | -8.64 pp | 52.21 % | +22.14 pp | 24.04 % | -6.03 pp | 21.97 % | -8.10 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 28.00 register/thread | +6 | 28.00 register/thread | +6 | 32.00 register/thread | +10 |
| Waves/SM | 0.35 | 0.35 | +0.00 | 0.70 | +0.35 | 0.35 | +0.00 | 0.35 | +0.00 |
| No Eligible (%) | 71.38 % | 60.07 % | -11.31 pp | 44.66 % | -26.72 pp | 63.49 % | -7.89 pp | 67.93 % | -3.45 pp |
| IPC Active | 1.12 inst/cycle | 1.54 inst/cycle | +0.42 | 2.34 inst/cycle | +1.22 | 1.38 inst/cycle | +0.26 | 1.34 inst/cycle | +0.22 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.981× | 0.969× | ✓ |  |
| B | persistent_tile_scheduling | 0.951× | 0.923× | ✓ |  |
| C | persistent_tile_scheduling | 0.951× | 0.728× | ✓ |  |
| D | persistent_tile_scheduling | 0.922× | 0.751× | ✓ |  |
| E | add_triton_autotune | 0.990× | 0.960× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 3200 ns | +100 ns (+3.2%) | 3360 ns | +260 ns (+8.4%) | 4260 ns | +1160 ns (+37.4%) | 4130 ns | +1030 ns (+33.2%) | 3230 ns | +130 ns (+4.2%) |
| SM Throughput (%) | 10.18 % | 8.02 % | -2.16 pp | 11.88 % | +1.70 pp | 5.76 % | -4.42 pp | 5.77 % | -4.41 pp | 10.18 % | +0.00 pp |
| Memory Throughput (%) | 5.02 % | 4.90 % | -0.12 pp | 4.65 % | -0.37 pp | 3.67 % | -1.35 pp | 3.77 % | -1.25 pp | 4.82 % | -0.20 pp |
| DRAM Throughput (%) | 5.02 % | 4.90 % | -0.12 pp | 4.65 % | -0.37 pp | 3.67 % | -1.35 pp | 3.77 % | -1.25 pp | 4.82 % | -0.20 pp |
| Achieved Occupancy (%) | 30.07 % | 18.51 % | -11.56 pp | 30.90 % | +0.83 pp | 11.60 % | -18.47 pp | 12.87 % | -17.20 pp | 29.95 % | -0.12 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 40.00 register/thread | +18 | 38.00 register/thread | +16 | 22.00 register/thread | +0 |
| Waves/SM | 0.35 | 0.35 | +0.00 | 0.35 | +0.00 | 0.46 | +0.11 | 0.46 | +0.11 | 0.35 | +0.00 |
| No Eligible (%) | 71.38 % | 79.50 % | +8.12 pp | 73.15 % | +1.77 pp | 88.20 % | +16.82 pp | 88.41 % | +17.03 pp | 70.66 % | -0.72 pp |
| IPC Active | 1.12 inst/cycle | 0.78 inst/cycle | -0.34 | 0.98 inst/cycle | -0.14 | 0.44 inst/cycle | -0.68 | 0.39 inst/cycle | -0.73 | 1.13 inst/cycle | +0.01 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.751×)

Adding a persistent while-loop inside a Triton kernel to reduce grid size increased register usage from 28 to 38 registers/thread, which dropped theoretical occupancy from 100% to 75% (Block Limit Registers fell from 16 to 12 blocks/SM). The reduced occupancy caused achieved occupancy to fall from 24% to 13%, warp cycles per issued instruction to nearly double (10.89→19.96), and eligible warps per scheduler to collapse from 0.66 to 0.13, resulting in a 33% slowdown despite the grid being slightly better utilized. The persistent loop pattern is counterproductive when the kernel is already register-heavy: the loop control variables and tile_id bookkeeping consume additional registers that spill the kernel past a register-limit occupancy boundary.

> **Applicable when**: Kernel has very low register usage (≤24 registers/thread) with significant headroom before hitting a register-limit occupancy boundary; grid is severely under-saturating SMs (waves_per_sm < 0.5) and the bottleneck is confirmed to be occupancy rather than memory bandwidth.

> **Avoid when**: Kernel already uses ≥28 registers/thread and is near a register-limit occupancy boundary (Block Limit Registers close to Block Limit Warps); adding loop control variables will push register count past the next boundary, reducing theoretical occupancy and worsening achieved occupancy and warp scheduling. Also avoid when the kernel is already memory-bandwidth bound rather than latency-bound.

#### ❌ Plan B: `persistent_tile_scheduling` (0.751×)

Adding a persistent while-loop inside a Triton kernel to reduce grid size increased register usage from 28 to 38 registers/thread, which dropped theoretical occupancy from 100% to 75% (Block Limit Registers fell from 16 to 12 blocks/SM). The reduced occupancy caused achieved occupancy to fall from 24% to 13%, warp cycles per issued instruction to nearly double (10.89→19.96), and eligible warps per scheduler to collapse from 0.66 to 0.13, resulting in a 33% slowdown despite the grid being slightly better utilized. The persistent loop pattern is counterproductive when the kernel is already register-heavy: the loop control variables and tile_id bookkeeping consume additional registers that spill the kernel past a register-limit occupancy boundary.

> **Applicable when**: Kernel has very low register usage (≤24 registers/thread) with significant headroom before hitting a register-limit occupancy boundary; grid is severely under-saturating SMs (waves_per_sm < 0.5) and the bottleneck is confirmed to be occupancy rather than memory bandwidth.

> **Avoid when**: Kernel already uses ≥28 registers/thread and is near a register-limit occupancy boundary (Block Limit Registers close to Block Limit Warps); adding loop control variables will push register count past the next boundary, reducing theoretical occupancy and worsening achieved occupancy and warp scheduling. Also avoid when the kernel is already memory-bandwidth bound rather than latency-bound.

#### ❌ Plan C: `persistent_tile_scheduling` (0.751×)

Adding a persistent while-loop inside a Triton kernel to reduce grid size increased register usage from 28 to 38 registers/thread, which dropped theoretical occupancy from 100% to 75% (Block Limit Registers fell from 16 to 12 blocks/SM). The reduced occupancy caused achieved occupancy to fall from 24% to 13%, warp cycles per issued instruction to nearly double (10.89→19.96), and eligible warps per scheduler to collapse from 0.66 to 0.13, resulting in a 33% slowdown despite the grid being slightly better utilized. The persistent loop pattern is counterproductive when the kernel is already register-heavy: the loop control variables and tile_id bookkeeping consume additional registers that spill the kernel past a register-limit occupancy boundary.

> **Applicable when**: Kernel has very low register usage (≤24 registers/thread) with significant headroom before hitting a register-limit occupancy boundary; grid is severely under-saturating SMs (waves_per_sm < 0.5) and the bottleneck is confirmed to be occupancy rather than memory bandwidth.

> **Avoid when**: Kernel already uses ≥28 registers/thread and is near a register-limit occupancy boundary (Block Limit Registers close to Block Limit Warps); adding loop control variables will push register count past the next boundary, reducing theoretical occupancy and worsening achieved occupancy and warp scheduling. Also avoid when the kernel is already memory-bandwidth bound rather than latency-bound.

#### ❌ Plan D: `persistent_tile_scheduling` (0.751×)

Adding a persistent while-loop inside a Triton kernel to reduce grid size increased register usage from 28 to 38 registers/thread, which dropped theoretical occupancy from 100% to 75% (Block Limit Registers fell from 16 to 12 blocks/SM). The reduced occupancy caused achieved occupancy to fall from 24% to 13%, warp cycles per issued instruction to nearly double (10.89→19.96), and eligible warps per scheduler to collapse from 0.66 to 0.13, resulting in a 33% slowdown despite the grid being slightly better utilized. The persistent loop pattern is counterproductive when the kernel is already register-heavy: the loop control variables and tile_id bookkeeping consume additional registers that spill the kernel past a register-limit occupancy boundary.

> **Applicable when**: Kernel has very low register usage (≤24 registers/thread) with significant headroom before hitting a register-limit occupancy boundary; grid is severely under-saturating SMs (waves_per_sm < 0.5) and the bottleneck is confirmed to be occupancy rather than memory bandwidth.

> **Avoid when**: Kernel already uses ≥28 registers/thread and is near a register-limit occupancy boundary (Block Limit Registers close to Block Limit Warps); adding loop control variables will push register count past the next boundary, reducing theoretical occupancy and worsening achieved occupancy and warp scheduling. Also avoid when the kernel is already memory-bandwidth bound rather than latency-bound.

---

## Cumulative Progress

```
Baseline:  3100 ns  (1.000×)

Final best: 3100 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=3),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 94080
@@ -33,8 +46,6 @@

     tmp17 = tmp15 + tmp16
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -44,7 +55,7 @@

     arg_5 = inputs[5]
     xnumel = 94080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 94080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 94080)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

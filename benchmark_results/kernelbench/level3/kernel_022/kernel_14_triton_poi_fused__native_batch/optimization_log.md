# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6300 ns — **Best latency**: 6300 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_14_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 22
- **Duration**: 6300 ns
- **SM Throughput**: 40.35 % | **Memory**: 19.58 % | **DRAM**: 19.58 %
- **Achieved Occupancy**: 61.41 %
- **IPC Active**: 2.39 inst/cycle | **No Eligible**: 38.87 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_eliminate_partial_wave | 0.859× | 0.775× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.857× | 0.275× | ✓ |  |
| C | autotune_xblock_and_num_warps_to_eliminate_partial_wave | 0.976× | 0.376× | ✓ |  |
| D | increase_xblock_with_autotune_to_hide_l1tex_latency_and_eliminate_partial_wave | 0.576× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_e... | Δ | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6300 ns | 8130 ns | +1830 ns (+29.0%) | 22880 ns | +16580 ns (+263.2%) | 16740 ns | +10440 ns (+165.7%) | 6370 ns | +70 ns (+1.1%) |
| SM Throughput (%) | 40.35 % | 60.60 % | +20.25 pp | 75.08 % | +34.73 pp | 71.47 % | +31.12 pp | 40.78 % | +0.43 pp |
| Memory Throughput (%) | 19.58 % | 18.69 % | -0.89 pp | 17.48 % | -2.10 pp | 23.44 % | +3.86 pp | 22.39 % | +2.81 pp |
| DRAM Throughput (%) | 19.58 % | 15.15 % | -4.43 pp | 5.37 % | -14.21 pp | 7.35 % | -12.23 pp | 22.39 % | +2.81 pp |
| Achieved Occupancy (%) | 61.41 % | 79.68 % | +18.27 pp | 43.76 % | -17.65 pp | 80.82 % | +19.41 pp | 63.38 % | +1.97 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 58.00 register/thread | +36 | 28.00 register/thread | +6 | 22.00 register/thread | +0 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 11.14 | +8.36 | 11.14 | +8.36 | 2.78 | +0.00 |
| No Eligible (%) | 38.87 % | 19.33 % | -19.54 pp | 17.45 % | -21.42 pp | 18.86 % | -20.01 pp | 39.56 % | +0.69 pp |
| IPC Active | 2.39 inst/cycle | 3.21 inst/cycle | +0.82 | 3.28 inst/cycle | +0.89 | 3.21 inst/cycle | +0.82 | 2.47 inst/cycle | +0.08 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_tile_size_with_autotune` (0.275×)

Adding @triton.autotune with large XBLOCK configs (512–2048) caused the compiler to allocate significantly more registers per thread (22 → 58), cutting theoretical occupancy from 100% to 50% and achieved occupancy from 61% to 44%. Although warp cycles per instruction improved (16.2 → 8.5 cycles) and compute throughput rose (40% → 75%), the 6.7x increase in total executed instructions and the 3.6x longer kernel duration (6.3 → 22.9 µs) produced a severe regression. The autotune configs selected a large tile that forced the compiler into a high-register-pressure code path, overwhelming the occupancy benefit of larger tiles.

> **Applicable when**: Never applicable as described — this is a negative result. The pattern of adding large-XBLOCK autotune configs to a low-register kernel that appears register-safe at the original tile size should be avoided when the compiler may increase register usage non-linearly with tile size.

> **Avoid when**: Do not increase XBLOCK via autotune when the kernel body contains enough live values that the compiler will allocate significantly more registers at larger tile sizes (e.g., registers/thread jumps from ~22 to ~58). This collapses theoretical occupancy (100% → 50%) and causes a net regression even if per-warp IPC improves. Specifically avoid when: (1) the kernel has many independent in-flight loads from multiple pointer streams, (2) the register budget analysis was done at the original tile size without accounting for compiler behavior at larger sizes, or (3) the grid is already producing multiple waves (waves_per_sm > 1).

#### ❌ Plan C: `autotune_xblock_and_num_warps_to_eliminate_partial_wave` (0.376×)

Adding @triton.autotune with larger XBLOCK values (512, 1024) selected a configuration that increased block size from 128 to 512 threads, which raised waves_per_sm from 2.78 to 11.14 and executed instructions from ~2M to ~9.4M — a 4.65x instruction count increase — causing duration to jump from 6.3 µs to 16.74 µs (2.66x slower). Although occupancy improved (61% → 81%) and compute throughput rose (40% → 71%), the massive instruction count increase dominated, indicating the autotune selected a config that redundantly reprocesses work or the grid was not updated to use a lambda (META-based) grid, causing the fixed grid of cdiv(xnumel, 128) blocks to each execute 512-thread tiles and over-cover the problem domain. When the grid is not updated to scale with XBLOCK, larger block sizes cause each element to be processed multiple times.

> **Applicable when**: Never applicable in this form. The pattern of using @triton.autotune with variable XBLOCK while keeping a hardcoded grid size is always wrong.

> **Avoid when**: Do not use @triton.autotune with variable XBLOCK unless the launch grid is also updated to a lambda that reads META['XBLOCK']. A fixed grid computed with the original XBLOCK will cause massive over-coverage and instruction explosion when autotune selects a larger block size.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.971× | 0.975× | ✓ |  |
| B | persistent_tile_scheduling | 0.961× | 0.946× | ✓ |  |
| C | fix_grid_lambda_and_fuse_rsqrt_fp_instructions | 0.992× | 1.010× | ✓ |  |
| D | persistent_tile_scheduling | 0.957× | 0.946× | ✓ |  |
| E | add_triton_autotune | 0.981× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fix_grid_lambda_a... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6300 ns | 6460 ns | +160 ns (+2.5%) | 6660 ns | +360 ns (+5.7%) | 6240 ns | -60 ns (-1.0%) | 6660 ns | +360 ns (+5.7%) | 6340 ns | +40 ns (+0.6%) |
| SM Throughput (%) | 40.35 % | 40.75 % | +0.40 pp | 44.16 % | +3.81 pp | 30.83 % | -9.52 pp | 43.51 % | +3.16 pp | 39.63 % | -0.72 pp |
| Memory Throughput (%) | 19.58 % | 19.00 % | -0.58 pp | 18.62 % | -0.96 pp | 19.77 % | +0.19 pp | 18.59 % | -0.99 pp | 19.50 % | -0.08 pp |
| DRAM Throughput (%) | 19.58 % | 19.00 % | -0.58 pp | 18.62 % | -0.96 pp | 19.77 % | +0.19 pp | 18.59 % | -0.99 pp | 19.50 % | -0.08 pp |
| Achieved Occupancy (%) | 61.41 % | 66.39 % | +4.98 pp | 68.75 % | +7.34 pp | 55.95 % | -5.46 pp | 70.39 % | +8.98 pp | 63.34 % | +1.93 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 21.00 register/thread | -1 | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 38.87 % | 37.82 % | -1.05 pp | 35.09 % | -3.78 pp | 58.39 % | +19.52 pp | 34.33 % | -4.54 pp | 37.58 % | -1.29 pp |
| IPC Active | 2.39 inst/cycle | 2.50 inst/cycle | +0.11 | 2.58 inst/cycle | +0.19 | 1.55 inst/cycle | -0.84 | 2.59 inst/cycle | +0.20 | 2.47 inst/cycle | +0.08 |

---

## Cumulative Progress

```
Baseline:  6300 ns  (1.000×)

Final best: 6300 ns  (1.0000×)
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
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=2),  # baseline
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 752640
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
     xnumel = 752640
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 752640, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 752640)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

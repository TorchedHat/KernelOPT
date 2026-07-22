# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 32260 ns — **Best latency**: 31680 ns (**1.0183×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_and_tune_num_warps_to_reduce_scheduler_stall`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_11_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (47040,1,1), Waves/SM: 22.27
- **Registers/thread**: 30
- **Duration**: 32260 ns
- **SM Throughput**: 73.05 % | **Memory**: 35.36 % | **DRAM**: 18.83 %
- **Achieved Occupancy**: 72.81 %
- **IPC Active**: 3.09 inst/cycle | **No Eligible**: 22.14 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.991× | 0.960× | ✓ |  |
| B | autotune_xblock_for_compute_and_l1_efficiency | 0.993× | 1.007× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads | 0.996× | 0.258× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_compute_bound_pointwise_cat_bn_relu | 0.862× | 0.760× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_f... | Δ | add_autotune_with... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32260 ns | 33600 ns | +1340 ns (+4.2%) | 32030 ns | -230 ns (-0.7%) | 125060 ns | +92800 ns (+287.7%) | 42460 ns | +10200 ns (+31.6%) |
| SM Throughput (%) | 73.05 % | 77.29 % | +4.24 pp | 71.79 % | -1.26 pp | 81.31 % | +8.26 pp | 74.41 % | +1.36 pp |
| Memory Throughput (%) | 35.36 % | 40.96 % | +5.60 pp | 34.73 % | -0.63 pp | 33.78 % | -1.58 pp | 34.73 % | -0.63 pp |
| DRAM Throughput (%) | 18.83 % | 17.95 % | -0.88 pp | 19.08 % | +0.25 pp | 4.83 % | -14.00 pp | 13.99 % | -4.84 pp |
| Achieved Occupancy (%) | 72.81 % | 79.62 % | +6.81 pp | 64.74 % | -8.07 pp | 80.64 % | +7.83 pp | 62.77 % | -10.04 pp |
| Registers/Thread | 30.00 register/thread | 32.00 register/thread | +2 | 30.00 register/thread | +0 | 32.00 register/thread | +2 | 38.00 register/thread | +8 |
| Waves/SM | 22.27 | 22.27 | +0.00 | 22.27 | +0.00 | 89.09 | +66.82 | 29.70 | +7.43 |
| No Eligible (%) | 22.14 % | 17.71 % | -4.43 pp | 22.59 % | +0.45 pp | 17.16 % | -4.98 pp | 21.80 % | -0.34 pp |
| IPC Active | 3.09 inst/cycle | 3.29 inst/cycle | +0.20 | 3.07 inst/cycle | -0.02 | 3.31 inst/cycle | +0.22 | 3.13 inst/cycle | +0.04 |


### Key Learnings

#### ❌ Plan D: `autotune_xblock_and_num_warps_for_compute_bound_pointwise_cat_bn_relu` (0.760×)

Adding a @triton.autotune decorator with larger XBLOCK configs (512–2048) caused the autotuner to select a configuration that increased registers per thread from 30 to 38, reducing theoretical occupancy from 100% to 75% and achieved occupancy from 72.8% to 62.8%. This register spill/pressure effect more than offset any benefit from reduced scheduler contention (Warp Cycles Per Issued Instruction improved from 15.78 to 12.74), because the total executed instruction count grew by ~36% (18.25M → 24.79M) and kernel duration increased from 32.26 µs to 42.46 µs. When a kernel is already register-limited at the baseline block size, increasing XBLOCK via autotune can trigger register pressure that caps occupancy and causes net regression.

> **Applicable when**: Never applicable as described — this is a negative result. Avoid autotuning XBLOCK upward when baseline registers_per_thread is already near the register-file limit (e.g., 30/thread with 128-thread blocks on H200 where Block Limit Registers = 16 blocks).

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs when: (1) registers_per_thread is already ≥28 at the baseline block size, (2) theoretical occupancy is already at 100% and Block Limit Registers is the binding constraint, or (3) the kernel body contains enough live values that scaling XBLOCK will increase register pressure and reduce theoretical occupancy below the achieved occupancy of the baseline.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | change_eviction_policy_streaming_loads_to_evict_first | 1.003× | 1.007× | ✓ | **✓ best** |
| B | fix_autotune_grid_lambda_to_enable_correct_xblock_selection | 0.998× | 0.995× | ✓ |  |
| C | cap_registers_with_maxnreg_to_restore_occupancy | 0.993× | 1.003× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_instruction_explosion | 0.995× | 0.259× | ✓ |  |


### Metric Comparison

| Metric | Baseline | change_eviction_p... | Δ | fix_autotune_grid... | Δ | cap_registers_wit... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32260 ns | 32030 ns | -230 ns (-0.7%) | 32420 ns | +160 ns (+0.5%) | 32160 ns | -100 ns (-0.3%) | 124800 ns | +92540 ns (+286.9%) |
| SM Throughput (%) | 73.05 % | 72.73 % | -0.32 pp | 72.47 % | -0.58 pp | 73.14 % | +0.09 pp | 81.39 % | +8.34 pp |
| Memory Throughput (%) | 35.36 % | 35.18 % | -0.18 pp | 35.07 % | -0.29 pp | 35.39 % | +0.03 pp | 33.81 % | -1.55 pp |
| DRAM Throughput (%) | 18.83 % | 17.33 % | -1.50 pp | 18.82 % | -0.01 pp | 18.95 % | +0.12 pp | 4.82 % | -14.01 pp |
| Achieved Occupancy (%) | 72.81 % | 77.61 % | +4.80 pp | 73.26 % | +0.45 pp | 71.72 % | -1.09 pp | 80.60 % | +7.79 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 32.00 register/thread | +2 |
| Waves/SM | 22.27 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 | 89.09 | +66.82 |
| No Eligible (%) | 22.14 % | 22.11 % | -0.03 pp | 21.61 % | -0.53 pp | 21.51 % | -0.63 pp | 17.15 % | -4.99 pp |
| IPC Active | 3.09 inst/cycle | 3.10 inst/cycle | +0.01 | 3.10 inst/cycle | +0.01 | 3.11 inst/cycle | +0.02 | 3.31 inst/cycle | +0.22 |


### Key Learnings

#### ❌ Plan D: `fix_autotune_grid_lambda_to_eliminate_instruction_explosion` (0.258×)

For the fifth consecutive iteration, adding @triton.autotune with XBLOCK configs {1024, 512, 256, 128} while leaving the grid hardcoded as `(triton.cdiv(xnumel, 128),)` produced the same catastrophic over-launch: executed instructions exploded from ~10M to ~80M (~8x), duration from 31.7 µs to 124.9 µs (~4x), and DRAM throughput collapsed from 19.1% to 4.8%. The autotuner selected XBLOCK=512, causing 512/128=4x more blocks than needed. The mandatory fix remains `grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` — no other change (eviction policy, config list, num_stages) prevents over-launch.

> **Applicable when**: Never applicable as a positive strategy — this is a negative result. Relevant for diagnosing kernels where: (1) autotune is present with variable XBLOCK, (2) executed instructions are ~N× higher than expected where N = selected_XBLOCK / hardcoded_grid_divisor, (3) DRAM throughput collapses while compute throughput rises, and (4) waves_per_sm is N× higher than baseline.

> **Avoid when**: Do not attempt to fix the instruction-explosion problem by only changing eviction policies, reordering configs, adding num_stages, or reducing the autotune config list. The grid lambda fix is mandatory and is the only change that prevents over-launch. Any autotune attempt that leaves the grid hardcoded will reproduce this failure regardless of which XBLOCK configs or num_stages values remain in the list.

---

## Iteration 3

**Best so far**: 1.0072× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_restrict_to_safe_xblock_configs | 1.007× | 1.022× | ✓ | **✓ best** |
| B | reduce_num_warps_to_cut_scheduler_not_selected_stalls | 1.000× | 1.007× | ✓ |  |
| C | fix_hardcoded_grid_to_lambda_to_unlock_existing_autotune | 0.998× | 1.004× | ✓ |  |
| D | fix_autotune_grid_lambda_to_correct_overlaunch | 0.991× | 0.259× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | reduce_num_warps_... | Δ | fix_hardcoded_gri... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32260 ns | 31550 ns | -710 ns (-2.2%) | 32030 ns | -230 ns (-0.7%) | 32130 ns | -130 ns (-0.4%) | 124700 ns | +92440 ns (+286.5%) |
| SM Throughput (%) | 73.05 % | 41.44 % | -31.61 pp | 72.69 % | -0.36 pp | 72.55 % | -0.50 pp | 81.39 % | +8.34 pp |
| Memory Throughput (%) | 35.36 % | 26.80 % | -8.56 pp | 35.18 % | -0.18 pp | 35.09 % | -0.27 pp | 33.83 % | -1.53 pp |
| DRAM Throughput (%) | 18.83 % | 17.52 % | -1.31 pp | 19.07 % | +0.24 pp | 18.86 % | +0.03 pp | 4.81 % | -14.02 pp |
| Achieved Occupancy (%) | 72.81 % | 29.97 % | -42.84 pp | 77.25 % | +4.44 pp | 76.03 % | +3.22 pp | 80.74 % | +7.93 pp |
| Registers/Thread | 30.00 register/thread | 32.00 register/thread | +2 | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 32.00 register/thread | +2 |
| Waves/SM | 22.27 | 11.14 | -11.13 | 22.27 | +0.00 | 22.27 | +0.00 | 89.09 | +66.82 |
| No Eligible (%) | 22.14 % | 53.02 % | +30.88 pp | 21.54 % | -0.60 pp | 21.54 % | -0.60 pp | 17.15 % | -4.99 pp |
| IPC Active | 3.09 inst/cycle | 1.85 inst/cycle | -1.24 | 3.12 inst/cycle | +0.03 | 3.12 inst/cycle | +0.03 | 3.31 inst/cycle | +0.22 |

---

## Iteration 4

**Best so far**: 1.0225× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_num_warps_2_config_to_restore_block_size_128 | 1.004× | 0.999× | ✓ |  |
| B | fuse_fp32_arithmetic_to_fma_instructions | 1.001× | 1.009× | ✓ |  |
| C | fix_grid_lambda_and_tune_num_warps_to_reduce_scheduler_stall | 1.010× | 1.018× | ✓ | **✓ best** |
| D | fix_hardcoded_grid_to_lambda_to_eliminate_overlaunch_regression | 0.994× | 0.259× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_num_warps_... | Δ | fuse_fp32_arithme... | Δ | fix_grid_lambda_a... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32260 ns | 32290 ns | +30 ns (+0.1%) | 31970 ns | -290 ns (-0.9%) | 31680 ns | -580 ns (-1.8%) | 124770 ns | +92510 ns (+286.8%) |
| SM Throughput (%) | 73.05 % | 72.90 % | -0.15 pp | 73.73 % | +0.68 pp | 41.47 % | -31.58 pp | 81.25 % | +8.20 pp |
| Memory Throughput (%) | 35.36 % | 35.25 % | -0.11 pp | 35.35 % | -0.01 pp | 26.83 % | -8.53 pp | 33.77 % | -1.59 pp |
| DRAM Throughput (%) | 18.83 % | 17.14 % | -1.69 pp | 19.00 % | +0.17 pp | 19.13 % | +0.30 pp | 4.83 % | -14.00 pp |
| Achieved Occupancy (%) | 72.81 % | 73.92 % | +1.11 pp | 76.93 % | +4.12 pp | 30.38 % | -42.43 pp | 80.66 % | +7.85 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 32.00 register/thread | +2 | 32.00 register/thread | +2 |
| Waves/SM | 22.27 | 22.27 | +0.00 | 22.27 | +0.00 | 11.14 | -11.13 | 89.09 | +66.82 |
| No Eligible (%) | 22.14 % | 21.51 % | -0.63 pp | 20.34 % | -1.80 pp | 53.14 % | +31.00 pp | 17.16 % | -4.98 pp |
| IPC Active | 3.09 inst/cycle | 3.11 inst/cycle | +0.02 | 3.17 inst/cycle | +0.08 | 1.86 inst/cycle | -1.23 | 3.31 inst/cycle | +0.22 |

---

## Iteration 5

**Best so far**: 1.0225× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_at_fixed_xblock | 1.000× | 1.001× | ✓ |  |
| B | pipeline_loads_with_num_stages_to_hide_l1tex_latency | 1.002× | 1.006× | ✓ |  |
| C | increase_num_stages_to_hide_l1tex_load_latency | 0.998× | 1.009× | ✓ |  |
| D | fix_hardcoded_grid_to_lambda_to_unlock_autotune_xblock_selection | 0.994× | 0.258× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | pipeline_loads_wi... | Δ | increase_num_stag... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32260 ns | 32220 ns | -40 ns (-0.1%) | 32060 ns | -200 ns (-0.6%) | 31970 ns | -290 ns (-0.9%) | 124860 ns | +92600 ns (+287.0%) |
| SM Throughput (%) | 73.05 % | 72.51 % | -0.54 pp | 73.71 % | +0.66 pp | 71.85 % | -1.20 pp | 81.39 % | +8.34 pp |
| Memory Throughput (%) | 35.36 % | 35.11 % | -0.25 pp | 35.33 % | -0.03 pp | 34.75 % | -0.61 pp | 33.82 % | -1.54 pp |
| DRAM Throughput (%) | 18.83 % | 18.95 % | +0.12 pp | 19.07 % | +0.24 pp | 17.29 % | -1.54 pp | 4.82 % | -14.01 pp |
| Achieved Occupancy (%) | 72.81 % | 73.86 % | +1.05 pp | 77.03 % | +4.22 pp | 69.88 % | -2.93 pp | 80.58 % | +7.77 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 32.00 register/thread | +2 |
| Waves/SM | 22.27 | 22.27 | +0.00 | 22.27 | +0.00 | 22.27 | +0.00 | 89.09 | +66.82 |
| No Eligible (%) | 22.14 % | 21.75 % | -0.39 pp | 20.54 % | -1.60 pp | 22.27 % | +0.13 pp | 17.14 % | -5.00 pp |
| IPC Active | 3.09 inst/cycle | 3.12 inst/cycle | +0.03 | 3.16 inst/cycle | +0.07 | 3.10 inst/cycle | +0.01 | 3.31 inst/cycle | +0.22 |

---

## Cumulative Progress

```
Baseline:  32260 ns  (1.000×)
Iter 2     32030 ns  (1.007×)  — change_eviction_policy_streaming_loads_to_evict_first
Iter 3     31550 ns  (1.022×)  — fix_autotune_grid_lambda_and_restrict_to_safe_xblock_configs

Final best: 31680 ns  (1.0183×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,14 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
     xnumel = 6021120
@@ -57,7 +65,6 @@

     tmp45 = tl.maximum(tmp44, tmp43)
     tl.store(in_out_ptr0 + (x2), tmp45, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -71,7 +78,7 @@

     arg_9 = inputs[9]
     xnumel = 6021120
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 6021120, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 6021120)
     return arg_9
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

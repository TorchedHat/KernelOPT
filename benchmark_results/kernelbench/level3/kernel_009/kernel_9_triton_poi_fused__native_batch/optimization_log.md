# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7460 ns — **Best latency**: 7420 ns (**1.0054×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_grid_lambda_to_enable_autotune_xblock_with_maxnreg_guard`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 17
- **Duration**: 7460 ns
- **SM Throughput**: 39.43 % | **Memory**: 17.28 % | **DRAM**: 11.05 %
- **Achieved Occupancy**: 57.52 %
- **IPC Active**: 2.22 inst/cycle | **No Eligible**: 42.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.998× | 0.279× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_improve_occupancy | 0.999× | 0.979× | ✓ |  |
| C | increase_xblock_with_autotune_for_vectorized_loads | 0.999× | 0.277× | ✓ |  |
| D | increase_xblock_autotune_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.708× | 0.639× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_w... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7460 ns | 26750 ns | +19290 ns (+258.6%) | 7620 ns | +160 ns (+2.1%) | 26980 ns | +19520 ns (+261.7%) | 11680 ns | +4220 ns (+56.6%) |
| SM Throughput (%) | 39.43 % | 81.43 % | +42.00 pp | 46.39 % | +6.96 pp | 81.83 % | +42.40 pp | 71.36 % | +31.93 pp |
| Memory Throughput (%) | 17.28 % | 17.13 % | -0.15 pp | 17.34 % | +0.06 pp | 17.21 % | -0.07 pp | 22.42 % | +5.14 pp |
| DRAM Throughput (%) | 11.05 % | 3.07 % | -7.98 pp | 10.77 % | -0.28 pp | 3.04 % | -8.01 pp | 7.04 % | -4.01 pp |
| Achieved Occupancy (%) | 57.52 % | 54.87 % | -2.65 pp | 60.52 % | +3.00 pp | 54.80 % | -2.72 pp | 81.87 % | +24.35 pp |
| Registers/Thread | 17.00 register/thread | 48.00 register/thread | +31 | 18.00 register/thread | +1 | 48.00 register/thread | +31 | 24.00 register/thread | +7 |
| Waves/SM | 3.71 | 11.88 | +8.17 | 3.71 | +0.00 | 11.88 | +8.17 | 7.42 | +3.71 |
| No Eligible (%) | 42.01 % | 11.14 % | -30.87 pp | 32.93 % | -9.08 pp | 11.13 % | -30.88 pp | 14.49 % | -27.52 pp |
| IPC Active | 2.22 inst/cycle | 3.52 inst/cycle | +1.30 | 2.65 inst/cycle | +0.43 | 3.54 inst/cycle | +1.32 | 3.38 inst/cycle | +1.16 |


### Key Learnings

#### ❌ Plan C: `increase_xblock_with_autotune_for_vectorized_loads` (0.277×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) caused the compiler to allocate significantly more registers per thread (17 → 48), which reduced Block Limit Registers from 21 to 5 and capped theoretical occupancy at 62.5%. Despite improving eligible warps per scheduler (1.4 → 4.98) and compute throughput (39% → 82%), the kernel duration increased 3.6x (7.46 µs → 26.98 µs) because the autotuner selected a larger block size that dramatically increased total instruction count (2.3M → 17.1M executed instructions) — the kernel was doing far more work per launch due to the restore_value=['in_out_ptr0'] autotune overhead and the selected config processing more elements with higher register pressure. When the original kernel has very low register usage (≤20 registers/thread) and is already near theoretical 100% occupancy, adding autotune wrappers with larger XBLOCK can trigger register spilling or compiler register allocation changes that negate any latency-hiding benefit.

> **Applicable when**: Never — this specific pattern (autotune on in-place kernel with restore_value) caused severe regression. Avoid when registers/thread is already low (≤20) and theoretical occupancy is near 100%.

> **Avoid when**: Kernel has low register count (≤20 registers/thread) and near-100% theoretical occupancy; adding autotune with larger XBLOCK configs causes register count to jump (e.g., 17→48), collapsing Block Limit Registers and theoretical occupancy. Also avoid when grid is not updated to use meta['XBLOCK'] — the fixed grid with larger autotuned XBLOCK causes each block to process more elements than the grid covers, leading to massive instruction count inflation. Do not use restore_value on in-place output tensors with autotune unless the grid is also made dynamic.

#### ❌ Plan D: `increase_xblock_autotune_to_eliminate_partial_wave_and_hide_l1tex_latency` (0.639×)

Doubling the block size (128→256 selected by autotune) increased register pressure from 17 to 24 registers/thread, which reduced the number of concurrent blocks per SM (Block Limit Registers dropped from 21 to 10). Although achieved occupancy improved from 57.5% to 81.9% and eligible warps per scheduler rose from 1.4 to 6.5, the total executed instruction count ballooned from 2.3M to 6.5M (2.8×), indicating the larger XBLOCK caused each thread to process more elements with redundant masked work, overwhelming the compute gains. The kernel duration increased from 7.46 µs to 11.68 µs (0.64× speedup), demonstrating that adding autotune with larger XBLOCK configs on a kernel where xnumel is already exactly divisible by the original block size can introduce unnecessary instruction overhead without proportional throughput benefit.

> **Applicable when**: Kernel has very low eligible warps per scheduler (<2) AND low achieved occupancy (<60%) AND the bottleneck is confirmed L1TEX scoreboard stalls AND register count per thread is low enough that larger blocks won't hit register-file limits.

> **Avoid when**: The original block size already divides xnumel exactly with no masking needed (xmask=True), because adding autotune with larger XBLOCK forces conditional masking that multiplies instruction count. Also avoid when the kernel is already compute-bound at the original block size, or when register pressure increase from larger blocks will reduce Block Limit Registers significantly. Do not apply when the autotune overhead (extra kernel launches for tuning) is not amortized by repeated calls.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_add_evict_first_on_streaming_load | 0.997× | 0.996× | ✓ |  |
| B | rewrite_arithmetic_chain_with_explicit_fma_to_reduce_nonfused_fp32_instructions | 0.912× | 0.932× | ✓ |  |
| C | reduce_register_pressure_via_maxnreg_to_restore_occupancy | 1.029× | 1.005× | ✓ | **✓ best** |
| D | remove_autotune_and_cap_registers_to_restore_occupancy | 0.987× | 0.979× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | rewrite_arithmeti... | Δ | reduce_register_p... | Δ | remove_autotune_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7460 ns | 7490 ns | +30 ns (+0.4%) | 8000 ns | +540 ns (+7.2%) | 7420 ns | -40 ns (-0.5%) | 7620 ns | +160 ns (+2.1%) |
| SM Throughput (%) | 39.43 % | 47.84 % | +8.41 pp | 58.68 % | +19.25 pp | 31.87 % | -7.56 pp | 39.51 % | +0.08 pp |
| Memory Throughput (%) | 17.28 % | 16.80 % | -0.48 pp | 34.27 % | +16.99 pp | 23.61 % | +6.33 pp | 17.32 % | +0.04 pp |
| DRAM Throughput (%) | 11.05 % | 10.99 % | -0.06 pp | 10.25 % | -0.80 pp | 11.07 % | +0.02 pp | 10.75 % | -0.30 pp |
| Achieved Occupancy (%) | 57.52 % | 61.82 % | +4.30 pp | 71.74 % | +14.22 pp | 41.37 % | -16.15 pp | 56.42 % | -1.10 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 24.00 register/thread | +7 | 17.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 7.42 | +3.71 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 42.01 % | 29.57 % | -12.44 pp | 23.89 % | -18.12 pp | 48.60 % | +6.59 pp | 42.44 % | +0.43 pp |
| IPC Active | 2.22 inst/cycle | 2.81 inst/cycle | +0.59 | 3.01 inst/cycle | +0.79 | 2.07 inst/cycle | -0.15 | 2.24 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.0054× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_arithmetic_to_fma_to_reduce_instruction_count | 1.002× | 1.010× | ✓ |  |
| B | collapse_reciprocal_sqrt_chain_to_rsqrt_to_reduce_nonfused_fp32_instructions | 1.024× | 1.022× | ✓ |  |
| C | fuse_rsqrt_and_increase_num_warps_to_hide_l1tex_latency | 1.020× | 1.028× | ✓ |  |
| D | add_evict_first_on_streaming_main_tensor_to_improve_l2_locality_for_broadcast_params | 0.889× | 0.897× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_arithme... | Δ | collapse_reciproc... | Δ | fuse_rsqrt_and_in... | Δ | add_evict_first_o... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7460 ns | 7390 ns | -70 ns (-0.9%) | 7300 ns | -160 ns (-2.1%) | 7260 ns | -200 ns (-2.7%) | 8320 ns | +860 ns (+11.5%) |
| SM Throughput (%) | 39.43 % | 32.28 % | -7.15 pp | 31.33 % | -8.10 pp | 26.34 % | -13.09 pp | 60.18 % | +20.75 pp |
| Memory Throughput (%) | 17.28 % | 23.13 % | +5.85 pp | 17.76 % | +0.48 pp | 17.11 % | -0.17 pp | 32.57 % | +15.29 pp |
| DRAM Throughput (%) | 11.05 % | 11.13 % | +0.08 pp | 11.31 % | +0.26 pp | 11.34 % | +0.29 pp | 9.88 % | -1.17 pp |
| Achieved Occupancy (%) | 57.52 % | 44.28 % | -13.24 pp | 47.35 % | -10.17 pp | 44.89 % | -12.63 pp | 73.67 % | +16.15 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 18.00 register/thread | +1 | 16.00 register/thread | -1 | 22.00 register/thread | +5 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 7.42 | +3.71 |
| No Eligible (%) | 42.01 % | 45.92 % | +3.91 pp | 53.11 % | +11.10 pp | 67.29 % | +25.28 pp | 21.91 % | -20.10 pp |
| IPC Active | 2.22 inst/cycle | 2.05 inst/cycle | -0.17 | 1.87 inst/cycle | -0.35 | 1.35 inst/cycle | -0.87 | 3.10 inst/cycle | +0.88 |

---

## Iteration 4

**Best so far**: 1.0054× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | align_grid_to_sm_multiple_to_eliminate_partial_wave_tail | 1.020× | 1.032× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_latency_at_fixed_xblock | 1.019× | 1.028× | ✓ |  |
| C | fix_grid_lambda_to_enable_autotune_xblock_with_maxnreg_guard | 1.023× | 1.005× | ✓ |  |
| D | reduce_num_warps_per_block_to_eliminate_not_selected_cpi_stall | 0.996× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | align_grid_to_sm_... | Δ | increase_num_warp... | Δ | fix_grid_lambda_t... | Δ | reduce_num_warps_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7460 ns | 7230 ns | -230 ns (-3.1%) | 7260 ns | -200 ns (-2.7%) | 7420 ns | -40 ns (-0.5%) | 7390 ns | -70 ns (-0.9%) |
| SM Throughput (%) | 39.43 % | 28.69 % | -10.74 pp | 31.31 % | -8.12 pp | 32.29 % | -7.14 pp | 34.50 % | -4.93 pp |
| Memory Throughput (%) | 17.28 % | 17.81 % | +0.53 pp | 17.80 % | +0.52 pp | 23.08 % | +5.80 pp | 23.90 % | +6.62 pp |
| DRAM Throughput (%) | 11.05 % | 11.33 % | +0.28 pp | 11.32 % | +0.27 pp | 11.04 % | -0.01 pp | 11.15 % | +0.10 pp |
| Achieved Occupancy (%) | 57.52 % | 46.66 % | -10.86 pp | 48.04 % | -9.48 pp | 41.04 % | -16.48 pp | 42.28 % | -15.24 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 22.00 register/thread | +5 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 42.01 % | 57.04 % | +15.03 pp | 52.82 % | +10.81 pp | 47.34 % | +5.33 pp | 44.58 % | +2.57 pp |
| IPC Active | 2.22 inst/cycle | 1.72 inst/cycle | -0.50 | 1.89 inst/cycle | -0.33 | 2.15 inst/cycle | -0.07 | 2.23 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  7460 ns  (1.000×)
Iter 2     7420 ns  (1.005×)  — reduce_register_pressure_via_maxnreg_to_restore_occupancy

Final best: 7420 ns  (1.0054×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,12 +5,12 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1003520
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x2 = xindex
     x0 = (xindex % 128)
     tmp0 = tl.load(in_out_ptr0 + (x2), None)
@@ -21,19 +21,13 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    tmp6 = libdevice.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, None)
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

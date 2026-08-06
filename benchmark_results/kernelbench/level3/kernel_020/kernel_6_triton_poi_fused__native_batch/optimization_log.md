# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 13730 ns — **Best latency**: 13470 ns (**1.0193×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_memory_access_hints_via_multiple_of_and_max_contiguous`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_6_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (18000,1,1), Waves/SM: 8.52
- **Registers/thread**: 18
- **Duration**: 13730 ns
- **SM Throughput**: 56.84 % | **Memory**: 21.16 % | **DRAM**: 13.71 %
- **Achieved Occupancy**: 59.21 %
- **IPC Active**: 2.73 inst/cycle | **No Eligible**: 31.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size_for_latency_hiding | 0.773× | 0.759× | ✓ |  |
| B | increase_xblock_with_autotune | 0.985× | 0.415× | ✓ |  |
| C | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 0.986× | 1.000× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_better_occupancy | 0.782× | 0.739× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13730 ns | 18080 ns | +4350 ns (+31.7%) | 33090 ns | +19360 ns (+141.0%) | 13730 ns | +0 ns (+0.0%) | 18590 ns | +4860 ns (+35.4%) |
| SM Throughput (%) | 56.84 % | 76.72 % | +19.88 pp | 82.60 % | +25.76 pp | 56.50 % | -0.34 pp | 77.21 % | +20.37 pp |
| Memory Throughput (%) | 21.16 % | 23.68 % | +2.52 pp | 19.52 % | -1.64 pp | 21.02 % | -0.14 pp | 22.97 % | +1.81 pp |
| DRAM Throughput (%) | 13.71 % | 10.40 % | -3.31 pp | 5.68 % | -8.03 pp | 13.70 % | -0.01 pp | 10.12 % | -3.59 pp |
| Achieved Occupancy (%) | 59.21 % | 84.79 % | +25.58 pp | 85.63 % | +26.42 pp | 59.94 % | +0.73 pp | 85.49 % | +26.28 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 32.00 register/thread | +14 |
| Waves/SM | 8.52 | 8.52 | +0.00 | 17.05 | +8.53 | 8.52 | +0.00 | 8.52 | +0.00 |
| No Eligible (%) | 31.12 % | 13.25 % | -17.87 pp | 11.82 % | -19.30 pp | 31.20 % | +0.08 pp | 13.64 % | -17.48 pp |
| IPC Active | 2.73 inst/cycle | 3.43 inst/cycle | +0.70 | 3.52 inst/cycle | +0.79 | 2.73 inst/cycle | +0.00 | 3.46 inst/cycle | +0.73 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_tile_size_for_latency_hiding` (0.759×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to allocate significantly more registers per thread (18→32), which reduced the block limit from registers (21→16 blocks/SM) and increased total executed instructions by ~78% (6.1M→10.9M). Although achieved occupancy improved (59%→85%) and eligible warps per scheduler rose (1.76→7.06), the net effect was a 32% slowdown because the instruction count explosion and higher warp cycles per instruction (13.6→15.6) outweighed the latency-hiding benefit. When a kernel's register count is near the boundary where doubling XBLOCK causes a register spill or significant register file pressure increase, the occupancy gain does not compensate for the extra instruction overhead.

> **Applicable when**: Kernel has low register count (≤20 registers/thread) AND low achieved occupancy (<65%) AND high L1TEX stall fraction (>40% of CPI stall) AND small fixed XBLOCK (≤128) with large grid

> **Avoid when**: Kernel register count is already moderate (18+ registers/thread) and increasing XBLOCK causes register count to jump significantly (e.g. 18→32), as the instruction count increase and register pressure will outweigh occupancy gains. Also avoid when the kernel body contains many independent FP32 operations that scale with XBLOCK, inflating instruction count super-linearly.

#### ❌ Plan B: `increase_xblock_with_autotune` (0.415×)

Adding @triton.autotune with larger XBLOCK configs (256–1024) to a pointwise kernel improved achieved occupancy (59% → 86%) and scheduler eligible warps (1.76 → 7.65), but caused a severe 2.4x slowdown (13.7 µs → 33.1 µs). The root cause is that the larger block size triggered a register pressure increase from 18 to 32 registers/thread, and the autotune selected XBLOCK=256 (not the intended 512–1024), which doubled waves_per_SM (8.5 → 17.05) and executed instructions (6.1M → 21.4M) without reducing grid size. The grid was still hardcoded to cdiv(xnumel, 128) in the launch wrapper, so the autotune's XBLOCK injection was overridden by the stale fixed grid, causing each block to process only 128 elements despite the larger XBLOCK tile — effectively doubling work.

> **Applicable when**: Never apply this exact pattern. The implementation flaw (stale fixed grid) invalidates the strategy.

> **Avoid when**: When adding @triton.autotune to change XBLOCK, the launch grid MUST be updated to a lambda: `grid = lambda meta: (cdiv(xnumel, meta['XBLOCK']),)`. Failing to do so causes the grid to remain sized for the old XBLOCK while the kernel processes a larger tile, resulting in redundant work and severe slowdown. Also avoid if register count per thread is near the block limit threshold — larger XBLOCK can push registers/thread from 18 to 32+, reducing theoretical occupancy via Block Limit Registers.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.042× | 0.035× | ✓ |  |
| B | persistent_tile_scheduling_with_fma_arithmetic_fix | 1.011× | 1.012× | ✓ | **✓ best** |
| C | fuse_fp32_instructions_via_rsqrt_and_fma | 0.934× | 0.956× | ✓ |  |
| D | fix_stale_autotune_grid_and_fuse_rsqrt_instructions | 0.998× | 0.722× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fuse_fp32_instruc... | Δ | fix_stale_autotun... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13730 ns | 393660 ns | +379930 ns (+2767.2%) | 13570 ns | -160 ns (-1.2%) | 14370 ns | +640 ns (+4.7%) | 19010 ns | +5280 ns (+38.5%) |
| SM Throughput (%) | 56.84 % | 92.66 % | +35.82 pp | 37.10 % | -19.74 pp | 45.72 % | -11.12 pp | 67.73 % | +10.89 pp |
| Memory Throughput (%) | 21.16 % | 42.53 % | +21.37 pp | 20.71 % | -0.45 pp | 31.33 % | +10.17 pp | 34.82 % | +13.66 pp |
| DRAM Throughput (%) | 13.71 % | 0.56 % | -13.15 pp | 13.88 % | +0.17 pp | 13.08 % | -0.63 pp | 9.89 % | -3.82 pp |
| Achieved Occupancy (%) | 59.21 % | 98.40 % | +39.19 pp | 45.57 % | -13.64 pp | 39.29 % | -19.92 pp | 78.99 % | +19.78 pp |
| Registers/Thread | 18.00 register/thread | 25.00 register/thread | +7 | 30.00 register/thread | +12 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 8.52 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 | 17.05 | +8.53 |
| No Eligible (%) | 31.12 % | 6.60 % | -24.52 pp | 54.70 % | +23.58 pp | 42.12 % | +11.00 pp | 24.34 % | -6.78 pp |
| IPC Active | 2.73 inst/cycle | 3.73 inst/cycle | +1.00 | 1.83 inst/cycle | -0.90 | 2.30 inst/cycle | -0.43 | 3.06 inst/cycle | +0.33 |

---

## Iteration 3

**Best so far**: 1.0118× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_streaming_load_to_improve_l2_hit_rate_for_coefficients | 0.995× | 1.012× | ✓ |  |
| B | streaming_eviction_policy_for_large_tensor_plus_reduce_num_warps | 0.998× | 1.017× | ✓ |  |
| C | fuse_fp32_multiply_add_pairs_via_libdevice_fma | 1.004× | 0.722× | ✓ |  |
| D | revert_persistent_loop_and_fuse_rsqrt_via_libdevice | 1.013× | 1.019× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | streaming_evictio... | Δ | fuse_fp32_multipl... | Δ | revert_persistent... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13730 ns | 13570 ns | -160 ns (-1.2%) | 13500 ns | -230 ns (-1.7%) | 19010 ns | +5280 ns (+38.5%) | 13470 ns | -260 ns (-1.9%) |
| SM Throughput (%) | 56.84 % | 41.22 % | -15.62 pp | 37.06 % | -19.78 pp | 68.34 % | +11.50 pp | 34.03 % | -22.81 pp |
| Memory Throughput (%) | 21.16 % | 20.97 % | -0.19 pp | 21.35 % | +0.19 pp | 35.01 % | +13.85 pp | 21.18 % | +0.02 pp |
| DRAM Throughput (%) | 13.71 % | 13.87 % | +0.16 pp | 13.93 % | +0.22 pp | 9.89 % | -3.82 pp | 13.95 % | +0.24 pp |
| Achieved Occupancy (%) | 59.21 % | 46.57 % | -12.64 pp | 43.35 % | -15.86 pp | 78.93 % | +19.72 pp | 42.18 % | -17.03 pp |
| Registers/Thread | 18.00 register/thread | 30.00 register/thread | +12 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 8.52 | 8.52 | +0.00 | 8.52 | +0.00 | 17.05 | +8.53 | 8.52 | +0.00 |
| No Eligible (%) | 31.12 % | 48.82 % | +17.70 pp | 54.35 % | +23.23 pp | 24.14 % | -6.98 pp | 57.68 % | +26.56 pp |
| IPC Active | 2.73 inst/cycle | 2.03 inst/cycle | -0.70 | 1.81 inst/cycle | -0.92 | 3.06 inst/cycle | +0.33 | 1.68 inst/cycle | -1.05 |

---

## Iteration 4

**Best so far**: 1.0193× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_without_register_pressure | 1.012× | 1.019× | ✓ |  |
| B | reduce_block_size_to_improve_l1_coefficient_reuse_and_warp_scheduling | 0.999× | 1.005× | ✓ |  |
| C | fix_persistent_kernel_grid_to_use_num_sms_ctas | 1.008× | 1.012× | ✓ |  |
| D | fix_stale_grid_and_reduce_warp_count_for_not_selected_stalls | 1.026× | 1.019× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | reduce_block_size... | Δ | fix_persistent_ke... | Δ | fix_stale_grid_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13730 ns | 13470 ns | -260 ns (-1.9%) | 13660 ns | -70 ns (-0.5%) | 13570 ns | -160 ns (-1.2%) | 13470 ns | -260 ns (-1.9%) |
| SM Throughput (%) | 56.84 % | 34.44 % | -22.40 pp | 37.02 % | -19.82 pp | 41.63 % | -15.21 pp | 34.42 % | -22.42 pp |
| Memory Throughput (%) | 21.16 % | 21.38 % | +0.22 pp | 21.32 % | +0.16 pp | 21.17 % | +0.01 pp | 21.39 % | +0.23 pp |
| DRAM Throughput (%) | 13.71 % | 13.95 % | +0.24 pp | 13.77 % | +0.06 pp | 13.87 % | +0.16 pp | 13.96 % | +0.25 pp |
| Achieved Occupancy (%) | 59.21 % | 43.06 % | -16.15 pp | 43.52 % | -15.69 pp | 46.75 % | -12.46 pp | 43.38 % | -15.83 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 18.00 register/thread | +0 |
| Waves/SM | 8.52 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 |
| No Eligible (%) | 31.12 % | 57.80 % | +26.68 pp | 54.23 % | +23.11 pp | 48.75 % | +17.63 pp | 57.51 % | +26.39 pp |
| IPC Active | 2.73 inst/cycle | 1.66 inst/cycle | -1.07 | 1.82 inst/cycle | -0.91 | 2.03 inst/cycle | -0.70 | 1.65 inst/cycle | -1.08 |

---

## Iteration 5

**Best so far**: 1.0193× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_memory_access_hints_via_multiple_of_and_max_contiguous | 0.996× | 1.019× | ✓ |  |
| B | fix_autotune_grid_lambda_and_explore_small_xblock_num_warps | 1.011× | 1.019× | ✓ |  |
| C | preload_small_coefficient_tensors_into_registers_before_persistent_loop | 0.999× | 1.005× | ✓ |  |
| D | vectorized_load_hints_via_tl_multiple_of_and_max_contiguous | 0.998× | 1.017× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_memory_access... | Δ | fix_autotune_grid... | Δ | preload_small_coe... | Δ | vectorized_load_h... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13730 ns | 13470 ns | -260 ns (-1.9%) | 13470 ns | -260 ns (-1.9%) | 13660 ns | -70 ns (-0.5%) | 13500 ns | -230 ns (-1.7%) |
| SM Throughput (%) | 56.84 % | 34.44 % | -22.40 pp | 34.41 % | -22.43 pp | 37.37 % | -19.47 pp | 36.93 % | -19.91 pp |
| Memory Throughput (%) | 21.16 % | 21.38 % | +0.22 pp | 21.37 % | +0.21 pp | 20.66 % | -0.50 pp | 21.29 % | +0.13 pp |
| DRAM Throughput (%) | 13.71 % | 13.95 % | +0.24 pp | 13.94 % | +0.23 pp | 13.78 % | +0.07 pp | 13.95 % | +0.24 pp |
| Achieved Occupancy (%) | 59.21 % | 42.45 % | -16.76 pp | 41.97 % | -17.24 pp | 44.22 % | -14.99 pp | 44.36 % | -14.85 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 16.00 register/thread | -2 | 18.00 register/thread | +0 |
| Waves/SM | 8.52 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 | 8.52 | +0.00 |
| No Eligible (%) | 31.12 % | 57.83 % | +26.71 pp | 57.70 % | +26.58 pp | 60.58 % | +29.46 pp | 54.21 % | +23.09 pp |
| IPC Active | 2.73 inst/cycle | 1.69 inst/cycle | -1.04 | 1.68 inst/cycle | -1.05 | 1.56 inst/cycle | -1.17 | 1.82 inst/cycle | -0.91 |

---

## Cumulative Progress

```
Baseline:  13730 ns  (1.000×)
Iter 2     13570 ns  (1.012×)  — persistent_tile_scheduling_with_fma_arithmetic_fix
Iter 3     13470 ns  (1.019×)  — revert_persistent_loop_and_fuse_rsqrt_via_libdevice

Final best: 13470 ns  (1.0193×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,8 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 128}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
-def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
+def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
     xnumel = 2304000
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
@@ -21,16 +30,11 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = libdevice.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -40,7 +44,7 @@

     arg_4 = inputs[4]
     xnumel = 2304000
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2304000, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2304000)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21980 ns — **Best latency**: 21820 ns (**1.0073×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_to_unlock_autotune_block_size_selection`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 14.85
- **Registers/thread**: 16
- **Duration**: 21980 ns
- **SM Throughput**: 35.63 % | **Memory**: 23.53 % | **DRAM**: 14.90 %
- **Achieved Occupancy**: 40.88 %
- **IPC Active**: 1.53 inst/cycle | **No Eligible**: 62.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.998× | 0.325× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 0.999× | 0.323× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_latency_hiding | 1.006× | 1.009× | ✓ |  |
| D | autotune_xblock_num_warps_num_stages | 1.010× | 1.004× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21980 ns | 67680 ns | +45700 ns (+207.9%) | 67970 ns | +45990 ns (+209.2%) | 21790 ns | -190 ns (-0.9%) | 21890 ns | -90 ns (-0.4%) |
| SM Throughput (%) | 35.63 % | 79.17 % | +43.54 pp | 79.05 % | +43.42 pp | 35.56 % | -0.07 pp | 24.22 % | -11.41 pp |
| Memory Throughput (%) | 23.53 % | 28.48 % | +4.95 pp | 28.44 % | +4.91 pp | 23.45 % | -0.08 pp | 21.09 % | -2.44 pp |
| DRAM Throughput (%) | 14.90 % | 4.83 % | -10.07 pp | 4.81 % | -10.09 pp | 15.03 % | +0.13 pp | 14.97 % | +0.07 pp |
| Achieved Occupancy (%) | 40.88 % | 77.03 % | +36.15 pp | 77.05 % | +36.17 pp | 40.46 % | -0.42 pp | 27.74 % | -13.14 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 16.00 register/thread | +0 | 24.00 register/thread | +8 |
| Waves/SM | 14.85 | 59.39 | +44.54 | 59.39 | +44.54 | 14.85 | +0.00 | 7.42 | -7.43 |
| No Eligible (%) | 62.00 % | 18.35 % | -43.65 pp | 18.61 % | -43.39 pp | 61.59 % | -0.41 pp | 70.33 % | +8.33 pp |
| IPC Active | 1.53 inst/cycle | 3.26 inst/cycle | +1.73 | 3.26 inst/cycle | +1.73 | 1.53 inst/cycle | +0.00 | 1.17 inst/cycle | -0.36 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.325×)

Increasing the tile size (XBLOCK) from 128 to 512 via autotune doubled registers per thread from 16 to 32, which halved the number of concurrent blocks per SM (Block Limit Registers dropped from 32 to 4), causing waves_per_sm to balloon from 14.85 to 59.39 and total executed instructions to increase 7.3x (5.77M to 42.2M). Although compute SM throughput improved (35% → 79%) and occupancy rose (41% → 77%), the kernel duration tripled (22us → 68us) because the autotuner selected a suboptimal config (XBLOCK=512) that dramatically increased total work and serialized execution through register-limited occupancy. The implementation flaw was that the grid was hardcoded to cdiv(xnumel, 128) while the kernel was called without XBLOCK, so the autotuner ran with a mismatched grid, and the added xmask predicate on the small-stride loads (in_ptr0..in_ptr3 indexed by x0=xindex%32) introduced unnecessary predication overhead on what were previously unmasked loads.

> **Applicable when**: Kernel has low achieved occupancy (<50%), high L1TEX scoreboard stalls (>60% of warp cycles), low issued warps per scheduler (<0.5), and small block size (<=128) with many waves per SM.

> **Avoid when**: Registers per thread are already at or near 16-24 and block size is small — increasing XBLOCK will double register usage, cut Block Limit Registers, and cause more serialized waves rather than fewer. Also avoid when the grid is hardcoded independently of XBLOCK (implementation flaw). Do not add xmask to loads indexed by a modulo of xindex (e.g., xindex % C for small C) when those loads were previously unmasked — the predication adds overhead without correctness benefit for the channel-stride dimension.

---

## Iteration 2

**Best so far**: 1.0041× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.000× | 1.004× | ✓ |  |
| B | fix_grid_lambda_to_enable_correct_autotune_xblock_selection | 1.015× | 1.015× | ✓ | **✓ best** |
| C | fix_grid_mismatch_and_restore_lean_xblock128_config | 0.999× | 1.012× | ✓ |  |
| D | fix_grid_bug_and_cap_registers_to_recover_baseline | 0.999× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_grid_lambda_t... | Δ | fix_grid_mismatch... | Δ | fix_grid_bug_and_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21980 ns | 21890 ns | -90 ns (-0.4%) | 21660 ns | -320 ns (-1.5%) | 21730 ns | -250 ns (-1.1%) | 21730 ns | -250 ns (-1.1%) |
| SM Throughput (%) | 35.63 % | 35.48 % | -0.15 pp | 21.24 % | -14.39 pp | 41.91 % | +6.28 pp | 42.73 % | +7.10 pp |
| Memory Throughput (%) | 23.53 % | 23.40 % | -0.13 pp | 11.20 % | -12.33 pp | 23.73 % | +0.20 pp | 23.75 % | +0.22 pp |
| DRAM Throughput (%) | 14.90 % | 14.94 % | +0.04 pp | 7.56 % | -7.34 pp | 15.07 % | +0.17 pp | 15.07 % | +0.17 pp |
| Achieved Occupancy (%) | 40.88 % | 40.36 % | -0.52 pp | 27.45 % | -13.43 pp | 42.19 % | +1.31 pp | 42.54 % | +1.66 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 17.00 register/thread | +1 | 18.00 register/thread | +2 | 17.00 register/thread | +1 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 7.42 | -7.43 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 62.00 % | 61.77 % | -0.23 pp | 74.14 % | +12.14 pp | 52.32 % | -9.68 pp | 51.59 % | -10.41 pp |
| IPC Active | 1.53 inst/cycle | 1.53 inst/cycle | +0.00 | 1.02 inst/cycle | -0.51 | 1.90 inst/cycle | +0.37 | 1.91 inst/cycle | +0.38 |

---

## Iteration 3

**Best so far**: 1.0148× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_unlock_autotune_block_size_selection | 1.015× | 1.007× | ✓ | **✓ best** |
| B | increase_num_warps_at_fixed_xblock_to_saturate_warp_schedulers | 0.999× | 1.001× | ✓ |  |
| C | cap_registers_and_remove_predication_overhead_on_channel_loads | 1.005× | 1.004× | ✓ |  |
| D | replace_explicit_fma_with_standard_arithmetic_for_compiler_fused_ffma | 1.001× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | increase_num_warp... | Δ | cap_registers_and... | Δ | replace_explicit_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21980 ns | 21820 ns | -160 ns (-0.7%) | 21950 ns | -30 ns (-0.1%) | 21890 ns | -90 ns (-0.4%) | 21760 ns | -220 ns (-1.0%) |
| SM Throughput (%) | 35.63 % | 21.24 % | -14.39 pp | 41.68 % | +6.05 pp | 41.94 % | +6.31 pp | 35.43 % | -0.20 pp |
| Memory Throughput (%) | 23.53 % | 11.22 % | -12.31 pp | 23.58 % | +0.05 pp | 23.71 % | +0.18 pp | 23.41 % | -0.12 pp |
| DRAM Throughput (%) | 14.90 % | 7.51 % | -7.39 pp | 14.91 % | +0.01 pp | 14.96 % | +0.06 pp | 15.07 % | +0.17 pp |
| Achieved Occupancy (%) | 40.88 % | 28.21 % | -12.67 pp | 42.19 % | +1.31 pp | 42.18 % | +1.30 pp | 40.55 % | -0.33 pp |
| Registers/Thread | 16.00 register/thread | 17.00 register/thread | +1 | 18.00 register/thread | +2 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 14.85 | 7.42 | -7.43 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 62.00 % | 74.16 % | +12.16 pp | 52.45 % | -9.55 pp | 52.41 % | -9.59 pp | 62.36 % | +0.36 pp |
| IPC Active | 1.53 inst/cycle | 1.00 inst/cycle | -0.53 | 1.90 inst/cycle | +0.37 | 1.90 inst/cycle | +0.37 | 1.50 inst/cycle | -0.03 |

---

## Iteration 4

**Best so far**: 1.0148× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | preload_channel_params_into_shared_memory_to_eliminate_l1tex_stalls | 0.988× | 0.998× | ✓ |  |
| B | fix_grid_lambda_and_cap_registers_for_true_occupancy_recovery | 1.009× | 1.015× | ✓ |  |
| C | preload_channel_params_into_shared_memory_to_eliminate_l1tex_stalls | 0.988× | 0.998× | ✓ |  |
| D | restructure_to_2d_channel_spatial_decomposition_eliminating_modulo | 0.018× | 0.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | preload_channel_p... | Δ | fix_grid_lambda_a... | Δ | preload_channel_p... | Δ | restructure_to_2d... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21980 ns | 22020 ns | +40 ns (+0.2%) | 21660 ns | -320 ns (-1.5%) | 22020 ns | +40 ns (+0.2%) | 1370000 ns | +1348020 ns (+6132.9%) |
| SM Throughput (%) | 35.63 % | 82.29 % | +46.66 pp | 20.87 % | -14.76 pp | 82.21 % | +46.58 pp | 0.63 % | -35.00 pp |
| Memory Throughput (%) | 23.53 % | 56.51 % | +32.98 pp | 11.23 % | -12.30 pp | 56.44 % | +32.91 pp | 7.51 % | -16.02 pp |
| DRAM Throughput (%) | 14.90 % | 14.87 % | -0.03 pp | 7.56 % | -7.34 pp | 14.86 % | -0.04 pp | 0.00 % | -14.90 pp |
| Achieved Occupancy (%) | 40.88 % | 53.96 % | +13.08 pp | 27.26 % | -13.62 pp | 55.49 % | +14.61 pp | 86.29 % | +45.41 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 7.42 | -7.43 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 62.00 % | 48.74 % | -13.26 pp | 74.64 % | +12.64 pp | 45.56 % | -16.44 pp | 99.54 % | +37.54 pp |
| IPC Active | 1.53 inst/cycle | 2.04 inst/cycle | +0.51 | 1.00 inst/cycle | -0.53 | 2.16 inst/cycle | +0.63 | 0.02 inst/cycle | -1.51 |


### Key Learnings

#### ❌ Plan D: `restructure_to_2d_channel_spatial_decomposition_eliminating_modulo` (0.016×)

Restructuring a 1D flat-index kernel into a 2D (channel × spatial) decomposition to eliminate per-element modulo and gather loads caused a 63x regression (22 µs → 1370 µs). The root cause is that the new access pattern `flat_idx = sidx * 32 + pid_c` accesses memory with a stride of 32 elements (128 bytes) between consecutive threads in the spatial dimension, producing 82% excessive sectors (7M of 8.5M total) due to severely uncoalesced global loads and stores. Warp cycles per issued instruction exploded from 17 to 2907, and eligible warps per scheduler collapsed from 0.58 to 0.01, indicating near-total stall on memory transactions. The original 1D layout with `xindex % 32` gather loads, despite their apparent inefficiency, produced coalesced accesses across the spatial dimension that the 2D decomposition destroyed entirely.

> **Applicable when**: Never: this decomposition pattern is fundamentally incompatible with channels-last (NHWC) memory layout where the channel dimension is the innermost stride. The stride-32 access pattern between consecutive spatial indices in a block causes catastrophic uncoalescing regardless of how many modulo operations are saved.

> **Avoid when**: Do NOT apply 2D channel-spatial decomposition (pid_c × pid_s) to channels-last (NHWC) tensors where the channel dimension is innermost (stride=1). The resulting access pattern `flat_idx = spatial_idx * num_channels + channel_id` creates a stride of num_channels between consecutive threads, producing near-100% uncoalesced global memory transactions. This is the opposite of the intended optimization: the modulo-indexed gather loads in the original 1D kernel are actually coalesced across the spatial dimension and well-cached in L1, while the 2D decomposition destroys coalescing entirely. Only apply channel-spatial decomposition when the channel dimension is the outermost stride (NCHW layout) so that spatial threads access contiguous memory.

---

## Iteration 5

**Best so far**: 1.0148× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_increase_warps_to_hide_l1tex_latency | 1.009× | 1.015× | ✓ |  |
| B | explicit_fma_via_tl_math_fma_for_norm_and_bias_add | 0.996× | 1.001× | ✓ |  |
| C | replace_gather_with_direct_indexed_loads_to_enable_ffma_fusion | 1.004× | 1.004× | ✓ |  |
| D | revert_to_1d_flat_index_with_evict_last_channel_params | 0.998× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | explicit_fma_via_... | Δ | replace_gather_wi... | Δ | revert_to_1d_flat... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21980 ns | 21660 ns | -320 ns (-1.5%) | 21950 ns | -30 ns (-0.1%) | 21890 ns | -90 ns (-0.4%) | 21730 ns | -250 ns (-1.1%) |
| SM Throughput (%) | 35.63 % | 20.86 % | -14.77 pp | 82.47 % | +46.84 pp | 41.35 % | +5.72 pp | 41.94 % | +6.31 pp |
| Memory Throughput (%) | 23.53 % | 11.20 % | -12.33 pp | 56.40 % | +32.87 pp | 23.37 % | -0.16 pp | 23.72 % | +0.19 pp |
| DRAM Throughput (%) | 14.90 % | 7.56 % | -7.34 pp | 14.92 % | +0.02 pp | 14.94 % | +0.04 pp | 15.08 % | +0.18 pp |
| Achieved Occupancy (%) | 40.88 % | 27.22 % | -13.66 pp | 50.18 % | +9.30 pp | 42.18 % | +1.30 pp | 42.17 % | +1.29 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 18.00 register/thread | +2 |
| Waves/SM | 14.85 | 7.42 | -7.43 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 62.00 % | 74.61 % | +12.61 pp | 47.43 % | -14.57 pp | 52.72 % | -9.28 pp | 52.39 % | -9.61 pp |
| IPC Active | 1.53 inst/cycle | 1.00 inst/cycle | -0.53 | 2.09 inst/cycle | +0.56 | 1.90 inst/cycle | +0.37 | 1.90 inst/cycle | +0.37 |

---

## Cumulative Progress

```
Baseline:  21980 ns  (1.000×)
Iter 1     21890 ns  (1.004×)  — autotune_xblock_num_warps_num_stages
Iter 2     21660 ns  (1.015×)  — fix_grid_lambda_to_enable_correct_autotune_xblock_selection

Final best: 21820 ns  (1.0073×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,33 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=2),
+    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=3),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    # xnumel=3010560 is exactly divisible by XBLOCK=128 (3010560/128=23520) — no mask needed
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 32)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_first')
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    # FMA-friendly BN normalization
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    # BN normalization
     inv_std = tl.rsqrt(tmp3 + 1e-5)
-    # tl.math.fma(a,b,c) = a*b+c as single FFMA PTX instruction
-    norm   = tl.math.fma(-tmp1, inv_std, tmp0 * inv_std)  # (tmp0-tmp1)*inv_std
-    out    = tl.math.fma(norm, tmp12, tmp14)               # norm*tmp12+tmp14
-    result = tl.maximum(0.0, out)                          # ReLU
-    tl.store(in_out_ptr0 + (x2), result, None, eviction_policy='evict_first')
-
+    norm   = (tmp0 - tmp1) * inv_std   # standard arithmetic -> compiler emits FFMA
+    out    = norm * tmp12 + tmp14      # standard arithmetic -> compiler emits FFMA
+    result = tl.maximum(0.0, out)      # ReLU
+    tl.store(in_out_ptr0 + (x2), result, xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -34,7 +41,7 @@

     arg_4 = inputs[4]
     xnumel = 4014080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4014080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4014080)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4450 ns — **Best latency**: 4450 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_52_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2940,1,1), Waves/SM: 1.39
- **Registers/thread**: 18
- **Duration**: 4450 ns
- **SM Throughput**: 25.79 % | **Memory**: 11.38 % | **DRAM**: 7.04 %
- **Achieved Occupancy**: 62.49 %
- **IPC Active**: 1.89 inst/cycle | **No Eligible**: 45.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_larger_xblock_autotune | 0.968× | 0.420× | ✓ |  |
| B | increase_xblock_with_autotune_to_eliminate_partial_wave | 1.000× | 1.000× | ✓ | **✓ best** |
| C | autotune_xblock_to_eliminate_partial_wave | 0.973× | 0.725× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_eliminate_partial_wave | 0.969× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4450 ns | 10590 ns | +6140 ns (+138.0%) | 4450 ns | +0 ns (+0.0%) | 6140 ns | +1690 ns (+38.0%) | 4480 ns | +30 ns (+0.7%) |
| SM Throughput (%) | 25.79 % | 67.77 % | +41.98 pp | 39.48 % | +13.69 pp | 53.65 % | +27.86 pp | 39.53 % | +13.74 pp |
| Memory Throughput (%) | 11.38 % | 29.19 % | +17.81 pp | 21.34 % | +9.96 pp | 26.53 % | +15.15 pp | 21.43 % | +10.05 pp |
| DRAM Throughput (%) | 7.04 % | 2.93 % | -4.11 pp | 6.97 % | -0.07 pp | 5.08 % | -1.96 pp | 6.97 % | -0.07 pp |
| Achieved Occupancy (%) | 62.49 % | 82.18 % | +19.69 pp | 67.31 % | +4.82 pp | 55.19 % | -7.30 pp | 66.71 % | +4.22 pp |
| Registers/Thread | 18.00 register/thread | 31.00 register/thread | +13 | 31.00 register/thread | +13 | 36.00 register/thread | +18 | 31.00 register/thread | +13 |
| Waves/SM | 1.39 | 5.57 | +4.18 | 1.39 | +0.00 | 1.86 | +0.47 | 1.39 | +0.00 |
| No Eligible (%) | 45.96 % | 15.42 % | -30.54 pp | 21.95 % | -24.01 pp | 19.91 % | -26.05 pp | 24.45 % | -21.51 pp |
| IPC Active | 1.89 inst/cycle | 3.37 inst/cycle | +1.48 | 3.00 inst/cycle | +1.11 | 3.18 inst/cycle | +1.29 | 3.08 inst/cycle | +1.19 |


### Key Learnings

#### ❌ Plan A: `eliminate_partial_wave_via_larger_xblock_autotune` (0.420×)

Adding @triton.autotune with larger XBLOCK configs (512, 1024, 2048) to a pointwise kernel that had a partial-wave launch (waves_per_sm=1.39, grid=2940, block=128) caused a 2.38x slowdown. The autotuner selected XBLOCK=512 but the grid remained at 2940 blocks (same as before), meaning the grid was not recomputed via a lambda — it was hardcoded to cdiv(xnumel, 128). This caused 4x more threads to be launched (1.5M vs 376K) with 31 registers/thread instead of 18, inflating total instruction count from 917K to 5.7M (6.2x) and duration from 4.45 µs to 10.59 µs. The root cause is a code bug: the grid was not updated to use meta['XBLOCK'], so the kernel ran with a fixed grid of 2940 blocks but each block now processed 4x more elements, causing massive redundant work and register pressure increase.

> **Applicable when**: Kernel has a partial wave (waves_per_sm between 1.0 and 2.0), low compute throughput (<30%), and NCU LaunchConfiguration rule fires with high estimated speedup (>30%). Registers per thread are low (<24) and no shared memory is used.

> **Avoid when**: The grid is hardcoded and not updated to use meta['XBLOCK'] — this is the exact failure mode observed here. Do not apply autotune for XBLOCK without simultaneously converting the grid to a lambda that uses meta['XBLOCK']. Also avoid when register count per thread is already high (>24), as larger blocks will further increase register pressure and reduce occupancy.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.955× | 0.980× | ✓ |  |
| B | persistent_tile_scheduling | 0.975× | 1.016× | ✓ |  |
| C | persistent_tile_scheduling | 0.976× | 1.007× | ✓ |  |
| D | persistent_tile_scheduling | 0.322× | 0.208× | ✓ |  |
| E | add_triton_autotune | 0.995× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4450 ns | 4540 ns | +90 ns (+2.0%) | 4380 ns | -70 ns (-1.6%) | 4420 ns | -30 ns (-0.7%) | 21380 ns | +16930 ns (+380.4%) | 4420 ns | -30 ns (-0.7%) |
| SM Throughput (%) | 25.79 % | 27.90 % | +2.11 pp | 27.69 % | +1.90 pp | 27.79 % | +2.00 pp | 59.50 % | +33.71 pp | 25.92 % | +0.13 pp |
| Memory Throughput (%) | 11.38 % | 11.04 % | -0.34 pp | 11.30 % | -0.08 pp | 11.26 % | -0.12 pp | 23.16 % | +11.78 pp | 11.22 % | -0.16 pp |
| DRAM Throughput (%) | 7.04 % | 6.89 % | -0.15 pp | 7.09 % | +0.05 pp | 7.05 % | +0.01 pp | 1.45 % | -5.59 pp | 7.06 % | +0.02 pp |
| Achieved Occupancy (%) | 62.49 % | 65.60 % | +3.11 pp | 64.38 % | +1.89 pp | 64.45 % | +1.96 pp | 85.76 % | +23.27 pp | 64.42 % | +1.93 pp |
| Registers/Thread | 18.00 register/thread | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 29.00 register/thread | +11 | 18.00 register/thread | +0 |
| Waves/SM | 1.39 | 1.39 | +0.00 | 1.39 | +0.00 | 1.39 | +0.00 | 1.39 | +0.00 | 1.39 | +0.00 |
| No Eligible (%) | 45.96 % | 44.27 % | -1.69 pp | 46.73 % | +0.77 pp | 46.64 % | +0.68 pp | 25.10 % | -20.86 pp | 45.13 % | -0.83 pp |
| IPC Active | 1.89 inst/cycle | 2.12 inst/cycle | +0.23 | 2.15 inst/cycle | +0.26 | 2.16 inst/cycle | +0.27 | 2.97 inst/cycle | +1.08 | 2.06 inst/cycle | +0.17 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.208×)

Converting a standard pointwise Triton kernel to a persistent while-loop kernel (launching NUM_SMS=132 CTAs each iterating over tiles) caused a 4.8x slowdown, with duration increasing from 4.45 µs to 21.38 µs and executed instructions exploding from 1.43M to 9.38M (6.6x). The root cause is a grid/dispatch mismatch: the kernel_function still launched grid=(triton.cdiv(xnumel, 128),)=2940 blocks instead of grid=(132,), so 2940 persistent CTAs each ran the full while-loop over all tiles, processing the data ~22x more than intended. Despite improved L1 hit rate (30% → 78%) and L2 hit rate (59% → 85%) from the evict_last reuse, the massive over-computation from the incorrect grid completely dominated. The persistent kernel pattern requires the grid to be exactly NUM_SMS; if the grid remains at cdiv(xnumel, XBLOCK), each CTA will iterate through all tiles instead of its assigned stripe.

> **Applicable when**: Never beneficial in this form. Persistent while-loop kernels require the grid to be exactly NUM_SMS; if the dispatch grid is not simultaneously changed to (NUM_SMS,), the while-loop causes each CTA to process all tiles, resulting in massive instruction inflation proportional to grid_size/NUM_SMS.

> **Avoid when**: The kernel_function dispatch grid is not updated to (NUM_SMS,) when adding the persistent while-loop body. This is the exact failure mode: the while-loop was added to the kernel body but grid=(triton.cdiv(xnumel, 128),)=2940 was left unchanged, causing 2940 CTAs to each iterate over all 23 tiles instead of 1 tile each. Also avoid for short-duration kernels (<10 µs) where the overhead of the while-loop control flow and the risk of grid mismatch bugs outweigh any cache reuse benefit.

#### ❌ Plan B: `persistent_tile_scheduling` (0.208×)

Converting a standard pointwise Triton kernel to a persistent while-loop kernel (launching NUM_SMS=132 CTAs each iterating over tiles) caused a 4.8x slowdown, with duration increasing from 4.45 µs to 21.38 µs and executed instructions exploding from 1.43M to 9.38M (6.6x). The root cause is a grid/dispatch mismatch: the kernel_function still launched grid=(triton.cdiv(xnumel, 128),)=2940 blocks instead of grid=(132,), so 2940 persistent CTAs each ran the full while-loop over all tiles, processing the data ~22x more than intended. Despite improved L1 hit rate (30% → 78%) and L2 hit rate (59% → 85%) from the evict_last reuse, the massive over-computation from the incorrect grid completely dominated. The persistent kernel pattern requires the grid to be exactly NUM_SMS; if the grid remains at cdiv(xnumel, XBLOCK), each CTA will iterate through all tiles instead of its assigned stripe.

> **Applicable when**: Never beneficial in this form. Persistent while-loop kernels require the grid to be exactly NUM_SMS; if the dispatch grid is not simultaneously changed to (NUM_SMS,), the while-loop causes each CTA to process all tiles, resulting in massive instruction inflation proportional to grid_size/NUM_SMS.

> **Avoid when**: The kernel_function dispatch grid is not updated to (NUM_SMS,) when adding the persistent while-loop body. This is the exact failure mode: the while-loop was added to the kernel body but grid=(triton.cdiv(xnumel, 128),)=2940 was left unchanged, causing 2940 CTAs to each iterate over all 23 tiles instead of 1 tile each. Also avoid for short-duration kernels (<10 µs) where the overhead of the while-loop control flow and the risk of grid mismatch bugs outweigh any cache reuse benefit.

#### ❌ Plan C: `persistent_tile_scheduling` (0.208×)

Converting a standard pointwise Triton kernel to a persistent while-loop kernel (launching NUM_SMS=132 CTAs each iterating over tiles) caused a 4.8x slowdown, with duration increasing from 4.45 µs to 21.38 µs and executed instructions exploding from 1.43M to 9.38M (6.6x). The root cause is a grid/dispatch mismatch: the kernel_function still launched grid=(triton.cdiv(xnumel, 128),)=2940 blocks instead of grid=(132,), so 2940 persistent CTAs each ran the full while-loop over all tiles, processing the data ~22x more than intended. Despite improved L1 hit rate (30% → 78%) and L2 hit rate (59% → 85%) from the evict_last reuse, the massive over-computation from the incorrect grid completely dominated. The persistent kernel pattern requires the grid to be exactly NUM_SMS; if the grid remains at cdiv(xnumel, XBLOCK), each CTA will iterate through all tiles instead of its assigned stripe.

> **Applicable when**: Never beneficial in this form. Persistent while-loop kernels require the grid to be exactly NUM_SMS; if the dispatch grid is not simultaneously changed to (NUM_SMS,), the while-loop causes each CTA to process all tiles, resulting in massive instruction inflation proportional to grid_size/NUM_SMS.

> **Avoid when**: The kernel_function dispatch grid is not updated to (NUM_SMS,) when adding the persistent while-loop body. This is the exact failure mode: the while-loop was added to the kernel body but grid=(triton.cdiv(xnumel, 128),)=2940 was left unchanged, causing 2940 CTAs to each iterate over all 23 tiles instead of 1 tile each. Also avoid for short-duration kernels (<10 µs) where the overhead of the while-loop control flow and the risk of grid mismatch bugs outweigh any cache reuse benefit.

#### ❌ Plan D: `persistent_tile_scheduling` (0.208×)

Converting a standard pointwise Triton kernel to a persistent while-loop kernel (launching NUM_SMS=132 CTAs each iterating over tiles) caused a 4.8x slowdown, with duration increasing from 4.45 µs to 21.38 µs and executed instructions exploding from 1.43M to 9.38M (6.6x). The root cause is a grid/dispatch mismatch: the kernel_function still launched grid=(triton.cdiv(xnumel, 128),)=2940 blocks instead of grid=(132,), so 2940 persistent CTAs each ran the full while-loop over all tiles, processing the data ~22x more than intended. Despite improved L1 hit rate (30% → 78%) and L2 hit rate (59% → 85%) from the evict_last reuse, the massive over-computation from the incorrect grid completely dominated. The persistent kernel pattern requires the grid to be exactly NUM_SMS; if the grid remains at cdiv(xnumel, XBLOCK), each CTA will iterate through all tiles instead of its assigned stripe.

> **Applicable when**: Never beneficial in this form. Persistent while-loop kernels require the grid to be exactly NUM_SMS; if the dispatch grid is not simultaneously changed to (NUM_SMS,), the while-loop causes each CTA to process all tiles, resulting in massive instruction inflation proportional to grid_size/NUM_SMS.

> **Avoid when**: The kernel_function dispatch grid is not updated to (NUM_SMS,) when adding the persistent while-loop body. This is the exact failure mode: the while-loop was added to the kernel body but grid=(triton.cdiv(xnumel, 128),)=2940 was left unchanged, causing 2940 CTAs to each iterate over all 23 tiles instead of 1 tile each. Also avoid for short-duration kernels (<10 µs) where the overhead of the while-loop control flow and the risk of grid mismatch bugs outweigh any cache reuse benefit.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_multiply_add_to_fma_and_increase_num_warps | 0.958× | 0.965× | ✓ |  |
| B | replace_division_with_reciprocal_multiply_for_fma_fusion | 0.961× | 1.007× | ✓ |  |
| C | fix_autotune_grid_lambda_to_enable_xblock_and_num_warps_tuning | 0.965× | 1.007× | ✓ |  |
| D | fix_persistent_kernel_grid_mismatch_and_enable_fma | 0.963× | 0.987× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_multipl... | Δ | replace_division_... | Δ | fix_autotune_grid... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4450 ns | 4610 ns | +160 ns (+3.6%) | 4420 ns | -30 ns (-0.7%) | 4420 ns | -30 ns (-0.7%) | 4510 ns | +60 ns (+1.3%) |
| SM Throughput (%) | 25.79 % | 27.75 % | +1.96 pp | 27.81 % | +2.02 pp | 25.37 % | -0.42 pp | 26.01 % | +0.22 pp |
| Memory Throughput (%) | 11.38 % | 10.81 % | -0.57 pp | 11.26 % | -0.12 pp | 11.21 % | -0.17 pp | 11.08 % | -0.30 pp |
| DRAM Throughput (%) | 7.04 % | 6.78 % | -0.26 pp | 7.04 % | +0.00 pp | 7.06 % | +0.02 pp | 6.92 % | -0.12 pp |
| Achieved Occupancy (%) | 62.49 % | 64.96 % | +2.47 pp | 67.51 % | +5.02 pp | 62.80 % | +0.31 pp | 64.89 % | +2.40 pp |
| Registers/Thread | 18.00 register/thread | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 1.39 | 1.39 | +0.00 | 1.39 | +0.00 | 1.39 | +0.00 | 1.39 | +0.00 |
| No Eligible (%) | 45.96 % | 42.86 % | -3.10 pp | 44.27 % | -1.69 pp | 45.63 % | -0.33 pp | 46.18 % | +0.22 pp |
| IPC Active | 1.89 inst/cycle | 2.15 inst/cycle | +0.26 | 2.03 inst/cycle | +0.14 | 2.06 inst/cycle | +0.17 | 2.02 inst/cycle | +0.13 |

---

## Cumulative Progress

```
Baseline:  4450 ns  (1.000×)

Final best: 4450 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,33 +5,38 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 376320
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x0 = (xindex % 32)
-    x1 = ((xindex // 32) % 1176)
-    x2 = xindex // 37632
-    x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 768*((x1 % 49)) + 37632*x2), xmask)
-    tmp1 = tl.load(in_ptr1 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
-    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 768.0, tl.float32)
-    tmp5 = (tmp3 / tmp4)
-    tmp6 = tl.full([1], 1e-05, tl.float32)
-    tmp7 = tmp5 + tmp6
-    tmp8 = libdevice.rsqrt(tmp7)
-    tmp9 = tmp2 * tmp8
-    tmp11 = tmp9 * tmp10
-    tmp13 = tmp11 + tmp12
-    tl.store(out_ptr0 + (x3), tmp13, xmask)
-
-
+    sm_id = tl.program_id(0)
+    num_sms = tl.num_programs(0)
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    tile_id = sm_id
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x0 = (xindex % 32)
+        x1 = ((xindex // 32) % 1176)
+        x2 = xindex // 37632
+        x3 = xindex
+        tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 768*((x1 % 49)) + 37632*x2), xmask)
+        tmp1 = tl.load(in_ptr1 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
+        tmp3 = tl.load(in_ptr2 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
+        tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
+        tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
+        tmp2 = tmp0 - tmp1
+        tmp4 = tl.full([1], 768.0, tl.float32)
+        tmp5 = (tmp3 / tmp4)
+        tmp6 = tl.full([1], 1e-05, tl.float32)
+        tmp7 = tmp5 + tmp6
+        tmp8 = libdevice.rsqrt(tmp7)
+        tmp9 = tmp2 * tmp8
+        tmp11 = tmp9 * tmp10
+        tmp13 = tmp11 + tmp12
+        tl.store(out_ptr0 + (x3), tmp13, xmask)
+        tile_id += num_sms
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

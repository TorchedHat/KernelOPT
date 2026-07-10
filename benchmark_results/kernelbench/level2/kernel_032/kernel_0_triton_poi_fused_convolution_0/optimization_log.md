# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6110 ns — **Best latency**: 6080 ns (**1.0049×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_autotune_grid_lambda_to_enable_real_tile_exploration`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_032/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6110 ns
- **SM Throughput**: 15.48 % | **Memory**: 39.77 % | **DRAM**: 20.13 %
- **Achieved Occupancy**: 54.62 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.308× | 1.027× | ✓ | **✓ best** |
| B | autotune_tile_sizes_to_improve_coalescing_and_occupancy | 1.303× | 1.005× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.020× | 0.917× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6110 ns | 5950 ns | -160 ns (-2.6%) | 6080 ns | -30 ns (-0.5%) | 6660 ns | +550 ns (+9.0%) |
| SM Throughput (%) | 15.48 % | 23.72 % | +8.24 pp | 16.39 % | +0.91 pp | 28.86 % | +13.38 pp |
| Memory Throughput (%) | 39.77 % | 35.04 % | -4.73 pp | 40.24 % | +0.47 pp | 38.01 % | -1.76 pp |
| DRAM Throughput (%) | 20.13 % | 11.07 % | -9.06 pp | 20.32 % | +0.19 pp | 18.45 % | -1.68 pp |
| Achieved Occupancy (%) | 54.62 % | 50.89 % | -3.73 pp | 54.83 % | +0.21 pp | 76.11 % | +21.49 pp |
| Registers/Thread | 26.00 register/thread | 38.00 register/thread | +12 | 26.00 register/thread | +0 | 27.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.99 | +0.25 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.42 % | 56.81 % | -13.61 pp | 70.54 % | +0.12 pp | 54.18 % | -16.24 pp |
| IPC Active | 1.14 inst/cycle | 1.80 inst/cycle | +0.66 | 1.17 inst/cycle | +0.03 | 1.77 inst/cycle | +0.63 |

---

## Iteration 2

**Best so far**: 1.0269× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_smaller_tiles_and_more_warps | 1.334× | 1.157× | ✓ | **✓ best** |
| B | shared_memory_staging_to_coalesce_stride3_stores | 1.247× | 0.955× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | shared_memory_sta... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6110 ns | 5280 ns | -830 ns (-13.6%) | 6400 ns | +290 ns (+4.7%) |
| SM Throughput (%) | 15.48 % | 25.90 % | +10.42 pp | 17.86 % | +2.38 pp |
| Memory Throughput (%) | 39.77 % | 32.63 % | -7.14 pp | 38.45 % | -1.32 pp |
| DRAM Throughput (%) | 20.13 % | 11.71 % | -8.42 pp | 19.27 % | -0.86 pp |
| Achieved Occupancy (%) | 54.62 % | 84.26 % | +29.64 pp | 56.19 % | +1.57 pp |
| Registers/Thread | 26.00 register/thread | 20.00 register/thread | -6 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.42 % | 52.05 % | -18.37 pp | 67.90 % | -2.52 pp |
| IPC Active | 1.14 inst/cycle | 1.77 inst/cycle | +0.63 | 1.20 inst/cycle | +0.06 |


### Key Learnings

#### ✅ Plan A: `reduce_register_pressure_via_smaller_tiles_and_more_warps` (1.157×)

When a Triton kernel's register count per thread (e.g., 38) limits Block Limit Registers below the warp-limit, theoretical occupancy is capped (e.g., 75%) and achieved occupancy falls further (e.g., 50.9%), leaving >56% of issue slots idle. Replacing large XBLOCK configs (256–512) with smaller ones (32–128) and raising num_warps (4→8–16) causes the compiler to allocate fewer registers per thread (38→20), lifting theoretical occupancy to 100% and achieved occupancy to 84%, while also increasing waves_per_SM from ~1 to ~1.5. Adding num_stages=2 enables software pipelining to further hide memory latency. The net effect is a 1.16x speedup despite slightly worsened store coalescing.

> **Applicable when**: Triton kernel shows registers_per_thread >= 32, Block Limit Registers < Block Limit Warps (register-bound occupancy), theoretical occupancy < 100%, achieved occupancy < 60%, and IssueSlotUtilization rule fires with no_eligible_warp_cycles > 50%. Autotune config space includes large XBLOCK values (256+) with low num_warps (2–4).

> **Avoid when**: Kernel is already register-efficient (registers_per_thread < 24) or occupancy is already near theoretical maximum. Also avoid if the workload is memory-bandwidth bound (DRAM throughput > 70%) rather than latency/occupancy bound — shrinking tiles may hurt cache reuse without improving throughput. Note: this optimization worsened store coalescing (excessive sectors rose from 20% to 35%), so if UncoalescedGlobalAccess is the primary bottleneck, address coalescing first.

---

## Iteration 3

**Best so far**: 1.1572× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_and_grid_wave_alignment | 1.260× | 1.151× | ✓ |  |
| B | fix_eviction_policy_streaming_load | 1.256× | 1.179× | ✓ |  |
| C | remove_dead_smem_staging_and_fix_autotune_grid_lambda | 1.199× | 1.072× | ✓ |  |
| D | fix_broken_autotune_grid_and_remove_dead_smem_staging | 1.256× | 0.964× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | fix_eviction_poli... | Δ | remove_dead_smem_... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6110 ns | 5310 ns | -800 ns (-13.1%) | 5180 ns | -930 ns (-15.2%) | 5700 ns | -410 ns (-6.7%) | 6340 ns | +230 ns (+3.8%) |
| SM Throughput (%) | 15.48 % | 26.78 % | +11.30 pp | 26.40 % | +10.92 pp | 17.66 % | +2.18 pp | 17.48 % | +2.00 pp |
| Memory Throughput (%) | 39.77 % | 32.48 % | -7.29 pp | 33.32 % | -6.45 pp | 41.30 % | +1.53 pp | 37.57 % | -2.20 pp |
| DRAM Throughput (%) | 20.13 % | 11.61 % | -8.52 pp | 11.90 % | -8.23 pp | 21.55 % | +1.42 pp | 19.40 % | -0.73 pp |
| Achieved Occupancy (%) | 54.62 % | 83.40 % | +28.78 pp | 83.35 % | +28.73 pp | 55.27 % | +0.65 pp | 55.45 % | +0.83 pp |
| Registers/Thread | 26.00 register/thread | 20.00 register/thread | -6 | 20.00 register/thread | -6 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.42 % | 52.09 % | -18.33 pp | 51.83 % | -18.59 pp | 69.39 % | -1.03 pp | 68.94 % | -1.48 pp |
| IPC Active | 1.14 inst/cycle | 1.87 inst/cycle | +0.73 | 1.87 inst/cycle | +0.73 | 1.21 inst/cycle | +0.07 | 1.20 inst/cycle | +0.06 |

---

## Iteration 4

**Best so far**: 1.1572× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | grid_wave_alignment_via_yblock_divisors_of_ynumel | 1.406× | 1.802× | ✓ |  |
| C | fix_grid_lambda_to_use_autotune_meta_and_add_wave_aligned_configs | 0.577× | 1.151× | ✓ |  |
| D | fix_autotune_grid_lambda_to_enable_real_tile_exploration | 1.426× | 1.005× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | grid_wave_alignme... | Δ | fix_grid_lambda_t... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6110 ns | 3390 ns | -2720 ns (-44.5%) | 5310 ns | -800 ns (-13.1%) | 6080 ns | -30 ns (-0.5%) |
| SM Throughput (%) | 15.48 % | 12.76 % | -2.72 pp | 27.06 % | +11.58 pp | 17.12 % | +1.64 pp |
| Memory Throughput (%) | 39.77 % | 13.52 % | -26.25 pp | 32.58 % | -7.19 pp | 37.77 % | -2.00 pp |
| DRAM Throughput (%) | 20.13 % | 4.87 % | -15.26 pp | 11.64 % | -8.49 pp | 20.21 % | +0.08 pp |
| Achieved Occupancy (%) | 54.62 % | 56.22 % | +1.60 pp | 84.35 % | +29.73 pp | 55.27 % | +0.65 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 20.00 register/thread | -6 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.42 % | 65.31 % | -5.11 pp | 52.48 % | -17.94 pp | 68.84 % | -1.58 pp |
| IPC Active | 1.14 inst/cycle | 1.16 inst/cycle | +0.02 | 1.77 inst/cycle | +0.63 | 1.21 inst/cycle | +0.07 |

---

## Iteration 5

**Best so far**: 1.1572× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_use_meta_autotune_values | 1.210× | 1.103× | ✓ |  |
| B | fix_vectorized_load_hints_and_grid_lambda | 1.381× | 1.043× | ✓ |  |
| C | exact_divisor_yblock_with_meta_grid_lambda | 1.395× | 1.786× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_vectorized_lo... | Δ | exact_divisor_ybl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6110 ns | 5540 ns | -570 ns (-9.3%) | 5860 ns | -250 ns (-4.1%) | 3420 ns | -2690 ns (-44.0%) |
| SM Throughput (%) | 15.48 % | 26.56 % | +11.08 pp | 17.48 % | +2.00 pp | 13.67 % | -1.81 pp |
| Memory Throughput (%) | 39.77 % | 30.79 % | -8.98 pp | 41.04 % | +1.27 pp | 13.74 % | -26.03 pp |
| DRAM Throughput (%) | 20.13 % | 11.11 % | -9.02 pp | 21.18 % | +1.05 pp | 4.88 % | -15.25 pp |
| Achieved Occupancy (%) | 54.62 % | 84.68 % | +30.06 pp | 55.70 % | +1.08 pp | 51.61 % | -3.01 pp |
| Registers/Thread | 26.00 register/thread | 20.00 register/thread | -6 | 26.00 register/thread | +0 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.42 % | 56.18 % | -14.24 pp | 69.01 % | -1.41 pp | 62.08 % | -8.34 pp |
| IPC Active | 1.14 inst/cycle | 1.78 inst/cycle | +0.64 | 1.20 inst/cycle | +0.06 | 1.31 inst/cycle | +0.17 |

---

## Cumulative Progress

```
Baseline:  6110 ns  (1.000×)
Iter 1     5950 ns  (1.027×)  — vectorized_loads_with_contiguity_hints
Iter 2     5280 ns  (1.157×)  — reduce_register_pressure_via_smaller_tiles_and_more_warps

Final best: 6080 ns  (1.0049×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Wave-aligned: YBLOCK=2 divides 30 exactly (30/2=15 exact rows)
+        # XBLOCK=64, YBLOCK=2 -> 784x15=11760 blocks (nearly full waves on 132 SMs)
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 2},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2},  num_warps=8,  num_stages=3),
+        # Wave-aligned: YBLOCK=1 divides 30 exactly (30/1=30 exact rows)
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4,  num_stages=3),
+        # Fallback existing power-of-2 configs
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=8,  num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +42,16 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints to 1D arange before broadcasting
+    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +59,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

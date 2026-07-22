# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5860 ns — **Best latency**: 4350 ns (**1.3471×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`grid_wave_alignment_via_autotune_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5860 ns
- **SM Throughput**: 15.85 % | **Memory**: 42.58 % | **DRAM**: 21.11 %
- **Achieved Occupancy**: 55.16 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 71.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_hide_l1tex_latency | 1.393× | 0.930× | ✓ | **✓ best** |
| C | memory_coalescing_via_autotune_tile_swap | 1.359× | 1.665× | ✓ |  |
| D | autotune_tile_sizes_for_underutilized_grid | 1.385× | 1.454× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 6300 ns | +440 ns (+7.5%) | 3520 ns | -2340 ns (-39.9%) | 4030 ns | -1830 ns (-31.2%) |
| SM Throughput (%) | 15.85 % | 16.17 % | +0.32 pp | 15.73 % | -0.12 pp | 19.64 % | +3.79 pp |
| Memory Throughput (%) | 42.58 % | 39.81 % | -2.77 pp | 19.77 % | -22.81 pp | 22.67 % | -19.91 pp |
| DRAM Throughput (%) | 21.11 % | 19.57 % | -1.54 pp | 9.32 % | -11.79 pp | 4.09 % | -17.02 pp |
| Achieved Occupancy (%) | 55.16 % | 54.88 % | -0.28 pp | 55.25 % | +0.09 pp | 48.97 % | -6.19 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 16.00 register/thread | -10 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.53 % | 72.22 % | +0.69 pp | 61.93 % | -9.60 pp | 52.06 % | -19.47 pp |
| IPC Active | 1.13 inst/cycle | 1.14 inst/cycle | +0.01 | 1.44 inst/cycle | +0.31 | 1.83 inst/cycle | +0.70 |


### Key Learnings

#### ✅ Plan D: `autotune_tile_sizes_for_underutilized_grid` (1.454×)

When a Triton kernel runs at <1.0 waves per SM (here 0.74), expanding the autotune config space to include larger block sizes (e.g., XBLOCK=256 instead of 128) doubles the waves per SM (0.74→1.48), substantially improves scheduler eligibility (Eligible Warps Per Scheduler: 1.31→3.83), and cuts No Eligible cycles from 61% to 40%. This drove Issue Slots Busy from 39% to 62% and improved L2 Hit Rate from 64% to 76%, yielding a 1.45x wall-time speedup despite NCU reporting a slightly longer per-kernel duration (3.74→4.03 µs). The key causal chain is: larger block size → more warps per SM → more eligible warps per scheduler → fewer stall cycles → higher throughput.

> **Applicable when**: Triton kernel with waves_per_sm < 1.0; IssueSlotUtilization triggered with No Eligible > 55%; Eligible Warps Per Scheduler < 2.0; autotune config space does not include configs that produce block_size=256 or larger; L2 Hit Rate below 70% suggesting larger tiles improve reuse

> **Avoid when**: Kernel already runs at ≥2.0 waves per SM; register pressure is near the per-SM limit (>30 regs/thread) and larger block sizes would further reduce occupancy below wave-filling threshold; shared memory per block is high and larger tiles would exceed SM shared memory limits; XBLOCK*YBLOCK product would exceed 1024

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_multiple_of_hints | 1.362× | 1.651× | ✓ |  |
| B | vectorized_loads_via_xindex_contiguity_hints | 1.380× | 1.338× | ✓ |  |
| C | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.335× | 0.954× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 3550 ns | -2310 ns (-39.4%) | 4380 ns | -1480 ns (-25.3%) | 6140 ns | +280 ns (+4.8%) |
| SM Throughput (%) | 15.85 % | 16.10 % | +0.25 pp | 18.73 % | +2.88 pp | 16.35 % | +0.50 pp |
| Memory Throughput (%) | 42.58 % | 19.83 % | -22.75 pp | 21.62 % | -20.96 pp | 40.86 % | -1.72 pp |
| DRAM Throughput (%) | 21.11 % | 9.28 % | -11.83 pp | 3.77 % | -17.34 pp | 20.09 % | -1.02 pp |
| Achieved Occupancy (%) | 55.16 % | 55.44 % | +0.28 pp | 49.43 % | -5.73 pp | 57.49 % | +2.33 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 25.00 register/thread | -1 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.53 % | 60.83 % | -10.70 pp | 56.97 % | -14.56 pp | 71.49 % | -0.04 pp |
| IPC Active | 1.13 inst/cycle | 1.47 inst/cycle | +0.34 | 1.84 inst/cycle | +0.71 | 0.97 inst/cycle | -0.16 |


### Key Learnings

#### ❌ Plan A: `vectorized_loads_via_contiguous_multiple_of_hints` (1.651×)

Adding tl.max_contiguous and tl.multiple_of vectorization hints to the load index of a 2D Triton kernel did not improve performance when the bottleneck is a strided store (stride-3 in the fast axis). Despite the load being contiguous and the hints being syntactically correct, the kernel duration increased slightly (3.52→3.55 µs) and Warp Cycles Per Issued Instruction worsened (23.31→27.1). The uncoalesced access count (25088 excess sectors) was unchanged, confirming the store—not the load—dominates the memory bottleneck. The eligible warps per scheduler improved marginally (1.31→1.62) but this did not translate to a wall-time gain, likely because the store-side stalls remain the binding constraint.

> **Applicable when**: Never — this strategy did not produce improvement even when the load is provably contiguous and hints are syntactically correct.

> **Avoid when**: When the kernel has a strided store (stride > 1 in the fast thread axis) that dominates the uncoalesced access count; the store bottleneck will not be relieved by load-side vectorization hints, and the hints may slightly increase warp stall cycles due to compiler code-gen changes. Do not apply tl.max_contiguous/tl.multiple_of hints to 2D index tensors when the store pattern is the primary source of uncoalesced accesses.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_enable_autotune_large_tiles | 1.362× | 1.567× | ✓ |  |
| B | software_pipelining_via_num_stages_in_autotune_configs | 1.392× | 1.317× | ✓ |  |
| C | retarget_autotune_configs_for_strided_store_coalescing | 1.445× | 1.052× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | software_pipelini... | Δ | retarget_autotune... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 3740 ns | -2120 ns (-36.2%) | 4450 ns | -1410 ns (-24.1%) | 5570 ns | -290 ns (-4.9%) |
| SM Throughput (%) | 15.85 % | 15.58 % | -0.27 pp | 18.96 % | +3.11 pp | 27.34 % | +11.49 pp |
| Memory Throughput (%) | 42.58 % | 18.91 % | -23.67 pp | 21.87 % | -20.71 pp | 29.97 % | -12.61 pp |
| DRAM Throughput (%) | 21.11 % | 8.80 % | -12.31 pp | 3.71 % | -17.40 pp | 2.97 % | -18.14 pp |
| Achieved Occupancy (%) | 55.16 % | 56.34 % | +1.18 pp | 49.22 % | -5.94 pp | 73.55 % | +18.39 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 25.00 register/thread | -1 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.53 % | 61.04 % | -10.49 pp | 51.03 % | -20.50 pp | 37.98 % | -33.55 pp |
| IPC Active | 1.13 inst/cycle | 1.48 inst/cycle | +0.35 | 1.82 inst/cycle | +0.69 | 2.42 inst/cycle | +1.29 |

---

## Iteration 4

**Best so far**: 1.0521× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_contiguity_hints_to_1d_arange_before_broadcast_plus_evict_first | 1.358× | 1.651× | ✓ |  |
| B | streaming_eviction_policy_for_layout_permutation | 1.377× | 1.429× | ✓ |  |
| C | fix_static_grid_and_add_yblock3_aligned_configs | 1.392× | 1.454× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_contiguity_hi... | Δ | streaming_evictio... | Δ | fix_static_grid_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 3550 ns | -2310 ns (-39.4%) | 4100 ns | -1760 ns (-30.0%) | 4030 ns | -1830 ns (-31.2%) |
| SM Throughput (%) | 15.85 % | 15.49 % | -0.36 pp | 20.37 % | +4.52 pp | 26.89 % | +11.04 pp |
| Memory Throughput (%) | 42.58 % | 19.68 % | -22.90 pp | 22.66 % | -19.92 pp | 21.28 % | -21.30 pp |
| DRAM Throughput (%) | 21.11 % | 9.28 % | -11.83 pp | 4.05 % | -17.06 pp | 4.10 % | -17.01 pp |
| Achieved Occupancy (%) | 55.16 % | 55.59 % | +0.43 pp | 51.33 % | -3.83 pp | 69.52 % | +14.36 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 25.00 register/thread | -1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.53 % | 61.99 % | -9.54 pp | 55.17 % | -16.36 pp | 39.67 % | -31.86 pp |
| IPC Active | 1.13 inst/cycle | 1.43 inst/cycle | +0.30 | 1.71 inst/cycle | +0.58 | 2.42 inst/cycle | +1.29 |

---

## Iteration 5

**Best so far**: 1.0521× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_on_store_for_streaming_permutation | 1.350× | 1.665× | ✓ |  |
| B | grid_wave_alignment_via_autotune_configs | 1.462× | 1.347× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | evict_first_on_st... | Δ | grid_wave_alignme... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 3520 ns | -2340 ns (-39.9%) | 4350 ns | -1510 ns (-25.8%) |
| SM Throughput (%) | 15.85 % | 15.26 % | -0.59 pp | 27.31 % | +11.46 pp |
| Memory Throughput (%) | 42.58 % | 20.08 % | -22.50 pp | 21.74 % | -20.84 pp |
| DRAM Throughput (%) | 21.11 % | 9.42 % | -11.69 pp | 3.78 % | -17.33 pp |
| Achieved Occupancy (%) | 55.16 % | 56.96 % | +1.80 pp | 69.42 % | +14.26 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.53 % | 62.46 % | -9.07 pp | 37.04 % | -34.49 pp |
| IPC Active | 1.13 inst/cycle | 1.44 inst/cycle | +0.31 | 2.41 inst/cycle | +1.28 |

---

## Cumulative Progress

```
Baseline:  5860 ns  (1.000×)
Iter 3     5570 ns  (1.052×)  — retarget_autotune_configs_for_strided_store_coalescing
Iter 5     4350 ns  (1.347×)  — grid_wave_alignment_via_autotune_configs

Final best: 4350 ns  (1.3471×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4,  num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 2}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 4}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,   'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 32,   'YBLOCK': 1}, num_warps=4,  num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +39,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +48,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
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

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_67` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6620 ns — **Best latency**: 6140 ns (**1.0782×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`reduce_register_pressure_via_xblock_right_sizing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_67`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_67_triton_poi_fused_convolution_6/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6400,1,1), Waves/SM: 3.03
- **Registers/thread**: 28
- **Duration**: 6620 ns
- **SM Throughput**: 50.70 % | **Memory**: 48.77 % | **DRAM**: 5.72 %
- **Achieved Occupancy**: 71.43 %
- **IPC Active**: 3.06 inst/cycle | **No Eligible**: 25.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_dimensions_and_eliminate_partial_wave | 1.087× | 1.078× | ✓ | **✓ best** |
| B | right_size_tile_blocks_and_add_autotune | 1.029× | 1.009× | ✓ |  |
| C | fix_tile_shape_for_coalesced_access_and_full_occupancy | 1.027× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_dimensio... | Δ | right_size_tile_b... | Δ | fix_tile_shape_fo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6620 ns | 6140 ns | -480 ns (-7.3%) | 6560 ns | -60 ns (-0.9%) | 6590 ns | -30 ns (-0.5%) |
| SM Throughput (%) | 50.70 % | 44.02 % | -6.68 pp | 29.42 % | -21.28 pp | 29.34 % | -21.36 pp |
| Memory Throughput (%) | 48.77 % | 22.17 % | -26.60 pp | 27.70 % | -21.07 pp | 27.64 % | -21.13 pp |
| DRAM Throughput (%) | 5.72 % | 6.18 % | +0.46 pp | 5.74 % | +0.02 pp | 5.74 % | +0.02 pp |
| Achieved Occupancy (%) | 71.43 % | 53.22 % | -18.21 pp | 55.75 % | -15.68 pp | 54.33 % | -17.10 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 30.00 register/thread | +2 | 30.00 register/thread | +2 |
| Waves/SM | 3.03 | 3.03 | +0.00 | 1.52 | -1.51 | 1.52 | -1.51 |
| No Eligible (%) | 25.24 % | 20.34 % | -4.90 pp | 35.75 % | +10.51 pp | 31.84 % | +6.60 pp |
| IPC Active | 3.06 inst/cycle | 3.14 inst/cycle | +0.08 | 2.41 inst/cycle | -0.65 | 2.48 inst/cycle | -0.58 |

---

## Iteration 2

**Best so far**: 1.0782× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.000× | 0.000× | ✓ |  |
| B | persistent_tile_scheduling | 0.002× | 0.002× | ✓ |  |
| C | persistent_tile_scheduling | 0.002× | 0.002× | ✓ |  |
| D | add_triton_autotune | 0.128× | 0.093× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6620 ns | 28160000 ns | +28153380 ns (+425277.6%) | 4070000 ns | +4063380 ns (+61380.4%) | 4220000 ns | +4213380 ns (+63646.2%) | 70850 ns | +64230 ns (+970.2%) |
| SM Throughput (%) | 50.70 % | 86.68 % | +35.98 pp | 59.96 % | +9.26 pp | 52.67 % | +1.97 pp | 58.94 % | +8.24 pp |
| Memory Throughput (%) | 48.77 % | 85.98 % | +37.21 pp | 80.91 % | +32.14 pp | 83.04 % | +34.27 pp | 77.12 % | +28.35 pp |
| DRAM Throughput (%) | 5.72 % | 0.01 % | -5.71 pp | 0.01 % | -5.71 pp | 0.01 % | -5.71 pp | 0.54 % | -5.18 pp |
| Achieved Occupancy (%) | 71.43 % | 88.29 % | +16.86 pp | 53.32 % | -18.11 pp | 82.43 % | +11.00 pp | 17.78 % | -53.65 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 56.00 register/thread | +28 | 32.00 register/thread | +4 | 164.00 register/thread | +136 |
| Waves/SM | 3.03 | 3.03 | +0.00 | 5.39 | +2.36 | 1.52 | -1.51 | 16.16 | +13.13 |
| No Eligible (%) | 25.24 % | 12.38 % | -12.86 pp | 42.58 % | +17.34 pp | 45.64 % | +20.40 pp | 38.95 % | +13.71 pp |
| IPC Active | 3.06 inst/cycle | 3.51 inst/cycle | +0.45 | 2.30 inst/cycle | -0.76 | 2.17 inst/cycle | -0.89 | 2.44 inst/cycle | -0.62 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.002×)

Converting a short-lived 2D-grid Triton kernel into a persistent 1D-grid kernel with an inner tile loop caused a ~625x slowdown (6.14 µs → 4.07 ms). The persistent loop forced each CTA to iterate over many tiles sequentially, exploding executed instructions from 2.15M to 1.82B — an 848x increase. Register pressure doubled (28 → 56 registers/thread), reducing theoretical occupancy from 100% to 56.25%, and the L2 became the bottleneck (L2 Hit Rate 99.92%, Mem Busy 80.91%) as all data was recycled through L2 repeatedly. The partial-wave tail overhead flagged by LaunchConfiguration (≤25% waste) was negligible compared to the cost of serializing tile work inside a loop.

> **Applicable when**: Never: persistent tile scheduling is an anti-pattern for short-lived element-wise or transpose kernels where the total work is small (< ~1M elements) and the baseline kernel already completes in single-digit microseconds.

> **Avoid when**: Any kernel with short baseline duration (< ~50 µs). Any kernel where total_tiles >> num_CTAs, causing each CTA to loop hundreds or thousands of times — this serializes work that was previously parallel. Any kernel where the LaunchConfiguration partial-wave overhead is small in absolute terms (e.g., 25% of 6 µs = 1.5 µs). Persistent kernels are only beneficial when kernel launch overhead dominates (e.g., many tiny kernels launched in sequence) or when inter-tile data reuse in shared memory is exploited.

#### ❌ Plan B: `persistent_tile_scheduling` (0.002×)

Converting a short-lived 2D-grid Triton kernel into a persistent 1D-grid kernel with an inner tile loop caused a ~625x slowdown (6.14 µs → 4.07 ms). The persistent loop forced each CTA to iterate over many tiles sequentially, exploding executed instructions from 2.15M to 1.82B — an 848x increase. Register pressure doubled (28 → 56 registers/thread), reducing theoretical occupancy from 100% to 56.25%, and the L2 became the bottleneck (L2 Hit Rate 99.92%, Mem Busy 80.91%) as all data was recycled through L2 repeatedly. The partial-wave tail overhead flagged by LaunchConfiguration (≤25% waste) was negligible compared to the cost of serializing tile work inside a loop.

> **Applicable when**: Never: persistent tile scheduling is an anti-pattern for short-lived element-wise or transpose kernels where the total work is small (< ~1M elements) and the baseline kernel already completes in single-digit microseconds.

> **Avoid when**: Any kernel with short baseline duration (< ~50 µs). Any kernel where total_tiles >> num_CTAs, causing each CTA to loop hundreds or thousands of times — this serializes work that was previously parallel. Any kernel where the LaunchConfiguration partial-wave overhead is small in absolute terms (e.g., 25% of 6 µs = 1.5 µs). Persistent kernels are only beneficial when kernel launch overhead dominates (e.g., many tiny kernels launched in sequence) or when inter-tile data reuse in shared memory is exploited.

#### ❌ Plan C: `persistent_tile_scheduling` (0.002×)

Converting a short-lived 2D-grid Triton kernel into a persistent 1D-grid kernel with an inner tile loop caused a ~625x slowdown (6.14 µs → 4.07 ms). The persistent loop forced each CTA to iterate over many tiles sequentially, exploding executed instructions from 2.15M to 1.82B — an 848x increase. Register pressure doubled (28 → 56 registers/thread), reducing theoretical occupancy from 100% to 56.25%, and the L2 became the bottleneck (L2 Hit Rate 99.92%, Mem Busy 80.91%) as all data was recycled through L2 repeatedly. The partial-wave tail overhead flagged by LaunchConfiguration (≤25% waste) was negligible compared to the cost of serializing tile work inside a loop.

> **Applicable when**: Never: persistent tile scheduling is an anti-pattern for short-lived element-wise or transpose kernels where the total work is small (< ~1M elements) and the baseline kernel already completes in single-digit microseconds.

> **Avoid when**: Any kernel with short baseline duration (< ~50 µs). Any kernel where total_tiles >> num_CTAs, causing each CTA to loop hundreds or thousands of times — this serializes work that was previously parallel. Any kernel where the LaunchConfiguration partial-wave overhead is small in absolute terms (e.g., 25% of 6 µs = 1.5 µs). Persistent kernels are only beneficial when kernel launch overhead dominates (e.g., many tiny kernels launched in sequence) or when inter-tile data reuse in shared memory is exploited.

---

## Iteration 3

**Best so far**: 1.0782× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_xblock_right_sizing | 1.072× | 1.078× | ✓ |  |
| B | revert_persistent_loop_to_direct_2d_grid_kernel | 1.037× | 1.078× | ✓ |  |
| C | revert_persistent_loop_to_2d_grid_with_correct_autotune_launch | 1.026× | 0.958× | ✓ |  |
| D | revert_persistent_loop_to_2d_grid_with_correct_tile_dims | 1.054× | 1.078× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | revert_persistent... | Δ | revert_persistent... | Δ | revert_persistent... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6620 ns | 6140 ns | -480 ns (-7.3%) | 6140 ns | -480 ns (-7.3%) | 6910 ns | +290 ns (+4.4%) | 6140 ns | -480 ns (-7.3%) |
| SM Throughput (%) | 50.70 % | 44.86 % | -5.84 pp | 44.21 % | -6.49 pp | 28.29 % | -22.41 pp | 44.98 % | -5.72 pp |
| Memory Throughput (%) | 48.77 % | 22.11 % | -26.66 pp | 22.30 % | -26.47 pp | 26.63 % | -22.14 pp | 22.17 % | -26.60 pp |
| DRAM Throughput (%) | 5.72 % | 6.16 % | +0.44 pp | 6.15 % | +0.43 pp | 5.47 % | -0.25 pp | 6.15 % | +0.43 pp |
| Achieved Occupancy (%) | 71.43 % | 56.80 % | -14.63 pp | 56.43 % | -15.00 pp | 54.05 % | -17.38 pp | 55.87 % | -15.56 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 30.00 register/thread | +2 | 28.00 register/thread | +0 |
| Waves/SM | 3.03 | 3.03 | +0.00 | 3.03 | +0.00 | 1.52 | -1.51 | 3.03 | +0.00 |
| No Eligible (%) | 25.24 % | 19.39 % | -5.85 pp | 20.03 % | -5.21 pp | 38.18 % | +12.94 pp | 20.12 % | -5.12 pp |
| IPC Active | 3.06 inst/cycle | 3.15 inst/cycle | +0.09 | 3.08 inst/cycle | +0.02 | 2.52 inst/cycle | -0.54 | 3.15 inst/cycle | +0.09 |


### Key Learnings

#### ✅ Plan B: `revert_persistent_loop_to_direct_2d_grid_kernel` (1.078×)

A Triton kernel using a persistent for-loop (`for tile_id in range(cta_id, total_tiles, num_ctas)`) with a 1D grid serialized tile work that was previously parallel, inflating executed instructions from 2.15M to 32.9M (15x), registers per thread from 28 to 164 (loop-carried variables kept live), and dynamic shared memory from 0 to 16 KB — collapsing theoretical occupancy from 100% to 18.75% (Block Limit Registers: 16 → 3 blocks/SM). Removing the persistent loop and restoring a direct 2D grid (each CTA handles exactly one tile via `tl.program_id(0)` and `tl.program_id(1)`) reversed all these effects: instructions dropped 15x, registers dropped from 164 to 28, theoretical occupancy recovered to 100%, and duration fell from 70.85 µs to 6.14 µs (~11.5x). The autotune config set was also updated to include small XBLOCK values (1, 8) appropriate for the narrow x-dimension (xnumel=9).

> **Applicable when**: Kernel has a persistent for-loop (`range(cta_id, total_tiles, num_ctas)`) over a 1D grid where total_tiles >> num_CTAs, causing each CTA to process many tiles sequentially. Symptoms: registers per thread >> 32 (loop-carried variables), theoretical occupancy << 50% (Block Limit Registers is the binding constraint), executed instructions >> expected (proportional to tiles-per-CTA), dynamic shared memory allocated despite no explicit shared memory use (compiler spill), and duration much longer than a simple element-wise kernel should take.

> **Avoid when**: Do not remove persistent loops when the kernel genuinely benefits from inter-tile data reuse in shared memory (e.g., matmul), or when the number of tiles is much smaller than num_SMs (each CTA does <1 tile on average — the loop body never executes). Also do not apply when the kernel is already a direct 2D grid.

---

## Cumulative Progress

```
Baseline:  6620 ns  (1.000×)
Iter 1     6140 ns  (1.078×)  — fix_tile_dimensions_and_eliminate_partial_wave

Final best: 6140 ns  (1.0782×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 32},  num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 16}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_67(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 51200
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +29,8 @@

     y3 = yindex
     y0 = (yindex % 160)
     y1 = yindex // 160
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +38,7 @@

     ynumel = 51200
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_67[grid](arg_0, arg_1, 51200, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_67[grid](arg_0, arg_1, 51200, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

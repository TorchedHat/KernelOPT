# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7140 ns — **Best latency**: 7200 ns (**0.9917×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_225_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7168,1,1), Waves/SM: 3.39
- **Registers/thread**: 28
- **Duration**: 7140 ns
- **SM Throughput**: 54.81 % | **Memory**: 51.84 % | **DRAM**: 5.94 %
- **Achieved Occupancy**: 70.71 %
- **IPC Active**: 3.10 inst/cycle | **No Eligible**: 19.68 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| E | add_triton_autotune | 1.002× | 0.992× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7140 ns | 7200 ns | +60 ns (+0.8%) |
| SM Throughput (%) | 54.81 % | 54.01 % | -0.80 pp |
| Memory Throughput (%) | 51.84 % | 51.02 % | -0.82 pp |
| DRAM Throughput (%) | 5.94 % | 5.89 % | -0.05 pp |
| Achieved Occupancy (%) | 70.71 % | 72.45 % | +1.74 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 |
| Waves/SM | 3.39 | 3.39 | +0.00 |
| No Eligible (%) | 19.68 % | 24.35 % | +4.67 pp |
| IPC Active | 3.10 inst/cycle | 3.12 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.000× | 0.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7140 ns | 27770000 ns | +27762860 ns (+388835.6%) |
| SM Throughput (%) | 54.81 % | 75.79 % | +20.98 pp |
| Memory Throughput (%) | 51.84 % | 91.14 % | +39.30 pp |
| DRAM Throughput (%) | 5.94 % | 0.01 % | -5.93 pp |
| Achieved Occupancy (%) | 70.71 % | 91.20 % | +20.49 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 |
| Waves/SM | 3.39 | 3.39 | +0.00 |
| No Eligible (%) | 19.68 % | 23.41 % | +3.73 pp |
| IPC Active | 3.10 inst/cycle | 3.06 inst/cycle | -0.04 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.000×)

Implementing a persistent kernel with a `tl.range(pid, total_tiles, num_ctas)` stride loop caused a ~5500x increase in executed instructions (3M → 16.6B) and a ~3860x increase in kernel duration (7.2 µs → 27.77 ms). The root cause is that the diff left the launch grid unchanged as `(cdiv(9,128), cdiv(57344,8)) = (1, 7168)` blocks, so 7168 CTAs each iterated over `ceil(896/7168)` ≈ 1 tile but the stride loop `tl.range(pid, total_tiles, num_ctas)` with num_ctas=7168 and total_tiles=896 caused each CTA to execute the loop body 0 or 1 times — yet the instruction count exploded, suggesting the loop itself or the `tl.num_programs(0)` call introduced massive overhead or the grid/tile math was wrong. The persistent loop pattern is only safe when the grid is actually reduced to num_sms CTAs; leaving the original large grid intact while adding a stride loop multiplies work rather than redistributing it.

> **Applicable when**: Never apply persistent stride-loop scheduling without simultaneously reducing the launch grid to num_sms. The two changes are inseparable.

> **Avoid when**: Do not apply a persistent tile stride loop if the launch grid is left at its original size. With grid=(1, 7168) and num_ctas=1 (from program_id(0) axis), each of the 7168 CTAs in axis-1 iterates over all 896 tiles (stride=1), multiplying total work by ~896x and causing catastrophic instruction explosion. The pattern requires grid=(num_sms,) as a 1D grid so that tl.num_programs(0)=num_sms and each CTA handles only total_tiles/num_sms tiles.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_xblock_yblock_with_synchronized_grid_formula_update | 0.001× | 0.000× | ✓ |  |
| B | fix_tile_shape_static_xblock16_yblock64 | 0.000× | 0.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_xblock_yblock... | Δ | fix_tile_shape_st... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 27810000 ns | +27802860 ns (+389395.8%) | 27740000 ns | +27732860 ns (+388415.4%) |
| SM Throughput (%) | 54.81 % | 75.79 % | +20.98 pp | 75.74 % | +20.93 pp |
| Memory Throughput (%) | 51.84 % | 91.15 % | +39.31 pp | 91.08 % | +39.24 pp |
| DRAM Throughput (%) | 5.94 % | 0.01 % | -5.93 pp | 0.01 % | -5.93 pp |
| Achieved Occupancy (%) | 70.71 % | 91.19 % | +20.48 pp | 91.19 % | +20.48 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 3.39 | 3.39 | +0.00 | 3.39 | +0.00 |
| No Eligible (%) | 19.68 % | 23.26 % | +3.58 pp | 23.34 % | +3.66 pp |
| IPC Active | 3.10 inst/cycle | 3.07 inst/cycle | -0.03 | 3.07 inst/cycle | -0.03 |


### Key Learnings

#### ❌ Plan A: `fix_xblock_yblock_with_synchronized_grid_formula_update` (0.000×)

Changing XBLOCK from 128 to 16 and YBLOCK from 8 to 64 in the launcher (with a synchronized grid formula update) produced zero measurable improvement: duration moved from 27.77ms to 27.81ms (effectively identical), all key metrics remained unchanged, and the instruction count was identical. The diff was empty, indicating the code change was not actually applied — the plan was correct in theory but the implementation produced no diff. When a unified diff is empty, the kernel was not modified and the profiling delta reflects only measurement noise, not the effect of the intended optimization.

> **Applicable when**: Never — this specific plan produced an empty diff and thus no change. The underlying strategy (reducing XBLOCK to match small xnumel and updating grid formula atomically) is theoretically sound but has failed to be implemented correctly across multiple iterations.

> **Avoid when**: Do not attempt further XBLOCK/YBLOCK launcher-only modifications for this kernel without verifying the diff is non-empty. All prior attempts (iterations 1-3) to fix the tile shape for this kernel have either produced empty diffs, correctness failures, or crashes. The kernel appears to resist modification via launcher parameter changes alone — the kernel body itself may need restructuring (e.g., rewriting as a true flat 1D kernel with correct index math).

---

## Cumulative Progress

```
Baseline:  7140 ns  (1.000×)

Final best: 7200 ns  (0.9917×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
+def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
     ynumel = 57344
     xnumel = 9
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
-    x2 = xindex
-    y3 = yindex
-    y0 = (yindex % 1792)
-    y1 = yindex // 1792
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 1792*x2 + 16128*y1), tmp0, xmask)
-
-
+    pid = tl.program_id(0)
+    num_xtiles = tl.cdiv(xnumel, XBLOCK)
+    num_ytiles = tl.cdiv(ynumel, YBLOCK)
+    total_tiles = num_xtiles * num_ytiles
+    num_ctas = tl.num_programs(0)
+    for tile_id in tl.range(pid, total_tiles, num_ctas):
+        ytile = tile_id // num_xtiles
+        xtile = tile_id % num_xtiles
+        yoffset = ytile * YBLOCK
+        xoffset = xtile * XBLOCK
+        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+        ymask = yindex < ynumel
+        xmask = xindex < xnumel
+        x2 = xindex
+        y3 = yindex
+        y0 = (yindex % 1792)
+        y1 = yindex // 1792
+        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=xmask & ymask, eviction_policy='evict_last')
+        tl.store(out_ptr0 + (y0 + 1792*x2 + 16128*y1), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

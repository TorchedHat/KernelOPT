# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_cat_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 42530 ns — **Best latency**: 42530 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_cat_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_0_triton_poi_fused_cat_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 16
- **Duration**: 42530 ns
- **SM Throughput**: 26.46 % | **Memory**: 23.34 % | **DRAM**: 23.34 %
- **Achieved Occupancy**: 39.90 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.94 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_launch_overhead_and_improve_latency_hiding | 0.846× | 0.858× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.910× | 0.927× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 42530 ns | 49600 ns | +7070 ns (+16.6%) | 45890 ns | +3360 ns (+7.9%) |
| SM Throughput (%) | 26.46 % | 57.55 % | +31.09 pp | 30.27 % | +3.81 pp |
| Memory Throughput (%) | 23.34 % | 21.56 % | -1.78 pp | 20.52 % | -2.82 pp |
| DRAM Throughput (%) | 23.34 % | 18.72 % | -4.62 pp | 20.52 % | -2.82 pp |
| Achieved Occupancy (%) | 39.90 % | 34.12 % | -5.78 pp | 34.50 % | -5.40 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 | 18.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 71.94 % | 46.32 % | -25.62 pp | 64.93 % | -7.01 pp |
| IPC Active | 1.12 inst/cycle | 2.15 inst/cycle | +1.03 | 1.36 inst/cycle | +0.24 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_for_fused_cat | 0.999× | 0.997× | ✓ |  |
| B | fix_grid_lambda_and_eviction_policy_for_streaming_cat | 0.894× | 0.932× | ✓ |  |
| C | persistent_tile_scheduling_with_streaming_eviction_policy | 0.015× | 0.014× | ✓ |  |
| D | reduce_register_pressure_with_maxnreg_and_fix_eviction_policy_for_streaming_cat | 0.830× | 0.858× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_grid_lambda_a... | Δ | persistent_tile_s... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42530 ns | 42660 ns | +130 ns (+0.3%) | 45630 ns | +3100 ns (+7.3%) | 3030000 ns | +2987470 ns (+7024.4%) | 49570 ns | +7040 ns (+16.6%) |
| SM Throughput (%) | 26.46 % | 42.38 % | +15.92 pp | 30.16 % | +3.70 pp | 67.19 % | +40.73 pp | 57.24 % | +30.78 pp |
| Memory Throughput (%) | 23.34 % | 23.32 % | -0.02 pp | 18.49 % | -4.85 pp | 58.89 % | +35.55 pp | 21.46 % | -1.88 pp |
| DRAM Throughput (%) | 23.34 % | 23.32 % | -0.02 pp | 18.49 % | -4.85 pp | 4.79 % | -18.55 pp | 16.84 % | -6.50 pp |
| Achieved Occupancy (%) | 39.90 % | 44.61 % | +4.71 pp | 34.52 % | -5.38 pp | 99.19 % | +59.29 pp | 34.17 % | -5.73 pp |
| Registers/Thread | 16.00 register/thread | 20.00 register/thread | +4 | 18.00 register/thread | +2 | 24.00 register/thread | +8 | 24.00 register/thread | +8 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 71.94 % | 54.75 % | -17.19 pp | 65.22 % | -6.72 pp | 32.75 % | -39.19 pp | 46.48 % | -25.46 pp |
| IPC Active | 1.12 inst/cycle | 1.80 inst/cycle | +0.68 | 1.37 inst/cycle | +0.25 | 2.69 inst/cycle | +1.57 | 2.14 inst/cycle | +1.02 |


### Key Learnings

#### ❌ Plan C: `persistent_tile_scheduling_with_streaming_eviction_policy` (0.014×)

Converting a flat 65536-CTA grid to a persistent 132-CTA kernel with an inner while-loop caused a ~71x slowdown (45.89 µs → 3030 µs). The root cause was an instruction count explosion from 10.9M to 1.61B executed instructions — a 147x increase — because the grid launch in kernel_function was NOT updated to use 132 CTAs; it still launched 65536 CTAs (cdiv(xnumel, 128)), each of which then ran the full inner while-loop over all 16384 tiles, causing every element to be processed 65536 times instead of once. The persistent kernel pattern requires the grid size to be reduced to NUM_SMS (132) simultaneously with adding the inner loop — failing to do so creates a catastrophic work multiplication. Even though occupancy improved (34.5% → 99.2%) and L2 hit rate rose (50.6% → 93.8%), the sheer volume of redundant work dominated all other effects.

> **Applicable when**: Never apply this specific broken pattern. The persistent kernel pattern itself is valid when: grid is correctly set to NUM_SMS CTAs, the inner loop strides by NUM_SMS, and num_tiles is passed as a runtime argument.

> **Avoid when**: Do NOT apply persistent kernel transformation if the grid launch site is not simultaneously updated to use NUM_SMS CTAs. If the grid remains at the original flat size (e.g., cdiv(xnumel, XBLOCK_small)) while the kernel body contains an inner while-loop, every CTA will redundantly process all tiles, causing O(original_grid_size) work multiplication. Also avoid when the kernel already runs many waves per SM (31+ waves) — the persistent pattern provides minimal benefit over the flat launch in that regime.

---

## Cumulative Progress

```
Baseline:  42530 ns  (1.000×)

Final best: 42530 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,12 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 4096}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_cat_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 8388608
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 32768)
     x1 = xindex // 32768
     x2 = xindex
@@ -19,15 +31,13 @@

     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 16384, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (16384*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (16384*x1 + (x0)), mask=xmask & tmp4, eviction_policy='evict_last', other=0.0)
     tmp6 = tmp0 >= tmp3
     tmp7 = tl.full([1], 32768, tl.int64)
     tmp8 = tmp0 < tmp7
-    tmp9 = tl.load(in_ptr1 + (16384*x1 + ((-16384) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
+    tmp9 = tl.load(in_ptr1 + (16384*x1 + ((-16384) + x0)), mask=xmask & tmp6, eviction_policy='evict_last', other=0.0)
     tmp10 = tl.where(tmp4, tmp5, tmp9)
-    tl.store(out_ptr0 + (x2), tmp10, None)
-
-
+    tl.store(out_ptr0 + (x2), tmp10, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -35,7 +45,7 @@

     arg_2 = inputs[2]
     xnumel = 8388608
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_cat_1[grid](arg_0, arg_1, arg_2, 8388608, XBLOCK=128)
+    triton_poi_fused_cat_1[grid](arg_0, arg_1, arg_2, 8388608)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

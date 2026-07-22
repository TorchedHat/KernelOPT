# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 244480 ns — **Best latency**: 244190 ns (**1.0012×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`increase_num_warps_for_l1tex_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_2_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 244480 ns
- **SM Throughput**: 32.53 % | **Memory**: 48.18 % | **DRAM**: 48.18 %
- **Achieved Occupancy**: 45.73 %
- **IPC Active**: 0.56 inst/cycle | **No Eligible**: 86.05 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_and_add_eviction_policies | 0.739× | 0.541× | ✓ |  |
| B | vectorized_loads_with_larger_xblock | 1.002× | 1.001× | ✓ | **✓ best** |
| C | memory_coalescing_via_larger_xblock_autotune | 0.753× | 0.754× | ✓ |  |
| D | increase_xblock_tile_size_via_autotune | 0.909× | 0.654× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244480 ns | 452060 ns | +207580 ns (+84.9%) | 244190 ns | -290 ns (-0.1%) | 324160 ns | +79680 ns (+32.6%) | 374080 ns | +129600 ns (+53.0%) |
| SM Throughput (%) | 32.53 % | 71.38 % | +38.85 pp | 32.74 % | +0.21 pp | 24.99 % | -7.54 pp | 34.90 % | +2.37 pp |
| Memory Throughput (%) | 48.18 % | 26.89 % | -21.29 pp | 48.20 % | +0.02 pp | 36.27 % | -11.91 pp | 31.42 % | -16.76 pp |
| DRAM Throughput (%) | 48.18 % | 26.01 % | -22.17 pp | 48.20 % | +0.02 pp | 36.27 % | -11.91 pp | 31.42 % | -16.76 pp |
| Achieved Occupancy (%) | 45.73 % | 59.06 % | +13.33 pp | 45.76 % | +0.03 pp | 43.63 % | -2.10 pp | 44.01 % | -1.72 pp |
| Registers/Thread | 16.00 register/thread | 22.00 register/thread | +6 | 16.00 register/thread | +0 | 22.00 register/thread | +6 | 18.00 register/thread | +2 |
| Waves/SM | 186.18 | 744.73 | +558.55 | 186.18 | +0.00 | 186.18 | +0.00 | 372.36 | +186.18 |
| No Eligible (%) | 86.05 % | 49.16 % | -36.89 pp | 86.04 % | -0.01 pp | 82.76 % | -3.29 pp | 65.78 % | -20.27 pp |
| IPC Active | 0.56 inst/cycle | 2.02 inst/cycle | +1.46 | 0.56 inst/cycle | +0.00 | 0.67 inst/cycle | +0.11 | 1.35 inst/cycle | +0.79 |


### Key Learnings

#### ❌ Plan D: `increase_xblock_tile_size_via_autotune` (0.654×)

Adding autotune with larger XBLOCK values (256 vs 128) caused a 53% slowdown despite reducing Warp Cycles Per Issued Instruction from 51.4 to 19.7 and improving eligible warps from 0.15 to 0.60. The regression occurred because the implementation flaw introduced masked loads/stores (xmask = xindex < xnumel) where the original used unconditional None masks, and the grid was not updated to use meta['XBLOCK'] — it remained hardcoded to cdiv(xnumel, 128), so the autotuner selected a larger block size (256) but the grid was still sized for XBLOCK=128, causing 2x more threads to be launched and doubling executed instructions from 26.7M to 98.7M. DRAM throughput dropped from 48% to 31%, indicating the extra thread overhead outweighed any latency-hiding benefit.

> **Applicable when**: Never apply this exact pattern. The grid must be updated to lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),) when using autotune with variable XBLOCK.

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK without simultaneously updating the grid to use meta['XBLOCK']. Do not introduce xmask guards when xnumel is exactly divisible by all candidate XBLOCK values — the unconditional None mask is correct and cheaper.

---

## Iteration 2

**Best so far**: 1.0012× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_cache_policy_for_streaming_loads_and_stores | 1.000× | 1.002× | ✓ |  |
| B | evict_first_on_write_only_stores_evict_last_on_input_load | 1.001× | 1.000× | ✓ |  |
| C | eviction_policies_for_streaming_pointwise | 0.891× | 0.652× | ✓ |  |
| D | fix_broken_autotune_grid_restore_correct_xblock | 1.000× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_cache... | Δ | evict_first_on_wr... | Δ | eviction_policies... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244480 ns | 243940 ns | -540 ns (-0.2%) | 244380 ns | -100 ns (-0.0%) | 374880 ns | +130400 ns (+53.3%) | 244260 ns | -220 ns (-0.1%) |
| SM Throughput (%) | 32.53 % | 32.53 % | +0.00 pp | 32.52 % | -0.01 pp | 44.33 % | +11.80 pp | 32.48 % | -0.05 pp |
| Memory Throughput (%) | 48.18 % | 48.21 % | +0.03 pp | 48.05 % | -0.13 pp | 31.37 % | -16.81 pp | 48.17 % | -0.01 pp |
| DRAM Throughput (%) | 48.18 % | 48.21 % | +0.03 pp | 48.05 % | -0.13 pp | 31.37 % | -16.81 pp | 48.17 % | -0.01 pp |
| Achieved Occupancy (%) | 45.73 % | 45.67 % | -0.06 pp | 48.28 % | +2.55 pp | 46.94 % | +1.21 pp | 48.62 % | +2.89 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 372.36 | +186.18 | 186.18 | +0.00 |
| No Eligible (%) | 86.05 % | 81.94 % | -4.11 pp | 71.20 % | -14.85 pp | 55.21 % | -30.84 pp | 76.88 % | -9.17 pp |
| IPC Active | 0.56 inst/cycle | 0.73 inst/cycle | +0.17 | 1.15 inst/cycle | +0.59 | 1.80 inst/cycle | +1.24 | 0.92 inst/cycle | +0.36 |

---

## Iteration 3

**Best so far**: 1.0012× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_for_l1tex_latency_hiding | 1.000× | 1.001× | ✓ |  |
| B | increase_num_warps_for_l1tex_latency_hiding | 1.000× | 1.002× | ✓ |  |
| C | increase_num_warps_for_l1tex_latency_hiding | 0.999× | 1.000× | ✓ |  |
| D | fix_autotune_grid_to_use_meta_xblock | 0.891× | 0.653× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244480 ns | 244190 ns | -290 ns (-0.1%) | 244100 ns | -380 ns (-0.2%) | 244420 ns | -60 ns (-0.0%) | 374430 ns | +129950 ns (+53.2%) |
| SM Throughput (%) | 32.53 % | 32.44 % | -0.09 pp | 32.54 % | +0.01 pp | 32.55 % | +0.02 pp | 44.27 % | +11.74 pp |
| Memory Throughput (%) | 48.18 % | 48.18 % | +0.00 pp | 48.18 % | +0.00 pp | 48.07 % | -0.11 pp | 31.39 % | -16.79 pp |
| DRAM Throughput (%) | 48.18 % | 48.18 % | +0.00 pp | 48.18 % | +0.00 pp | 48.07 % | -0.11 pp | 31.39 % | -16.79 pp |
| Achieved Occupancy (%) | 45.73 % | 48.66 % | +2.93 pp | 48.50 % | +2.77 pp | 48.29 % | +2.56 pp | 47.07 % | +1.34 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 372.36 | +186.18 |
| No Eligible (%) | 86.05 % | 77.02 % | -9.03 pp | 76.87 % | -9.18 pp | 71.11 % | -14.94 pp | 54.89 % | -31.16 pp |
| IPC Active | 0.56 inst/cycle | 0.92 inst/cycle | +0.36 | 0.93 inst/cycle | +0.37 | 1.15 inst/cycle | +0.59 | 1.78 inst/cycle | +1.22 |


### Key Learnings

#### ❌ Plan D: `fix_autotune_grid_to_use_meta_xblock` (0.653×)

Even when the plan correctly identified the grid-mismatch bug and proposed using a lambda grid (`lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`), the actual diff shows the grid was NOT updated — it remained hardcoded to `(triton.cdiv(xnumel, 128),)` while autotune selected XBLOCK=256, causing the same 2x thread over-launch seen in prior iterations (executed instructions: 34.6M → 130.2M, DRAM throughput: 48% → 31%). The implementation flaw is that the plan was correct but the code change was incomplete: the `@triton.autotune` decorator was added and the kernel body was updated, but the call-site grid definition was left unchanged. This confirms that adding autotune without fixing the grid is always a regression regardless of the kernel body changes.

> **Applicable when**: Never applicable as a speedup. Documents the recurring failure mode: adding @triton.autotune with variable XBLOCK while leaving the launch grid hardcoded always causes instruction count explosion proportional to XBLOCK_selected / XBLOCK_hardcoded.

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK configs without simultaneously changing the launch grid to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. This anti-pattern has now caused regressions in iterations 1, 2, and 3 with nearly identical profiling signatures (executed instructions ~130M, DRAM throughput ~31%, block_size=256, waves_per_sm=372).

---

## Cumulative Progress

```
Baseline:  244480 ns  (1.000×)
Iter 1     244190 ns  (1.001×)  — vectorized_loads_with_larger_xblock

Final best: 244190 ns  (1.0012×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,20 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_convolution_relu_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
     xnumel = 50331648
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
+    # Alignment + contiguity hints unlock 128-bit (float4) vectorized loads
+    xindex = tl.max_contiguous(tl.multiple_of(xindex, XBLOCK), XBLOCK)
     xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x0 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None)
+    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_first')
     tmp1 = tl.full([1], 0, tl.int32)
     tmp2 = tl.maximum(tmp1, tmp0)
-    tl.store(out_ptr0 + (x0), tmp2, None)
-    tl.store(out_ptr1 + (x0), tmp2, None)
-
-
+    tl.store(out_ptr0 + (x0), tmp2, None, eviction_policy='evict_first')
+    tl.store(out_ptr1 + (x0), tmp2, None, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2530 ns — **Best latency**: 2560 ns (**0.9883×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_3_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.12
- **Registers/thread**: 28
- **Duration**: 2530 ns
- **SM Throughput**: 5.24 % | **Memory**: 4.90 % | **DRAM**: 0.64 %
- **Achieved Occupancy**: 11.17 %
- **IPC Active**: 0.61 inst/cycle | **No Eligible**: 87.75 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune_block_sizes | 0.991× | 0.941× | ✓ |  |
| C | increase_tile_sizes_for_occupancy_and_parallelism | 1.009× | 0.966× | ✓ |  |
| D | autotune_tile_sizes_to_increase_grid_and_occupancy | — | fail | ✓ |  |
| E | add_triton_autotune | 0.993× | 0.941× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | increase_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2530 ns | 2690 ns | +160 ns (+6.3%) | 2620 ns | +90 ns (+3.6%) | 2690 ns | +160 ns (+6.3%) |
| SM Throughput (%) | 5.24 % | 2.79 % | -2.45 pp | 4.75 % | -0.49 pp | 4.82 % | -0.42 pp |
| Memory Throughput (%) | 4.90 % | 3.96 % | -0.94 pp | 5.32 % | +0.42 pp | 4.51 % | -0.39 pp |
| DRAM Throughput (%) | 0.64 % | 2.30 % | +1.66 pp | 0.63 % | -0.01 pp | 0.60 % | -0.04 pp |
| Achieved Occupancy (%) | 11.17 % | 5.62 % | -5.55 pp | 5.66 % | -5.51 pp | 11.27 % | +0.10 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 32.00 register/thread | +4 | 28.00 register/thread | +0 |
| Waves/SM | 0.12 | 0.06 | -0.06 | 0.06 | -0.06 | 0.12 | +0.00 |
| No Eligible (%) | 87.75 % | 89.99 % | +2.24 pp | 79.65 % | -8.10 pp | 86.36 % | -1.39 pp |
| IPC Active | 0.61 inst/cycle | 0.33 inst/cycle | -0.28 | 0.71 inst/cycle | +0.10 | 0.58 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launch_grid_to_match_autotune_blocks_and_improve_coalescing | 0.959× | 0.941× | ✓ |  |
| B | fix_grid_launch_to_use_autotuned_block_sizes | 0.997× | 0.861× | ✓ |  |
| C | reduce_xblock_increase_yblock_for_grid_expansion | 1.025× | 0.988× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_launch_grid_t... | Δ | fix_grid_launch_t... | Δ | reduce_xblock_inc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2530 ns | 2690 ns | +160 ns (+6.3%) | 2940 ns | +410 ns (+16.2%) | 2560 ns | +30 ns (+1.2%) |
| SM Throughput (%) | 5.24 % | 4.41 % | -0.83 pp | 2.83 % | -2.41 pp | 3.50 % | -1.74 pp |
| Memory Throughput (%) | 4.90 % | 4.94 % | +0.04 pp | 3.46 % | -1.44 pp | 3.20 % | -1.70 pp |
| DRAM Throughput (%) | 0.64 % | 0.61 % | -0.03 pp | 2.09 % | +1.45 pp | 1.21 % | +0.57 pp |
| Achieved Occupancy (%) | 11.17 % | 5.67 % | -5.50 pp | 5.71 % | -5.46 pp | 11.03 % | -0.14 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 17.00 register/thread | -11 |
| Waves/SM | 0.12 | 0.06 | -0.06 | 0.06 | -0.06 | 0.12 | +0.00 |
| No Eligible (%) | 87.75 % | 79.92 % | -7.83 pp | 89.54 % | +1.79 pp | 87.52 % | -0.23 pp |
| IPC Active | 0.61 inst/cycle | 0.67 inst/cycle | +0.06 | 0.28 inst/cycle | -0.33 | 0.39 inst/cycle | -0.22 |


### Key Learnings

#### ❌ Plan B: `fix_grid_launch_to_use_autotuned_block_sizes` (0.861×)

Expanding the autotune config set and replacing a conditional ymask with tl.full([YBLOCK], True) caused a 16% regression. Despite reducing executed instructions from 97,280 to 59,392, elapsed cycles increased from 3,844 to 4,370 because Warp Cycles Per Issued Instruction doubled from 4.91 to 10.03 — a new CPIStall rule appeared attributing 45.5% of stall cycles to L1TEX scoreboard dependencies. The grid size and achieved occupancy remained essentially unchanged (~5.7%), so the bottleneck shifted from compute underutilization to memory latency stalls, yielding a net slowdown.

> **Applicable when**: Never apply this specific combination: replacing a conditional ymask with tl.full(True) while simultaneously expanding autotune configs for a 2D kernel where ynumel is a power of 2 and the grid remains wave-starved (waves_per_sm ~0.06). The always-true mask does not eliminate the L1TEX stall bottleneck and the expanded config set selects configs that increase memory latency exposure.

> **Avoid when**: Do not replace conditional ymask with tl.full([YBLOCK], True) when the kernel is already wave-starved (waves_per_sm < 0.1) and the grid size does not change — the mask simplification alone cannot compensate for the increased L1TEX scoreboard stall cycles it introduces. Do not expand autotune configs without also fixing the grid launcher to use META['XBLOCK']/META['YBLOCK'] so the grid actually scales with the config.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launcher_to_use_meta_block_sizes_only | 1.002× | 0.930× | ✓ |  |
| B | fix_launch_grid_to_respect_autotuned_block_sizes_only | 0.989× | 0.951× | ✓ |  |
| C | restore_conditional_ymask_and_streaming_eviction_to_reduce_l1tex_stall | 0.965× | 0.941× | ✓ |  |
| D | add_triton_autotune | 0.922× | 0.841× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launcher... | Δ | fix_launch_grid_t... | Δ | restore_condition... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2530 ns | 2720 ns | +190 ns (+7.5%) | 2660 ns | +130 ns (+5.1%) | 2690 ns | +160 ns (+6.3%) | 3010 ns | +480 ns (+19.0%) |
| SM Throughput (%) | 5.24 % | 3.53 % | -1.71 pp | 4.74 % | -0.50 pp | 3.07 % | -2.17 pp | 10.01 % | +4.77 pp |
| Memory Throughput (%) | 4.90 % | 3.07 % | -1.83 pp | 5.31 % | +0.41 pp | 2.81 % | -2.09 pp | 12.69 % | +7.79 pp |
| DRAM Throughput (%) | 0.64 % | 1.15 % | +0.51 pp | 0.61 % | -0.03 pp | 0.60 % | -0.04 pp | 8.17 % | +7.53 pp |
| Achieved Occupancy (%) | 11.17 % | 11.07 % | -0.10 pp | 5.71 % | -5.46 pp | 5.17 % | -6.00 pp | 11.33 % | +0.16 pp |
| Registers/Thread | 28.00 register/thread | 17.00 register/thread | -11 | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 32.00 register/thread | +4 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.06 | -0.06 | 0.06 | -0.06 | 0.12 | +0.00 |
| No Eligible (%) | 87.75 % | 88.20 % | +0.45 pp | 79.65 % | -8.10 pp | 84.75 % | -3.00 pp | 73.96 % | -13.79 pp |
| IPC Active | 0.61 inst/cycle | 0.43 inst/cycle | -0.18 | 0.57 inst/cycle | -0.04 | 0.49 inst/cycle | -0.12 | 1.00 inst/cycle | +0.39 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_with_sm_filling_small_tile_configs | 0.974× | 1.041× | ✓ |  |
| B | persistent_sm_loop_strided_transpose | 0.039× | 0.019× | ✓ |  |
| C | persistent_sm_loop_over_y_tiles | 0.959× | 0.951× | ✓ |  |
| D | fix_grid_launcher_with_small_xblock_configs_to_eliminate_thread_waste | 0.988× | 1.067× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_w... | Δ | persistent_sm_loo... | Δ | persistent_sm_loo... | Δ | fix_grid_launcher... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2530 ns | 2430 ns | -100 ns (-4.0%) | 129820 ns | +127290 ns (+5031.2%) | 2660 ns | +130 ns (+5.1%) | 2370 ns | -160 ns (-6.3%) |
| SM Throughput (%) | 5.24 % | 0.45 % | -4.79 pp | 30.86 % | +25.62 pp | 1.17 % | -4.07 pp | 0.85 % | -4.39 pp |
| Memory Throughput (%) | 4.90 % | 0.79 % | -4.11 pp | 29.82 % | +24.92 pp | 1.62 % | -3.28 pp | 1.38 % | -3.52 pp |
| DRAM Throughput (%) | 0.64 % | 0.65 % | +0.01 pp | 0.01 % | -0.63 pp | 0.61 % | -0.03 pp | 1.31 % | +0.67 pp |
| Achieved Occupancy (%) | 11.17 % | 2.13 % | -9.04 pp | 12.12 % | +0.95 pp | 3.79 % | -7.38 pp | 5.57 % | -5.60 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 28.00 register/thread | +0 | 16.00 register/thread | -12 |
| Waves/SM | 0.12 | 0.06 | -0.06 | 0.12 | +0.00 | 0.06 | -0.06 | 0.06 | -0.06 |
| No Eligible (%) | 87.75 % | 94.68 % | +6.93 pp | 68.90 % | -18.85 pp | 93.57 % | +5.82 pp | 96.01 % | +8.26 pp |
| IPC Active | 0.61 inst/cycle | 0.07 inst/cycle | -0.54 | 1.25 inst/cycle | +0.64 | 0.16 inst/cycle | -0.45 | 0.10 inst/cycle | -0.51 |


### Key Learnings

#### ❌ Plan B: `persistent_sm_loop_strided_transpose` (0.019×)

Converting a 2D-grid Triton kernel to a persistent while-loop over flat tile indices caused a 51x regression (2.66us → 129.82us). Although the persistent loop was intended to fix wave starvation, the grid launcher was not updated to use the new num_programs logic — it still used the old broken grid formula (cdiv(9,128), cdiv(2048,8)) = (1, 256), so 256 CTAs each executed the while-loop over all 128 tiles, exploding executed instructions from 97,280 to 31,247,360 (321x increase). The CPIStall rule shifted from instruction-cache misses to CTA barrier stalls (36.6% of cycles), and DRAM elapsed cycles grew from 404,992 to 19,933,184. The implementation flaw was that the grid launch was never updated to match the persistent kernel's expected num_programs, causing every CTA to redundantly process all tiles.

> **Applicable when**: Never apply this specific implementation: persistent while-loop with the old 2D grid formula left unchanged. The grid must be updated to (num_programs,) when converting to a persistent 1D-grid kernel.

> **Avoid when**: Do not convert a 2D-grid kernel to a persistent 1D while-loop without simultaneously updating the grid launcher to use a 1D grid of size num_programs = min(num_tiles, num_SMs). Leaving the old 2D grid in place causes every CTA to iterate over all tiles, multiplying instruction count by grid_size, producing catastrophic regression. Also avoid this pattern when total work is tiny (< 20K elements) — even a correct persistent kernel may not help because the kernel remains wave-starved.

#### ✅ Plan D: `fix_grid_launcher_with_small_xblock_configs_to_eliminate_thread_waste` (1.067×)

When a 2D Triton kernel has a very small X dimension (xnumel=9) but large XBLOCK configs (e.g., 128), the vast majority of threads per CTA are masked out (93% waste), and the grid remains wave-starved. Replacing large-XBLOCK autotune configs with small-XBLOCK configs (XBLOCK=1,4,8,16) matched to the actual xnumel, combined with replacing a hardcoded grid tuple with a lambda using META['XBLOCK']/META['YBLOCK'], allows the grid to scale dynamically with the autotuned config. This reduced registers per thread from 32 to 16 and eliminated dynamic shared memory (4 KB → 0), cutting elapsed cycles from 3,956 to 3,523 (11% reduction) and yielding a 1.07x speedup despite the kernel remaining wave-starved due to its small total element count.

> **Applicable when**: A 2D Triton kernel has a very small X dimension (xnumel << XBLOCK, e.g., xnumel=9 with XBLOCK=128 causing >80% thread waste per CTA). The grid launcher uses hardcoded block sizes instead of META['XBLOCK']/META['YBLOCK'], so the grid does not scale with the autotuned config. Registers per thread are high (≥32) and dynamic shared memory is non-zero despite the kernel being a simple memory permutation. The kernel is wave-starved (waves_per_sm < 0.2).

> **Avoid when**: The kernel's X dimension is large relative to XBLOCK (thread waste is low). The grid launcher already uses a META lambda. Reducing XBLOCK would cause excessive grid overhead (too many tiny CTAs). Note: this optimization yields modest gains (~7%) even when correctly applied because the kernel remains wave-starved due to small total element count — the grid cannot expand enough to fill all SMs.

---

## Cumulative Progress

```
Baseline:  2530 ns  (1.000×)

Final best: 2560 ns  (0.9883×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 16},  num_warps=1),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=1),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 2048
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +35,8 @@

     y3 = yindex
     y0 = (yindex % 64)
     y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 2048
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3[grid](arg_0, arg_1, 2048, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3[grid](arg_0, arg_1, 2048, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

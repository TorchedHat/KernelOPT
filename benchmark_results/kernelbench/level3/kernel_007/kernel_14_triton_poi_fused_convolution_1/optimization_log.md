# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2660 ns — **Best latency**: 2400 ns (**1.1083×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`expand_autotune_configs_with_xblock1_for_sm_coverage`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_14_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.03
- **Registers/thread**: 28
- **Duration**: 2660 ns
- **SM Throughput**: 1.27 % | **Memory**: 1.21 % | **DRAM**: 0.43 %
- **Achieved Occupancy**: 6.13 %
- **IPC Active**: 0.31 inst/cycle | **No Eligible**: 89.63 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 0.958× | 0.875× | ✓ |  |
| B | autotune_tile_sizes_for_small_grid_utilization | 1.022× | 1.172× | ✓ |  |
| C | increase_grid_coverage_via_tile_size_rebalancing | 1.034× | 1.108× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_grid_coverage | 0.953× | 1.081× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ | increase_grid_cov... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 3040 ns | +380 ns (+14.3%) | 2270 ns | -390 ns (-14.7%) | 2400 ns | -260 ns (-9.8%) | 2460 ns | -200 ns (-7.5%) |
| SM Throughput (%) | 1.27 % | 0.28 % | -0.99 pp | 0.12 % | -1.15 pp | 0.92 % | -0.35 pp | 1.14 % | -0.13 pp |
| Memory Throughput (%) | 1.21 % | 1.50 % | +0.29 pp | 1.18 % | -0.03 pp | 1.16 % | -0.05 pp | 0.91 % | -0.30 pp |
| DRAM Throughput (%) | 0.43 % | 0.37 % | -0.06 pp | 0.09 % | -0.34 pp | 0.47 % | +0.04 pp | 0.46 % | +0.03 pp |
| Achieved Occupancy (%) | 6.13 % | 1.56 % | -4.57 pp | 1.56 % | -4.57 pp | 6.19 % | +0.06 pp | 6.21 % | +0.08 pp |
| Registers/Thread | 28.00 register/thread | 22.00 register/thread | -6 | 16.00 register/thread | -12 | 20.00 register/thread | -8 | 21.00 register/thread | -7 |
| Waves/SM | 0.03 | 0.02 | -0.01 | 0.02 | -0.01 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 89.63 % | 94.04 % | +4.41 pp | 95.62 % | +5.99 pp | 89.87 % | +0.24 pp | 90.73 % | +1.10 pp |
| IPC Active | 0.31 inst/cycle | 0.05 inst/cycle | -0.26 | 0.03 inst/cycle | -0.28 | 0.30 inst/cycle | -0.01 | 0.45 inst/cycle | +0.14 |


### Key Learnings

#### ✅ Plan D: `autotune_tile_sizes_for_grid_coverage` (1.081×)

When a Triton kernel has a small problem dimension (e.g., xnumel=25) but uses an oversized tile (XBLOCK=128), the autotune decorator can discover a smaller tile configuration that reduces register usage per thread (28→21) and shared memory per block (4KB→2KB), lowering instruction count and warp stall cycles even when grid size remains unchanged. The primary measurable effect here was a reduction in Warp Cycles Per Issued Instruction (10.26→7.53) and elapsed cycles (3912→3643), driven by fewer instructions executed (25856→21184) due to the autotuner selecting a more efficient tile configuration. Grid size and occupancy remained nearly constant, indicating the speedup came from reduced per-thread work and register pressure rather than improved SM coverage.

> **Applicable when**: Triton kernel has hardcoded tile sizes (XBLOCK, YBLOCK) that are mismatched to the actual problem dimensions (e.g., XBLOCK >> xnumel), causing high register usage per thread, excessive wasted lanes, and elevated Warp Cycles Per Issued Instruction. NCU rules LaunchConfiguration and/or AchievedOccupancy are triggered. Registers Per Thread is notably high (>24).

> **Avoid when**: The kernel already uses autotune, or the problem dimensions are large enough that the hardcoded tile fills the dimension efficiently. Also avoid if the kernel body contains tile-size-dependent logic that cannot be generalized across configs.

---

## Iteration 2

**Best so far**: 1.1083× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_with_xblock1_for_sm_coverage | 1.031× | 1.108× | ✓ |  |
| C | expand_grid_coverage_via_minimal_yblock_configs | 1.023× | 1.172× | ✓ |  |
| D | persistent_sm_saturating_grid_scheduling | 0.975× | 0.711× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | expand_grid_cover... | Δ | persistent_sm_sat... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2400 ns | -260 ns (-9.8%) | 2270 ns | -390 ns (-14.7%) | 3740 ns | +1080 ns (+40.6%) |
| SM Throughput (%) | 1.27 % | 0.10 % | -1.17 pp | 0.22 % | -1.05 pp | 0.28 % | -0.99 pp |
| Memory Throughput (%) | 1.21 % | 0.64 % | -0.57 pp | 0.53 % | -0.68 pp | 2.43 % | +1.22 pp |
| DRAM Throughput (%) | 0.43 % | 0.31 % | -0.12 pp | 0.07 % | -0.36 pp | 0.30 % | -0.13 pp |
| Achieved Occupancy (%) | 6.13 % | 1.56 % | -4.57 pp | 1.56 % | -4.57 pp | 1.56 % | -4.57 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 18.00 register/thread | -10 |
| Waves/SM | 0.03 | 0.02 | -0.01 | 0.02 | -0.01 | 0.02 | -0.01 |
| No Eligible (%) | 89.63 % | 93.95 % | +4.32 pp | 96.66 % | +7.03 pp | 95.62 % | +5.99 pp |
| IPC Active | 0.31 inst/cycle | 0.05 inst/cycle | -0.26 | 0.04 inst/cycle | -0.27 | 0.05 inst/cycle | -0.26 |


### Key Learnings

#### ❌ Plan D: `persistent_sm_saturating_grid_scheduling` (0.711×)

Adding a persistent Y-stride loop to a kernel that already has a grid smaller than SM count (64 blocks on 132 SMs) does not increase the grid size if the autotune configs still produce the same grid shape. In this case, the grid remained at 64 blocks (waves_per_sm=0.02, achieved occupancy=1.56%) while executed instructions ballooned from 1920 to 8512 (4.4x more work per CTA due to the persistent loop iterating multiple Y-tiles), causing elapsed cycles to increase from 3364 to 5596 and duration to regress from 2.27 µs to 3.74 µs. The persistent loop overhead (loop control, repeated index recomputation) adds instruction count without any SM saturation benefit when the grid formula does not actually cap at SM_COUNT.

> **Applicable when**: Never: this pattern is a structural anti-pattern for small kernels where the grid cannot be increased beyond the number of Y-tiles. The persistent loop only helps when the grid is genuinely capped below the natural tile count AND the cap is enforced at launch.

> **Avoid when**: The kernel's total tile count (cdiv(ynumel, YBLOCK)) is already <= SM_COUNT, meaning the persistent loop adds overhead without distributing more work. When ynumel=512 and YBLOCK>=4, cdiv(512,4)=128 tiles already fits within 132 SMs without a persistent loop. Adding a persistent loop in this regime forces each CTA to process multiple tiles sequentially, increasing instruction count without parallelism gains.

---

## Iteration 3

**Best so far**: 1.1083× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_enable_autotune_sm_coverage | 0.992× | 1.172× | ✓ |  |
| B | fix_load_eviction_policy_for_streaming_transpose | 0.918× | 1.000× | ✓ |  |
| C | fix_uncoalesced_store_via_output_index_reordering | 0.992× | 1.064× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_load_eviction... | Δ | fix_uncoalesced_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2270 ns | -390 ns (-14.7%) | 2660 ns | +0 ns (+0.0%) | 2500 ns | -160 ns (-6.0%) |
| SM Throughput (%) | 1.27 % | 0.22 % | -1.05 pp | 0.11 % | -1.16 pp | 0.24 % | -1.03 pp |
| Memory Throughput (%) | 1.21 % | 0.56 % | -0.65 pp | 0.49 % | -0.72 pp | 0.99 % | -0.22 pp |
| DRAM Throughput (%) | 0.43 % | 0.07 % | -0.36 pp | 0.28 % | -0.15 pp | 0.31 % | -0.12 pp |
| Achieved Occupancy (%) | 6.13 % | 1.56 % | -4.57 pp | 1.56 % | -4.57 pp | 1.56 % | -4.57 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 24.00 register/thread | -4 |
| Waves/SM | 0.03 | 0.02 | -0.01 | 0.02 | -0.01 | 0.02 | -0.01 |
| No Eligible (%) | 89.63 % | 97.32 % | +7.69 pp | 96.37 % | +6.74 pp | 93.43 % | +3.80 pp |
| IPC Active | 0.31 inst/cycle | 0.03 inst/cycle | -0.28 | 0.05 inst/cycle | -0.26 | 0.06 inst/cycle | -0.25 |

---

## Cumulative Progress

```
Baseline:  2660 ns  (1.000×)
Iter 1     2400 ns  (1.108×)  — increase_grid_coverage_via_tile_size_rebalancing

Final best: 2400 ns  (1.1083×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK':  1, 'YBLOCK':   1}, num_warps=1),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':   2}, num_warps=1),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':   4}, num_warps=1),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':   8}, num_warps=1),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':  16}, num_warps=2),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':   1}, num_warps=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':   4}, num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':   8}, num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  16}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  32}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 512
@@ -22,15 +41,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 16*x2 + 400*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 512
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_14[grid](arg_0, arg_1, 512, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_14[grid](arg_0, arg_1, 512, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

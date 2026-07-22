# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_permute_view_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16860 ns — **Best latency**: 16860 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_permute_view_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_1_triton_poi_fused_clone_permute/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 16
- **Duration**: 16860 ns
- **SM Throughput**: 24.06 % | **Memory**: 18.29 % | **DRAM**: 14.59 %
- **Achieved Occupancy**: 39.26 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.66 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_larger_tiles | 0.917× | 0.929× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.921× | 0.951× | ✓ |  |
| C | autotune_xblock_num_warps_for_latency_hiding | 0.996× | 0.606× | ✓ |  |
| D | vectorized_loads_stores_via_autotune_larger_xblock | 0.996× | 0.608× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | autotune_xblock_a... | Δ | autotune_xblock_n... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16860 ns | 18140 ns | +1280 ns (+7.6%) | 17730 ns | +870 ns (+5.2%) | 27810 ns | +10950 ns (+64.9%) | 27740 ns | +10880 ns (+64.5%) |
| SM Throughput (%) | 24.06 % | 25.78 % | +1.72 pp | 25.37 % | +1.31 pp | 63.36 % | +39.30 pp | 63.80 % | +39.74 pp |
| Memory Throughput (%) | 18.29 % | 16.81 % | -1.48 pp | 17.18 % | -1.11 pp | 17.80 % | -0.49 pp | 17.96 % | -0.33 pp |
| DRAM Throughput (%) | 14.59 % | 13.56 % | -1.03 pp | 13.85 % | -0.74 pp | 8.83 % | -5.76 pp | 8.85 % | -5.74 pp |
| Achieved Occupancy (%) | 39.26 % | 34.21 % | -5.05 pp | 35.44 % | -3.82 pp | 66.26 % | +27.00 pp | 65.90 % | +26.64 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 44.55 | +33.41 | 44.55 | +33.41 |
| No Eligible (%) | 71.66 % | 65.35 % | -6.31 pp | 66.44 % | -5.22 pp | 32.01 % | -39.65 pp | 31.03 % | -40.63 pp |
| IPC Active | 1.12 inst/cycle | 1.34 inst/cycle | +0.22 | 1.28 inst/cycle | +0.16 | 2.74 inst/cycle | +1.62 | 2.74 inst/cycle | +1.62 |


### Key Learnings

#### ❌ Plan C: `autotune_xblock_num_warps_for_latency_hiding` (0.606×)

Increasing block size (128→512) for a memory-bandwidth-bound pointwise permute kernel improved compute utilization (24%→63%) and occupancy (39%→66%), but caused a 65% increase in kernel duration (16.86→27.81 µs). The autotune selected a larger block size that dramatically increased total instruction count (3.2M→14M instructions) and waves per SM (11→44), overwhelming the memory subsystem: DRAM throughput dropped from 14.6% to 8.8% and memory throughput fell from 715 to 433 GB/s, indicating the larger blocks created more serialized memory traffic rather than hiding latency. For kernels where the bottleneck is memory bandwidth (not latency hiding), increasing block size increases total work without proportionally increasing memory throughput, resulting in regression.

> **Applicable when**: Kernel is compute-latency-bound (high CPIStall on L1TEX, low eligible warps per scheduler <1.0, low achieved occupancy <50%), with very low instruction count per block and minimal memory bandwidth utilization (<20% DRAM throughput). The bottleneck is truly latency hiding, not bandwidth saturation.

> **Avoid when**: Kernel is already achieving meaningful memory bandwidth (>10% DRAM throughput) and the total work is fixed — increasing block size will multiply instruction count (waves per SM) without proportionally increasing memory throughput, causing regression. Also avoid when the grid size is already large (>10K blocks) and waves per SM are already >10, as autotune may select configs that increase waves dramatically.

#### ❌ Plan D: `vectorized_loads_stores_via_autotune_larger_xblock` (0.608×)

Increasing block size (XBLOCK) from 128 to 512 via @triton.autotune on a strided permute kernel caused a 1.65x slowdown despite improving occupancy (39% → 66%) and eligible warps (0.43 → 3.61). The root cause is that the larger block size increased total executed instructions by 4.4x (3.2M → 14.0M) and shifted the bottleneck from memory latency to compute (SM throughput 24% → 64%), while actual memory throughput dropped (714 GB/s → 434 GB/s) and DRAM throughput halved (14.6% → 8.85%). For a pure data-movement permute with no arithmetic, the extra instruction overhead from bounds-checking (xmask = xindex < xnumel replacing tl.full(..., True)) and larger tile indexing math dominated, making the kernel compute-bound rather than memory-latency-bound.

> **Applicable when**: Never apply this specific pattern (autotune larger XBLOCK + bounds mask) to pure data-movement permute kernels where xnumel is already divisible by the original block size and no bounds check is needed.

> **Avoid when**: Kernel is a pure strided permute/copy with no arithmetic; xnumel is divisible by the original XBLOCK (making xmask=True always); the original kernel already uses tl.full([XBLOCK], True) mask (no bounds check overhead). Adding a real bounds-check mask (xindex < xnumel) to a larger XBLOCK inflates instruction count and shifts bottleneck to compute even when memory latency was the stated problem.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.041× | 0.035× | ✓ |  |
| B | persistent_tile_scheduling | 0.993× | 0.996× | ✓ |  |
| D | persistent_tile_scheduling | 0.984× | 1.000× | ✓ |  |
| E | add_triton_autotune | 0.997× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16860 ns | 474300 ns | +457440 ns (+2713.2%) | 16930 ns | +70 ns (+0.4%) | 16860 ns | +0 ns (+0.0%) | 16860 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 24.06 % | 82.48 % | +58.42 pp | 35.25 % | +11.19 pp | 32.51 % | +8.45 pp | 28.33 % | +4.27 pp |
| Memory Throughput (%) | 18.29 % | 52.47 % | +34.18 pp | 18.21 % | -0.08 pp | 18.16 % | -0.13 pp | 18.27 % | -0.02 pp |
| DRAM Throughput (%) | 14.59 % | 0.58 % | -14.01 pp | 14.53 % | -0.06 pp | 14.56 % | -0.03 pp | 14.57 % | -0.02 pp |
| Achieved Occupancy (%) | 39.26 % | 98.81 % | +59.55 pp | 43.94 % | +4.68 pp | 44.18 % | +4.92 pp | 39.92 % | +0.66 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 71.66 % | 17.12 % | -54.54 pp | 58.68 % | -12.98 pp | 61.99 % | -9.67 pp | 66.69 % | -4.97 pp |
| IPC Active | 1.12 inst/cycle | 3.31 inst/cycle | +2.19 | 1.65 inst/cycle | +0.53 | 1.52 inst/cycle | +0.40 | 1.31 inst/cycle | +0.19 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.035×)

Converting a one-CTA-per-tile grid (23,520 CTAs) to a persistent kernel with 132 CTAs each iterating over tiles in a while-loop caused a 28x slowdown (17.73 µs → 474.3 µs). The persistent loop increased executed instructions by 87.7x (3.5M → 309.5M), shifting the bottleneck from memory latency to compute (SM throughput 25% → 82%), while actual memory throughput collapsed from 679 GB/s to 28 GB/s and DRAM throughput dropped from 13.85% to 0.58%. Although occupancy improved dramatically (35% → 99%) and eligible warps increased (0.52 → 5.61), the while-loop control flow and tile_id arithmetic in Triton generated massive instruction overhead per tile iteration, making the kernel ALU-bound rather than memory-latency-bound. The implementation bug was that the grid remained at 23,520 (not reduced to 132), so 23,520 CTAs each ran the full while-loop of 23,520 iterations, executing 23,520× more work than intended.

> **Applicable when**: Never apply persistent while-loop scheduling to this kernel pattern without verifying the grid is correctly reduced to NUM_SMS. Even when correctly implemented, persistent kernels for pure memory-copy/permute workloads may not help because the bottleneck is memory bandwidth, not CTA launch overhead.

> **Avoid when**: Kernel is a pure data-movement (permute/copy) with no arithmetic and the bottleneck is memory latency or bandwidth. Persistent while-loop kernels in Triton generate significant ALU instruction overhead per loop iteration (tile_id arithmetic, loop condition check, increment) that can dominate for kernels with trivial per-tile compute. Also avoid when the grid reduction (from N tiles to NUM_SMS) is not correctly propagated to the kernel launch call — leaving the grid at the original size causes each CTA to redundantly process all tiles, multiplying total work by N.

#### ❌ Plan B: `persistent_tile_scheduling` (0.035×)

Converting a one-CTA-per-tile grid (23,520 CTAs) to a persistent kernel with 132 CTAs each iterating over tiles in a while-loop caused a 28x slowdown (17.73 µs → 474.3 µs). The persistent loop increased executed instructions by 87.7x (3.5M → 309.5M), shifting the bottleneck from memory latency to compute (SM throughput 25% → 82%), while actual memory throughput collapsed from 679 GB/s to 28 GB/s and DRAM throughput dropped from 13.85% to 0.58%. Although occupancy improved dramatically (35% → 99%) and eligible warps increased (0.52 → 5.61), the while-loop control flow and tile_id arithmetic in Triton generated massive instruction overhead per tile iteration, making the kernel ALU-bound rather than memory-latency-bound. The implementation bug was that the grid remained at 23,520 (not reduced to 132), so 23,520 CTAs each ran the full while-loop of 23,520 iterations, executing 23,520× more work than intended.

> **Applicable when**: Never apply persistent while-loop scheduling to this kernel pattern without verifying the grid is correctly reduced to NUM_SMS. Even when correctly implemented, persistent kernels for pure memory-copy/permute workloads may not help because the bottleneck is memory bandwidth, not CTA launch overhead.

> **Avoid when**: Kernel is a pure data-movement (permute/copy) with no arithmetic and the bottleneck is memory latency or bandwidth. Persistent while-loop kernels in Triton generate significant ALU instruction overhead per loop iteration (tile_id arithmetic, loop condition check, increment) that can dominate for kernels with trivial per-tile compute. Also avoid when the grid reduction (from N tiles to NUM_SMS) is not correctly propagated to the kernel launch call — leaving the grid at the original size causes each CTA to redundantly process all tiles, multiplying total work by N.

#### ❌ Plan D: `persistent_tile_scheduling` (0.035×)

Converting a one-CTA-per-tile grid (23,520 CTAs) to a persistent kernel with 132 CTAs each iterating over tiles in a while-loop caused a 28x slowdown (17.73 µs → 474.3 µs). The persistent loop increased executed instructions by 87.7x (3.5M → 309.5M), shifting the bottleneck from memory latency to compute (SM throughput 25% → 82%), while actual memory throughput collapsed from 679 GB/s to 28 GB/s and DRAM throughput dropped from 13.85% to 0.58%. Although occupancy improved dramatically (35% → 99%) and eligible warps increased (0.52 → 5.61), the while-loop control flow and tile_id arithmetic in Triton generated massive instruction overhead per tile iteration, making the kernel ALU-bound rather than memory-latency-bound. The implementation bug was that the grid remained at 23,520 (not reduced to 132), so 23,520 CTAs each ran the full while-loop of 23,520 iterations, executing 23,520× more work than intended.

> **Applicable when**: Never apply persistent while-loop scheduling to this kernel pattern without verifying the grid is correctly reduced to NUM_SMS. Even when correctly implemented, persistent kernels for pure memory-copy/permute workloads may not help because the bottleneck is memory bandwidth, not CTA launch overhead.

> **Avoid when**: Kernel is a pure data-movement (permute/copy) with no arithmetic and the bottleneck is memory latency or bandwidth. Persistent while-loop kernels in Triton generate significant ALU instruction overhead per loop iteration (tile_id arithmetic, loop condition check, increment) that can dominate for kernels with trivial per-tile compute. Also avoid when the grid reduction (from N tiles to NUM_SMS) is not correctly propagated to the kernel launch call — leaving the grid at the original size causes each CTA to redundantly process all tiles, multiplying total work by N.

---

## Cumulative Progress

```
Baseline:  16860 ns  (1.000×)

Final best: 16860 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,28 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_clone_permute_view_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3010560
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 672)
     x1 = ((xindex // 672) % 7)
     x2 = ((xindex // 4704) % 8)
     x3 = xindex // 37632
     x4 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 672*x2 + 5376*x1 + 37632*x3), None)
-    tl.store(out_ptr0 + (x4), tmp0, None)
-
-
+    tmp0 = tl.load(in_ptr0 + (x0 + 672*x2 + 5376*x1 + 37632*x3), xmask)
+    tl.store(out_ptr0 + (x4), tmp0, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 3010560
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_clone_permute_view_1[grid](arg_0, arg_1, 3010560, XBLOCK=128)
+    triton_poi_fused_clone_permute_view_1[grid](arg_0, arg_1, 3010560)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

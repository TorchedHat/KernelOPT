# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_add_convolution_mul_native_group_norm_sigmoid_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 640700 ns — **Best latency**: 641020 ns (**0.9995×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_add_convolution_mul_native_group_norm_sigmoid_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_021/kernel_2_triton_red_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 1.94
- **Registers/thread**: 127
- **Duration**: 640700 ns
- **SM Throughput**: 69.68 % | **Memory**: 77.27 % | **DRAM**: 33.81 %
- **Achieved Occupancy**: 21.93 %
- **IPC Active**: 2.88 inst/cycle | **No Eligible**: 27.84 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_r0block_to_reduce_register_pressure_and_improve_coalescing | 0.203× | 0.150× | ✓ |  |
| C | add_autotune_with_larger_xblock_and_vectorized_access | 0.384× | 0.130× | ✓ |  |
| D | autotune_xblock_r0block_to_reduce_register_pressure_and_improve_occupancy | 0.997× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_r... | Δ | add_autotune_with... | Δ | autotune_xblock_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 640700 ns | 4280000 ns | +3639300 ns (+568.0%) | 4940000 ns | +4299300 ns (+671.0%) | 641760 ns | +1060 ns (+0.2%) |
| SM Throughput (%) | 69.68 % | 80.27 % | +10.59 pp | 71.60 % | +1.92 pp | 69.81 % | +0.13 pp |
| Memory Throughput (%) | 77.27 % | 17.59 % | -59.68 pp | 15.31 % | -61.96 pp | 77.40 % | +0.13 pp |
| DRAM Throughput (%) | 33.81 % | 5.03 % | -28.78 pp | 4.37 % | -29.44 pp | 33.93 % | +0.12 pp |
| Achieved Occupancy (%) | 21.93 % | 17.12 % | -4.81 pp | 11.92 % | -10.01 pp | 21.91 % | -0.02 pp |
| Registers/Thread | 127.00 register/thread | 155.00 register/thread | +28 | 244.00 register/thread | +117 | 127.00 register/thread | +0 |
| Waves/SM | 1.94 | 2.59 | +0.65 | 3.88 | +1.94 | 1.94 | +0.00 |
| No Eligible (%) | 27.84 % | 17.28 % | -10.56 pp | 26.25 % | -1.59 pp | 27.92 % | +0.08 pp |
| IPC Active | 2.88 inst/cycle | 3.31 inst/cycle | +0.43 | 2.95 inst/cycle | +0.07 | 2.89 inst/cycle | +0.01 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_r0block_to_reduce_register_pressure_and_improve_coalescing` (0.150×)

Adding @triton.autotune with larger XBLOCK values (8–32) and smaller R0_BLOCK values (256–1024) to a reduction kernel caused a severe 6.7x slowdown. Despite the intent to reduce register pressure, the autotuner selected a configuration that increased registers/thread from 127 to 155, further reducing theoretical occupancy from 25% to 18.75% and Block Limit Registers from 4 to 3. Critically, executed instructions exploded from 348M to 2.72B (7.8x), indicating the autotuner selected a configuration with far more loop iterations or inefficient tiling that massively increased total work. The grid was not updated to use the autotuned XBLOCK (it remained fixed at triton.cdiv(xnumel, 1)=1024), so the autotuner's XBLOCK changes only affected the inner tile size without reducing grid size, causing each block to do redundant work.

> **Applicable when**: Never apply this exact pattern. The grid must be updated to a lambda using meta['XBLOCK'] when autotuning XBLOCK in a Triton reduction kernel.

> **Avoid when**: Do not add @triton.autotune for XBLOCK without simultaneously changing the grid to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. Failing to do so causes the autotuner to select large XBLOCK values that increase register pressure while the grid remains oversized, exploding total instruction count and causing severe regressions (7.8x more instructions, 6.7x slower).

#### ❌ Plan C: `add_autotune_with_larger_xblock_and_vectorized_access` (0.130×)

Replacing autotune configs with larger XBLOCK values (8–16) and num_stages=2 on a Welford reduction kernel caused registers per thread to jump from 127 to 244, cutting theoretical occupancy from 25% to 12.5% and executed instructions to balloon from 348M to 2799M (8x more work). The root cause is that the launch grid remained fixed at (cdiv(xnumel, 1),) = 1024 blocks rather than being updated to a lambda using META['XBLOCK'], so the autotuner selected larger XBLOCK values but the grid did not shrink proportionally — each CTA processed far more elements than intended. When adding or modifying autotune configs with variable XBLOCK in a Triton reduction kernel, the launch grid must be a lambda: `lambda META: (cdiv(xnumel, META['XBLOCK']),)`; failing to do so causes each block to redundantly process XBLOCK times more data, exploding both register pressure and total instruction count.

> **Applicable when**: Triton reduction kernel with XBLOCK=1 and high register pressure limiting occupancy; autotune configs are being added or modified to explore larger XBLOCK values.

> **Avoid when**: The launch grid is not simultaneously updated to a lambda using META['XBLOCK'] — this causes each CTA to process XBLOCK times more elements than intended, exploding instruction count and register pressure. Also avoid when the kernel body uses hardcoded tile-size assumptions that don't scale with XBLOCK.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_to_lambda_to_enable_xblock_scaling | 0.998× | 0.998× | ✓ |  |
| B | fix_grid_lambda_to_match_xblock_and_reduce_instruction_explosion | 1.226× | 0.928× | ✓ | **✓ best** |
| C | fix_launch_grid_to_use_meta_xblock_lambda | 0.384× | 0.130× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_grid_lambda_t... | Δ | fix_launch_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 640700 ns | 641790 ns | +1090 ns (+0.2%) | 690370 ns | +49670 ns (+7.8%) | 4930000 ns | +4289300 ns (+669.5%) |
| SM Throughput (%) | 69.68 % | 69.45 % | -0.23 pp | 68.42 % | -1.26 pp | 71.60 % | +1.92 pp |
| Memory Throughput (%) | 77.27 % | 77.69 % | +0.42 pp | 69.10 % | -8.17 pp | 15.30 % | -61.97 pp |
| DRAM Throughput (%) | 33.81 % | 33.92 % | +0.11 pp | 33.05 % | -0.76 pp | 4.37 % | -29.44 pp |
| Achieved Occupancy (%) | 21.93 % | 21.91 % | -0.02 pp | 39.24 % | +17.31 pp | 11.92 % | -10.01 pp |
| Registers/Thread | 127.00 register/thread | 127.00 register/thread | +0 | 61.00 register/thread | -66 | 244.00 register/thread | +117 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 0.97 | -0.97 | 3.88 | +1.94 |
| No Eligible (%) | 27.84 % | 27.93 % | +0.09 pp | 28.26 % | +0.42 pp | 26.26 % | -1.58 pp |
| IPC Active | 2.88 inst/cycle | 2.88 inst/cycle | +0.00 | 2.87 inst/cycle | -0.01 | 2.95 inst/cycle | +0.07 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_unlock_existing_autotune_configs | 0.997× | 0.997× | ✓ |  |
| B | fix_grid_lambda_and_increase_num_warps_to_improve_store_coalescing | 1.227× | 0.938× | ✓ | **✓ best** |
| C | fix_grid_lambda_and_restore_low_register_configs | 1.050× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 640700 ns | 642370 ns | +1670 ns (+0.3%) | 682980 ns | +42280 ns (+6.6%) | 639870 ns | -830 ns (-0.1%) |
| SM Throughput (%) | 69.68 % | 69.16 % | -0.52 pp | 68.50 % | -1.18 pp | 69.58 % | -0.10 pp |
| Memory Throughput (%) | 77.27 % | 77.40 % | +0.13 pp | 69.79 % | -7.48 pp | 77.97 % | +0.70 pp |
| DRAM Throughput (%) | 33.81 % | 33.97 % | +0.16 pp | 33.34 % | -0.47 pp | 34.01 % | +0.20 pp |
| Achieved Occupancy (%) | 21.93 % | 21.95 % | +0.02 pp | 39.26 % | +17.33 pp | 21.92 % | -0.01 pp |
| Registers/Thread | 127.00 register/thread | 127.00 register/thread | +0 | 61.00 register/thread | -66 | 127.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 0.97 | -0.97 | 1.94 | +0.00 |
| No Eligible (%) | 27.84 % | 27.89 % | +0.05 pp | 28.27 % | +0.43 pp | 27.73 % | -0.11 pp |
| IPC Active | 2.88 inst/cycle | 2.89 inst/cycle | +0.01 | 2.87 inst/cycle | -0.01 | 2.88 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_r0block_and_num_warps_to_cut_register_pressure_and_improve_occupancy | 1.183× | 0.726× | ✓ |  |
| B | fix_grid_lambda_to_enable_autotune_xblock_scaling | 0.999× | 0.996× | ✓ |  |
| C | fix_grid_lambda_to_enable_xblock_autotune_and_increase_r0block_for_fewer_iterations | 1.228× | 0.411× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_r0block_an... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 640700 ns | 882080 ns | +241380 ns (+37.7%) | 643420 ns | +2720 ns (+0.4%) | 1560000 ns | +919300 ns (+143.5%) |
| SM Throughput (%) | 69.68 % | 47.29 % | -22.39 pp | 69.46 % | -0.22 pp | 30.32 % | -39.36 pp |
| Memory Throughput (%) | 77.27 % | 52.27 % | -25.00 pp | 77.14 % | -0.13 pp | 29.70 % | -47.57 pp |
| DRAM Throughput (%) | 33.81 % | 24.44 % | -9.37 pp | 33.80 % | -0.01 pp | 13.81 % | -20.00 pp |
| Achieved Occupancy (%) | 21.93 % | 23.56 % | +1.63 pp | 21.90 % | -0.03 pp | 12.01 % | -9.92 pp |
| Registers/Thread | 127.00 register/thread | 34.00 register/thread | -93 | 127.00 register/thread | +0 | 255.00 register/thread | +128 |
| Waves/SM | 1.94 | 0.32 | -1.62 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 27.84 % | 51.39 % | +23.55 pp | 27.80 % | -0.04 pp | 69.10 % | +41.26 pp |
| IPC Active | 2.88 inst/cycle | 1.93 inst/cycle | -0.95 | 2.89 inst/cycle | +0.01 | 1.23 inst/cycle | -1.65 |


### Key Learnings

#### ❌ Plan A: `reduce_r0block_and_num_warps_to_cut_register_pressure_and_improve_occupancy` (0.726×)

Reducing R0_BLOCK from 2048 to 256/512 and num_warps from 4/8 to 2 successfully cut registers/thread from 127 to 34 and raised theoretical occupancy from 25% to 75%, but caused a 38% slowdown. The critical failure was that waves_per_sm collapsed from 1.94 to 0.32 — with only 1024 grid blocks and 132 SMs, the smaller block size (64 threads vs 128) meant the GPU was severely underutilized, with most SMs idle. Despite higher theoretical occupancy, achieved occupancy barely improved (21.92% → 23.56%), and warp cycles per issued instruction nearly doubled (4.85 → 7.82 cycles), indicating increased L1TEX stall latency (CPIStall rule triggered at 62.4% of stall cycles). When a reduction kernel's grid is fixed and small relative to SM count, reducing block size to improve occupancy backfires: fewer total threads means fewer waves, leaving the GPU starved of work and amplifying memory latency stalls.

> **Applicable when**: Never apply this pattern when the grid is fixed and small (waves_per_sm < 2) relative to SM count. Reducing block size in this regime collapses waves further and starves the GPU.

> **Avoid when**: Do not reduce num_warps or block_size to improve theoretical occupancy when waves_per_sm is already near or below 2.0 with a fixed grid. The resulting collapse in total thread count leaves SMs idle and amplifies memory latency stalls (warp cycles per issued instruction increases). The correct fix for register-pressure-limited occupancy in this regime is to reduce register usage within the kernel body (e.g., spill accumulators to shared memory, restructure the reduction loop) rather than shrinking block size.

#### ❌ Plan C: `fix_grid_lambda_to_enable_xblock_autotune_and_increase_r0block_for_fewer_iterations` (0.411×)

Adding num_warps=2 (64-thread block) configs to a Triton reduction kernel autotune caused the autotuner to select a 64-thread configuration that pushed registers per thread from 127 to 255 (the hardware maximum), halving theoretical occupancy from 25% to 12.5% and tripling kernel duration from 640us to 1560us. Despite the intent to allow more blocks per SM by reducing per-block register budget, the compiler instead allocated more registers per thread when given fewer threads per block, resulting in the same Block Limit Registers=4 constraint but with half the warps per block — net occupancy dropped. The 'No Eligible' scheduler stall rate jumped from 27.7% to 69.1%, indicating severe latency-hiding loss due to the occupancy collapse.

> **Applicable when**: Never: reducing num_warps below 4 in a high-register-pressure Triton reduction kernel does not improve occupancy when the compiler compensates by allocating more registers per thread.

> **Avoid when**: Do not add num_warps=2 (64-thread block) configs to autotune when the kernel already uses 127+ registers per thread at num_warps=4. The compiler will allocate up to 255 registers per thread with fewer threads, keeping Block Limit Registers the same but halving warps per block, collapsing occupancy and causing severe latency-hiding loss (No Eligible stalls jump from ~28% to ~69%). Also, XBLOCK>1 configs without a corrected grid lambda remain broken (same root cause as exp_001 and exp_002).

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | pipeline_reduction_loop_with_num_stages_2_and_evict_first_stores | 1.227× | 0.998× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_latency_with_fixed_xblock1_grid | 1.204× | 0.652× | ✓ |  |
| C | purge_num_warps2_and_xblock_gt1_configs_to_restore_128thread_occupancy | 1.262× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | pipeline_reductio... | Δ | increase_num_warp... | Δ | purge_num_warps2_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 640700 ns | 642110 ns | +1410 ns (+0.2%) | 982210 ns | +341510 ns (+53.3%) | 641020 ns | +320 ns (+0.0%) |
| SM Throughput (%) | 69.68 % | 69.26 % | -0.42 pp | 63.75 % | -5.93 pp | 69.49 % | -0.19 pp |
| Memory Throughput (%) | 77.27 % | 78.08 % | +0.81 pp | 47.68 % | -29.59 pp | 77.92 % | +0.65 pp |
| DRAM Throughput (%) | 33.81 % | 34.01 % | +0.20 pp | 22.39 % | -11.42 pp | 34.12 % | +0.31 pp |
| Achieved Occupancy (%) | 21.93 % | 21.90 % | -0.03 pp | 74.20 % | +52.27 pp | 21.93 % | +0.00 pp |
| Registers/Thread | 127.00 register/thread | 127.00 register/thread | +0 | 26.00 register/thread | -101 | 127.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 0.97 | -0.97 | 1.94 | +0.00 |
| No Eligible (%) | 27.84 % | 27.87 % | +0.03 pp | 30.95 % | +3.11 pp | 27.86 % | +0.02 pp |
| IPC Active | 2.88 inst/cycle | 2.88 inst/cycle | +0.00 | 2.76 inst/cycle | -0.12 | 2.89 inst/cycle | +0.01 |


### Key Learnings

#### ❌ Plan B: `increase_num_warps_to_hide_l1tex_latency_with_fixed_xblock1_grid` (0.652×)

Replacing autotune configs with num_warps=8/16 (block_size=256/512) on a XBLOCK=1 reduction kernel with a fixed 1024-block grid caused a 35% slowdown despite dramatically improving theoretical occupancy (25%→100%) and achieved occupancy (21.9%→74.2%). The critical failure was that waves_per_sm collapsed from 1.94 to 0.97 — with only 1024 grid blocks and 132 SMs, doubling block size halved the number of waves, leaving the GPU underutilized in terms of total work. Warp cycles per issued instruction tripled from 4.86 to 17.18, and a new CPIStall rule triggered at 51.6% of stall cycles attributed to L1TEX scoreboard waits, indicating that despite more warps per SM, the memory access pattern remained uncoalesced and the larger blocks amplified total instruction count (348M→498M) without proportional throughput gain.

> **Applicable when**: Never apply when the grid is fixed and small (waves_per_sm near or below 2.0) relative to SM count. Increasing num_warps in this regime halves waves_per_sm further and amplifies per-warp memory latency stalls.

> **Avoid when**: Do not increase num_warps (block_size) to hide L1TEX latency when the grid is fixed at ~1024 blocks across 132 SMs and waves_per_sm is already near 2.0. Doubling block size halves waves_per_sm to ~1.0, leaving half the SMs underutilized during the partial wave. The resulting CPI stall increase (4.86→17.18 cycles) overwhelms any latency-hiding benefit from more warps per SM. The uncoalesced access pattern (33M excessive sectors) is the root bottleneck that more warps cannot fix.

---

## Cumulative Progress

```
Baseline:  640700 ns  (1.000×)

Final best: 641020 ns  (0.9995×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_add_convolution_mul_native_group_norm_sigmoid_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 1024
@@ -29,10 +45,10 @@

         rindex = r0_index
         r0_2 = (r0_index % 4)
         r0_3 = r0_index // 4
-        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 32*r0_3 + 2064512*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (r0_2 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp3 = tl.load(in_ptr2 + (r0_2 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp5 = tl.load(in_ptr3 + (r0_2 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 32*r0_3 + 2064512*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (r0_2 + 4*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp3 = tl.load(in_ptr2 + (r0_2 + 4*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp5 = tl.load(in_ptr3 + (r0_2 + 4*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp4 = tmp2 + tmp3
         tmp6 = tmp4 * tmp5
@@ -48,10 +64,8 @@

     tmp9 = tmp10[:, None]
     tmp13 = tmp11[:, None]
     tmp14 = tmp12[:, None]
-    tl.store(out_ptr0 + (x4), tmp9, xmask)
-    tl.store(out_ptr1 + (x4), tmp13, xmask)
-
-
+    tl.store(out_ptr0 + (x4), tmp9, mask=xmask)
+    tl.store(out_ptr1 + (x4), tmp13, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -63,7 +77,7 @@

     xnumel = 1024
     r0_numel = 258064
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_add_convolution_mul_native_group_norm_sigmoid_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1024, 258064, XBLOCK=1, R0_BLOCK=2048)
+    triton_red_fused_add_convolution_mul_native_group_norm_sigmoid_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1024, 258064)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4480 ns — **Best latency**: 3580 ns (**1.2514×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_sqrt_reciprocal_with_rsqrt_and_fix_persistent_loop`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_17_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2450,1,1), Waves/SM: 1.16
- **Registers/thread**: 18
- **Duration**: 4480 ns
- **SM Throughput**: 23.90 % | **Memory**: 11.48 % | **DRAM**: 5.81 %
- **Achieved Occupancy**: 62.43 %
- **IPC Active**: 2.03 inst/cycle | **No Eligible**: 46.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_larger_xblock_autotune | 0.986× | 1.000× | ✓ |  |
| B | increase_xblock_eliminate_partial_wave | 0.957× | 1.007× | ✓ |  |
| C | fix_launch_configuration_eliminate_partial_wave | 0.989× | 1.000× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_eliminate_partial_wave | 1.034× | 0.432× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | increase_xblock_e... | Δ | fix_launch_config... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4480 ns | 4480 ns | +0 ns (+0.0%) | 4450 ns | -30 ns (-0.7%) | 4480 ns | +0 ns (+0.0%) | 10370 ns | +5890 ns (+131.5%) |
| SM Throughput (%) | 23.90 % | 41.12 % | +17.22 pp | 42.57 % | +18.67 pp | 42.06 % | +18.16 pp | 68.85 % | +44.95 pp |
| Memory Throughput (%) | 11.48 % | 13.94 % | +2.46 pp | 14.52 % | +3.04 pp | 14.31 % | +2.83 pp | 14.91 % | +3.43 pp |
| DRAM Throughput (%) | 5.81 % | 5.76 % | -0.05 pp | 5.82 % | +0.01 pp | 5.78 % | -0.03 pp | 2.49 % | -3.32 pp |
| Achieved Occupancy (%) | 62.43 % | 69.57 % | +7.14 pp | 69.51 % | +7.08 pp | 69.60 % | +7.17 pp | 82.20 % | +19.77 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 1.16 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 | 4.64 | +3.48 |
| No Eligible (%) | 46.90 % | 24.18 % | -22.72 pp | 25.68 % | -21.22 pp | 23.90 % | -23.00 pp | 16.09 % | -30.81 pp |
| IPC Active | 2.03 inst/cycle | 2.98 inst/cycle | +0.95 | 2.98 inst/cycle | +0.95 | 2.97 inst/cycle | +0.94 | 3.42 inst/cycle | +1.39 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.982× | 1.037× | ✓ |  |
| B | persistent_tile_scheduling | 0.357× | 0.241× | ✓ |  |
| C | persistent_tile_scheduling | 0.982× | 1.014× | ✓ |  |
| D | persistent_tile_scheduling_with_corrected_grid | 1.096× | 1.251× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.998× | 1.077× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4480 ns | 4320 ns | -160 ns (-3.6%) | 18560 ns | +14080 ns (+314.3%) | 4420 ns | -60 ns (-1.3%) | 3580 ns | -900 ns (-20.1%) | 4160 ns | -320 ns (-7.1%) |
| SM Throughput (%) | 23.90 % | 26.41 % | +2.51 pp | 56.77 % | +32.87 pp | 26.47 % | +2.57 pp | 19.40 % | -4.50 pp | 23.98 % | +0.08 pp |
| Memory Throughput (%) | 11.48 % | 11.48 % | +0.00 pp | 26.22 % | +14.74 pp | 11.48 % | +0.00 pp | 11.58 % | +0.10 pp | 11.41 % | -0.07 pp |
| DRAM Throughput (%) | 5.81 % | 5.98 % | +0.17 pp | 1.38 % | -4.43 pp | 5.86 % | +0.05 pp | 7.21 % | +1.40 pp | 6.18 % | +0.37 pp |
| Achieved Occupancy (%) | 62.43 % | 63.15 % | +0.72 pp | 86.45 % | +24.02 pp | 65.02 % | +2.59 pp | 57.32 % | -5.11 pp | 60.07 % | -2.36 pp |
| Registers/Thread | 18.00 register/thread | 30.00 register/thread | +12 | 28.00 register/thread | +10 | 30.00 register/thread | +12 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 1.16 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 | 2.32 | +1.16 | 1.16 | +0.00 |
| No Eligible (%) | 46.90 % | 44.81 % | -2.09 pp | 20.56 % | -26.34 pp | 44.26 % | -2.64 pp | 69.21 % | +22.31 pp | 47.17 % | +0.27 pp |
| IPC Active | 2.03 inst/cycle | 2.07 inst/cycle | +0.04 | 3.15 inst/cycle | +1.12 | 2.12 inst/cycle | +0.09 | 1.16 inst/cycle | -0.87 | 2.01 inst/cycle | -0.02 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.241×)

Converting a pointwise kernel to a persistent while-loop (each CTA iterates over multiple tiles via `tile_id += NUM_SMS`) caused a 4.2x slowdown despite the intent to eliminate the partial-wave tail. The critical bug: the grid was NOT changed from `(triton.cdiv(xnumel, 128),) = 2450` to `(NUM_SMS,) = 132`. With 2450 CTAs each running a while-loop that iterates ~19 tiles, every element was processed approximately 19x redundantly, causing executed instructions to balloon from 1.52M to 7.65M (5x increase) and duration to grow from 4.45µs to 18.56µs. The L1 hit rate improved (67% → 92%) and L2 hit rate improved (54% → 85%) as predicted, confirming the cache-reuse hypothesis was correct, but the grid mismatch completely negated any benefit.

> **Applicable when**: Never apply the persistent while-loop body transformation without simultaneously changing the grid to (NUM_SMS,). The two changes are inseparable.

> **Avoid when**: The grid is not updated from (cdiv(xnumel, XBLOCK),) to (NUM_SMS,). Applying only the while-loop body without the grid change causes every element to be processed NUM_TILES times redundantly (5x instruction explosion observed here). Also avoid when the kernel duration is already very short (<5µs) and latency-dominated, as the while-loop overhead may offset partial-wave savings.

#### ❌ Plan B: `persistent_tile_scheduling` (0.241×)

Converting a pointwise kernel to a persistent while-loop (each CTA iterates over multiple tiles via `tile_id += NUM_SMS`) caused a 4.2x slowdown despite the intent to eliminate the partial-wave tail. The critical bug: the grid was NOT changed from `(triton.cdiv(xnumel, 128),) = 2450` to `(NUM_SMS,) = 132`. With 2450 CTAs each running a while-loop that iterates ~19 tiles, every element was processed approximately 19x redundantly, causing executed instructions to balloon from 1.52M to 7.65M (5x increase) and duration to grow from 4.45µs to 18.56µs. The L1 hit rate improved (67% → 92%) and L2 hit rate improved (54% → 85%) as predicted, confirming the cache-reuse hypothesis was correct, but the grid mismatch completely negated any benefit.

> **Applicable when**: Never apply the persistent while-loop body transformation without simultaneously changing the grid to (NUM_SMS,). The two changes are inseparable.

> **Avoid when**: The grid is not updated from (cdiv(xnumel, XBLOCK),) to (NUM_SMS,). Applying only the while-loop body without the grid change causes every element to be processed NUM_TILES times redundantly (5x instruction explosion observed here). Also avoid when the kernel duration is already very short (<5µs) and latency-dominated, as the while-loop overhead may offset partial-wave savings.

#### ❌ Plan C: `persistent_tile_scheduling` (0.241×)

Converting a pointwise kernel to a persistent while-loop (each CTA iterates over multiple tiles via `tile_id += NUM_SMS`) caused a 4.2x slowdown despite the intent to eliminate the partial-wave tail. The critical bug: the grid was NOT changed from `(triton.cdiv(xnumel, 128),) = 2450` to `(NUM_SMS,) = 132`. With 2450 CTAs each running a while-loop that iterates ~19 tiles, every element was processed approximately 19x redundantly, causing executed instructions to balloon from 1.52M to 7.65M (5x increase) and duration to grow from 4.45µs to 18.56µs. The L1 hit rate improved (67% → 92%) and L2 hit rate improved (54% → 85%) as predicted, confirming the cache-reuse hypothesis was correct, but the grid mismatch completely negated any benefit.

> **Applicable when**: Never apply the persistent while-loop body transformation without simultaneously changing the grid to (NUM_SMS,). The two changes are inseparable.

> **Avoid when**: The grid is not updated from (cdiv(xnumel, XBLOCK),) to (NUM_SMS,). Applying only the while-loop body without the grid change causes every element to be processed NUM_TILES times redundantly (5x instruction explosion observed here). Also avoid when the kernel duration is already very short (<5µs) and latency-dominated, as the while-loop overhead may offset partial-wave savings.

---

## Iteration 3

**Best so far**: 1.2514× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_to_reduce_instruction_count_and_register_pressure | 1.114× | 1.262× | ✓ |  |
| B | replace_sqrt_reciprocal_with_rsqrt_and_fix_persistent_loop | 1.141× | 1.251× | ✓ | **✓ best** |
| C | replace_sqrt_reciprocal_chain_with_rsqrt_for_fma_fusion | 1.052× | 1.103× | ✓ |  |
| D | fix_persistent_grid_mismatch_and_replace_sqrt_with_rsqrt | 1.014× | 1.093× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_to_red... | Δ | replace_sqrt_reci... | Δ | replace_sqrt_reci... | Δ | fix_persistent_gr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4480 ns | 3550 ns | -930 ns (-20.8%) | 3580 ns | -900 ns (-20.1%) | 4060 ns | -420 ns (-9.4%) | 4100 ns | -380 ns (-8.5%) |
| SM Throughput (%) | 23.90 % | 20.70 % | -3.20 pp | 20.60 % | -3.30 pp | 20.03 % | -3.87 pp | 20.10 % | -3.80 pp |
| Memory Throughput (%) | 11.48 % | 12.68 % | +1.20 pp | 12.16 % | +0.68 pp | 12.62 % | +1.14 pp | 12.60 % | +1.12 pp |
| DRAM Throughput (%) | 5.81 % | 7.27 % | +1.46 pp | 7.23 % | +1.42 pp | 6.35 % | +0.54 pp | 6.34 % | +0.53 pp |
| Achieved Occupancy (%) | 62.43 % | 52.93 % | -9.50 pp | 58.69 % | -3.74 pp | 65.52 % | +3.09 pp | 65.57 % | +3.14 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 30.00 register/thread | +12 | 30.00 register/thread | +12 |
| Waves/SM | 1.16 | 2.32 | +1.16 | 2.32 | +1.16 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 46.90 % | 68.39 % | +21.49 pp | 69.71 % | +22.81 pp | 56.90 % | +10.00 pp | 56.56 % | +9.66 pp |
| IPC Active | 2.03 inst/cycle | 1.15 inst/cycle | -0.88 | 1.13 inst/cycle | -0.90 | 1.64 inst/cycle | -0.39 | 1.70 inst/cycle | -0.33 |


### Key Learnings

#### ✅ Plan A: `fma_fusion_to_reduce_instruction_count_and_register_pressure` (1.262×)

Replacing a three-step multiply-multiply-add chain (tmp11=a*b; tmp13=tmp11*c; tmp15=tmp13+d) with a two-step FMA sequence (tmp_combined=b*c; tmp15=libdevice.fma(a, tmp_combined, d)) reduced total SM elapsed cycles from 775K to 726K (~6.3% reduction) and duration from 3.58µs to 3.55µs, yielding a 1.26x speedup. The instruction count (363K) remained identical, suggesting the gain came from shortening the critical dependency chain by eliminating one intermediate live register (tmp13), reducing the latency path through the arithmetic. The FPInstructions NCU rule flagged 29,472 non-fused FP32 instructions (75% of total) as the optimization target; libdevice.fma() maps to PTX fma.rn.f32 which executes as a single fused operation on H200 ALU pipelines.

> **Applicable when**: Triton kernel where the FPInstructions NCU rule is triggered with a low FMA fusion rate (<50% fused instructions), and the arithmetic contains back-to-back mul+add pairs (a*b + c pattern). Especially effective when one operand in the chain is a broadcast scalar that can be pre-combined to expose the FMA structure. libdevice must already be imported.

> **Avoid when**: The PTX compiler already performs automatic FMA fusion (verify via NCU FPInstructions rule — if it is not triggered, fusion is already happening). Also avoid when catastrophic cancellation is a concern (though for BN normalization with variance >= 0, fma.rn.f32 is numerically safe). Do not apply if the dependency chain restructuring changes the order of floating-point operations in a precision-sensitive context.

---

## Iteration 4

**Best so far**: 1.2514× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_activate_persistent_loop_tile_reduction | 1.107× | 1.238× | ✓ |  |
| B | fix_persistent_grid_to_use_num_sms_not_cdiv | 1.139× | 1.238× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_stall_latency | 1.051× | 1.093× | ✓ |  |
| D | fix_persistent_grid_launch_and_fma_fusion | 1.049× | 1.103× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_persistent_gr... | Δ | increase_num_warp... | Δ | fix_persistent_gr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4480 ns | 3620 ns | -860 ns (-19.2%) | 3620 ns | -860 ns (-19.2%) | 4100 ns | -380 ns (-8.5%) | 4060 ns | -420 ns (-9.4%) |
| SM Throughput (%) | 23.90 % | 20.56 % | -3.34 pp | 20.49 % | -3.41 pp | 19.65 % | -4.25 pp | 20.10 % | -3.80 pp |
| Memory Throughput (%) | 11.48 % | 12.52 % | +1.04 pp | 12.11 % | +0.63 pp | 12.34 % | +0.86 pp | 12.58 % | +1.10 pp |
| DRAM Throughput (%) | 5.81 % | 7.18 % | +1.37 pp | 7.15 % | +1.34 pp | 6.29 % | +0.48 pp | 6.36 % | +0.55 pp |
| Achieved Occupancy (%) | 62.43 % | 53.18 % | -9.25 pp | 52.75 % | -9.68 pp | 63.30 % | +0.87 pp | 66.11 % | +3.68 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 30.00 register/thread | +12 | 30.00 register/thread | +12 |
| Waves/SM | 1.16 | 2.32 | +1.16 | 2.32 | +1.16 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 46.90 % | 67.59 % | +20.69 pp | 68.75 % | +21.85 pp | 59.52 % | +12.62 pp | 55.77 % | +8.87 pp |
| IPC Active | 2.03 inst/cycle | 1.26 inst/cycle | -0.77 | 1.25 inst/cycle | -0.78 | 1.64 inst/cycle | -0.39 | 1.71 inst/cycle | -0.32 |

---

## Iteration 5

**Best so far**: 1.2514× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_activate_autotune_and_persistent_loop | 1.066× | 1.238× | ✓ |  |
| B | fix_kernel_function_launch_wrapper_to_activate_persistent_loop | 1.085× | 1.217× | ✓ |  |
| C | fix_persistent_grid_launch_to_num_sms_only | 1.012× | 1.052× | ✓ |  |
| D | fma_fusion_precompute_broadcast_scalar_product | 1.052× | 1.069× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_kernel_functi... | Δ | fix_persistent_gr... | Δ | fma_fusion_precom... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4480 ns | 3620 ns | -860 ns (-19.2%) | 3680 ns | -800 ns (-17.9%) | 4260 ns | -220 ns (-4.9%) | 4190 ns | -290 ns (-6.5%) |
| SM Throughput (%) | 23.90 % | 20.92 % | -2.98 pp | 20.67 % | -3.23 pp | 18.55 % | -5.35 pp | 19.89 % | -4.01 pp |
| Memory Throughput (%) | 11.48 % | 12.56 % | +1.08 pp | 12.21 % | +0.73 pp | 11.67 % | +0.19 pp | 12.53 % | +1.05 pp |
| DRAM Throughput (%) | 5.81 % | 7.17 % | +1.36 pp | 7.04 % | +1.23 pp | 6.09 % | +0.28 pp | 6.16 % | +0.35 pp |
| Achieved Occupancy (%) | 62.43 % | 53.17 % | -9.26 pp | 58.52 % | -3.91 pp | 64.79 % | +2.36 pp | 63.70 % | +1.27 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 30.00 register/thread | +12 | 28.00 register/thread | +10 |
| Waves/SM | 1.16 | 2.32 | +1.16 | 2.32 | +1.16 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 46.90 % | 67.66 % | +20.76 pp | 68.51 % | +21.61 pp | 55.66 % | +8.76 pp | 56.69 % | +9.79 pp |
| IPC Active | 2.03 inst/cycle | 1.25 inst/cycle | -0.78 | 1.14 inst/cycle | -0.89 | 1.69 inst/cycle | -0.34 | 1.63 inst/cycle | -0.40 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_to_activate_autotune_and_persistent_loop` (1.238×)

Changing the Triton launch grid from a hardcoded static tuple (e.g., `(triton.cdiv(xnumel, 128),)`) to a lambda (`lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`) enables autotune to properly select XBLOCK and scale the grid accordingly. In this case, the individual kernel's NCU metrics were nearly identical before and after (same grid size 2450, same block size 256, same duration 3.62µs, same instruction count 377K), indicating autotune selected the same config — but the system-level benchmark showed 1.2376x speedup, likely from reduced autotune dispatch overhead or improved scheduling of surrounding kernels. The key structural insight is that a hardcoded grid with a persistent loop body makes `tl.num_programs(0)` equal to the full tile count (not NUM_SMS), rendering the persistent loop a no-op; the grid lambda is a prerequisite for autotune to shrink the grid and activate the persistent pattern.

> **Applicable when**: Triton kernel with @triton.autotune decorator and a persistent loop body (for/while tile_id in range(pid, num_tiles, tl.num_programs(0))), where the launch grid is hardcoded as a static tuple using a fixed block size instead of a lambda using meta['XBLOCK']. The IssueSlotUtilization or LaunchConfiguration NCU rules are triggered.

> **Avoid when**: The grid is already a lambda using meta['XBLOCK']. Also avoid if the kernel does not have a persistent loop body — changing to a lambda grid without a persistent loop will cause each element to be processed multiple times if XBLOCK > the original fixed block size (redundant work). Do not apply if autotune is not present on the kernel.

---

## Cumulative Progress

```
Baseline:  4480 ns  (1.000×)
Iter 2     3580 ns  (1.251×)  — persistent_tile_scheduling_with_corrected_grid

Final best: 3580 ns  (1.2514×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,33 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
-def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 313600
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x2 = xindex
-    x0 = (xindex % 40)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
-    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
-
-
+def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
+    pid = tl.program_id(0)
+    num_sms = tl.num_programs(0)
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    for tile_id in range(pid, num_tiles, num_sms):
+        xoffset = tile_id * XBLOCK
+        xindex  = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask   = xindex < xnumel
+        x0      = xindex % 40
+        tmp0  = tl.load(in_out_ptr0 + xindex, xmask, eviction_policy='evict_first')
+        tmp1  = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
+        tmp3  = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
+        tmp12 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
+        tmp14 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
+        tmp2  = tmp0 - tmp1
+        tmp5  = tmp3 + tl.full([1], 1e-05, tl.float32)
+        tmp10 = tl.rsqrt(tmp5)
+        tmp_combined = tmp10 * tmp12
+        tmp15 = libdevice.fma(tmp2, tmp_combined, tmp14)
+        tl.store(in_out_ptr0 + xindex, tmp15, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +45,7 @@

     arg_4 = inputs[4]
     xnumel = 313600
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 313600, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 313600)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

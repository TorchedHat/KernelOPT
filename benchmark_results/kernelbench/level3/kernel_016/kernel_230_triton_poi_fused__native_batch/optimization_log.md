# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_230` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7140 ns — **Best latency**: 6850 ns (**1.0423×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`streaming_evict_first_on_main_load_to_protect_broadcast_cache`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_230`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_230_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7105,1,1), Waves/SM: 3.36
- **Registers/thread**: 22
- **Duration**: 7140 ns
- **SM Throughput**: 42.38 % | **Memory**: 15.99 % | **DRAM**: 10.53 %
- **Achieved Occupancy**: 62.71 %
- **IPC Active**: 2.61 inst/cycle | **No Eligible**: 34.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.908× | 0.778× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_occupancy_and_latency_hiding | 0.965× | 1.029× | ✓ |  |
| C | vectorized_loads_via_larger_xblock_autotune | 0.996× | 0.469× | ✓ |  |
| D | autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.972× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 9180 ns | +2040 ns (+28.6%) | 6940 ns | -200 ns (-2.8%) | 15230 ns | +8090 ns (+113.3%) | 7070 ns | -70 ns (-1.0%) |
| SM Throughput (%) | 42.38 % | 63.63 % | +21.25 pp | 34.02 % | -8.36 pp | 72.13 % | +29.75 pp | 43.61 % | +1.23 pp |
| Memory Throughput (%) | 15.99 % | 17.17 % | +1.18 pp | 23.28 % | +7.29 pp | 15.85 % | -0.14 pp | 16.69 % | +0.70 pp |
| DRAM Throughput (%) | 10.53 % | 8.18 % | -2.35 pp | 10.81 % | +0.28 pp | 4.92 % | -5.61 pp | 10.60 % | +0.07 pp |
| Achieved Occupancy (%) | 62.71 % | 80.98 % | +18.27 pp | 43.33 % | -19.38 pp | 84.63 % | +21.92 pp | 62.58 % | -0.13 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 24.00 register/thread | +2 | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 3.36 | 3.36 | +0.00 | 3.36 | +0.00 | 6.73 | +3.37 | 3.36 | +0.00 |
| No Eligible (%) | 34.55 % | 21.57 % | -12.98 pp | 48.49 % | +13.94 pp | 15.03 % | -19.52 pp | 35.47 % | +0.92 pp |
| IPC Active | 2.61 inst/cycle | 3.26 inst/cycle | +0.65 | 2.08 inst/cycle | -0.53 | 3.38 inst/cycle | +0.77 | 2.60 inst/cycle | -0.01 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.792×)

Replacing tl.rsqrt() with a manual sqrt_rn + integer division sequence, combined with adding @triton.autotune with larger XBLOCK configs (512–2048), increased registers per thread from 24 to 32 and inflated executed instructions from 1.7M to 4.6M (2.7x), extending kernel duration from 6.94µs to 9.02µs despite occupancy improving from 43.3% to 80.9%. The kernel body rewrite introduced redundant integer/float type conversions and replaced a single high-throughput intrinsic (tl.rsqrt) with a multi-step sequence, causing the compiler to allocate more registers and emit far more instructions. Additionally, the grid was not updated to a lambda reading meta['XBLOCK'], so the fixed grid at cdiv(xnumel, 128) caused each CTA to process more elements than intended under larger XBLOCK configs, further inflating instruction count. Both the kernel body change and the grid mismatch contributed to the regression.

> **Applicable when**: Kernel uses tl.rsqrt() and has low register count; temptation to replace with manual sqrt+reciprocal for precision or compatibility reasons.

> **Avoid when**: Always avoid replacing tl.rsqrt() with manual sqrt_rn + integer division — this is strictly worse: it increases register pressure, introduces type conversion overhead, and inflates instruction count. Never mix int32 and float32 in arithmetic sequences that feed into tl.maximum() with float tensors. Do not apply autotune XBLOCK changes without simultaneously updating the grid to a lambda.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_correct_xblock_selection | 1.014× | 1.038× | ✓ | **✓ best** |
| B | increase_num_warps_and_num_stages_for_latency_hiding | 1.001× | 1.010× | ✓ |  |
| C | fix_autotune_grid_lambda_to_match_xblock | 0.904× | 0.792× | ✓ |  |
| D | eliminate_nonfused_fp32_instructions_via_rsqrt_and_fp32_constants | 1.013× | 0.746× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | increase_num_warp... | Δ | fix_autotune_grid... | Δ | eliminate_nonfuse... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 6880 ns | -260 ns (-3.6%) | 7070 ns | -70 ns (-1.0%) | 9020 ns | +1880 ns (+26.3%) | 9570 ns | +2430 ns (+34.0%) |
| SM Throughput (%) | 42.38 % | 33.57 % | -8.81 pp | 43.83 % | +1.45 pp | 64.32 % | +21.94 pp | 57.49 % | +15.11 pp |
| Memory Throughput (%) | 15.99 % | 16.93 % | +0.94 pp | 16.75 % | +0.76 pp | 17.37 % | +1.38 pp | 27.57 % | +11.58 pp |
| DRAM Throughput (%) | 10.53 % | 10.94 % | +0.41 pp | 10.64 % | +0.11 pp | 8.32 % | -2.21 pp | 7.85 % | -2.68 pp |
| Achieved Occupancy (%) | 62.71 % | 48.85 % | -13.86 pp | 60.89 % | -1.82 pp | 80.86 % | +18.15 pp | 80.17 % | +17.46 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 3.36 | 3.36 | +0.00 | 3.36 | +0.00 | 3.36 | +0.00 | 6.73 | +3.37 |
| No Eligible (%) | 34.55 % | 55.98 % | +21.43 pp | 34.86 % | +0.31 pp | 18.89 % | -15.66 pp | 27.17 % | -7.38 pp |
| IPC Active | 2.61 inst/cycle | 1.72 inst/cycle | -0.89 | 2.55 inst/cycle | -0.06 | 3.27 inst/cycle | +0.66 | 2.81 inst/cycle | +0.20 |

---

## Iteration 3

**Best so far**: 1.0378× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_enable_autotune_xblock_selection | 0.976× | 1.038× | ✓ |  |
| B | collapse_redundant_reciprocal_sqrt_to_tl_rsqrt_intrinsic | 0.983× | 1.042× | ✓ |  |
| C | fix_kernel_body_and_grid_lambda_to_restore_rsqrt_intrinsic | 1.001× | 0.992× | ✓ |  |
| D | pin_xblock_128_to_restore_low_register_baseline | 1.014× | 1.042× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | collapse_redundan... | Δ | fix_kernel_body_a... | Δ | pin_xblock_128_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 6880 ns | -260 ns (-3.6%) | 6850 ns | -290 ns (-4.1%) | 7200 ns | +60 ns (+0.8%) | 6850 ns | -290 ns (-4.1%) |
| SM Throughput (%) | 42.38 % | 33.07 % | -9.31 pp | 34.12 % | -8.26 pp | 37.98 % | -4.40 pp | 34.08 % | -8.30 pp |
| Memory Throughput (%) | 15.99 % | 16.63 % | +0.64 pp | 17.23 % | +1.24 pp | 24.41 % | +8.42 pp | 17.23 % | +1.24 pp |
| DRAM Throughput (%) | 10.53 % | 10.95 % | +0.42 pp | 10.97 % | +0.44 pp | 10.40 % | -0.13 pp | 10.97 % | +0.44 pp |
| Achieved Occupancy (%) | 62.71 % | 49.36 % | -13.35 pp | 49.42 % | -13.29 pp | 47.12 % | -15.59 pp | 49.47 % | -13.24 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 32.00 register/thread | +10 | 20.00 register/thread | -2 |
| Waves/SM | 3.36 | 3.36 | +0.00 | 3.36 | +0.00 | 3.36 | +0.00 | 3.36 | +0.00 |
| No Eligible (%) | 34.55 % | 58.85 % | +24.30 pp | 56.15 % | +21.60 pp | 36.49 % | +1.94 pp | 57.62 % | +23.07 pp |
| IPC Active | 2.61 inst/cycle | 1.74 inst/cycle | -0.87 | 1.70 inst/cycle | -0.91 | 2.49 inst/cycle | -0.12 | 1.74 inst/cycle | -0.87 |


### Key Learnings

#### ✅ Plan B: `collapse_redundant_reciprocal_sqrt_to_tl_rsqrt_intrinsic` (1.042×)

When a pointwise kernel is latency-bound (CPIStall ~53%, low DRAM utilization ~11%) and register pressure is already near the block-limit boundary (20 registers/thread, Block Limit Registers = 21 blocks/SM), locking XBLOCK to the known-good value (128) while sweeping only num_stages and num_warps in the autotune configs avoids register inflation and instruction count explosion. The actual causal change in this diff was replacing the 5-instruction manual reciprocal-sqrt chain (sqrt_rn + int32 constant + division + float constant + multiply) with tl.rsqrt() AND simultaneously constraining autotune to XBLOCK=128-only configs with varying num_stages (2/3/4) and num_warps (4/8). Duration dropped marginally (6.88→6.85 µs) with no change in executed instruction count or registers per thread, suggesting the speedup came primarily from the autotune finding a better num_stages/num_warps combination rather than the arithmetic simplification alone. The key insight is: when prior iterations showed XBLOCK increases cause register spill, fix XBLOCK and only tune pipeline depth (num_stages) and warp count.

> **Applicable when**: Kernel is latency-bound (CPIStall >50%, low DRAM throughput <15%), registers per thread is near the Block Limit Registers boundary, and prior autotune experiments with larger XBLOCK caused register inflation and instruction count explosion. The kernel contains a manual reciprocal-sqrt sequence (sqrt_rn + int division + float multiply) that can be collapsed to tl.rsqrt().

> **Avoid when**: Kernel is memory-bandwidth-bound (DRAM throughput >60%) where larger XBLOCK would improve coalescing. Do not apply when registers per thread is well below the block limit (e.g., <16 registers with limit at 32+), as there is headroom to increase XBLOCK safely. Do not apply when the kernel duration is dominated by compute throughput rather than L1TEX scoreboard stalls.

---

## Iteration 4

**Best so far**: 1.0423× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | align_grid_to_wave_boundary_to_eliminate_partial_wave | 1.012× | 1.038× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_scoreboard_latency | 1.011× | 1.010× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_scoreboard_stalls | 1.014× | 1.038× | ✓ |  |
| D | eliminate_partial_wave_via_grid_alignment_to_sm_count | 1.017× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | align_grid_to_wav... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 6880 ns | -260 ns (-3.6%) | 7070 ns | -70 ns (-1.0%) | 6880 ns | -260 ns (-3.6%) | 7140 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 42.38 % | 34.07 % | -8.31 pp | 33.97 % | -8.41 pp | 33.57 % | -8.81 pp | 34.09 % | -8.29 pp |
| Memory Throughput (%) | 15.99 % | 17.24 % | +1.25 pp | 17.21 % | +1.22 pp | 16.93 % | +0.94 pp | 17.21 % | +1.22 pp |
| DRAM Throughput (%) | 10.53 % | 10.91 % | +0.38 pp | 10.58 % | +0.05 pp | 10.91 % | +0.38 pp | 10.52 % | -0.01 pp |
| Achieved Occupancy (%) | 62.71 % | 48.89 % | -13.82 pp | 49.03 % | -13.68 pp | 48.64 % | -14.07 pp | 51.04 % | -11.67 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 20.00 register/thread | -2 |
| Waves/SM | 3.36 | 3.36 | +0.00 | 3.36 | +0.00 | 3.36 | +0.00 | 3.36 | +0.00 |
| No Eligible (%) | 34.55 % | 56.11 % | +21.56 pp | 56.54 % | +21.99 pp | 56.24 % | +21.69 pp | 58.69 % | +24.14 pp |
| IPC Active | 2.61 inst/cycle | 1.73 inst/cycle | -0.88 | 1.74 inst/cycle | -0.87 | 1.71 inst/cycle | -0.90 | 1.65 inst/cycle | -0.96 |

---

## Iteration 5

**Best so far**: 1.0423× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_evict_first_on_main_load_to_protect_broadcast_cache | 1.018× | 1.042× | ✓ | **✓ best** |
| B | add_evict_first_on_streaming_load_to_improve_l2_reuse_for_channel_params | 1.012× | 1.047× | ✓ |  |
| C | streaming_eviction_policy_for_main_tensor_to_improve_l1_hit_rate_for_channel_params | 1.016× | 1.042× | ✓ |  |
| D | fuse_fp32_multiply_add_chains_to_fma_instructions | 1.014× | 1.042× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evict_f... | Δ | add_evict_first_o... | Δ | streaming_evictio... | Δ | fuse_fp32_multipl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 6850 ns | -290 ns (-4.1%) | 6820 ns | -320 ns (-4.5%) | 6850 ns | -290 ns (-4.1%) | 6850 ns | -290 ns (-4.1%) |
| SM Throughput (%) | 42.38 % | 33.55 % | -8.83 pp | 33.74 % | -8.64 pp | 33.21 % | -9.17 pp | 33.86 % | -8.52 pp |
| Memory Throughput (%) | 15.99 % | 16.92 % | +0.93 pp | 17.08 % | +1.09 pp | 16.74 % | +0.75 pp | 17.19 % | +1.20 pp |
| DRAM Throughput (%) | 10.53 % | 10.95 % | +0.42 pp | 11.04 % | +0.51 pp | 10.95 % | +0.42 pp | 10.94 % | +0.41 pp |
| Achieved Occupancy (%) | 62.71 % | 51.54 % | -11.17 pp | 50.84 % | -11.87 pp | 49.17 % | -13.54 pp | 49.34 % | -13.37 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 18.00 register/thread | -4 |
| Waves/SM | 3.36 | 3.36 | +0.00 | 3.36 | +0.00 | 3.36 | +0.00 | 3.36 | +0.00 |
| No Eligible (%) | 34.55 % | 51.98 % | +17.43 pp | 52.55 % | +18.00 pp | 55.34 % | +20.79 pp | 57.64 % | +23.09 pp |
| IPC Active | 2.61 inst/cycle | 1.90 inst/cycle | -0.71 | 1.80 inst/cycle | -0.81 | 1.88 inst/cycle | -0.73 | 1.74 inst/cycle | -0.87 |


### Key Learnings

#### ✅ Plan C: `streaming_eviction_policy_for_main_tensor_to_improve_l1_hit_rate_for_channel_params` (1.042×)

When a kernel loads a large streaming tensor (read-once, no reuse) alongside small channel-parameter tensors that are reused across spatial positions, marking the streaming load with eviction_policy='evict_first' (maps to CUDA .cs cache hint) reduces L1 cache pollution and preserves capacity for the reused tensors. In this case, Warp Cycles Per Issued Instruction dropped from 18.15 to 16.44 cycles and eligible warps per scheduler improved from 1.11 to 1.21, indicating reduced L1TEX scoreboard stall pressure. The change is zero-cost in terms of registers and instruction count (no arithmetic change), making it a safe micro-optimization for latency-bound kernels dominated by CPIStall from L1TEX dependencies. Note: the L2 hit rate slightly decreased (62.46→60.94%) and executed instructions increased (1.51M→1.65M), suggesting the compiler may have reordered or expanded some instructions; the net effect was a modest 1.04x speedup.

> **Applicable when**: Kernel loads a large streaming tensor (read-once, much larger than L1 cache) alongside small tensors that are reused across spatial positions or waves. CPIStall rule is triggered with L1TEX scoreboard stalls >50% of warp cycles. L1/TEX hit rate is ~50% or below. The kernel is latency-bound (low DRAM throughput <15%, low compute throughput <40%).

> **Avoid when**: All tensors are similarly sized and accessed with similar reuse patterns — evict_first on one tensor will not selectively benefit others. Do not apply when the kernel is DRAM bandwidth-bound (DRAM throughput >60%), as cache eviction policy changes have negligible effect in that regime. Do not apply when the 'streaming' tensor actually has intra-CTA reuse (e.g., accessed multiple times per thread block).

#### ✅ Plan D: `fuse_fp32_multiply_add_chains_to_fma_instructions` (1.042×)

Replacing a 4-step BN arithmetic chain (sub→mul→mul→add) with two explicit libdevice.fma() calls reduced registers per thread from 20 to 18, slightly improved achieved occupancy (48.9%→49.3%), and cut Warp Cycles Per Issued Instruction from 18.15 to 17.62. Notably, executed instruction count did not decrease (it actually rose slightly from 1,506,260 to 1,534,680), indicating the compiler was already partially fusing the arithmetic — the benefit came from reduced register pressure rather than instruction count reduction. The speedup of ~1.04x is modest and mirrors prior iterations, suggesting this kernel is fundamentally latency-bound by L1TEX scoreboard stalls that are not addressable through arithmetic rewriting alone.

> **Applicable when**: Kernel has a multi-step FP32 arithmetic chain (e.g., BN normalization: sub→mul→mul→add) and registers per thread are near the Block Limit Registers boundary. FPInstructions NCU rule reports a high ratio of non-fused to fused FP32 instructions (>3:1). Warp Cycles Per Issued Instruction is elevated (>15) due to L1TEX scoreboard stalls.

> **Avoid when**: Kernel is already at low register count (well below block limit boundary) and the primary bottleneck is DRAM bandwidth rather than L1TEX stalls. Do not expect large instruction count reductions — the Triton compiler may already fuse arithmetic internally, so the main benefit is register pressure reduction (2 registers saved in this case). Do not apply if the arithmetic chain cannot be safely rewritten as FMA without changing numerical semantics.

---

## Cumulative Progress

```
Baseline:  7140 ns  (1.000×)
Iter 2     6880 ns  (1.038×)  — fix_autotune_grid_lambda_to_enable_correct_xblock_selection
Iter 3     6850 ns  (1.042×)  — pin_xblock_128_to_restore_low_register_baseline

Final best: 6850 ns  (1.0423×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_230(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 909440
@@ -13,7 +25,7 @@

     xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 1856)
-    tmp0 = tl.load(in_ptr0 + (x2), xmask)
+    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
     tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
@@ -21,19 +33,13 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = tl.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -44,7 +50,7 @@

     arg_5 = inputs[5]
     xnumel = 909440
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_230[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 909440, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_230[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 909440)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

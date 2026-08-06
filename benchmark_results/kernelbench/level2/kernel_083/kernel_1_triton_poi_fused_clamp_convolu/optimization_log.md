# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clamp_convolution_lift_fresh_minimum_native_group_norm_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 531550 ns — **Best latency**: 531650 ns (**0.9998×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clamp_convolution_lift_fresh_minimum_native_group_norm_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_083/kernel_1_triton_poi_fused_clamp_convolu/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (861056,1,1), Waves/SM: 407.70
- **Registers/thread**: 22
- **Duration**: 531550 ns
- **SM Throughput**: 58.98 % | **Memory**: 33.06 % | **DRAM**: 33.06 %
- **Achieved Occupancy**: 47.06 %
- **IPC Active**: 1.72 inst/cycle | **No Eligible**: 57.05 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_for_latency_hiding | 0.998× | 0.991× | ✓ |  |
| B | increase_xblock_with_autotune_to_reduce_launch_overhead_and_hide_l1tex_latency | 0.655× | 0.526× | ✓ |  |
| C | vectorized_loads_via_larger_xblock_autotune | 0.838× | 0.693× | ✓ |  |
| D | autotune_xblock_increase_for_latency_hiding | 0.838× | 0.696× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_w... | Δ | vectorized_loads_... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 531550 ns | 536640 ns | +5090 ns (+1.0%) | 1010000 ns | +478450 ns (+90.0%) | 767550 ns | +236000 ns (+44.4%) | 763970 ns | +232420 ns (+43.7%) |
| SM Throughput (%) | 58.98 % | 77.62 % | +18.64 pp | 81.56 % | +22.58 pp | 83.60 % | +24.62 pp | 83.86 % | +24.88 pp |
| Memory Throughput (%) | 33.06 % | 46.69 % | +13.63 pp | 45.61 % | +12.55 pp | 56.30 % | +23.24 pp | 56.48 % | +23.42 pp |
| DRAM Throughput (%) | 33.06 % | 32.73 % | -0.33 pp | 17.42 % | -15.64 pp | 22.80 % | -10.26 pp | 22.90 % | -10.16 pp |
| Achieved Occupancy (%) | 47.06 % | 50.23 % | +3.17 pp | 77.47 % | +30.41 pp | 83.60 % | +36.54 pp | 83.71 % | +36.65 pp |
| Registers/Thread | 22.00 register/thread | 30.00 register/thread | +8 | 30.00 register/thread | +8 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 407.70 | 407.70 | +0.00 | 815.39 | +407.69 | 407.70 | +0.00 | 407.70 | +0.00 |
| No Eligible (%) | 57.05 % | 21.64 % | -35.41 pp | 18.32 % | -38.73 pp | 16.27 % | -40.78 pp | 15.96 % | -41.09 pp |
| IPC Active | 1.72 inst/cycle | 3.12 inst/cycle | +1.40 | 3.27 inst/cycle | +1.55 | 3.35 inst/cycle | +1.63 | 3.36 inst/cycle | +1.64 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_with_autotune_to_reduce_launch_overhead_and_hide_l1tex_latency` (0.526×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel that already has a fixed-size grid caused a 3.6x increase in executed instructions (179M → 647M) and nearly doubled kernel duration (531 µs → 1010 µs). The regression occurred because the launch grid was not updated to use meta['XBLOCK'] — it remained hardcoded to cdiv(xnumel, 128) — so each CTA processed 256 threads (the autotuner selected block_size=256) over the same 128-element tile, causing massive redundant work. Additionally, changing the mask from tl.full([XBLOCK], True) to xindex < xnumel introduced per-element predication overhead that disabled compiler optimizations relying on the unconditional mask. When autotune is added, the grid lambda MUST be updated to use meta['XBLOCK']; failing to do so causes each block to execute far more instructions than needed.

> **Applicable when**: Never applicable in this form — this is a negative example documenting an implementation flaw.

> **Avoid when**: Do NOT add @triton.autotune with variable XBLOCK configs without simultaneously updating the launch grid from a static cdiv(xnumel, FIXED_BLOCK) to a dynamic lambda meta: (cdiv(xnumel, meta['XBLOCK']),). Also do NOT change tl.full([XBLOCK], True, tl.int1) to xindex < xnumel when the problem size is guaranteed divisible by all candidate XBLOCK values — the unconditional mask enables compiler optimizations.

#### ❌ Plan C: `vectorized_loads_via_larger_xblock_autotune` (0.693×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to allocate more registers per thread (22→32), which reduced the block limit from 21 to 16 blocks/SM. More critically, the executed instruction count ballooned by ~2.8x (179M→501M instructions), indicating the wider tiles generated significantly more work per element rather than reducing it. Although occupancy improved (47%→84%) and scheduler efficiency improved, the net effect was a 44% slowdown because the instruction count increase dominated. The autotune grid was also not updated to use a lambda with meta['XBLOCK'], so the grid remained fixed at cdiv(xnumel, 128) regardless of the selected XBLOCK, causing massive over-subscription and redundant computation.

> **Applicable when**: Never applicable in this broken form. The correct form (lambda grid) may help when: bytes_utilized_per_sector < 20, L1TEX stall fraction > 50% of warp cycles, kernel is memory-bound with DRAM throughput < 50%, and current XBLOCK < 512.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK without simultaneously changing the grid to a lambda: `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. A fixed grid with larger XBLOCK causes each block to process XBLOCK elements but the grid launches cdiv(xnumel, old_XBLOCK) blocks, resulting in massive redundant computation and instruction count inflation. Also avoid when register pressure is already near block-limit-registers threshold, as larger tiles increase register usage and reduce occupancy.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_correct_xblock_scaling | 0.998× | 0.991× | ✓ |  |
| D | reduce_register_pressure_via_smaller_num_warps_to_improve_occupancy | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 531550 ns | 536420 ns | +4870 ns (+0.9%) | 531740 ns | +190 ns (+0.0%) |
| SM Throughput (%) | 58.98 % | 77.62 % | +18.64 pp | 58.94 % | -0.04 pp |
| Memory Throughput (%) | 33.06 % | 46.71 % | +13.65 pp | 33.04 % | -0.02 pp |
| DRAM Throughput (%) | 33.06 % | 32.74 % | -0.32 pp | 33.04 % | -0.02 pp |
| Achieved Occupancy (%) | 47.06 % | 51.14 % | +4.08 pp | 49.02 % | +1.96 pp |
| Registers/Thread | 22.00 register/thread | 30.00 register/thread | +8 | 24.00 register/thread | +2 |
| Waves/SM | 407.70 | 407.70 | +0.00 | 407.70 | +0.00 |
| No Eligible (%) | 57.05 % | 21.93 % | -35.12 pp | 44.06 % | -12.99 pp |
| IPC Active | 1.72 inst/cycle | 3.13 inst/cycle | +1.41 | 2.24 inst/cycle | +0.52 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fuse_non_fused_fp32_instructions_via_explicit_fma_and_rsqrt_chain_simplification | 1.000× | 1.000× | ✓ |  |
| C | fix_autotune_grid_lambda_to_enable_correct_xblock_scaling_with_proper_mask | 0.998× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_non_fused_fp... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 531550 ns | 531520 ns | -30 ns (-0.0%) | 537470 ns | +5920 ns (+1.1%) |
| SM Throughput (%) | 58.98 % | 58.93 % | -0.05 pp | 78.05 % | +19.07 pp |
| Memory Throughput (%) | 33.06 % | 33.05 % | -0.01 pp | 46.96 % | +13.90 pp |
| DRAM Throughput (%) | 33.06 % | 33.05 % | -0.01 pp | 32.67 % | -0.39 pp |
| Achieved Occupancy (%) | 47.06 % | 49.11 % | +2.05 pp | 50.16 % | +3.10 pp |
| Registers/Thread | 22.00 register/thread | 24.00 register/thread | +2 | 30.00 register/thread | +8 |
| Waves/SM | 407.70 | 407.70 | +0.00 | 407.70 | +0.00 |
| No Eligible (%) | 57.05 % | 44.11 % | -12.94 pp | 21.52 % | -35.53 pp |
| IPC Active | 1.72 inst/cycle | 2.24 inst/cycle | +0.52 | 3.12 inst/cycle | +1.40 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_autotune_grid_lambda_and_add_num_stages_for_l1tex_latency_hiding | 1.000× | 1.000× | ✓ | **✓ best** |
| C | fix_autotune_grid_lambda_to_enable_correct_xblock_scaling_final | 0.998× | 0.993× | ✓ |  |
| D | reduce_num_warps_to_double_occupancy_via_register_pressure_relief | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | reduce_num_warps_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 531550 ns | 531650 ns | +100 ns (+0.0%) | 535170 ns | +3620 ns (+0.7%) | 531390 ns | -160 ns (-0.0%) |
| SM Throughput (%) | 58.98 % | 58.91 % | -0.07 pp | 77.61 % | +18.63 pp | 58.94 % | -0.04 pp |
| Memory Throughput (%) | 33.06 % | 33.04 % | -0.02 pp | 46.69 % | +13.63 pp | 33.05 % | -0.01 pp |
| DRAM Throughput (%) | 33.06 % | 33.04 % | -0.02 pp | 32.80 % | -0.26 pp | 33.05 % | -0.01 pp |
| Achieved Occupancy (%) | 47.06 % | 49.07 % | +2.01 pp | 50.54 % | +3.48 pp | 49.07 % | +2.01 pp |
| Registers/Thread | 22.00 register/thread | 24.00 register/thread | +2 | 30.00 register/thread | +8 | 24.00 register/thread | +2 |
| Waves/SM | 407.70 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 |
| No Eligible (%) | 57.05 % | 44.06 % | -12.99 pp | 22.06 % | -34.99 pp | 44.02 % | -13.03 pp |
| IPC Active | 1.72 inst/cycle | 2.24 inst/cycle | +0.52 | 3.12 inst/cycle | +1.40 | 2.24 inst/cycle | +0.52 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_large_input_to_improve_l1tex_hit_rate | 0.998× | 0.970× | ✓ |  |
| B | fix_grid_lambda_to_enable_autotune_xblock_scaling | 1.000× | 1.000× | ✓ |  |
| C | fix_grid_lambda_to_correctly_scale_with_autotuned_xblock | 0.998× | 0.991× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 531550 ns | 547970 ns | +16420 ns (+3.1%) | 531390 ns | -160 ns (-0.0%) | 536380 ns | +4830 ns (+0.9%) |
| SM Throughput (%) | 58.98 % | 79.96 % | +20.98 pp | 58.97 % | -0.01 pp | 78.45 % | +19.47 pp |
| Memory Throughput (%) | 33.06 % | 45.83 % | +12.77 pp | 33.05 % | -0.01 pp | 47.19 % | +14.13 pp |
| DRAM Throughput (%) | 33.06 % | 31.80 % | -1.26 pp | 33.05 % | -0.01 pp | 32.73 % | -0.33 pp |
| Achieved Occupancy (%) | 47.06 % | 64.94 % | +17.88 pp | 49.12 % | +2.06 pp | 50.94 % | +3.88 pp |
| Registers/Thread | 22.00 register/thread | 30.00 register/thread | +8 | 24.00 register/thread | +2 | 30.00 register/thread | +8 |
| Waves/SM | 407.70 | 407.70 | +0.00 | 407.70 | +0.00 | 407.70 | +0.00 |
| No Eligible (%) | 57.05 % | 19.54 % | -37.51 pp | 44.06 % | -12.99 pp | 21.83 % | -35.22 pp |
| IPC Active | 1.72 inst/cycle | 3.22 inst/cycle | +1.50 | 2.24 inst/cycle | +0.52 | 3.13 inst/cycle | +1.41 |

---

## Cumulative Progress

```
Baseline:  531550 ns  (1.000×)

Final best: 531650 ns  (0.9998×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,21 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_clamp_convolution_lift_fresh_minimum_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 110215168
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x4 = xindex
     x2 = ((xindex // 53816) % 16)
     x6 = xindex // 53816
-    tmp0 = tl.load(in_ptr0 + (x4), None)
-    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (x6 // 2), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr3 + (x6 // 2), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x4), xmask)
+    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (x6 // 2), xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr3 + (x6 // 2), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
     tmp6 = tl.full([1], 107632.0, tl.float32)
@@ -35,9 +36,7 @@

     tmp18 = tl.maximum(tmp17, tmp16)
     tmp19 = tl.full([1], 1.0, tl.float32)
     tmp20 = tl.minimum(tmp18, tmp19)
-    tl.store(out_ptr0 + (x4), tmp20, None)
-
-
+    tl.store(out_ptr0 + (x4), tmp20, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

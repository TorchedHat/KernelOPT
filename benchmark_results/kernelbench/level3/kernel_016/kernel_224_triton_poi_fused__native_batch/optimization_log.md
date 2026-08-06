# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_224` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6940 ns — **Best latency**: 6880 ns (**1.0087×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_improve_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_224`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_224_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6860,1,1), Waves/SM: 3.25
- **Registers/thread**: 22
- **Duration**: 6940 ns
- **SM Throughput**: 42.86 % | **Memory**: 15.87 % | **DRAM**: 10.42 %
- **Achieved Occupancy**: 59.49 %
- **IPC Active**: 2.50 inst/cycle | **No Eligible**: 35.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_autotune_to_eliminate_partial_wave_and_reduce_l1tex_stalls | 0.973× | 0.262× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_improve_occupancy | 1.013× | 1.009× | ✓ | **✓ best** |
| C | vectorized_loads_larger_xblock_autotune | 0.908× | 0.271× | ✓ |  |
| D | autotune_xblock_tile_size_to_eliminate_partial_wave_and_improve_occupancy | 0.907× | 0.783× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6940 ns | 26500 ns | +19560 ns (+281.8%) | 6880 ns | -60 ns (-0.9%) | 25570 ns | +18630 ns (+268.4%) | 8860 ns | +1920 ns (+27.7%) |
| SM Throughput (%) | 42.86 % | 79.14 % | +36.28 pp | 33.75 % | -9.11 pp | 78.81 % | +35.95 pp | 63.11 % | +20.25 pp |
| Memory Throughput (%) | 15.87 % | 15.24 % | -0.63 pp | 17.01 % | +1.14 pp | 15.69 % | -0.18 pp | 17.03 % | +1.16 pp |
| DRAM Throughput (%) | 10.42 % | 2.73 % | -7.69 pp | 10.56 % | +0.14 pp | 2.83 % | -7.59 pp | 8.18 % | -2.24 pp |
| Achieved Occupancy (%) | 59.49 % | 85.64 % | +26.15 pp | 50.25 % | -9.24 pp | 54.78 % | -4.71 pp | 80.91 % | +21.42 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 20.00 register/thread | -2 | 48.00 register/thread | +26 | 32.00 register/thread | +10 |
| Waves/SM | 3.25 | 12.99 | +9.74 | 3.25 | +0.00 | 10.39 | +7.14 | 3.25 | +0.00 |
| No Eligible (%) | 35.25 % | 13.70 % | -21.55 pp | 58.01 % | +22.76 pp | 13.53 % | -21.72 pp | 19.50 % | -15.75 pp |
| IPC Active | 2.50 inst/cycle | 3.44 inst/cycle | +0.94 | 1.74 inst/cycle | -0.76 | 3.45 inst/cycle | +0.95 | 3.28 inst/cycle | +0.78 |


### Key Learnings

#### ❌ Plan D: `autotune_xblock_tile_size_to_eliminate_partial_wave_and_improve_occupancy` (0.783×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel that was already register-constrained caused registers/thread to increase from 22 to 32, reducing Block Limit Registers from 21 to 16 blocks/SM. Although achieved occupancy improved (59.5% → 80.9%), the executed instruction count nearly doubled (2.33M → 4.42M), indicating the autotuner selected a configuration that introduced significant extra computation or loop overhead. The net effect was a 27% slowdown (6.94 µs → 8.86 µs) despite higher compute throughput utilization (42.9% → 63.1%), because the kernel became compute-bound with more instructions rather than reducing wall time.

> **Applicable when**: Never — this specific approach caused regression. The general idea of autotuning XBLOCK is valid, but not when: (1) the kernel is already register-constrained (registers/thread near 22–24 with block_limit_registers near 21), (2) the grid is not updated to use meta['XBLOCK'] (fixed grid with variable XBLOCK causes incorrect tiling), and (3) the instruction count doubles suggesting the autotuner selected a config that unrolls excessively.

> **Avoid when**: Kernel is register-constrained (registers/thread > 20 and block_limit_registers < 22); the grid launch is not updated to use lambda meta: (cdiv(n, meta['XBLOCK']),) when XBLOCK is autotuned; the kernel body has non-trivial per-element work that scales super-linearly with XBLOCK due to unrolling. Also avoid when the instruction count increase from larger tiles outweighs the wave-reduction benefit.

---

## Iteration 2

**Best so far**: 1.0087× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_main_input_to_improve_l1tex_cache_hit_rate | 0.991× | 1.037× | ✓ |  |
| B | replace_integer_reciprocal_with_rsqrt_to_enable_fma_fusion | 0.950× | 0.990× | ✓ |  |
| C | replace_integer_reciprocal_with_float_rsqrt_to_enable_fma_fusion | 1.002× | 0.475× | ✓ |  |
| D | fix_grid_lambda_to_match_autotune_xblock_and_fuse_bn_arithmetic_to_fma | 0.949× | 0.259× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | replace_integer_r... | Δ | replace_integer_r... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6940 ns | 6690 ns | -250 ns (-3.6%) | 7010 ns | +70 ns (+1.0%) | 14620 ns | +7680 ns (+110.7%) | 26780 ns | +19840 ns (+285.9%) |
| SM Throughput (%) | 42.86 % | 33.75 % | -9.11 pp | 37.92 % | -4.94 pp | 65.91 % | +23.05 pp | 77.84 % | +34.98 pp |
| Memory Throughput (%) | 15.87 % | 16.96 % | +1.09 pp | 24.27 % | +8.40 pp | 28.45 % | +12.58 pp | 15.02 % | -0.85 pp |
| DRAM Throughput (%) | 10.42 % | 10.82 % | +0.40 pp | 10.33 % | -0.09 pp | 4.95 % | -5.47 pp | 2.70 % | -7.72 pp |
| Achieved Occupancy (%) | 59.49 % | 52.23 % | -7.26 pp | 47.90 % | -11.59 pp | 62.24 % | +2.75 pp | 86.82 % | +27.33 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 32.00 register/thread | +10 | 38.00 register/thread | +16 | 32.00 register/thread | +10 |
| Waves/SM | 3.25 | 3.25 | +0.00 | 3.25 | +0.00 | 8.66 | +5.41 | 12.99 | +9.74 |
| No Eligible (%) | 35.25 % | 53.10 % | +17.85 pp | 36.45 % | +1.20 pp | 23.08 % | -12.17 pp | 15.18 % | -20.07 pp |
| IPC Active | 2.50 inst/cycle | 1.80 inst/cycle | -0.70 | 2.53 inst/cycle | +0.03 | 3.07 inst/cycle | +0.57 | 3.39 inst/cycle | +0.89 |


### Key Learnings

#### ❌ Plan C: `replace_integer_reciprocal_with_float_rsqrt_to_enable_fma_fusion` (0.430×)

Despite the plan claiming to apply the rsqrt arithmetic fix 'in isolation', the diff reveals the autotune configs were reordered (placing large XBLOCK configs first) and the eviction policy on in_ptr0 was removed — the autotuner selected XBLOCK=512 with 32 registers/thread, up from 20 at XBLOCK=128. Block_Limit_Registers collapsed from 21 to 4, instruction count ballooned from 1.59M to 8.36M (5.3x), and memory throughput dropped from 530 GB/s to 220 GB/s, causing a 2.4x slowdown. This is the third consecutive iteration where any autotune config change (even reordering) on this low-register kernel causes the autotuner to select a larger XBLOCK, inflating register pressure and multiplying instruction count far beyond any arithmetic benefit.

> **Applicable when**: Never — this is a negative example. Any modification to the autotune config list (including reordering) on this kernel causes the autotuner to select a larger XBLOCK, inflating register pressure and instruction count.

> **Avoid when**: Do not reorder, add, or remove autotune configs on kernels where: (1) baseline registers/thread is ≤24 and theoretical occupancy is near 100%; (2) the smallest XBLOCK config is the performance winner; (3) prior iterations have already demonstrated that larger XBLOCK configs cause register pressure collapse. Even 'cosmetic' reordering of configs can change which config the autotuner selects as the winner.

---

## Iteration 3

**Best so far**: 1.0087× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_and_fuse_bn_arithmetic_to_fma | 0.987× | 1.033× | ✓ |  |
| B | fix_grid_lambda_to_use_meta_xblock_enabling_correct_autotune_selection | 0.972× | 0.994× | ✓ |  |
| C | fix_grid_mismatch_and_cap_registers_with_maxnreg_to_restore_occupancy | 1.012× | 0.878× | ✓ |  |
| D | fuse_bn_reciprocal_sqrt_to_rsqrt_for_fma_fusion_isolated | 0.971× | 0.430× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_grid_lambda_t... | Δ | fix_grid_mismatch... | Δ | fuse_bn_reciproca... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6940 ns | 6720 ns | -220 ns (-3.2%) | 6980 ns | +40 ns (+0.6%) | 7900 ns | +960 ns (+13.8%) | 16130 ns | +9190 ns (+132.4%) |
| SM Throughput (%) | 42.86 % | 33.41 % | -9.45 pp | 37.80 % | -5.06 pp | 58.19 % | +15.33 pp | 65.87 % | +23.01 pp |
| Memory Throughput (%) | 15.87 % | 16.71 % | +0.84 pp | 24.14 % | +8.27 pp | 30.89 % | +15.02 pp | 26.00 % | +10.13 pp |
| DRAM Throughput (%) | 10.42 % | 10.82 % | +0.40 pp | 10.36 % | -0.06 pp | 9.18 % | -1.24 pp | 4.48 % | -5.94 pp |
| Achieved Occupancy (%) | 59.49 % | 50.54 % | -8.95 pp | 49.92 % | -9.57 pp | 74.31 % | +14.82 pp | 78.69 % | +19.20 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 32.00 register/thread | +10 | 24.00 register/thread | +2 | 32.00 register/thread | +10 |
| Waves/SM | 3.25 | 3.25 | +0.00 | 3.25 | +0.00 | 6.50 | +3.25 | 12.99 | +9.74 |
| No Eligible (%) | 35.25 % | 52.05 % | +16.80 pp | 36.67 % | +1.42 pp | 31.37 % | -3.88 pp | 24.54 % | -10.71 pp |
| IPC Active | 2.50 inst/cycle | 1.88 inst/cycle | -0.62 | 2.42 inst/cycle | -0.08 | 2.82 inst/cycle | +0.32 | 3.00 inst/cycle | +0.50 |

---

## Cumulative Progress

```
Baseline:  6940 ns  (1.000×)
Iter 1     6880 ns  (1.009×)  — autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_improve_occupancy

Final best: 6880 ns  (1.0087×)
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
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_224(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 878080
@@ -13,7 +25,7 @@

     xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 1792)
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
-    tmp11 = tmp2 * tmp10
+    tmp6 = tl.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
+    tmp16 = tl.full([1], 0, tl.float32)
     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -44,7 +50,7 @@

     arg_5 = inputs[5]
     xnumel = 878080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_224[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 878080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_224[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 878080)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

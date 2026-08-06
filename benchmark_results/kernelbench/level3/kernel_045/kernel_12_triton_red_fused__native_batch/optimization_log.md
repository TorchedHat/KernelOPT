# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 48640 ns — **Best latency**: 12130 ns (**4.0099×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_r0_block_oversize_and_add_autotune_for_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_12_triton_red_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 255
- **Duration**: 48640 ns
- **SM Throughput**: 21.33 % | **Memory**: 28.89 % | **DRAM**: 17.54 %
- **Achieved Occupancy**: 12.08 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 77.64 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_r0_block_oversize_and_add_autotune_for_occupancy | — | 4.010× | ✓ | **✓ best** |
| B | reduce_register_pressure_via_autotune_tile_sizes | — | 5.447× | ✓ |  |
| C | fix_r0_block_register_pressure_increase_xblock_parallelism | — | 0.076× | ✓ |  |
| D | reduce_register_pressure_via_tile_size_correction | — | 10.483× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_r0_block_over... | Δ | reduce_register_p... | Δ | fix_r0_block_regi... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48640 ns | 12130 ns | -36510 ns (-75.1%) | 8930 ns | -39710 ns (-81.6%) | 640290 ns | +591650 ns (+1216.4%) | 4640 ns | -44000 ns (-90.5%) |
| SM Throughput (%) | 21.33 % | 20.95 % | -0.38 pp | 31.07 % | +9.74 pp | 11.73 % | -9.60 pp | 3.34 % | -17.99 pp |
| Memory Throughput (%) | 28.89 % | 16.52 % | -12.37 pp | 21.88 % | -7.01 pp | 34.93 % | +6.04 pp | 23.16 % | -5.73 pp |
| DRAM Throughput (%) | 17.54 % | 7.17 % | -10.37 pp | 9.71 % | -7.83 pp | 34.93 % | +17.39 pp | 0.63 % | -16.91 pp |
| Achieved Occupancy (%) | 12.08 % | 6.05 % | -6.03 pp | 11.84 % | -0.24 pp | 11.17 % | -0.91 pp | 5.70 % | -6.38 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 166.00 register/thread | -89 | 32.00 register/thread | -223 | 29.00 register/thread | -226 |
| Waves/SM | 0.97 | 0.48 | -0.49 | 0.65 | -0.32 | 0.12 | -0.85 | 0.06 | -0.91 |
| No Eligible (%) | 77.64 % | 71.12 % | -6.52 pp | 58.61 % | -19.03 pp | 93.77 % | +16.13 pp | 92.75 % | +15.11 pp |
| IPC Active | 0.92 inst/cycle | 1.13 inst/cycle | +0.21 | 1.62 inst/cycle | +0.70 | 0.25 inst/cycle | -0.67 | 0.28 inst/cycle | -0.64 |


### Key Learnings

#### ✅ Plan D: `reduce_register_pressure_via_tile_size_correction` (10.483×)

When a Triton reduction kernel is launched with R0_BLOCK (reduction tile size) larger than the actual reduction dimension (r0_numel), the compiler allocates registers for the full tile even though most elements are masked out. This drives registers/thread to the hardware maximum (255), collapsing Block Limit Registers to 2 and theoretical occupancy to ~12.5%. Correcting R0_BLOCK to match r0_numel exactly (or using autotune to explore valid configs) drops registers/thread from 255 to ~29, raises Block Limit Registers from 2 to 32, and reduces kernel duration by ~10x. The fix is to add @triton.autotune with configs that cap R0_BLOCK at r0_numel and remove the hardcoded oversized tile from the launch call.

> **Applicable when**: Registers per thread is at or near hardware maximum (255); Block Limit Registers is 2 or very low; Theoretical Occupancy is ≤12.5%; NCU TheoreticalOccupancy rule fires with high speedup_pct (>50%); R0_BLOCK (or equivalent reduction tile constexpr) is hardcoded to a value larger than the actual reduction dimension r0_numel.

> **Avoid when**: R0_BLOCK already matches or is smaller than r0_numel; register pressure is not the occupancy bottleneck (Block Limit Registers is not the binding constraint); the kernel is memory-bandwidth bound rather than occupancy/latency bound.

---

## Iteration 2

**Best so far**: 4.0099× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_via_loop_dimension_swap | — | 5.465× | ✓ |  |
| C | fix_hardcoded_grid_to_enable_autotune_xblock_exploration | — | 9.806× | ✓ |  |
| D | fix_autotune_grid_lambda_to_restore_correct_parallelism | — | 0.076× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | fix_hardcoded_gri... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48640 ns | 8900 ns | -39740 ns (-81.7%) | 4960 ns | -43680 ns (-89.8%) | 635550 ns | +586910 ns (+1206.6%) |
| SM Throughput (%) | 21.33 % | 31.17 % | +9.84 pp | 3.63 % | -17.70 pp | 11.72 % | -9.61 pp |
| Memory Throughput (%) | 28.89 % | 21.97 % | -6.92 pp | 21.50 % | -7.39 pp | 35.12 % | +6.23 pp |
| DRAM Throughput (%) | 17.54 % | 9.73 % | -7.81 pp | 0.60 % | -16.94 pp | 35.12 % | +17.58 pp |
| Achieved Occupancy (%) | 12.08 % | 11.83 % | -0.25 pp | 5.70 % | -6.38 pp | 11.18 % | -0.90 pp |
| Registers/Thread | 255.00 register/thread | 166.00 register/thread | -89 | 26.00 register/thread | -229 | 32.00 register/thread | -223 |
| Waves/SM | 0.97 | 0.65 | -0.32 | 0.06 | -0.91 | 0.12 | -0.85 |
| No Eligible (%) | 77.64 % | 58.67 % | -18.97 pp | 92.34 % | +14.70 pp | 93.77 % | +16.13 pp |
| IPC Active | 0.92 inst/cycle | 1.67 inst/cycle | +0.75 | 0.28 inst/cycle | -0.64 | 0.25 inst/cycle | -0.67 |

---

## Iteration 3

**Best so far**: 4.0099× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_via_small_xblock_autotune_configs | — | 10.551× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 48640 ns | 4610 ns | -44030 ns (-90.5%) |
| SM Throughput (%) | 21.33 % | 3.80 % | -17.53 pp |
| Memory Throughput (%) | 28.89 % | 23.28 % | -5.61 pp |
| DRAM Throughput (%) | 17.54 % | 0.64 % | -16.90 pp |
| Achieved Occupancy (%) | 12.08 % | 5.70 % | -6.38 pp |
| Registers/Thread | 255.00 register/thread | 26.00 register/thread | -229 |
| Waves/SM | 0.97 | 0.06 | -0.91 |
| No Eligible (%) | 77.64 % | 91.87 % | +14.23 pp |
| IPC Active | 0.92 inst/cycle | 0.29 inst/cycle | -0.63 |

---

## Cumulative Progress

```
Baseline:  48640 ns  (1.000×)
Iter 1     12130 ns  (4.010×)  — fix_r0_block_oversize_and_add_autotune_for_occupancy

Final best: 12130 ns  (4.0099×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 2,   'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 32768
@@ -13,16 +31,16 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_base = tl.arange(0, R0_BLOCK)[None, :]
     rbase = r0_base
     x0 = (xindex % 256)
     x1 = xindex // 256
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr0 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
     _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
     _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     x3 = xindex
@@ -32,7 +50,7 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp0 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp4 = tmp2 - tmp3
         tmp6 = tl.full([1, 1], 1e-05, tl.float32)
@@ -53,7 +71,7 @@

 
         _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
         _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
-        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp17, r0_mask)
+        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp17, mask=xmask & r0_mask)
 
     tmp19, tmp20 = triton_helpers.online_softmax_reduce(
         _tmp19_max, _tmp19_sum, 1, False)
@@ -65,13 +83,11 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp21 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp21 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
         tmp22 = tmp21 - tmp19
         tmp23 = tl.exp(tmp22)
         tmp24 = (tmp23 / tmp20)
-        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp24, r0_mask)
-
-
+        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp24, mask=xmask & r0_mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -83,7 +99,7 @@

     xnumel = 32768
     r0_numel = 128
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 128, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 128)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

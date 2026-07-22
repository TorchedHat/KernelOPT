# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 906590 ns — **Best latency**: 76450 ns (**11.8586×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`memory_coalescing_reorder_tile_dimensions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_7_triton_red_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.12
- **Registers/thread**: 32
- **Duration**: 906590 ns
- **SM Throughput**: 11.51 % | **Memory**: 39.35 % | **DRAM**: 39.35 %
- **Achieved Occupancy**: 11.92 %
- **IPC Active**: 0.31 inst/cycle | **No Eligible**: 92.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_reorder_tile_dimensions | — | 11.859× | ✓ | **✓ best** |
| B | online_softmax_single_pass_eliminate_second_global_read | — | 224.960× | ✓ |  |
| C | increase_grid_parallelism_via_autotune_tile_sizes | — | 97.065× | ✓ |  |
| D | autotune_tile_sizes_for_sm_utilization | — | 236.091× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | online_softmax_si... | Δ | increase_grid_par... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 906590 ns | 76450 ns | -830140 ns (-91.6%) | 4030 ns | -902560 ns (-99.6%) | 9340 ns | -897250 ns (-99.0%) | 3840 ns | -902750 ns (-99.6%) |
| SM Throughput (%) | 11.51 % | 46.41 % | +34.90 pp | 7.27 % | -4.24 pp | 37.15 % | +25.64 pp | 10.75 % | -0.76 pp |
| Memory Throughput (%) | 39.35 % | 35.13 % | -4.22 pp | 3.25 % | -36.10 pp | 30.43 % | -8.92 pp | 6.14 % | -33.21 pp |
| DRAM Throughput (%) | 39.35 % | 19.45 % | -19.90 pp | 0.73 % | -38.62 pp | 4.69 % | -34.66 pp | 0.76 % | -38.59 pp |
| Achieved Occupancy (%) | 11.92 % | 11.03 % | -0.89 pp | 5.94 % | -5.98 pp | 11.95 % | +0.03 pp | 11.69 % | -0.23 pp |
| Registers/Thread | 32.00 register/thread | 255.00 register/thread | +223 | 39.00 register/thread | +7 | 255.00 register/thread | +223 | 30.00 register/thread | -2 |
| Waves/SM | 0.12 | 0.97 | +0.85 | 0.08 | -0.04 | 0.97 | +0.85 | 0.12 | +0.00 |
| No Eligible (%) | 92.08 % | 42.71 % | -49.37 pp | 84.98 % | -7.10 pp | 51.62 % | -40.46 pp | 80.26 % | -11.82 pp |
| IPC Active | 0.31 inst/cycle | 2.25 inst/cycle | +1.94 | 0.54 inst/cycle | +0.23 | 1.94 inst/cycle | +1.63 | 0.81 inst/cycle | +0.50 |

---

## Iteration 2

**Best so far**: 11.8586× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_wiring_and_coalesce_via_loop_order_swap | — | 163.644× | ✓ |  |
| B | fix_oversized_r0block_and_dynamic_grid_launch | — | 226.648× | ✓ |  |
| C | reduce_register_pressure_via_small_xblock_autotune_configs | — | 221.119× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_to_improve_occupancy | — | 6.522× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | fix_oversized_r0b... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 906590 ns | 5540 ns | -901050 ns (-99.4%) | 4000 ns | -902590 ns (-99.6%) | 4100 ns | -902490 ns (-99.5%) | 139010 ns | -767580 ns (-84.7%) |
| SM Throughput (%) | 11.51 % | 6.38 % | -5.13 pp | 7.19 % | -4.32 pp | 7.45 % | -4.06 pp | 25.87 % | +14.36 pp |
| Memory Throughput (%) | 39.35 % | 37.05 % | -2.30 pp | 3.41 % | -35.94 pp | 3.90 % | -35.45 pp | 42.93 % | +3.58 pp |
| DRAM Throughput (%) | 39.35 % | 1.01 % | -38.34 pp | 0.74 % | -38.61 pp | 0.72 % | -38.63 pp | 22.32 % | -17.03 pp |
| Achieved Occupancy (%) | 11.92 % | 10.83 % | -1.09 pp | 6.02 % | -5.90 pp | 5.97 % | -5.95 pp | 10.66 % | -1.26 pp |
| Registers/Thread | 32.00 register/thread | 28.00 register/thread | -4 | 39.00 register/thread | +7 | 39.00 register/thread | +7 | 128.00 register/thread | +96 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.08 | -0.04 | 0.08 | -0.04 | 0.48 | +0.36 |
| No Eligible (%) | 92.08 % | 88.96 % | -3.12 pp | 85.19 % | -6.89 pp | 85.52 % | -6.56 pp | 69.03 % | -23.05 pp |
| IPC Active | 0.31 inst/cycle | 0.41 inst/cycle | +0.10 | 0.56 inst/cycle | +0.25 | 0.52 inst/cycle | +0.21 | 1.23 inst/cycle | +0.92 |

---

## Iteration 3

**Best so far**: 11.8586× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_add_pid_swizzle_for_l2_locality | — | 219.513× | ✓ |  |
| B | eliminate_intermediate_global_store_via_register_buffer_and_fix_grid_launch | — | 224.960× | ✓ |  |
| C | eliminate_intermediate_global_store_reload_between_softmax_passes | — | 179.168× | ✓ |  |
| D | remove_maxnreg_cap_to_eliminate_register_spill_to_local_memory | — | 12.508× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | eliminate_interme... | Δ | eliminate_interme... | Δ | remove_maxnreg_ca... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 906590 ns | 4130 ns | -902460 ns (-99.5%) | 4030 ns | -902560 ns (-99.6%) | 5060 ns | -901530 ns (-99.4%) | 72480 ns | -834110 ns (-92.0%) |
| SM Throughput (%) | 11.51 % | 7.69 % | -3.82 pp | 7.32 % | -4.19 pp | 6.36 % | -5.15 pp | 47.89 % | +36.38 pp |
| Memory Throughput (%) | 39.35 % | 3.34 % | -36.01 pp | 3.47 % | -35.88 pp | 24.11 % | -15.24 pp | 37.05 % | -2.30 pp |
| DRAM Throughput (%) | 39.35 % | 0.72 % | -38.63 pp | 0.73 % | -38.62 pp | 1.10 % | -38.25 pp | 20.51 % | -18.84 pp |
| Achieved Occupancy (%) | 11.92 % | 6.01 % | -5.91 pp | 5.97 % | -5.95 pp | 11.07 % | -0.85 pp | 11.08 % | -0.84 pp |
| Registers/Thread | 32.00 register/thread | 39.00 register/thread | +7 | 39.00 register/thread | +7 | 29.00 register/thread | -3 | 255.00 register/thread | +223 |
| Waves/SM | 0.12 | 0.08 | -0.04 | 0.08 | -0.04 | 0.12 | +0.00 | 0.97 | +0.85 |
| No Eligible (%) | 92.08 % | 85.66 % | -6.42 pp | 85.48 % | -6.60 pp | 87.95 % | -4.13 pp | 42.64 % | -49.44 pp |
| IPC Active | 0.31 inst/cycle | 0.55 inst/cycle | +0.24 | 0.55 inst/cycle | +0.24 | 0.48 inst/cycle | +0.17 | 2.25 inst/cycle | +1.94 |


### Key Learnings

#### ✅ Plan D: `remove_maxnreg_cap_to_eliminate_register_spill_to_local_memory` (12.508×)

When a Triton kernel has a maxnreg cap that forces register spill to local (stack) memory, removing the cap allows the compiler to allocate registers naturally, eliminating the spill. However, if the kernel body has been restructured (e.g., loop-invariant loads hoisted, intermediate buffers added, dimension swaps applied) such that the natural register demand is very high (e.g., 255 registers/thread), removing maxnreg trades spill-induced instruction bloat (29M instructions) for a register-limited occupancy regime (Block Limit Registers=2, 12.5% theoretical occupancy) while still achieving better scheduler utilization (No Eligible drops from 85% to 43%) and higher compute throughput (7% to 48%). The net result is a 12.5x speedup over the spilling baseline (4us→72us is a regression vs the 3.84us pre-spill state, but 12.5x better than the 139us maxnreg-spill state), confirming that eliminating spill is correct even when it reduces theoretical occupancy.

> **Applicable when**: Triton autotune configs carry maxnreg keyword arguments AND profiling shows MemoryCacheAccessPattern firing with ~1/32 bytes utilized per sector on local loads AND stores (definitive register spill signature), AND executed instructions are dramatically inflated (10x-100x above expected for the problem size), AND the kernel body has been restructured with hoisted loads or register buffers that increase natural register demand. Removing maxnreg eliminates spill even if it reduces theoretical occupancy.

> **Avoid when**: The kernel's natural register count without the cap is low (e.g., <64 registers/thread) — the cap has no effect and removing it changes nothing. Also do not apply if MemoryCacheAccessPattern is NOT firing (no spill present) and the maxnreg cap is actually helping occupancy without causing spill. If the kernel body is simple enough that the cap does not cause spill, removing it may reduce occupancy without benefit.

---

## Cumulative Progress

```
Baseline:  906590 ns  (1.000×)
Iter 1     76450 ns  (11.859×)  — memory_coalescing_reorder_tile_dimensions

Final best: 76450 ns  (11.8586×)
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
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 256}, num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 256}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 2,   'R0_BLOCK': 256}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 256}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 256}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 256}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 256}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 256}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 256}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 256}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 256}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 256}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 256}, num_warps=8,  num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 32768
@@ -21,6 +40,7 @@

     x_b = xindex // (256 * 32)
     _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
     _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
+    tmp17_buf = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     x3 = xindex
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
@@ -54,7 +74,7 @@

 
         _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
         _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
-        tl.store(in_out_ptr0 + (r0_c + 128*x_w + 32768*x_h + 1048576*x_b), tmp17, mask=xmask & r0_mask)
+        tmp17_buf = tl.where(r0_mask, tmp17, tmp17_buf)
 
     tmp19, tmp20 = triton_helpers.online_softmax_reduce(
         _tmp19_max, _tmp19_sum, 1, False)
@@ -66,12 +86,11 @@

         roffset = r0_offset
         rindex = r0_index
         r0_c = r0_index
-        tmp21 = tl.load(in_out_ptr0 + (r0_c + 128*x_w + 32768*x_h + 1048576*x_b), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp21 = tmp17_buf
         tmp22 = tmp21 - tmp19
         tmp23 = tl.exp(tmp22)
         tmp24 = (tmp23 / tmp20)
         tl.store(in_out_ptr0 + (r0_c + 128*x_w + 32768*x_h + 1048576*x_b), tmp24, mask=xmask & r0_mask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -83,7 +102,7 @@

     xnumel = 32768
     r0_numel = 256
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 256, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 256)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

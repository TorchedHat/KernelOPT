# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10090000 ns — **Best latency**: 7460 ns (**1352.5469×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`swap_eviction_policies_to_improve_l2_compression_and_reduce_l1tex_stalls`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_6_triton_red_fused__softmax_eq_e/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 32
- **Duration**: 10090000 ns
- **SM Throughput**: 13.98 % | **Memory**: 68.55 % | **DRAM**: 68.55 %
- **Achieved Occupancy**: 93.08 %
- **IPC Active**: 0.44 inst/cycle | **No Eligible**: 88.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_swap_tile_dimensions | 738.369× | 890.556× | ✓ |  |
| B | online_softmax_attention_single_pass | 925.771× | 1370.924× | ✓ | **✓ best** |
| C | autotune_xblock_r0block_to_fix_memory_coalescing_and_partial_wave | 739.536× | 909.009× | ✓ |  |
| D | autotune_xblock_r0block_to_eliminate_partial_wave_and_reduce_memory_stalls | 746.448× | 905.745× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | online_softmax_at... | Δ | autotune_xblock_r... | Δ | autotune_xblock_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10090000 ns | 11330 ns | -10078670 ns (-99.9%) | 7360 ns | -10082640 ns (-99.9%) | 11100 ns | -10078900 ns (-99.9%) | 11140 ns | -10078860 ns (-99.9%) |
| SM Throughput (%) | 13.98 % | 60.26 % | +46.28 pp | 41.64 % | +27.66 pp | 59.44 % | +45.46 pp | 61.33 % | +47.35 pp |
| Memory Throughput (%) | 68.55 % | 32.31 % | -36.24 pp | 45.28 % | -23.27 pp | 32.87 % | -35.68 pp | 32.90 % | -35.65 pp |
| DRAM Throughput (%) | 68.55 % | 17.02 % | -51.53 pp | 26.22 % | -42.33 pp | 17.40 % | -51.15 pp | 17.35 % | -51.20 pp |
| Achieved Occupancy (%) | 93.08 % | 78.91 % | -14.17 pp | 81.39 % | -11.69 pp | 79.07 % | -14.01 pp | 78.91 % | -14.17 pp |
| Registers/Thread | 32.00 register/thread | 28.00 register/thread | -4 | 20.00 register/thread | -12 | 32.00 register/thread | +0 | 28.00 register/thread | -4 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 88.97 % | 24.70 % | -64.27 pp | 42.57 % | -46.40 pp | 27.40 % | -61.57 pp | 24.54 % | -64.43 pp |
| IPC Active | 0.44 inst/cycle | 3.02 inst/cycle | +2.58 | 2.28 inst/cycle | +1.84 | 2.95 inst/cycle | +2.51 | 3.03 inst/cycle | +2.59 |

---

## Iteration 2

**Best so far**: 1370.9239× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_num_stages_and_num_warps_to_hide_l1tex_latency_and_partial_wave | 919.115× | 1347.130× | ✓ |  |
| B | eliminate_partial_wave_via_wider_xblock_configs_and_eviction_policy | 716.881× | 927.390× | ✓ |  |
| C | add_num_stages_2_to_autotune_configs_for_pipeline_and_fma_fusion | 747.322× | 892.920× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_num_stag... | Δ | eliminate_partial... | Δ | add_num_stages_2_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10090000 ns | 7490 ns | -10082510 ns (-99.9%) | 10880 ns | -10079120 ns (-99.9%) | 11300 ns | -10078700 ns (-99.9%) |
| SM Throughput (%) | 13.98 % | 41.38 % | +27.40 pp | 62.41 % | +48.43 pp | 60.65 % | +46.67 pp |
| Memory Throughput (%) | 68.55 % | 45.06 % | -23.49 pp | 33.93 % | -34.62 pp | 32.63 % | -35.92 pp |
| DRAM Throughput (%) | 68.55 % | 25.76 % | -42.79 pp | 17.69 % | -50.86 pp | 17.07 % | -51.48 pp |
| Achieved Occupancy (%) | 93.08 % | 81.27 % | -11.81 pp | 79.24 % | -13.84 pp | 79.29 % | -13.79 pp |
| Registers/Thread | 32.00 register/thread | 20.00 register/thread | -12 | 32.00 register/thread | +0 | 28.00 register/thread | -4 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 88.97 % | 42.04 % | -46.93 pp | 22.26 % | -66.71 pp | 24.27 % | -64.70 pp |
| IPC Active | 0.44 inst/cycle | 2.28 inst/cycle | +1.84 | 3.09 inst/cycle | +2.65 | 2.97 inst/cycle | +2.53 |

---

## Iteration 3

**Best so far**: 1370.9239× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | swap_eviction_policies_to_improve_l2_compression_and_reduce_l1tex_stalls | 956.216× | 1352.547× | ✓ | **✓ best** |
| C | fuse_fp32_mul_sub_into_fma_to_reduce_non_fused_instruction_count | 751.159× | 911.472× | ✓ |  |


### Metric Comparison

| Metric | Baseline | swap_eviction_pol... | Δ | fuse_fp32_mul_sub... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 10090000 ns | 7460 ns | -10082540 ns (-99.9%) | 11070 ns | -10078930 ns (-99.9%) |
| SM Throughput (%) | 13.98 % | 41.77 % | +27.79 pp | 61.12 % | +47.14 pp |
| Memory Throughput (%) | 68.55 % | 45.84 % | -22.71 pp | 33.03 % | -35.52 pp |
| DRAM Throughput (%) | 68.55 % | 25.95 % | -42.60 pp | 17.37 % | -51.18 pp |
| Achieved Occupancy (%) | 93.08 % | 81.33 % | -11.75 pp | 79.00 % | -14.08 pp |
| Registers/Thread | 32.00 register/thread | 20.00 register/thread | -12 | 28.00 register/thread | -4 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 88.97 % | 42.49 % | -46.48 pp | 24.47 % | -64.50 pp |
| IPC Active | 0.44 inst/cycle | 2.27 inst/cycle | +1.83 | 3.00 inst/cycle | +2.56 |

---

## Iteration 4

**Best so far**: 1370.9239× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | reduce_cpi_not_selected_stall_via_fewer_warps_and_r0block_exploration | 772.319× | 913.949× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_cpi_not_se... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 10090000 ns | 11040 ns | -10078960 ns (-99.9%) |
| SM Throughput (%) | 13.98 % | 60.79 % | +46.81 pp |
| Memory Throughput (%) | 68.55 % | 32.87 % | -35.68 pp |
| DRAM Throughput (%) | 68.55 % | 17.48 % | -51.07 pp |
| Achieved Occupancy (%) | 93.08 % | 78.80 % | -14.28 pp |
| Registers/Thread | 32.00 register/thread | 28.00 register/thread | -4 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 88.97 % | 25.19 % | -63.78 pp |
| IPC Active | 0.44 inst/cycle | 3.00 inst/cycle | +2.56 |

---

## Iteration 5

**Best so far**: 1370.9239× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  10090000 ns  (1.000×)
Iter 1     7360 ns  (1370.924×)  — online_softmax_attention_single_pass

Final best: 7460 ns  (1352.5469×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,68 +5,81 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # XBLOCK=1: grid=524288 full waves, vary num_warps and num_stages
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=16, num_stages=3),
+        # XBLOCK=2
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=16, num_stages=2),
+        # XBLOCK=4
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=16, num_stages=2),
+        # XBLOCK=8
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
+        # XBLOCK=16
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
+        # XBLOCK=32
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
+        # Larger XBLOCK (kept from original)
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 512}, num_warps=16, num_stages=3),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 524288
     r0_numel = 512
-    rnumel = r0_numel
-    RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
-    r0_base = tl.arange(0, R0_BLOCK)[None, :]
-    rbase = r0_base
+    xmask = xindex < xnumel
+    r0_index = tl.arange(0, R0_BLOCK)[None, :]
+    r0_mask = r0_index < r0_numel
     x0 = (xindex % 512)
     x3 = xindex
-    _tmp9_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
-    _tmp9_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
-    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
-        r0_index = r0_offset + r0_base
-        r0_mask = r0_index < r0_numel
-        roffset = r0_offset
-        rindex = r0_index
-        r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_2 + 1024*x0), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp3 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.full([1, 1], 0.0, tl.float32)
-        tmp2 = tmp0 == tmp1
-        tmp4 = tl.full([1, 1], 0.10206207261596577, tl.float32)
-        tmp5 = tmp3 * tmp4
-        tmp6 = tl.full([1, 1], float("-inf"), tl.float32)
-        tmp7 = tl.where(tmp2, tmp6, tmp5)
-        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
-
-        _tmp9_max_next, _tmp9_sum_next = triton_helpers.online_softmax_combine(
-            _tmp9_max, _tmp9_sum, tmp8, False
-        )
-
-        _tmp9_max = tl.where(r0_mask, _tmp9_max_next, _tmp9_max)
-        _tmp9_sum = tl.where(r0_mask, _tmp9_sum_next, _tmp9_sum)
-
-    tmp9, tmp10 = triton_helpers.online_softmax_reduce(
-        _tmp9_max, _tmp9_sum, 1, False)
-    tmp9 = tmp9[:, None]
-    tmp10 = tmp10[:, None]
-    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
-        r0_index = r0_offset + r0_base
-        r0_mask = r0_index < r0_numel
-        roffset = r0_offset
-        rindex = r0_index
-        r0_2 = r0_index
-        tmp11 = tl.load(in_ptr0 + (r0_2 + 1024*x0), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp14 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), r0_mask, eviction_policy='evict_first', other=0.0)
-        tmp12 = tl.full([1, 1], 0.0, tl.float32)
-        tmp13 = tmp11 == tmp12
-        tmp15 = tl.full([1, 1], 0.10206207261596577, tl.float32)
-        tmp16 = tmp14 * tmp15
-        tmp17 = tl.full([1, 1], float("-inf"), tl.float32)
-        tmp18 = tl.where(tmp13, tmp17, tmp16)
-        tmp19 = tmp18 - tmp9
-        tmp20 = tl.exp(tmp19)
-        tmp21 = (tmp20 / tmp10)
-        tl.store(in_out_ptr0 + (r0_2 + 512*x3), tmp21, r0_mask)
-
-
+    # --- SINGLE LOAD of both inputs ---
+    tmp0 = tl.load(in_ptr0 + (r0_index + 1024*x0), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
+    tmp3 = tl.load(in_out_ptr0 + (r0_index + 512*x3), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
+    # --- Apply mask and scale ---
+    tmp1 = tl.full([1, 1], 0.0, tl.float32)
+    tmp2 = tmp0 == tmp1
+    tmp4 = tl.full([1, 1], 0.10206207261596577, tl.float32)
+    tmp5 = tmp3 * tmp4
+    tmp6 = tl.full([1, 1], float("-inf"), tl.float32)
+    tmp7 = tl.where(tmp2, tmp6, tmp5)
+    # --- Compute softmax in registers (single pass) ---
+    row_max = tl.max(tmp7, axis=1, keep_dims=True)    # [XBLOCK, 1]
+    exp_val = tl.exp(tmp7 - row_max)                  # [XBLOCK, R0_BLOCK]
+    row_sum = tl.sum(exp_val, axis=1, keep_dims=True)  # [XBLOCK, 1]
+    out = exp_val / row_sum                            # [XBLOCK, R0_BLOCK]
+    # --- SINGLE STORE ---
+    tl.store(in_out_ptr0 + (r0_index + 512*x3), out, mask=xmask & r0_mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -74,7 +87,7 @@

     xnumel = 524288
     r0_numel = 512
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6[grid](arg_0, arg_1, 524288, 512, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6[grid](arg_0, arg_1, 524288, 512)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

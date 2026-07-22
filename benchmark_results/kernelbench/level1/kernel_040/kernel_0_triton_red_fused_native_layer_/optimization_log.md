# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_native_layer_norm_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 97060 ns — **Best latency**: 128000 ns (**0.7583×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_native_layer_norm_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_040/kernel_0_triton_red_fused_native_layer_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.97
- **Registers/thread**: 124
- **Duration**: 97060 ns
- **SM Throughput**: 63.70 % | **Memory**: 56.95 % | **DRAM**: 56.95 %
- **Achieved Occupancy**: 23.29 %
- **IPC Active**: 2.71 inst/cycle | **No Eligible**: 31.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_and_autotune_tile_sizes | 1.008× | 1.009× | ✓ |  |
| B | reduce_register_pressure_via_autotune_tile_sizes | 1.046× | 1.040× | ✓ |  |
| C | add_autotune_increase_xblock_reduce_register_pressure | 1.004× | 1.001× | ✓ |  |
| D | autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy | 1.275× | 0.997× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | reduce_register_p... | Δ | add_autotune_incr... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 97060 ns | 96220 ns | -840 ns (-0.9%) | 93310 ns | -3750 ns (-3.9%) | 96960 ns | -100 ns (-0.1%) | 97310 ns | +250 ns (+0.3%) |
| SM Throughput (%) | 63.70 % | 63.71 % | +0.01 pp | 44.55 % | -19.15 pp | 63.59 % | -0.11 pp | 63.78 % | +0.08 pp |
| Memory Throughput (%) | 56.95 % | 57.27 % | +0.32 pp | 59.09 % | +2.14 pp | 56.84 % | -0.11 pp | 56.62 % | -0.33 pp |
| DRAM Throughput (%) | 56.95 % | 57.27 % | +0.32 pp | 59.09 % | +2.14 pp | 56.84 % | -0.11 pp | 56.62 % | -0.33 pp |
| Achieved Occupancy (%) | 23.29 % | 23.30 % | +0.01 pp | 23.90 % | +0.61 pp | 23.30 % | +0.01 pp | 23.30 % | +0.01 pp |
| Registers/Thread | 124.00 register/thread | 124.00 register/thread | +0 | 31.00 register/thread | -93 | 124.00 register/thread | +0 | 124.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.24 | -0.73 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 31.96 % | 31.98 % | +0.02 pp | 52.97 % | +21.01 pp | 32.84 % | +0.88 pp | 32.87 % | +0.91 pp |
| IPC Active | 2.71 inst/cycle | 2.69 inst/cycle | -0.02 | 1.91 inst/cycle | -0.80 | 2.74 inst/cycle | +0.03 | 2.71 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_uncoalesced_output_stores_with_evict_last_and_num_warps_sweep | 1.312× | 0.371× | ✓ | **✓ best** |
| B | add_num_stages_software_pipelining_to_hide_l1tex_stall_latency | 1.008× | 0.994× | ✓ |  |
| C | fix_uncoalesced_output_stores_with_cache_streaming_eviction | 1.001× | 1.002× | ✓ |  |
| D | add_num_stages2_and_num_warps16_to_hide_l1tex_load_latency | 1.289× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_uncoalesced_o... | Δ | add_num_stages_so... | Δ | fix_uncoalesced_o... | Δ | add_num_stages2_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 97060 ns | 261660 ns | +164600 ns (+169.6%) | 97660 ns | +600 ns (+0.6%) | 96900 ns | -160 ns (-0.2%) | 96580 ns | -480 ns (-0.5%) |
| SM Throughput (%) | 63.70 % | 18.94 % | -44.76 pp | 63.92 % | +0.22 pp | 64.05 % | +0.35 pp | 63.77 % | +0.07 pp |
| Memory Throughput (%) | 56.95 % | 21.04 % | -35.91 pp | 56.46 % | -0.49 pp | 56.85 % | -0.10 pp | 57.07 % | +0.12 pp |
| DRAM Throughput (%) | 56.95 % | 21.04 % | -35.91 pp | 56.46 % | -0.49 pp | 56.85 % | -0.10 pp | 57.07 % | +0.12 pp |
| Achieved Occupancy (%) | 23.29 % | 12.04 % | -11.25 pp | 23.24 % | -0.05 pp | 23.29 % | +0.00 pp | 23.26 % | -0.03 pp |
| Registers/Thread | 124.00 register/thread | 32.00 register/thread | -92 | 124.00 register/thread | +0 | 124.00 register/thread | +0 | 124.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.12 | -0.85 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 31.96 % | 80.47 % | +48.51 pp | 31.87 % | -0.09 pp | 32.68 % | +0.72 pp | 32.88 % | +0.92 pp |
| IPC Active | 2.71 inst/cycle | 0.78 inst/cycle | -1.93 | 2.71 inst/cycle | +0.00 | 2.70 inst/cycle | -0.01 | 2.68 inst/cycle | -0.03 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | convert_non_fused_fp32_to_fma_in_welford_loop | 1.178× | 1.194× | ✓ |  |
| B | replace_welford_with_fused_fma_two_pass_variance | 0.537× | 0.414× | ✓ |  |
| C | restore_xblock_gt1_configs_to_increase_waves_and_occupancy | 1.047× | 1.033× | ✓ |  |
| D | add_triton_autotune | 1.041× | 1.037× | ✓ |  |


### Metric Comparison

| Metric | Baseline | convert_non_fused... | Δ | replace_welford_w... | Δ | restore_xblock_gt... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 97060 ns | 81280 ns | -15780 ns (-16.3%) | 234140 ns | +137080 ns (+141.2%) | 93920 ns | -3140 ns (-3.2%) | 93630 ns | -3430 ns (-3.5%) |
| SM Throughput (%) | 63.70 % | 64.68 % | +0.98 pp | 9.30 % | -54.40 pp | 44.32 % | -19.38 pp | 44.21 % | -19.49 pp |
| Memory Throughput (%) | 56.95 % | 67.84 % | +10.89 pp | 45.34 % | -11.61 pp | 58.66 % | +1.71 pp | 58.86 % | +1.91 pp |
| DRAM Throughput (%) | 56.95 % | 67.84 % | +10.89 pp | 45.34 % | -11.61 pp | 58.66 % | +1.71 pp | 58.86 % | +1.91 pp |
| Achieved Occupancy (%) | 23.29 % | 23.29 % | +0.00 pp | 24.04 % | +0.75 pp | 23.92 % | +0.63 pp | 23.89 % | +0.60 pp |
| Registers/Thread | 124.00 register/thread | 95.00 register/thread | -29 | 21.00 register/thread | -103 | 31.00 register/thread | -93 | 31.00 register/thread | -93 |
| Waves/SM | 0.97 | 0.78 | -0.19 | 0.24 | -0.73 | 0.24 | -0.73 | 0.24 | -0.73 |
| No Eligible (%) | 31.96 % | 32.51 % | +0.55 pp | 90.41 % | +58.45 pp | 53.09 % | +21.13 pp | 53.18 % | +21.22 pp |
| IPC Active | 2.71 inst/cycle | 2.76 inst/cycle | +0.05 | 0.38 inst/cycle | -2.33 | 1.89 inst/cycle | -0.82 | 1.88 inst/cycle | -0.83 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_scalar_reciprocal_out_of_welford_tile_broadcast | — | fail | ✓ |  |
| B | inline_welford_fma_with_loop_unroll_factor2 | 1.198× | 0.751× | ✓ |  |
| C | replace_two_pass_variance_with_single_pass_welford_fma_inline | 1.130× | 1.015× | ✓ | **✓ best** |
| D | inline_welford_with_libdevice_fma_and_loop_unroll_factor2 | 1.197× | 0.749× | ✓ |  |
| E | add_triton_autotune | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | inline_welford_fm... | Δ | replace_two_pass_... | Δ | inline_welford_wi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 97060 ns | 129250 ns | +32190 ns (+33.2%) | 95620 ns | -1440 ns (-1.5%) | 129570 ns | +32510 ns (+33.5%) |
| SM Throughput (%) | 63.70 % | 45.10 % | -18.60 pp | 56.02 % | -7.68 pp | 44.33 % | -19.37 pp |
| Memory Throughput (%) | 56.95 % | 42.66 % | -14.29 pp | 57.67 % | +0.72 pp | 42.54 % | -14.41 pp |
| DRAM Throughput (%) | 56.95 % | 42.66 % | -14.29 pp | 57.67 % | +0.72 pp | 42.54 % | -14.41 pp |
| Achieved Occupancy (%) | 23.29 % | 23.98 % | +0.69 pp | 23.80 % | +0.51 pp | 23.97 % | +0.68 pp |
| Registers/Thread | 124.00 register/thread | 29.00 register/thread | -95 | 40.00 register/thread | -84 | 29.00 register/thread | -95 |
| Waves/SM | 0.97 | 0.24 | -0.73 | 0.32 | -0.65 | 0.24 | -0.73 |
| No Eligible (%) | 31.96 % | 53.71 % | +21.75 pp | 40.44 % | +8.48 pp | 53.82 % | +21.86 pp |
| IPC Active | 2.71 inst/cycle | 1.86 inst/cycle | -0.85 | 2.33 inst/cycle | -0.38 | 1.86 inst/cycle | -0.85 |

---

## Iteration 5

**Best so far**: 1.0151× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_welford_reciprocal_to_scalar_mufu_rcp_per_iteration | 1.230× | 0.758× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | hoist_welford_rec... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 97060 ns | 128000 ns | +30940 ns (+31.9%) |
| SM Throughput (%) | 63.70 % | 32.02 % | -31.68 pp |
| Memory Throughput (%) | 56.95 % | 43.06 % | -13.89 pp |
| DRAM Throughput (%) | 56.95 % | 43.06 % | -13.89 pp |
| Achieved Occupancy (%) | 23.29 % | 23.99 % | +0.70 pp |
| Registers/Thread | 124.00 register/thread | 25.00 register/thread | -99 |
| Waves/SM | 0.97 | 0.24 | -0.73 |
| No Eligible (%) | 31.96 % | 66.46 % | +34.50 pp |
| IPC Active | 2.71 inst/cycle | 1.34 inst/cycle | -1.37 |


### Key Learnings

#### ❌ Plan A: `hoist_welford_reciprocal_to_scalar_mufu_rcp_per_iteration` (0.758×)

Replacing the per-tile `1.0 / tmp2_weight_next` FDIV with a scalar `1.0 / scalar_weight` computed once per loop iteration reduced total executed instructions by ~21% (40.4M → 31.9M) and register pressure from 40 to 25 regs/thread, raising theoretical occupancy from 75% to 100%. However, achieved occupancy barely changed (23.8% → 24.0%) because the grid was already too small (waves_per_sm dropped from 0.32 to 0.24 due to expanded autotune configs selecting larger tiles), and Warp Cycles Per Issued Instruction nearly doubled (6.41 → 11.46 cycles), with scheduler no-eligible rate rising from 40% to 66%. The instruction reduction freed no additional latency-hiding capacity because the kernel is L1TEX-stall-dominated (CPIStall L1TEX stalls grew from 3.7 to 9.0 cycles/warp), and the simultaneous expansion of the autotune config space (adding larger R0_BLOCK=1024 and XBLOCK=16 configs) likely caused autotune to select a config with more work per block but fewer waves, compounding the regression.

> **Applicable when**: Never: hoisting the Welford weight reciprocal to a scalar is mathematically sound but does not improve performance when the kernel is already L1TEX-stall-dominated (CPIStall L1TEX > 50% of stall cycles) and the grid is small (waves_per_sm < 0.5). Reducing instruction count does not help when the bottleneck is memory latency, not compute throughput.

> **Avoid when**: Any kernel where: (1) CPIStall L1TEX stalls account for >50% of warp cycles (the bottleneck is memory latency, not compute), (2) waves_per_sm < 0.5 and achieved occupancy is already near its practical ceiling (~24%), meaning theoretical occupancy improvements from lower register pressure do not translate to higher achieved occupancy, or (3) the autotune config space is being simultaneously expanded — the interaction between scalar weight hoisting and larger tile configs can cause autotune to select configs with fewer waves, compounding the regression. Do not conflate instruction count reduction with latency reduction in memory-bound kernels.

---

## Cumulative Progress

```
Baseline:  97060 ns  (1.000×)
Iter 4     95620 ns  (1.015×)  — replace_two_pass_variance_with_single_pass_welford_fma_inline

Final best: 128000 ns  (0.7583×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=16, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 512
@@ -19,30 +39,27 @@

     x0 = xindex
     tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
-    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
-    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
+    scalar_weight = 0.0
+    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, loop_unroll_factor=2):
         r0_index = r0_offset + r0_base
-        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
-        roffset = r0_offset
-        rindex = r0_index
-        r0_1 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_1 + 131072*x0), xmask, eviction_policy='evict_first', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_index + 131072*x0), xmask, eviction_policy='evict_first', other=0.0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
-        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
-            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
-        )
+        scalar_weight = scalar_weight + 1.0
+        _inv_w_scalar = 1.0 / scalar_weight
+        _delta = tmp1 - tmp2_mean
+        tmp2_mean_next = libdevice.fma(_delta, _inv_w_scalar, tmp2_mean)
+        _delta2 = tmp1 - tmp2_mean_next
+        tmp2_m2_next = libdevice.fma(_delta, _delta2, tmp2_m2)
         tmp2_mean = tl.where(xmask, tmp2_mean_next, tmp2_mean)
         tmp2_m2 = tl.where(xmask, tmp2_m2_next, tmp2_m2)
-        tmp2_weight = tl.where(xmask, tmp2_weight_next, tmp2_weight)
-    tmp3, tmp4, tmp5 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
+    tmp_weight_tile = tl.full([XBLOCK, R0_BLOCK], scalar_weight, tl.float32)
+    tmp3, tmp4, tmp5 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp_weight_tile, 1)
     tmp2 = tmp3[:, None]
     tmp6 = tmp4[:, None]
     tmp7 = tmp5[:, None]
-    tl.store(out_ptr0 + (x0), tmp2, xmask)
-    tl.store(out_ptr1 + (x0), tmp6, xmask)
-    tl.store(out_ptr2 + (x0), tmp7, xmask)
-
-
+    tl.store(out_ptr0 + (x0), tmp2, xmask, eviction_policy='evict_last')
+    tl.store(out_ptr1 + (x0), tmp6, xmask, eviction_policy='evict_last')
+    tl.store(out_ptr2 + (x0), tmp7, xmask, eviction_policy='evict_last')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -52,7 +69,7 @@

     xnumel = 512
     r0_numel = 131072
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_native_layer_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 512, 131072, XBLOCK=1, R0_BLOCK=2048)
+    triton_red_fused_native_layer_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 512, 131072)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

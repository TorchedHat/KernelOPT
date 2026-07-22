# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_min_native_group_norm_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7490 ns — **Best latency**: 6560 ns (**1.1418×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_r0block_with_correctness_fixes_and_autotune_grid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_min_native_group_norm_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_075/kernel_3_triton_per_fused_min_native_gr/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.01
- **Registers/thread**: 36
- **Duration**: 7490 ns
- **SM Throughput**: 0.53 % | **Memory**: 0.75 % | **DRAM**: 0.75 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.30 inst/cycle | **No Eligible**: 92.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_xblock_to_maximize_grid_parallelism | 1.021× | 1.009× | ✓ | **✓ best** |
| D | reduce_xblock_to_fill_sm_grid | 1.021× | 1.040× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_xblock_to_... | Δ | reduce_xblock_to_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7420 ns | -70 ns (-0.9%) | 7200 ns | -290 ns (-3.9%) |
| SM Throughput (%) | 0.53 % | 0.53 % | +0.00 pp | 0.52 % | -0.01 pp |
| Memory Throughput (%) | 0.75 % | 0.75 % | +0.00 pp | 0.78 % | +0.03 pp |
| DRAM Throughput (%) | 0.75 % | 0.75 % | +0.00 pp | 0.78 % | +0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 36.00 register/thread | 36.00 register/thread | +0 | 36.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 92.44 % | 92.52 % | +0.08 pp | 92.58 % | +0.14 pp |
| IPC Active | 0.30 inst/cycle | 0.29 inst/cycle | -0.01 | 0.29 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0094× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_xblock_as_constexpr_with_correct_num_warps | 1.934× | 2.996× | ✓ | **✓ best** |
| C | increase_num_warps_and_autotune_to_improve_sm_occupancy | 1.730× | 1.036× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 2500 ns | -4990 ns (-66.6%) | 7230 ns | -260 ns (-3.5%) |
| SM Throughput (%) | 0.53 % | 0.02 % | -0.51 pp | 0.53 % | +0.00 pp |
| Memory Throughput (%) | 0.75 % | 0.39 % | -0.36 pp | 0.78 % | +0.03 pp |
| DRAM Throughput (%) | 0.75 % | 0.04 % | -0.71 pp | 0.78 % | +0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 36.00 register/thread | 16.00 register/thread | -20 | 36.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 |
| No Eligible (%) | 92.44 % | 95.62 % | +3.18 pp | 92.58 % | +0.14 pp |
| IPC Active | 0.30 inst/cycle | 0.04 inst/cycle | -0.26 | 0.29 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 2.996× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_r0block_and_fix_autotune_grid_lambda | 2.019× | 1.152× | ✓ |  |
| B | fix_callsite_to_enable_autotune_grid_expansion | 2.024× | 3.045× | ✓ | **✓ best** |
| C | autotune_r0_block_and_xblock_to_increase_tile_coverage | 1.942× | 1.049× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_r0block_... | Δ | fix_callsite_to_e... | Δ | autotune_r0_block... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 6500 ns | -990 ns (-13.2%) | 2460 ns | -5030 ns (-67.2%) | 7140 ns | -350 ns (-4.7%) |
| SM Throughput (%) | 0.53 % | 0.56 % | +0.03 pp | 0.02 % | -0.51 pp | 0.52 % | -0.01 pp |
| Memory Throughput (%) | 0.75 % | 0.87 % | +0.12 pp | 0.41 % | -0.34 pp | 0.78 % | +0.03 pp |
| DRAM Throughput (%) | 0.75 % | 0.87 % | +0.12 pp | 0.04 % | -0.71 pp | 0.78 % | +0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 1.56 % | -4.69 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 36.00 register/thread | 36.00 register/thread | +0 | 16.00 register/thread | -20 | 36.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.00 | -0.01 | 0.01 | +0.00 |
| No Eligible (%) | 92.44 % | 90.33 % | -2.11 pp | 96.25 % | +3.81 pp | 92.63 % | +0.19 pp |
| IPC Active | 0.30 inst/cycle | 0.38 inst/cycle | +0.08 | 0.04 inst/cycle | -0.26 | 0.29 inst/cycle | -0.01 |

---

## Iteration 4

**Best so far**: 3.0447× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_callsite_grid_lambda_and_r0mask_for_correct_autotune_grid_expansion | 1.931× | 2.601× | ✓ |  |
| B | fix_callsite_grid_lambda_to_use_autotuned_xblock | 2.021× | 1.018× | ✓ |  |
| C | fix_callsite_grid_lambda_to_enable_xblock_driven_grid_expansion | 1.942× | 1.142× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_callsite_grid... | Δ | fix_callsite_grid... | Δ | fix_callsite_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 2880 ns | -4610 ns (-61.5%) | 7360 ns | -130 ns (-1.7%) | 6560 ns | -930 ns (-12.4%) |
| SM Throughput (%) | 0.53 % | 0.01 % | -0.52 pp | 0.53 % | +0.00 pp | 0.57 % | +0.04 pp |
| Memory Throughput (%) | 0.75 % | 0.34 % | -0.41 pp | 0.76 % | +0.01 pp | 0.86 % | +0.11 pp |
| DRAM Throughput (%) | 0.75 % | 0.04 % | -0.71 pp | 0.76 % | +0.01 pp | 0.86 % | +0.11 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 36.00 register/thread | 16.00 register/thread | -20 | 36.00 register/thread | +0 | 36.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 92.44 % | 95.42 % | +2.98 pp | 92.60 % | +0.16 pp | 90.50 % | -1.94 pp |
| IPC Active | 0.30 inst/cycle | 0.04 inst/cycle | -0.26 | 0.29 inst/cycle | -0.01 | 0.39 inst/cycle | +0.09 |

---

## Iteration 5

**Best so far**: 3.0447× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_callsite_grid_lambda_to_enable_autotune_xblock_selection | 2.015× | 1.137× | ✓ |  |
| B | increase_r0block_with_correctness_fixes_and_autotune_grid | 2.028× | 1.142× | ✓ | **✓ best** |
| C | expand_autotune_configs_with_num_stages_and_fix_callsite_grid_lambda | 1.955× | 3.082× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_callsite_grid... | Δ | increase_r0block_... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 6590 ns | -900 ns (-12.0%) | 6560 ns | -930 ns (-12.4%) | 2430 ns | -5060 ns (-67.6%) |
| SM Throughput (%) | 0.53 % | 0.58 % | +0.05 pp | 0.57 % | +0.04 pp | 0.02 % | -0.51 pp |
| Memory Throughput (%) | 0.75 % | 0.85 % | +0.10 pp | 0.86 % | +0.11 pp | 0.39 % | -0.36 pp |
| DRAM Throughput (%) | 0.75 % | 0.85 % | +0.10 pp | 0.86 % | +0.11 pp | 0.05 % | -0.70 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 36.00 register/thread | 36.00 register/thread | +0 | 36.00 register/thread | +0 | 16.00 register/thread | -20 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.00 | -0.01 |
| No Eligible (%) | 92.44 % | 90.41 % | -2.03 pp | 90.41 % | -2.03 pp | 95.68 % | +3.24 pp |
| IPC Active | 0.30 inst/cycle | 0.38 inst/cycle | +0.08 | 0.39 inst/cycle | +0.09 | 0.04 inst/cycle | -0.26 |

---

## Cumulative Progress

```
Baseline:  7490 ns  (1.000×)
Iter 1     7420 ns  (1.009×)  — reduce_xblock_to_maximize_grid_parallelism
Iter 2     2500 ns  (2.996×)  — autotune_xblock_as_constexpr_with_correct_num_warps
Iter 3     2460 ns  (3.045×)  — fix_callsite_to_enable_autotune_grid_expansion

Final best: 6560 ns  (1.1418×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,8 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK':  64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK':  32}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK':  16}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK':   8}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK':   4}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK':   2}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK':   1}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK':  64}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK':  32}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK':  16}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK':   8}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK':   4}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK':   2}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK':   1}, num_warps=1, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
-def triton_per_fused_min_native_group_norm_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
+def triton_per_fused_min_native_group_norm_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
     R0_BLOCK: tl.constexpr = 64
     rnumel = r0_numel
     RBLOCK: tl.constexpr = R0_BLOCK
@@ -15,17 +37,16 @@

     xmask = xindex < xnumel
     r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
-    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
+    r0_mask = r0_index < r0_numel
     roffset = r0_offset
     rindex = r0_index
     r0_1 = r0_index
     x0 = xindex
-    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
+    tmp0 = tl.load(in_ptr0 + (r0_1 + r0_numel * x0), xmask & r0_mask, other=float('inf'))
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
-    tmp3 = tl.where(xmask, tmp1, float("inf"))
+    tmp3 = tl.where(xmask & r0_mask, tmp1, float("inf"))
     tmp4 = triton_helpers.min2(tmp3, 1)[:, None].to(tl.float32)
     tl.store(out_ptr0 + (x0), tmp4, xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +54,7 @@

     xnumel = 1024
     r0_numel = 64
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_min_native_group_norm_3[grid](arg_0, arg_1, 1024, 64, XBLOCK=128)
+    triton_per_fused_min_native_group_norm_3[grid](arg_0, arg_1, 1024, 64)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

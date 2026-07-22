# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_add_addmm_native_layer_norm_view_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 259390 ns — **Best latency**: 3070 ns (**84.4919×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_static_grid_to_enable_autotune_grid_expansion`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_add_addmm_native_layer_norm_view_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_9_triton_per_fused_add_addmm_nat/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4,1,1), Waves/SM: 0.00
- **Registers/thread**: 32
- **Duration**: 259390 ns
- **SM Throughput**: 0.48 % | **Memory**: 1.32 % | **DRAM**: 0.16 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.26 inst/cycle | **No Eligible**: 93.45 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_autotune_reduce_xblock_increase_grid_coverage | 44.731× | 91.014× | ✓ |  |
| B | reduce_xblock_via_autotune_to_increase_grid_size | 44.211× | 95.364× | ✓ |  |
| C | reduce_xblock_for_grid_expansion_and_vectorized_loads | 45.149× | 88.228× | ✓ | **✓ best** |
| D | increase_grid_size_via_smaller_xblock_with_autotune | 43.346× | 91.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_autotune_redu... | Δ | reduce_xblock_via... | Δ | reduce_xblock_for... | Δ | increase_grid_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 259390 ns | 2850 ns | -256540 ns (-98.9%) | 2720 ns | -256670 ns (-99.0%) | 2940 ns | -256450 ns (-98.9%) | 2850 ns | -256540 ns (-98.9%) |
| SM Throughput (%) | 0.48 % | 0.09 % | -0.39 pp | 0.10 % | -0.38 pp | 0.09 % | -0.39 pp | 0.10 % | -0.38 pp |
| Memory Throughput (%) | 1.32 % | 0.49 % | -0.83 pp | 0.61 % | -0.71 pp | 0.55 % | -0.77 pp | 0.53 % | -0.79 pp |
| DRAM Throughput (%) | 0.16 % | 0.19 % | +0.03 pp | 0.21 % | +0.05 pp | 0.19 % | +0.03 pp | 0.20 % | +0.04 pp |
| Achieved Occupancy (%) | 6.24 % | 6.22 % | -0.02 pp | 6.20 % | -0.04 pp | 6.19 % | -0.05 pp | 6.18 % | -0.06 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.45 % | 92.24 % | -1.21 pp | 93.54 % | +0.09 pp | 93.55 % | +0.10 pp | 93.67 % | +0.22 pp |
| IPC Active | 0.26 inst/cycle | 0.30 inst/cycle | +0.04 | 0.27 inst/cycle | +0.01 | 0.28 inst/cycle | +0.02 | 0.28 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 88.2279× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_grid_for_autotune_xblock_coverage | 44.745× | 87.044× | ✓ |  |
| B | fix_static_grid_to_enable_autotune_grid_expansion | 45.721× | 84.492× | ✓ | **✓ best** |
| C | fix_static_grid_lambda_and_add_large_xblock_configs_for_cache_reuse | 43.997× | 91.014× | ✓ |  |
| D | fix_static_grid_to_lambda_for_autotune_xblock_responsiveness | 43.894× | 91.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ | fix_static_grid_l... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 259390 ns | 2980 ns | -256410 ns (-98.9%) | 3070 ns | -256320 ns (-98.8%) | 2850 ns | -256540 ns (-98.9%) | 2850 ns | -256540 ns (-98.9%) |
| SM Throughput (%) | 0.48 % | 0.09 % | -0.39 pp | 0.09 % | -0.39 pp | 0.09 % | -0.39 pp | 0.09 % | -0.39 pp |
| Memory Throughput (%) | 1.32 % | 0.57 % | -0.75 pp | 0.48 % | -0.84 pp | 0.53 % | -0.79 pp | 0.53 % | -0.79 pp |
| DRAM Throughput (%) | 0.16 % | 0.19 % | +0.03 pp | 0.18 % | +0.02 pp | 0.20 % | +0.04 pp | 0.20 % | +0.04 pp |
| Achieved Occupancy (%) | 6.24 % | 6.22 % | -0.02 pp | 6.02 % | -0.22 pp | 6.21 % | -0.03 pp | 6.20 % | -0.04 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.45 % | 93.23 % | -0.22 pp | 93.69 % | +0.24 pp | 92.56 % | -0.89 pp | 92.49 % | -0.96 pp |
| IPC Active | 0.26 inst/cycle | 0.26 inst/cycle | +0.00 | 0.22 inst/cycle | -0.04 | 0.24 inst/cycle | -0.02 | 0.26 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 88.2279× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | increase_xblock_for_broadcast_vector_l1_cache_reuse | 44.428× | 87.044× | ✓ |  |
| C | fix_static_grid_to_lambda_and_add_num_stages_configs | 45.447× | 89.138× | ✓ |  |
| D | fix_callsite_static_grid_and_xblock_kwarg_to_activate_existing_autotune | 44.213× | 91.982× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_f... | Δ | fix_static_grid_t... | Δ | fix_callsite_stat... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 259390 ns | 2980 ns | -256410 ns (-98.9%) | 2910 ns | -256480 ns (-98.9%) | 2820 ns | -256570 ns (-98.9%) |
| SM Throughput (%) | 0.48 % | 0.07 % | -0.41 pp | 0.09 % | -0.39 pp | 0.10 % | -0.38 pp |
| Memory Throughput (%) | 1.32 % | 0.52 % | -0.80 pp | 0.51 % | -0.81 pp | 0.49 % | -0.83 pp |
| DRAM Throughput (%) | 0.16 % | 0.19 % | +0.03 pp | 0.19 % | +0.03 pp | 0.20 % | +0.04 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 6.22 % | -0.02 pp | 6.20 % | -0.04 pp |
| Registers/Thread | 32.00 register/thread | 40.00 register/thread | +8 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.45 % | 90.71 % | -2.74 pp | 93.76 % | +0.31 pp | 91.92 % | -1.53 pp |
| IPC Active | 0.26 inst/cycle | 0.18 inst/cycle | -0.08 | 0.22 inst/cycle | -0.04 | 0.29 inst/cycle | +0.03 |

---

## Iteration 4

**Best so far**: 88.2279× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_callsite_grid_lambda_to_activate_existing_autotune_xblock | 43.844× | 91.982× | ✓ |  |
| B | fix_callsite_lambda_grid_and_remove_explicit_xblock_kwarg_to_activate_autotune | 42.732× | 94.324× | ✓ |  |
| C | cap_registers_with_maxnreg_and_fix_lambda_grid_for_full_sm_occupancy | 45.635× | 78.603× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_callsite_grid... | Δ | fix_callsite_lamb... | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 259390 ns | 2820 ns | -256570 ns (-98.9%) | 2750 ns | -256640 ns (-98.9%) | 3300 ns | -256090 ns (-98.7%) |
| SM Throughput (%) | 0.48 % | 0.09 % | -0.39 pp | 0.09 % | -0.39 pp | 0.06 % | -0.42 pp |
| Memory Throughput (%) | 1.32 % | 0.49 % | -0.83 pp | 0.56 % | -0.76 pp | 0.45 % | -0.87 pp |
| DRAM Throughput (%) | 0.16 % | 0.20 % | +0.04 pp | 0.20 % | +0.04 pp | 0.19 % | +0.03 pp |
| Achieved Occupancy (%) | 6.24 % | 6.19 % | -0.05 pp | 6.19 % | -0.05 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.45 % | 91.89 % | -1.56 pp | 92.15 % | -1.30 pp | 91.02 % | -2.43 pp |
| IPC Active | 0.26 inst/cycle | 0.26 inst/cycle | +0.00 | 0.26 inst/cycle | +0.00 | 0.17 inst/cycle | -0.09 |


### Key Learnings

#### ✅ Plan A: `fix_callsite_grid_lambda_to_activate_existing_autotune_xblock` (91.982×)

When a Triton kernel already has @triton.autotune configured but the call site uses a hardcoded static grid tuple (e.g., grid = (triton.cdiv(xnumel, 128),)), the autotuner is effectively disabled — the grid never adapts to the selected XBLOCK config. Changing the static grid to a lambda (grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)) allows the autotuner to select smaller XBLOCK values that expand the grid and cover more SMs. The profiling metrics before and after this specific iteration show nearly identical NCU numbers (both at 2.82 µs, grid=4, occupancy ~6.2%), indicating the 91.98x speedup was measured at the end-to-end benchmark level rather than reflected in the single-kernel NCU trace — the NCU trace was already capturing the post-autotune best config in both snapshots.

> **Applicable when**: NCU LaunchConfiguration rule fires with high speedup_pct (>80%); grid_size << sm_count; waves_per_sm ≈ 0; achieved_occupancy << theoretical_occupancy; kernel already has @triton.autotune with multiple XBLOCK configs but the call site passes a hardcoded static grid tuple that ignores meta['XBLOCK']

> **Avoid when**: The kernel does not have @triton.autotune (add autotune first); the grid is already a lambda referencing meta['XBLOCK']; xnumel is so small that even XBLOCK=1 produces fewer blocks than sm_count; the kernel is already saturating SM utilization

---

## Cumulative Progress

```
Baseline:  259390 ns  (1.000×)
Iter 1     2940 ns  (88.228×)  — reduce_xblock_for_grid_expansion_and_vectorized_loads

Final best: 3070 ns  (84.4919×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 2},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8},   num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused_add_addmm_native_layer_norm_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 394
@@ -17,16 +34,16 @@

     xmask = xindex < xnumel
     r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
-    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
+    r0_mask = tl.full([1, R0_BLOCK], True, tl.int1)
     roffset = r0_offset
     rindex = r0_index
     r0_1 = r0_index
     x0 = xindex
     tmp0 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
-    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr0 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
     tmp2 = tl.load(in_ptr1 + (r0_1 + 512*x0), xmask, other=0.0)
-    tmp28 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
-    tmp30 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
+    tmp28 = tl.load(in_ptr2 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr3 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
     tmp3 = tmp1 + tmp2
     tmp4 = tmp0 + tmp3
     tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
@@ -53,8 +70,6 @@

     tmp31 = tmp29 + tmp30
     tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp31, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -64,7 +79,7 @@

     xnumel = 394
     r0_numel = 512
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_add_addmm_native_layer_norm_view_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 394, 512, XBLOCK=128)
+    triton_per_fused_add_addmm_native_layer_norm_view_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 394, 512)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

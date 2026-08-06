# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_convolution_mean_native_group_norm_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2240 ns — **Best latency**: 2370 ns (**0.9451×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_convolution_mean_native_group_norm_3`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_023/kernel_3_triton_per_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 22
- **Duration**: 2240 ns
- **SM Throughput**: 0.02 % | **Memory**: 0.50 % | **DRAM**: 0.06 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.22 inst/cycle | **No Eligible**: 94.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_xblock_to_increase_grid_parallelism | 1.036× | 1.047× | ✓ | **✓ best** |
| C | vectorized_loads_with_autotune_tile_sizes | 1.010× | 1.014× | ✓ |  |
| D | add_triton_autotune | 1.004× | 0.945× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_xblock_to_... | Δ | vectorized_loads_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2240 ns | 2140 ns | -100 ns (-4.5%) | 2210 ns | -30 ns (-1.3%) | 2370 ns | +130 ns (+5.8%) |
| SM Throughput (%) | 0.02 % | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp | 0.02 % | +0.00 pp |
| Memory Throughput (%) | 0.50 % | 0.49 % | -0.01 pp | 0.40 % | -0.10 pp | 0.42 % | -0.08 pp |
| DRAM Throughput (%) | 0.06 % | 0.03 % | -0.03 pp | 0.03 % | -0.03 pp | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 22.00 register/thread | 16.00 register/thread | -6 | 16.00 register/thread | -6 | 22.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 94.31 % | 96.64 % | +2.33 pp | 96.12 % | +1.81 pp | 92.26 % | -2.05 pp |
| IPC Active | 0.22 inst/cycle | 0.03 inst/cycle | -0.19 | 0.03 inst/cycle | -0.19 | 0.27 inst/cycle | +0.05 |

---

## Iteration 2

**Best so far**: 1.0467× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_enable_multi_sm_dispatch | 0.994× | 1.014× | ✓ |  |
| B | fix_hardcoded_grid_to_enable_autotune_parallelism | 1.071× | 0.945× | ✓ | **✓ best** |
| C | fix_broken_autotune_configs_to_enable_multi_block_grid | 0.988× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_hardcoded_gri... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2240 ns | 2210 ns | -30 ns (-1.3%) | 2370 ns | +130 ns (+5.8%) | 2300 ns | +60 ns (+2.7%) |
| SM Throughput (%) | 0.02 % | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp |
| Memory Throughput (%) | 0.50 % | 0.42 % | -0.08 pp | 0.37 % | -0.13 pp | 0.39 % | -0.11 pp |
| DRAM Throughput (%) | 0.06 % | 0.03 % | -0.03 pp | 0.03 % | -0.03 pp | 0.03 % | -0.03 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp |
| Registers/Thread | 22.00 register/thread | 16.00 register/thread | -6 | 16.00 register/thread | -6 | 16.00 register/thread | -6 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 94.31 % | 96.78 % | +2.47 pp | 95.92 % | +1.61 pp | 96.70 % | +2.39 pp |
| IPC Active | 0.22 inst/cycle | 0.03 inst/cycle | -0.19 | 0.03 inst/cycle | -0.19 | 0.03 inst/cycle | -0.19 |

---

## Iteration 3

**Best so far**: 1.0467× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_expand_autotune_configs_for_full_sm_utilization | 0.985× | 1.062× | ✓ |  |
| B | increase_xblock_tile_size_with_correct_grid_lambda | 1.030× | 1.047× | ✓ |  |
| C | remove_evict_last_and_debug_barrier_to_reduce_l1_thrashing | 1.041× | 1.062× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | increase_xblock_t... | Δ | remove_evict_last... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2240 ns | 2110 ns | -130 ns (-5.8%) | 2140 ns | -100 ns (-4.5%) | 2110 ns | -130 ns (-5.8%) |
| SM Throughput (%) | 0.02 % | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp |
| Memory Throughput (%) | 0.50 % | 0.41 % | -0.09 pp | 0.42 % | -0.08 pp | 0.42 % | -0.08 pp |
| DRAM Throughput (%) | 0.06 % | 0.03 % | -0.03 pp | 0.03 % | -0.03 pp | 0.03 % | -0.03 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp |
| Registers/Thread | 22.00 register/thread | 16.00 register/thread | -6 | 16.00 register/thread | -6 | 16.00 register/thread | -6 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 94.31 % | 96.13 % | +1.82 pp | 96.20 % | +1.89 pp | 96.30 % | +1.99 pp |
| IPC Active | 0.22 inst/cycle | 0.04 inst/cycle | -0.18 | 0.03 inst/cycle | -0.19 | 0.03 inst/cycle | -0.19 |

---

## Iteration 4

**Best so far**: 1.0467× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_debug_barrier_and_add_num_stages_to_autotune_configs | 1.030× | 0.957× | ✓ |  |
| B | add_num_stages_software_pipelining_to_hide_l1tex_latency | 1.032× | 1.047× | ✓ |  |
| C | remove_debug_barrier_and_fix_grid_lambda_for_full_sm_dispatch | 1.023× | 1.047× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_debug_barr... | Δ | add_num_stages_so... | Δ | remove_debug_barr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2240 ns | 2340 ns | +100 ns (+4.5%) | 2140 ns | -100 ns (-4.5%) | 2140 ns | -100 ns (-4.5%) |
| SM Throughput (%) | 0.02 % | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp |
| Memory Throughput (%) | 0.50 % | 0.38 % | -0.12 pp | 0.41 % | -0.09 pp | 0.41 % | -0.09 pp |
| DRAM Throughput (%) | 0.06 % | 0.03 % | -0.03 pp | 0.03 % | -0.03 pp | 0.03 % | -0.03 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp |
| Registers/Thread | 22.00 register/thread | 16.00 register/thread | -6 | 16.00 register/thread | -6 | 16.00 register/thread | -6 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 94.31 % | 96.39 % | +2.08 pp | 96.25 % | +1.94 pp | 96.28 % | +1.97 pp |
| IPC Active | 0.22 inst/cycle | 0.03 inst/cycle | -0.19 | 0.03 inst/cycle | -0.19 | 0.03 inst/cycle | -0.19 |

---

## Cumulative Progress

```
Baseline:  2240 ns  (1.000×)
Iter 1     2140 ns  (1.047×)  — reduce_xblock_to_increase_grid_parallelism

Final best: 2370 ns  (0.9451×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},   num_warps=1),
+        triton.Config({'XBLOCK': 2},   num_warps=1),
+        triton.Config({'XBLOCK': 4},   num_warps=1),
+        triton.Config({'XBLOCK': 8},   num_warps=2),
+        triton.Config({'XBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 64},  num_warps=4),  # NEW
+        triton.Config({'XBLOCK': 128}, num_warps=4),  # NEW
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused_convolution_mean_native_group_norm_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 128
@@ -22,16 +36,14 @@

     rindex = r0_index
     r0_1 = r0_index
     x0 = xindex
-    tmp0 = tl.load(in_ptr0 + (r0_1 + 6*x0), r0_mask & xmask, other=0.0)
+    tmp0 = tl.load(in_ptr0 + (r0_1 + 6*x0), mask=r0_mask & xmask, other=0.0)
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
     tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
     tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
     tmp5 = tl.full([1, 1], 475200.0, tl.float32)
     tmp6 = (tmp4 / tmp5)
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
-
-
+    tl.store(in_out_ptr0 + (x0), tmp6, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -39,7 +51,7 @@

     xnumel = 128
     r0_numel = 6
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_convolution_mean_native_group_norm_3[grid](arg_0, arg_1, 128, 6, XBLOCK=128)
+    triton_per_fused_convolution_mean_native_group_norm_3[grid](arg_0, arg_1, 128, 6)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

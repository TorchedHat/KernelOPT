# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_native_group_norm_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 281280 ns — **Best latency**: 2840000 ns (**0.0990×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_native_group_norm_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_092/kernel_2_triton_red_fused_convolution_n/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 1.94
- **Registers/thread**: 61
- **Duration**: 281280 ns
- **SM Throughput**: 53.91 % | **Memory**: 90.82 % | **DRAM**: 41.86 %
- **Achieved Occupancy**: 46.89 %
- **IPC Active**: 2.25 inst/cycle | **No Eligible**: 43.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_via_xblock_increase_and_autotune | 0.999× | 0.003× | ✓ |  |
| B | eliminate_partial_wave_via_xblock_increase_and_pipelining | 1.137× | 0.982× | ✓ |  |
| C | increase_xblock_to_reduce_partial_wave_and_improve_coalescing | 1.154× | 0.974× | ✓ | **✓ best** |
| D | autotune_xblock_r0block_to_fix_partial_wave_and_coalescing | 1.043× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | eliminate_partial... | Δ | increase_xblock_t... | Δ | autotune_xblock_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 281280 ns | 90620000 ns | +90338720 ns (+32117.0%) | 286460 ns | +5180 ns (+1.8%) | 288830 ns | +7550 ns (+2.7%) | 284320 ns | +3040 ns (+1.1%) |
| SM Throughput (%) | 53.91 % | 14.01 % | -39.90 pp | 54.74 % | +0.83 pp | 53.53 % | -0.38 pp | 53.76 % | -0.15 pp |
| Memory Throughput (%) | 90.82 % | 67.18 % | -23.64 pp | 88.62 % | -2.20 pp | 88.30 % | -2.52 pp | 89.25 % | -1.57 pp |
| DRAM Throughput (%) | 41.86 % | 67.18 % | +25.32 pp | 42.43 % | +0.57 pp | 41.64 % | -0.22 pp | 41.34 % | -0.52 pp |
| Achieved Occupancy (%) | 46.89 % | 96.55 % | +49.66 pp | 46.91 % | +0.02 pp | 46.85 % | -0.04 pp | 46.87 % | -0.02 pp |
| Registers/Thread | 61.00 register/thread | 32.00 register/thread | -29 | 61.00 register/thread | +0 | 61.00 register/thread | +0 | 61.00 register/thread | +0 |
| Waves/SM | 1.94 | 0.97 | -0.97 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 43.10 % | 94.33 % | +51.23 pp | 44.41 % | +1.31 pp | 43.46 % | +0.36 pp | 43.48 % | +0.38 pp |
| IPC Active | 2.25 inst/cycle | 0.23 inst/cycle | -2.02 | 2.23 inst/cycle | -0.02 | 2.28 inst/cycle | +0.03 | 2.28 inst/cycle | +0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_enable_xblock_coalescing_and_eliminate_partial_wave | 1.045× | 0.998× | ✓ |  |
| B | fix_autotune_grid_lambda_and_reduce_register_pressure | 1.135× | 0.985× | ✓ |  |
| C | pipeline_reduction_loop_to_hide_l1tex_latency_via_num_stages | 1.155× | 0.987× | ✓ | **✓ best** |
| D | fix_autotune_grid_lambda_to_eliminate_cta_inflation | 0.998× | 0.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_autotune_grid... | Δ | pipeline_reductio... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 281280 ns | 281890 ns | +610 ns (+0.2%) | 285660 ns | +4380 ns (+1.6%) | 284930 ns | +3650 ns (+1.3%) | 90640000 ns | +90358720 ns (+32124.1%) |
| SM Throughput (%) | 53.91 % | 53.07 % | -0.84 pp | 53.98 % | +0.07 pp | 53.42 % | -0.49 pp | 14.03 % | -39.88 pp |
| Memory Throughput (%) | 90.82 % | 90.84 % | +0.02 pp | 90.23 % | -0.59 pp | 90.29 % | -0.53 pp | 67.17 % | -23.65 pp |
| DRAM Throughput (%) | 41.86 % | 42.01 % | +0.15 pp | 41.97 % | +0.11 pp | 41.78 % | -0.08 pp | 67.17 % | +25.31 pp |
| Achieved Occupancy (%) | 46.89 % | 46.86 % | -0.03 pp | 46.86 % | -0.03 pp | 46.77 % | -0.12 pp | 96.57 % | +49.68 pp |
| Registers/Thread | 61.00 register/thread | 61.00 register/thread | +0 | 61.00 register/thread | +0 | 61.00 register/thread | +0 | 32.00 register/thread | -29 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 0.97 | -0.97 |
| No Eligible (%) | 43.10 % | 42.43 % | -0.67 pp | 44.73 % | +1.63 pp | 43.81 % | +0.71 pp | 94.33 % | +51.23 pp |
| IPC Active | 2.25 inst/cycle | 2.26 inst/cycle | +0.01 | 2.24 inst/cycle | -0.01 | 2.25 inst/cycle | +0.00 | 0.23 inst/cycle | -2.02 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_to_double_occupancy_and_hide_l2_latency | 0.262× | 0.202× | ✓ |  |
| B | fix_grid_lambda_and_add_evict_first_on_stores_for_coalesced_output_writes | 1.154× | 0.992× | ✓ |  |
| C | fix_grid_lambda_and_coalesce_stores_via_maxnreg_autotune | 1.135× | 0.991× | ✓ |  |
| D | fix_grid_lambda_to_match_autotuned_xblock | 0.999× | 0.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 281280 ns | 1390000 ns | +1108720 ns (+394.2%) | 283580 ns | +2300 ns (+0.8%) | 283810 ns | +2530 ns (+0.9%) | 90640000 ns | +90358720 ns (+32124.1%) |
| SM Throughput (%) | 53.91 % | 19.52 % | -34.39 pp | 54.79 % | +0.88 pp | 52.79 % | -1.12 pp | 14.02 % | -39.89 pp |
| Memory Throughput (%) | 90.82 % | 52.61 % | -38.21 pp | 90.30 % | -0.52 pp | 89.38 % | -1.44 pp | 67.17 % | -23.65 pp |
| DRAM Throughput (%) | 41.86 % | 40.18 % | -1.68 pp | 41.28 % | -0.58 pp | 42.02 % | +0.16 pp | 67.17 % | +25.31 pp |
| Achieved Occupancy (%) | 46.89 % | 89.53 % | +42.64 pp | 46.95 % | +0.06 pp | 46.90 % | +0.01 pp | 96.56 % | +49.67 pp |
| Registers/Thread | 61.00 register/thread | 32.00 register/thread | -29 | 61.00 register/thread | +0 | 61.00 register/thread | +0 | 32.00 register/thread | -29 |
| Waves/SM | 1.94 | 0.97 | -0.97 | 1.94 | +0.00 | 1.94 | +0.00 | 0.97 | -0.97 |
| No Eligible (%) | 43.10 % | 83.29 % | +40.19 pp | 43.12 % | +0.02 pp | 43.29 % | +0.19 pp | 94.33 % | +51.23 pp |
| IPC Active | 2.25 inst/cycle | 0.68 inst/cycle | -1.57 | 2.25 inst/cycle | +0.00 | 2.25 inst/cycle | +0.00 | 0.23 inst/cycle | -2.02 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_global_stores_via_grid_lambda_fix_and_load_reorder | 1.156× | 0.995× | ✓ |  |
| B | fix_grid_lambda_to_enable_xblock_coalescing_and_eliminate_partial_wave_v2 | 1.135× | 0.995× | ✓ |  |
| C | fix_grid_lambda_to_coalesce_output_stores_via_xblock_autotune | 0.262× | 0.201× | ✓ |  |
| D | fix_grid_lambda_and_reduce_r0block_to_eliminate_register_spill_and_cta_inflation | 1.456× | 0.099× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | coalesce_global_s... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 281280 ns | 282750 ns | +1470 ns (+0.5%) | 282780 ns | +1500 ns (+0.5%) | 1400000 ns | +1118720 ns (+397.7%) | 2840000 ns | +2558720 ns (+909.7%) |
| SM Throughput (%) | 53.91 % | 53.41 % | -0.50 pp | 53.64 % | -0.27 pp | 19.40 % | -34.51 pp | 72.42 % | +18.51 pp |
| Memory Throughput (%) | 90.82 % | 89.71 % | -1.11 pp | 89.90 % | -0.92 pp | 52.37 % | -38.45 pp | 12.60 % | -78.22 pp |
| DRAM Throughput (%) | 41.86 % | 41.41 % | -0.45 pp | 40.39 % | -1.47 pp | 40.12 % | -1.74 pp | 3.74 % | -38.12 pp |
| Achieved Occupancy (%) | 46.89 % | 46.89 % | +0.00 pp | 46.88 % | -0.01 pp | 89.17 % | +42.28 pp | 12.20 % | -34.69 pp |
| Registers/Thread | 61.00 register/thread | 61.00 register/thread | +0 | 61.00 register/thread | +0 | 32.00 register/thread | -29 | 255.00 register/thread | +194 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 0.97 | -0.97 | 7.76 | +5.82 |
| No Eligible (%) | 43.10 % | 43.17 % | +0.07 pp | 42.55 % | -0.55 pp | 83.21 % | +40.11 pp | 25.37 % | -17.73 pp |
| IPC Active | 2.25 inst/cycle | 2.27 inst/cycle | +0.02 | 2.27 inst/cycle | +0.02 | 0.68 inst/cycle | -1.57 | 2.98 inst/cycle | +0.73 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_to_coalesce_scalar_output_stores | 1.283× | 0.962× | ✓ |  |
| C | remove_maxnreg_spill_and_fix_grid_lambda_for_welford_reduction | 1.127× | 1.131× | ✓ |  |
| D | fix_grid_lambda_and_cap_welford_tile_to_eliminate_register_spill | 1.377× | 1.114× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | remove_maxnreg_sp... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 281280 ns | 292420 ns | +11140 ns (+4.0%) | 248610 ns | -32670 ns (-11.6%) | 252580 ns | -28700 ns (-10.2%) |
| SM Throughput (%) | 53.91 % | 50.98 % | -2.93 pp | 52.53 % | -1.38 pp | 51.14 % | -2.77 pp |
| Memory Throughput (%) | 90.82 % | 78.03 % | -12.79 pp | 93.11 % | +2.29 pp | 91.75 % | +0.93 pp |
| DRAM Throughput (%) | 41.86 % | 36.35 % | -5.51 pp | 42.79 % | +0.93 pp | 42.11 % | +0.25 pp |
| Achieved Occupancy (%) | 46.89 % | 22.62 % | -24.27 pp | 95.08 % | +48.19 pp | 47.35 % | +0.46 pp |
| Registers/Thread | 61.00 register/thread | 62.00 register/thread | +1 | 32.00 register/thread | -29 | 32.00 register/thread | -29 |
| Waves/SM | 1.94 | 0.48 | -1.46 | 0.97 | -0.97 | 0.48 | -1.46 |
| No Eligible (%) | 43.10 % | 47.08 % | +3.98 pp | 44.45 % | +1.35 pp | 47.15 % | +4.05 pp |
| IPC Active | 2.25 inst/cycle | 2.08 inst/cycle | -0.17 | 2.22 inst/cycle | -0.03 | 2.11 inst/cycle | -0.14 |

---

## Cumulative Progress

```
Baseline:  281280 ns  (1.000×)

Final best: 2840000 ns  (0.0990×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 512}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 512}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 256}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 64}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 256}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_native_group_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 2048
@@ -47,9 +73,8 @@

     tmp4 = tmp5[:, None]
     tmp8 = tmp6[:, None]
     tmp9 = tmp7[:, None]
-    tl.store(out_ptr0 + (x4), tmp4, mask=xmask)
-    tl.store(out_ptr1 + (x4), tmp8, mask=xmask)
-
+    tl.store(out_ptr0 + (x4), tmp4, mask=xmask, eviction_policy='evict_first')
+    tl.store(out_ptr1 + (x4), tmp8, mask=xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -59,7 +84,7 @@

     xnumel = 2048
     r0_numel = 63504
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_convolution_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, 2048, 63504, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused_convolution_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, 2048, 63504)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

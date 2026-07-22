# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__native_batch_norm_legit_convolution_mul_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 43420 ns — **Best latency**: 46140 ns (**0.9410×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__native_batch_norm_legit_convolution_mul_view_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_079/kernel_0_triton_red_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 1.55
- **Registers/thread**: 48
- **Duration**: 43420 ns
- **SM Throughput**: 60.07 % | **Memory**: 49.53 % | **DRAM**: 49.53 %
- **Achieved Occupancy**: 49.85 %
- **IPC Active**: 2.67 inst/cycle | **No Eligible**: 33.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_reduce_partial_wave_and_improve_store_coalescing | 0.996× | 0.986× | ✓ |  |
| B | autotune_xblock_r0block_tile_sizes | 1.001× | 1.012× | ✓ | **✓ best** |
| C | vectorized_loads_stores_via_xblock_increase_and_autotune | 0.996× | 0.996× | ✓ |  |
| D | add_autotune_increase_xblock_reduce_grid_size | 0.996× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_r... | Δ | vectorized_loads_... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43420 ns | 44030 ns | +610 ns (+1.4%) | 42910 ns | -510 ns (-1.2%) | 43580 ns | +160 ns (+0.4%) | 42910 ns | -510 ns (-1.2%) |
| SM Throughput (%) | 60.07 % | 59.86 % | -0.21 pp | 60.82 % | +0.75 pp | 59.76 % | -0.31 pp | 59.43 % | -0.64 pp |
| Memory Throughput (%) | 49.53 % | 48.83 % | -0.70 pp | 50.08 % | +0.55 pp | 49.26 % | -0.27 pp | 50.07 % | +0.54 pp |
| DRAM Throughput (%) | 49.53 % | 48.83 % | -0.70 pp | 50.08 % | +0.55 pp | 49.26 % | -0.27 pp | 50.07 % | +0.54 pp |
| Achieved Occupancy (%) | 49.85 % | 50.12 % | +0.27 pp | 49.94 % | +0.09 pp | 49.93 % | +0.08 pp | 49.94 % | +0.09 pp |
| Registers/Thread | 48.00 register/thread | 48.00 register/thread | +0 | 48.00 register/thread | +0 | 48.00 register/thread | +0 | 48.00 register/thread | +0 |
| Waves/SM | 1.55 | 1.55 | +0.00 | 1.55 | +0.00 | 1.55 | +0.00 | 1.55 | +0.00 |
| No Eligible (%) | 33.42 % | 33.61 % | +0.19 pp | 33.15 % | -0.27 pp | 33.53 % | +0.11 pp | 33.10 % | -0.32 pp |
| IPC Active | 2.67 inst/cycle | 2.62 inst/cycle | -0.05 | 2.64 inst/cycle | -0.03 | 2.66 inst/cycle | -0.01 | 2.65 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.0119× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_increase_occupancy_via_num_warps_and_fix_grid | 1.025× | 1.013× | ✓ |  |
| B | reduce_register_pressure_and_hide_l1tex_latency_via_num_warps2_and_num_stages2 | 1.033× | 1.000× | ✓ | **✓ best** |
| C | expand_autotune_configs_larger_xblock_fewer_warps_more_stages | 0.996× | 1.015× | ✓ |  |
| D | fix_grid_xblock_and_add_pipelined_stages_to_hide_l1tex_latency | 1.030× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | expand_autotune_c... | Δ | fix_grid_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43420 ns | 42850 ns | -570 ns (-1.3%) | 43420 ns | +0 ns (+0.0%) | 42780 ns | -640 ns (-1.5%) | 43300 ns | -120 ns (-0.3%) |
| SM Throughput (%) | 60.07 % | 58.82 % | -1.25 pp | 59.31 % | -0.76 pp | 61.01 % | +0.94 pp | 59.25 % | -0.82 pp |
| Memory Throughput (%) | 49.53 % | 50.16 % | +0.63 pp | 49.52 % | -0.01 pp | 50.42 % | +0.89 pp | 49.59 % | +0.06 pp |
| DRAM Throughput (%) | 49.53 % | 50.16 % | +0.63 pp | 49.52 % | -0.01 pp | 50.42 % | +0.89 pp | 49.59 % | +0.06 pp |
| Achieved Occupancy (%) | 49.85 % | 49.87 % | +0.02 pp | 49.86 % | +0.01 pp | 49.89 % | +0.04 pp | 49.96 % | +0.11 pp |
| Registers/Thread | 48.00 register/thread | 48.00 register/thread | +0 | 48.00 register/thread | +0 | 48.00 register/thread | +0 | 48.00 register/thread | +0 |
| Waves/SM | 1.55 | 1.55 | +0.00 | 1.55 | +0.00 | 1.55 | +0.00 | 1.55 | +0.00 |
| No Eligible (%) | 33.42 % | 33.21 % | -0.21 pp | 34.35 % | +0.93 pp | 33.93 % | +0.51 pp | 34.10 % | +0.68 pp |
| IPC Active | 2.67 inst/cycle | 2.67 inst/cycle | +0.00 | 2.66 inst/cycle | -0.01 | 2.65 inst/cycle | -0.02 | 2.62 inst/cycle | -0.05 |

---

## Iteration 3

**Best so far**: 1.0119× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_to_respect_xblock_autotune_and_eliminate_partial_wave | 0.234× | 0.003× | ✓ |  |
| B | fix_grid_launch_to_use_xblock_and_add_larger_xblock_configs | 1.033× | 1.013× | ✓ |  |
| C | fix_grid_formula_and_add_large_xblock_configs_for_coalesced_stores | 1.025× | 1.016× | ✓ |  |
| D | fix_grid_to_use_autotuned_xblock_via_lambda | 1.025× | 1.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_to_respe... | Δ | fix_grid_launch_t... | Δ | fix_grid_formula_... | Δ | fix_grid_to_use_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43420 ns | 15110000 ns | +15066580 ns (+34699.6%) | 42880 ns | -540 ns (-1.2%) | 42720 ns | -700 ns (-1.6%) | 42750 ns | -670 ns (-1.5%) |
| SM Throughput (%) | 60.07 % | 15.94 % | -44.13 pp | 59.98 % | -0.09 pp | 60.40 % | +0.33 pp | 60.64 % | +0.57 pp |
| Memory Throughput (%) | 49.53 % | 67.57 % | +18.04 pp | 50.08 % | +0.55 pp | 50.30 % | +0.77 pp | 50.16 % | +0.63 pp |
| DRAM Throughput (%) | 49.53 % | 67.57 % | +18.04 pp | 50.08 % | +0.55 pp | 50.30 % | +0.77 pp | 50.16 % | +0.63 pp |
| Achieved Occupancy (%) | 49.85 % | 96.49 % | +46.64 pp | 49.81 % | -0.04 pp | 49.85 % | +0.00 pp | 49.85 % | +0.00 pp |
| Registers/Thread | 48.00 register/thread | 32.00 register/thread | -16 | 48.00 register/thread | +0 | 48.00 register/thread | +0 | 48.00 register/thread | +0 |
| Waves/SM | 1.55 | 0.97 | -0.58 | 1.55 | +0.00 | 1.55 | +0.00 | 1.55 | +0.00 |
| No Eligible (%) | 33.42 % | 93.32 % | +59.90 pp | 33.46 % | +0.04 pp | 33.15 % | -0.27 pp | 34.60 % | +1.18 pp |
| IPC Active | 2.67 inst/cycle | 0.27 inst/cycle | -2.40 | 2.66 inst/cycle | -0.01 | 2.68 inst/cycle | +0.01 | 2.66 inst/cycle | -0.01 |

---

## Iteration 4

**Best so far**: 1.0119× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_add_mul_into_fma_in_welford_reduction_body | 0.912× | 0.915× | ✓ |  |
| B | fix_grid_lambda_with_spill_safe_large_xblock_small_r0block_configs | 1.030× | 0.983× | ✓ |  |
| C | coalesce_output_stores_via_moderate_xblock_with_small_r0block_and_fixed_grid | 1.029× | 1.013× | ✓ |  |
| D | fix_grid_lambda_with_spill_safe_xblock1_configs | 1.111× | 0.941× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fuse_fp32_add_mul... | Δ | fix_grid_lambda_w... | Δ | coalesce_output_s... | Δ | fix_grid_lambda_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43420 ns | 47460 ns | +4040 ns (+9.3%) | 44160 ns | +740 ns (+1.7%) | 42880 ns | -540 ns (-1.2%) | 46140 ns | +2720 ns (+6.3%) |
| SM Throughput (%) | 60.07 % | 62.66 % | +2.59 pp | 61.24 % | +1.17 pp | 59.91 % | -0.16 pp | 66.06 % | +5.99 pp |
| Memory Throughput (%) | 49.53 % | 45.24 % | -4.29 pp | 48.67 % | -0.86 pp | 50.13 % | +0.60 pp | 46.48 % | -3.05 pp |
| DRAM Throughput (%) | 49.53 % | 45.24 % | -4.29 pp | 48.67 % | -0.86 pp | 50.13 % | +0.60 pp | 46.48 % | -3.05 pp |
| Achieved Occupancy (%) | 49.85 % | 37.12 % | -12.73 pp | 49.52 % | -0.33 pp | 49.62 % | -0.23 pp | 55.59 % | +5.74 pp |
| Registers/Thread | 48.00 register/thread | 66.00 register/thread | +18 | 48.00 register/thread | +0 | 48.00 register/thread | +0 | 48.00 register/thread | +0 |
| Waves/SM | 1.55 | 2.22 | +0.67 | 1.55 | +0.00 | 1.55 | +0.00 | 3.10 | +1.55 |
| No Eligible (%) | 33.42 % | 31.85 % | -1.57 pp | 34.82 % | +1.40 pp | 33.25 % | -0.17 pp | 26.24 % | -7.18 pp |
| IPC Active | 2.67 inst/cycle | 2.70 inst/cycle | +0.03 | 2.61 inst/cycle | -0.06 | 2.61 inst/cycle | -0.06 | 2.91 inst/cycle | +0.24 |

---

## Iteration 5

**Best so far**: 1.0119× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | pipeline_reduction_loop_with_num_stages2_and_fix_grid_lambda | 1.110× | 1.013× | ✓ |  |
| B | inline_welford_with_fma_friendly_arithmetic_to_reduce_nonfused_fp32_instructions | 1.041× | 0.985× | ✓ |  |
| C | increase_r0block_with_pipelining_to_hide_l1tex_latency_and_exploit_fma | 1.107× | 0.557× | ✓ |  |
| D | add_num_stages3_configs_to_hide_l1tex_latency_and_reduce_register_spill | 1.095× | 0.920× | ✓ |  |


### Metric Comparison

| Metric | Baseline | pipeline_reductio... | Δ | inline_welford_wi... | Δ | increase_r0block_... | Δ | add_num_stages3_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43420 ns | 42850 ns | -570 ns (-1.3%) | 44100 ns | +680 ns (+1.6%) | 77890 ns | +34470 ns (+79.4%) | 47200 ns | +3780 ns (+8.7%) |
| SM Throughput (%) | 60.07 % | 60.22 % | +0.15 pp | 64.44 % | +4.37 pp | 52.68 % | -7.39 pp | 62.17 % | +2.10 pp |
| Memory Throughput (%) | 49.53 % | 50.26 % | +0.73 pp | 48.86 % | -0.67 pp | 27.60 % | -21.93 pp | 45.72 % | -3.81 pp |
| DRAM Throughput (%) | 49.53 % | 50.26 % | +0.73 pp | 48.86 % | -0.67 pp | 27.60 % | -21.93 pp | 45.72 % | -3.81 pp |
| Achieved Occupancy (%) | 49.85 % | 49.93 % | +0.08 pp | 49.60 % | -0.25 pp | 12.37 % | -37.48 pp | 37.10 % | -12.75 pp |
| Registers/Thread | 48.00 register/thread | 48.00 register/thread | +0 | 48.00 register/thread | +0 | 253.00 register/thread | +205 | 66.00 register/thread | +18 |
| Waves/SM | 1.55 | 1.55 | +0.00 | 1.55 | +0.00 | 15.52 | +13.97 | 2.22 | +0.67 |
| No Eligible (%) | 33.42 % | 33.79 % | +0.37 pp | 26.73 % | -6.69 pp | 41.47 % | +8.05 pp | 32.75 % | -0.67 pp |
| IPC Active | 2.67 inst/cycle | 2.64 inst/cycle | -0.03 | 2.94 inst/cycle | +0.27 | 2.33 inst/cycle | -0.34 | 2.71 inst/cycle | +0.04 |

---

## Cumulative Progress

```
Baseline:  43420 ns  (1.000×)
Iter 1     42910 ns  (1.012×)  — autotune_xblock_r0block_tile_sizes

Final best: 46140 ns  (0.9410×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    # XBLOCK=1 fallback configs with larger R0_BLOCK (safe, no spill risk)
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=2, num_stages=1),
+    # XBLOCK=2 with small R0_BLOCK (XBLOCK*R0_BLOCK <= 512)
+    triton.Config({'XBLOCK': 2, 'R0_BLOCK': 256}, num_warps=2, num_stages=1),
+    triton.Config({'XBLOCK': 2, 'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+    # XBLOCK=4 with small R0_BLOCK (XBLOCK*R0_BLOCK <= 1024)
+    triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256}, num_warps=2, num_stages=1),
+    triton.Config({'XBLOCK': 4, 'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+    triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
+    # XBLOCK=8 with small R0_BLOCK (XBLOCK*R0_BLOCK <= 2048)
+    triton.Config({'XBLOCK': 8, 'R0_BLOCK': 256}, num_warps=2, num_stages=1),
+    triton.Config({'XBLOCK': 8, 'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+    triton.Config({'XBLOCK': 8, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
+    # XBLOCK=16 with small R0_BLOCK (XBLOCK*R0_BLOCK <= 2048)
+    triton.Config({'XBLOCK': 16, 'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+    triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
+], key=['xnumel', 'r0_numel'])
 @triton.jit
 def triton_red_fused__native_batch_norm_legit_convolution_mul_view_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 2048
@@ -17,8 +37,8 @@

     r0_base = tl.arange(0, R0_BLOCK)[None, :]
     rbase = r0_base
     x0 = xindex
-    tmp1 = tl.load(in_ptr1 + ((x0 % 16)), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + ((x0 % 16)), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + ((x0 % 16)), mask=xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + ((x0 % 16)), mask=xmask, eviction_policy='evict_last')
     tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
@@ -28,7 +48,7 @@

         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_1 + 12600*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_1 + 12600*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp4 = tmp2 * tmp3
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
@@ -42,10 +62,8 @@

     tmp6 = tmp7[:, None]
     tmp10 = tmp8[:, None]
     tmp11 = tmp9[:, None]
-    tl.store(out_ptr0 + (x0), tmp6, xmask)
-    tl.store(out_ptr1 + (x0), tmp10, xmask)
-
-
+    tl.store(out_ptr0 + (x0), tmp6, mask=xmask, eviction_policy='evict_last')
+    tl.store(out_ptr1 + (x0), tmp10, mask=xmask, eviction_policy='evict_last')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -56,7 +74,7 @@

     xnumel = 2048
     r0_numel = 12600
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused__native_batch_norm_legit_convolution_mul_view_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2048, 12600, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused__native_batch_norm_legit_convolution_mul_view_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2048, 12600)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

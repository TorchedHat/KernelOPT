# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_max_pool2d_with_indices_75` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4990 ns — **Best latency**: 5250 ns (**0.9505×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_max_pool2d_with_indices_75`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_75_triton_poi_fused_max_pool2d_wi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3185,1,1), Waves/SM: 1.51
- **Registers/thread**: 24
- **Duration**: 4990 ns
- **SM Throughput**: 31.94 % | **Memory**: 30.03 % | **DRAM**: 26.70 %
- **Achieved Occupancy**: 70.87 %
- **IPC Active**: 2.04 inst/cycle | **No Eligible**: 47.54 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave | 0.977× | 0.945× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 1.030× | 0.986× | ✓ | **✓ best** |
| C | add_autotune_with_larger_xblock_for_vectorized_loads_and_wave_elimination | 0.999× | 0.986× | ✓ |  |
| D | eliminate_partial_wave_via_larger_xblock_autotune | 1.006× | 0.980× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_e... | Δ | add_autotune_with... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4990 ns | 5280 ns | +290 ns (+5.8%) | 5060 ns | +70 ns (+1.4%) | 5060 ns | +70 ns (+1.4%) | 5090 ns | +100 ns (+2.0%) |
| SM Throughput (%) | 31.94 % | 41.50 % | +9.56 pp | 36.85 % | +4.91 pp | 36.31 % | +4.37 pp | 31.39 % | -0.55 pp |
| Memory Throughput (%) | 30.03 % | 27.04 % | -2.99 pp | 27.37 % | -2.66 pp | 27.30 % | -2.73 pp | 29.27 % | -0.76 pp |
| DRAM Throughput (%) | 26.70 % | 25.34 % | -1.36 pp | 26.58 % | -0.12 pp | 26.44 % | -0.26 pp | 26.14 % | -0.56 pp |
| Achieved Occupancy (%) | 70.87 % | 75.32 % | +4.45 pp | 70.89 % | +0.02 pp | 71.55 % | +0.68 pp | 71.16 % | +0.29 pp |
| Registers/Thread | 24.00 register/thread | 32.00 register/thread | +8 | 32.00 register/thread | +8 | 32.00 register/thread | +8 | 24.00 register/thread | +0 |
| Waves/SM | 1.51 | 1.51 | +0.00 | 1.51 | +0.00 | 1.51 | +0.00 | 1.51 | +0.00 |
| No Eligible (%) | 47.54 % | 24.16 % | -23.38 pp | 33.78 % | -13.76 pp | 33.17 % | -14.37 pp | 47.70 % | +0.16 pp |
| IPC Active | 2.04 inst/cycle | 2.97 inst/cycle | +0.93 | 2.56 inst/cycle | +0.52 | 2.58 inst/cycle | +0.54 | 2.04 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_increase_num_warps_for_latency_hiding | 0.843× | 0.722× | ✓ |  |
| B | fix_autotune_grid_and_increase_num_warps_to_hide_l1tex_latency | 1.024× | 0.712× | ✓ |  |
| C | fix_autotune_grid_and_add_num_stages_for_latency_hiding | 1.023× | 0.986× | ✓ |  |
| D | fix_autotune_grid_and_increase_num_warps_for_occupancy | 1.024× | 0.986× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4990 ns | 6910 ns | +1920 ns (+38.5%) | 7010 ns | +2020 ns (+40.5%) | 5060 ns | +70 ns (+1.4%) | 5060 ns | +70 ns (+1.4%) |
| SM Throughput (%) | 31.94 % | 53.20 % | +21.26 pp | 46.98 % | +15.04 pp | 31.22 % | -0.72 pp | 36.68 % | +4.74 pp |
| Memory Throughput (%) | 30.03 % | 25.46 % | -4.57 pp | 21.52 % | -8.51 pp | 29.57 % | -0.46 pp | 27.17 % | -2.86 pp |
| DRAM Throughput (%) | 26.70 % | 19.30 % | -7.40 pp | 18.99 % | -7.71 pp | 26.42 % | -0.28 pp | 26.37 % | -0.33 pp |
| Achieved Occupancy (%) | 70.87 % | 77.20 % | +6.33 pp | 77.34 % | +6.47 pp | 72.52 % | +1.65 pp | 72.91 % | +2.04 pp |
| Registers/Thread | 24.00 register/thread | 32.00 register/thread | +8 | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 32.00 register/thread | +8 |
| Waves/SM | 1.51 | 3.02 | +1.51 | 3.02 | +1.51 | 1.51 | +0.00 | 1.51 | +0.00 |
| No Eligible (%) | 47.54 % | 24.33 % | -23.21 pp | 33.63 % | -13.91 pp | 47.85 % | +0.31 pp | 32.85 % | -14.69 pp |
| IPC Active | 2.04 inst/cycle | 2.99 inst/cycle | +0.95 | 2.61 inst/cycle | +0.57 | 1.92 inst/cycle | -0.12 | 2.41 inst/cycle | +0.37 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_cache_eviction_policy_to_promote_l1_reuse_on_scattered_loads | 1.065× | 0.969× | ✓ | **✓ best** |
| B | wave_align_xblock_via_correct_grid_lambda | 1.036× | 0.994× | ✓ |  |
| C | fix_grid_lambda_and_autotune_small_xblock_to_eliminate_partial_wave | 1.063× | 1.075× | ✓ |  |
| D | reduce_instruction_count_via_interior_fast_path_for_pooling_boundary_checks | 0.901× | 0.530× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_cache_evictio... | Δ | wave_align_xblock... | Δ | fix_grid_lambda_a... | Δ | reduce_instructio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4990 ns | 5150 ns | +160 ns (+3.2%) | 5020 ns | +30 ns (+0.6%) | 4640 ns | -350 ns (-7.0%) | 9410 ns | +4420 ns (+88.6%) |
| SM Throughput (%) | 31.94 % | 37.74 % | +5.80 pp | 36.52 % | +4.58 pp | 16.81 % | -15.13 pp | 53.63 % | +21.69 pp |
| Memory Throughput (%) | 30.03 % | 29.20 % | -0.83 pp | 27.35 % | -2.68 pp | 16.55 % | -13.48 pp | 31.29 % | +1.26 pp |
| DRAM Throughput (%) | 26.70 % | 25.94 % | -0.76 pp | 26.56 % | -0.14 pp | 14.40 % | -12.30 pp | 14.21 % | -12.49 pp |
| Achieved Occupancy (%) | 70.87 % | 71.17 % | +0.30 pp | 70.66 % | -0.21 pp | 48.15 % | -22.72 pp | 80.09 % | +9.22 pp |
| Registers/Thread | 24.00 register/thread | 23.00 register/thread | -1 | 32.00 register/thread | +8 | 24.00 register/thread | +0 | 28.00 register/thread | +4 |
| Waves/SM | 1.51 | 1.51 | +0.00 | 1.51 | +0.00 | 0.75 | -0.76 | 3.02 | +1.51 |
| No Eligible (%) | 47.54 % | 34.52 % | -13.02 pp | 34.17 % | -13.37 pp | 57.20 % | +9.66 pp | 29.16 % | -18.38 pp |
| IPC Active | 2.04 inst/cycle | 2.55 inst/cycle | +0.51 | 2.61 inst/cycle | +0.57 | 1.64 inst/cycle | -0.40 | 2.82 inst/cycle | +0.78 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | promote_l2_reuse_via_evict_last_on_scattered_pooling_loads | 1.120× | 1.033× | ✓ | **✓ best** |
| B | reduce_xblock_to_64_fix_grid_lambda_and_set_num_stages_1 | 1.103× | 1.099× | ✓ |  |
| C | reduce_xblock_to_64_with_correct_grid_lambda_to_cut_instructions | 1.119× | 1.075× | ✓ |  |
| D | remove_dual_path_bloat_and_use_small_xblock_single_path | 1.089× | 1.062× | ✓ |  |


### Metric Comparison

| Metric | Baseline | promote_l2_reuse_... | Δ | reduce_xblock_to_... | Δ | reduce_xblock_to_... | Δ | remove_dual_path_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4990 ns | 4830 ns | -160 ns (-3.2%) | 4540 ns | -450 ns (-9.0%) | 4640 ns | -350 ns (-7.0%) | 4700 ns | -290 ns (-5.8%) |
| SM Throughput (%) | 31.94 % | 21.16 % | -10.78 pp | 17.55 % | -14.39 pp | 20.71 % | -11.23 pp | 16.53 % | -15.41 pp |
| Memory Throughput (%) | 30.03 % | 15.94 % | -14.09 pp | 8.96 % | -21.07 pp | 16.58 % | -13.45 pp | 16.25 % | -13.78 pp |
| DRAM Throughput (%) | 26.70 % | 13.90 % | -12.80 pp | 7.35 % | -19.35 pp | 14.47 % | -12.23 pp | 14.21 % | -12.49 pp |
| Achieved Occupancy (%) | 70.87 % | 48.99 % | -21.88 pp | 47.04 % | -23.83 pp | 49.01 % | -21.86 pp | 49.12 % | -21.75 pp |
| Registers/Thread | 24.00 register/thread | 23.00 register/thread | -1 | 24.00 register/thread | +0 | 23.00 register/thread | -1 | 24.00 register/thread | +0 |
| Waves/SM | 1.51 | 0.75 | -0.76 | 0.75 | -0.76 | 0.75 | -0.76 | 0.75 | -0.76 |
| No Eligible (%) | 47.54 % | 46.94 % | -0.60 pp | 53.32 % | +5.78 pp | 52.55 % | +5.01 pp | 57.02 % | +9.48 pp |
| IPC Active | 2.04 inst/cycle | 2.01 inst/cycle | -0.03 | 1.92 inst/cycle | -0.12 | 2.02 inst/cycle | -0.02 | 1.62 inst/cycle | -0.42 |

---

## Iteration 5

**Best so far**: 1.0331× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_evict_last_to_all_nine_pooling_loads_to_reduce_l1tex_stall | 1.158× | 1.089× | ✓ |  |
| B | increase_num_warps_within_xblock64_to_hide_l1tex_latency | 1.161× | 0.951× | ✓ | **✓ best** |
| C | evict_last_on_repeated_loads_with_smaller_xblock_for_l1tex_stall_reduction | 1.137× | 1.089× | ✓ |  |
| D | add_xblock32_configs_fix_grid_lambda_and_increase_num_warps | 1.120× | 1.089× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_evict_last_to... | Δ | increase_num_warp... | Δ | evict_last_on_rep... | Δ | add_xblock32_conf... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4990 ns | 4580 ns | -410 ns (-8.2%) | 5250 ns | +260 ns (+5.2%) | 4580 ns | -410 ns (-8.2%) | 4580 ns | -410 ns (-8.2%) |
| SM Throughput (%) | 31.94 % | 21.74 % | -10.20 pp | 38.74 % | +6.80 pp | 21.80 % | -10.14 pp | 21.89 % | -10.05 pp |
| Memory Throughput (%) | 30.03 % | 8.91 % | -21.12 pp | 13.26 % | -16.77 pp | 8.90 % | -21.13 pp | 8.89 % | -21.14 pp |
| DRAM Throughput (%) | 26.70 % | 7.33 % | -19.37 pp | 6.39 % | -20.31 pp | 7.31 % | -19.39 pp | 7.32 % | -19.38 pp |
| Achieved Occupancy (%) | 70.87 % | 46.88 % | -23.99 pp | 71.25 % | +0.38 pp | 48.88 % | -21.99 pp | 48.04 % | -22.83 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 24.00 register/thread | +0 |
| Waves/SM | 1.51 | 0.75 | -0.76 | 1.51 | +0.00 | 0.75 | -0.76 | 0.75 | -0.76 |
| No Eligible (%) | 47.54 % | 43.98 % | -3.56 pp | 35.48 % | -12.06 pp | 48.96 % | +1.42 pp | 41.40 % | -6.14 pp |
| IPC Active | 2.04 inst/cycle | 2.09 inst/cycle | +0.05 | 2.61 inst/cycle | +0.57 | 2.22 inst/cycle | +0.18 | 2.18 inst/cycle | +0.14 |

---

## Cumulative Progress

```
Baseline:  4990 ns  (1.000×)
Iter 4     4830 ns  (1.033×)  — promote_l2_reuse_via_evict_last_on_scattered_pooling_loads

Final best: 5250 ns  (0.9505×)
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
+        triton.Config({'XBLOCK': 32},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_max_pool2d_with_indices_75(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 407680
@@ -27,57 +41,55 @@

     tmp8 = tmp6 < tmp3
     tmp9 = tmp7 & tmp8
     tmp10 = tmp5 & tmp9
-    tmp11 = tl.load(in_ptr0 + ((-12480) + x0 + 1664*x1 + 23296*x5), tmp10 & xmask, other=float("-inf"))
+    tmp11 = tl.load(in_ptr0 + ((-12480) + x0 + 1664*x1 + 23296*x5), mask=tmp10 & xmask, other=float("-inf"), eviction_policy='evict_last')
     tmp12 = 2*x1
     tmp13 = tmp12 >= tmp1
     tmp14 = tmp12 < tmp3
     tmp15 = tmp13 & tmp14
     tmp16 = tmp5 & tmp15
-    tmp17 = tl.load(in_ptr0 + ((-11648) + x0 + 1664*x1 + 23296*x5), tmp16 & xmask, other=float("-inf"))
+    tmp17 = tl.load(in_ptr0 + ((-11648) + x0 + 1664*x1 + 23296*x5), mask=tmp16 & xmask, other=float("-inf"), eviction_policy='evict_last')
     tmp18 = tl.maximum(tmp11, tmp17)
     tmp19 = 1 + 2*x1
     tmp20 = tmp19 >= tmp1
     tmp21 = tmp19 < tmp3
     tmp22 = tmp20 & tmp21
     tmp23 = tmp5 & tmp22
-    tmp24 = tl.load(in_ptr0 + ((-10816) + x0 + 1664*x1 + 23296*x5), tmp23 & xmask, other=float("-inf"))
+    tmp24 = tl.load(in_ptr0 + ((-10816) + x0 + 1664*x1 + 23296*x5), mask=tmp23 & xmask, other=float("-inf"), eviction_policy='evict_last')
     tmp25 = tl.maximum(tmp18, tmp24)
     tmp26 = 2*x2
     tmp27 = tmp26 >= tmp1
     tmp28 = tmp26 < tmp3
     tmp29 = tmp27 & tmp28
     tmp30 = tmp29 & tmp9
-    tmp31 = tl.load(in_ptr0 + ((-832) + x0 + 1664*x1 + 23296*x5), tmp30 & xmask, other=float("-inf"))
+    tmp31 = tl.load(in_ptr0 + ((-832) + x0 + 1664*x1 + 23296*x5), mask=tmp30 & xmask, other=float("-inf"), eviction_policy='evict_last')
     tmp32 = tl.maximum(tmp25, tmp31)
     tmp33 = tmp29 & tmp15
-    tmp34 = tl.load(in_ptr0 + (x0 + 1664*x1 + 23296*x5), tmp33 & xmask, other=float("-inf"))
+    tmp34 = tl.load(in_ptr0 + (x0 + 1664*x1 + 23296*x5), mask=tmp33 & xmask, other=float("-inf"), eviction_policy='evict_last')
     tmp35 = tl.maximum(tmp32, tmp34)
     tmp36 = tmp29 & tmp22
-    tmp37 = tl.load(in_ptr0 + (832 + x0 + 1664*x1 + 23296*x5), tmp36 & xmask, other=float("-inf"))
+    tmp37 = tl.load(in_ptr0 + (832 + x0 + 1664*x1 + 23296*x5), mask=tmp36 & xmask, other=float("-inf"), eviction_policy='evict_last')
     tmp38 = tl.maximum(tmp35, tmp37)
     tmp39 = 1 + 2*x2
     tmp40 = tmp39 >= tmp1
     tmp41 = tmp39 < tmp3
     tmp42 = tmp40 & tmp41
     tmp43 = tmp42 & tmp9
-    tmp44 = tl.load(in_ptr0 + (10816 + x0 + 1664*x1 + 23296*x5), tmp43 & xmask, other=float("-inf"))
+    tmp44 = tl.load(in_ptr0 + (10816 + x0 + 1664*x1 + 23296*x5), mask=tmp43 & xmask, other=float("-inf"), eviction_policy='evict_last')
     tmp45 = tl.maximum(tmp38, tmp44)
     tmp46 = tmp42 & tmp15
-    tmp47 = tl.load(in_ptr0 + (11648 + x0 + 1664*x1 + 23296*x5), tmp46 & xmask, other=float("-inf"))
+    tmp47 = tl.load(in_ptr0 + (11648 + x0 + 1664*x1 + 23296*x5), mask=tmp46 & xmask, other=float("-inf"), eviction_policy='evict_last')
     tmp48 = tl.maximum(tmp45, tmp47)
     tmp49 = tmp42 & tmp22
-    tmp50 = tl.load(in_ptr0 + (12480 + x0 + 1664*x1 + 23296*x5), tmp49 & xmask, other=float("-inf"))
+    tmp50 = tl.load(in_ptr0 + (12480 + x0 + 1664*x1 + 23296*x5), mask=tmp49 & xmask, other=float("-inf"), eviction_policy='evict_last')
     tmp51 = tl.maximum(tmp48, tmp50)
-    tl.store(out_ptr0 + (x6), tmp51, xmask)
-
-
+    tl.store(out_ptr0 + (x6), tmp51, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 407680
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_max_pool2d_with_indices_75[grid](arg_0, arg_1, 407680, XBLOCK=128)
+    triton_poi_fused_max_pool2d_with_indices_75[grid](arg_0, arg_1, 407680)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

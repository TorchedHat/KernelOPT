# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_clone_split_transpose_view_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 243900 ns — **Best latency**: 278690 ns (**0.8752×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_clone_split_transpose_view_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_043/kernel_3_triton_poi_fused_addmm_clone_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 243900 ns
- **SM Throughput**: 34.93 % | **Memory**: 32.04 % | **DRAM**: 32.04 %
- **Achieved Occupancy**: 43.70 %
- **IPC Active**: 1.41 inst/cycle | **No Eligible**: 64.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_num_warps_for_latency_hiding | 0.999× | 0.999× | ✓ |  |
| B | increase_xblock_via_autotune | 0.793× | 0.313× | ✓ |  |
| C | autotune_xblock_and_num_warps_for_pointwise_bias_add | 0.876× | 0.886× | ✓ |  |
| D | vectorized_loads_stores_via_larger_xblock | 0.877× | 0.885× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_n... | Δ | increase_xblock_v... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 243900 ns | 244060 ns | +160 ns (+0.1%) | 779300 ns | +535400 ns (+219.5%) | 275260 ns | +31360 ns (+12.9%) | 275620 ns | +31720 ns (+13.0%) |
| SM Throughput (%) | 34.93 % | 37.41 % | +2.48 pp | 74.42 % | +39.49 pp | 38.16 % | +3.23 pp | 37.90 % | +2.97 pp |
| Memory Throughput (%) | 32.04 % | 32.02 % | -0.02 pp | 27.50 % | -4.54 pp | 28.13 % | -3.91 pp | 28.08 % | -3.96 pp |
| DRAM Throughput (%) | 32.04 % | 32.02 % | -0.02 pp | 9.89 % | -22.15 pp | 28.13 % | -3.91 pp | 28.08 % | -3.96 pp |
| Achieved Occupancy (%) | 43.70 % | 44.32 % | +0.62 pp | 85.53 % | +41.83 pp | 35.59 % | -8.11 pp | 35.63 % | -8.07 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 32.00 register/thread | +16 | 18.00 register/thread | +2 | 18.00 register/thread | +2 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 372.36 | +186.18 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 64.65 % | 62.17 % | -2.48 pp | 25.51 % | -39.14 pp | 60.41 % | -4.24 pp | 60.36 % | -4.29 pp |
| IPC Active | 1.41 inst/cycle | 1.51 inst/cycle | +0.10 | 2.98 inst/cycle | +1.57 | 1.56 inst/cycle | +0.15 | 1.56 inst/cycle | +0.15 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.999× | 0.999× | ✓ |  |
| B | persistent_tile_scheduling | 0.757× | 0.762× | ✓ |  |
| C | persistent_tile_scheduling | 0.877× | 0.883× | ✓ |  |
| D | fix_grid_lambda_for_autotune_correctness_and_persistent_tile_scheduling | 1.005× | 0.885× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fix_grid_lambda_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 243900 ns | 244130 ns | +230 ns (+0.1%) | 320030 ns | +76130 ns (+31.2%) | 276160 ns | +32260 ns (+13.2%) | 275680 ns | +31780 ns (+13.0%) |
| SM Throughput (%) | 34.93 % | 46.29 % | +11.36 pp | 15.49 % | -19.44 pp | 14.00 % | -20.93 pp | 38.61 % | +3.68 pp |
| Memory Throughput (%) | 32.04 % | 32.01 % | -0.03 pp | 24.08 % | -7.96 pp | 28.02 % | -4.02 pp | 28.06 % | -3.98 pp |
| DRAM Throughput (%) | 32.04 % | 32.01 % | -0.03 pp | 24.08 % | -7.96 pp | 28.02 % | -4.02 pp | 28.06 % | -3.98 pp |
| Achieved Occupancy (%) | 43.70 % | 48.03 % | +4.33 pp | 34.81 % | -8.89 pp | 38.33 % | -5.37 pp | 35.40 % | -8.30 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 32.00 register/thread | +16 | 22.00 register/thread | +6 | 18.00 register/thread | +2 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 64.65 % | 53.21 % | -11.44 pp | 84.59 % | +19.94 pp | 80.69 % | +16.04 pp | 59.44 % | -5.21 pp |
| IPC Active | 1.41 inst/cycle | 1.87 inst/cycle | +0.46 | 0.55 inst/cycle | -0.86 | 0.68 inst/cycle | -0.73 | 1.58 inst/cycle | +0.17 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cache_eviction_policy_streaming_load_evict_first | 1.000× | 0.999× | ✓ |  |
| B | streaming_eviction_policy_for_large_matrix_load | 1.005× | 0.875× | ✓ |  |
| C | fix_grid_lambda_and_simplify_to_single_tile_pattern | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cache_eviction_po... | Δ | streaming_evictio... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 243900 ns | 244130 ns | +230 ns (+0.1%) | 278750 ns | +34850 ns (+14.3%) | 244030 ns | +130 ns (+0.1%) |
| SM Throughput (%) | 34.93 % | 51.21 % | +16.28 pp | 41.97 % | +7.04 pp | 37.43 % | +2.50 pp |
| Memory Throughput (%) | 32.04 % | 31.43 % | -0.61 pp | 27.55 % | -4.49 pp | 32.02 % | -0.02 pp |
| DRAM Throughput (%) | 32.04 % | 31.43 % | -0.61 pp | 27.55 % | -4.49 pp | 32.02 % | -0.02 pp |
| Achieved Occupancy (%) | 43.70 % | 47.73 % | +4.03 pp | 35.96 % | -7.74 pp | 44.35 % | +0.65 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 64.65 % | 48.15 % | -16.50 pp | 57.27 % | -7.38 pp | 62.21 % | -2.44 pp |
| IPC Active | 1.41 inst/cycle | 2.07 inst/cycle | +0.66 | 1.67 inst/cycle | +0.26 | 1.51 inst/cycle | +0.10 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_autotune_num_warps_for_latency_hiding | 1.000× | 0.999× | ✓ |  |
| B | shared_memory_preload_broadcast_bias_to_eliminate_l1tex_stalls | 0.999× | 0.999× | ✓ |  |
| C | add_num_stages_to_autotune_configs_for_l1tex_latency_hiding | 1.005× | 0.875× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | shared_memory_pre... | Δ | add_num_stages_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 243900 ns | 244060 ns | +160 ns (+0.1%) | 244190 ns | +290 ns (+0.1%) | 278690 ns | +34790 ns (+14.3%) |
| SM Throughput (%) | 34.93 % | 37.42 % | +2.49 pp | 51.23 % | +16.30 pp | 41.04 % | +6.11 pp |
| Memory Throughput (%) | 32.04 % | 32.04 % | +0.00 pp | 31.45 % | -0.59 pp | 27.58 % | -4.46 pp |
| DRAM Throughput (%) | 32.04 % | 32.04 % | +0.00 pp | 31.45 % | -0.59 pp | 27.58 % | -4.46 pp |
| Achieved Occupancy (%) | 43.70 % | 44.38 % | +0.68 pp | 47.65 % | +3.95 pp | 35.96 % | -7.74 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 18.00 register/thread | +2 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 64.65 % | 62.21 % | -2.44 pp | 48.13 % | -16.52 pp | 57.45 % | -7.20 pp |
| IPC Active | 1.41 inst/cycle | 1.51 inst/cycle | +0.10 | 2.07 inst/cycle | +0.66 | 1.68 inst/cycle | +0.27 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_to_autotune_for_software_pipelining | 1.000× | 0.999× | ✓ |  |
| B | increase_num_warps_for_l1tex_latency_hiding | 1.000× | 0.999× | ✓ |  |
| C | fix_grid_lambda_and_enable_fma_via_multiply_add_rewrite | 1.005× | 0.875× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_to... | Δ | increase_num_warp... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 243900 ns | 244060 ns | +160 ns (+0.1%) | 244160 ns | +260 ns (+0.1%) | 278780 ns | +34880 ns (+14.3%) |
| SM Throughput (%) | 34.93 % | 37.43 % | +2.50 pp | 51.25 % | +16.32 pp | 40.98 % | +6.05 pp |
| Memory Throughput (%) | 32.04 % | 32.03 % | -0.01 pp | 31.44 % | -0.60 pp | 27.57 % | -4.47 pp |
| DRAM Throughput (%) | 32.04 % | 32.03 % | -0.01 pp | 31.44 % | -0.60 pp | 27.57 % | -4.47 pp |
| Achieved Occupancy (%) | 43.70 % | 44.30 % | +0.60 pp | 47.75 % | +4.05 pp | 35.71 % | -7.99 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 18.00 register/thread | +2 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 64.65 % | 62.21 % | -2.44 pp | 48.25 % | -16.40 pp | 56.96 % | -7.69 pp |
| IPC Active | 1.41 inst/cycle | 1.51 inst/cycle | +0.10 | 2.07 inst/cycle | +0.66 | 1.70 inst/cycle | +0.29 |

---

## Cumulative Progress

```
Baseline:  243900 ns  (1.000×)

Final best: 278690 ns  (0.8752×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,23 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 512}, num_warps=4),
+        triton.Config({'XBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 512}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_addmm_clone_split_transpose_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 50331648
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 96)
     x2 = ((xindex // 49152) % 8)
     x1 = ((xindex // 96) % 512)
     x3 = xindex // 393216
-    x4 = xindex
-    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 96*x2), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (1536 + x0 + 96*x2 + 2304*x1 + 1179648*x3), None)
+    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 96*x2), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (1536 + x0 + 96*x2 + 2304*x1 + 1179648*x3), xmask)
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x4), tmp2, None)
-
-
+    tl.store(out_ptr0 + xindex, tmp2, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -29,7 +38,7 @@

     arg_2 = inputs[2]
     xnumel = 50331648
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_clone_split_transpose_view_3[grid](arg_0, arg_1, arg_2, 50331648, XBLOCK=128)
+    triton_poi_fused_addmm_clone_split_transpose_view_3[grid](arg_0, arg_1, arg_2, 50331648)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

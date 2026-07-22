# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 33500 ns — **Best latency**: 33280 ns (**1.0066×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`remove_low_warp_configs_and_fix_autotune_grid_to_restore_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (50176,1,1), Waves/SM: 23.76
- **Registers/thread**: 16
- **Duration**: 33500 ns
- **SM Throughput**: 36.05 % | **Memory**: 74.37 % | **DRAM**: 74.37 %
- **Achieved Occupancy**: 63.75 %
- **IPC Active**: 1.51 inst/cycle | **No Eligible**: 62.30 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads_and_larger_xblock | 0.709× | 0.672× | ✓ |  |
| B | vectorized_loads_with_larger_xblock | 0.701× | 0.662× | ✓ |  |
| C | increase_xblock_for_l2_reuse_and_coalescing | 0.998× | 0.312× | ✓ |  |
| D | increase_xblock_tile_size_for_memory_latency_hiding | 0.993× | 0.492× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | increase_xblock_f... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33500 ns | 49820 ns | +16320 ns (+48.7%) | 50590 ns | +17090 ns (+51.0%) | 107390 ns | +73890 ns (+220.6%) | 68100 ns | +34600 ns (+103.3%) |
| SM Throughput (%) | 36.05 % | 37.07 % | +1.02 pp | 41.18 % | +5.13 pp | 71.05 % | +35.00 pp | 61.44 % | +25.39 pp |
| Memory Throughput (%) | 74.37 % | 46.49 % | -27.88 pp | 47.97 % | -26.40 pp | 23.13 % | -51.24 pp | 35.67 % | -38.70 pp |
| DRAM Throughput (%) | 74.37 % | 46.49 % | -27.88 pp | 47.97 % | -26.40 pp | 22.31 % | -52.06 pp | 35.67 % | -38.70 pp |
| Achieved Occupancy (%) | 63.75 % | 54.46 % | -9.29 pp | 54.87 % | -8.88 pp | 88.72 % | +24.97 pp | 82.15 % | +18.40 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 32.00 register/thread | +16 |
| Waves/SM | 23.76 | 23.76 | +0.00 | 23.76 | +0.00 | 47.52 | +23.76 | 47.52 | +23.76 |
| No Eligible (%) | 62.30 % | 61.31 % | -0.99 pp | 64.99 % | +2.69 pp | 44.34 % | -17.96 pp | 49.62 % | -12.68 pp |
| IPC Active | 1.51 inst/cycle | 1.57 inst/cycle | +0.06 | 1.40 inst/cycle | -0.11 | 2.23 inst/cycle | +0.72 | 2.01 inst/cycle | +0.50 |


### Key Learnings

#### ❌ Plan C: `increase_xblock_for_l2_reuse_and_coalescing` (0.312×)

Adding autotune with larger XBLOCK configs caused the compiler to select a 256-thread block that doubled registers per thread from 16 to 32, which halved the number of blocks per SM and forced 47.5 waves instead of 23.8. Critically, executed instructions exploded from 9.4M to 46.2M (nearly 5x), indicating the larger tile introduced significant extra computation or predicated work overhead that dwarfed any L2 reuse benefit. The L2 hit rate was essentially unchanged (20.84% → 20.99%), confirming the spatial reuse hypothesis was wrong for this kernel's access pattern, and the net effect was a 3.2x slowdown. Do not apply autotune with larger block sizes when the kernel's memory access pattern does not exhibit spatial locality that would benefit from larger tiles, or when increasing block size causes register count to double (indicating the compiler is spilling or unrolling more aggressively), as the instruction count explosion will dominate any latency-hiding benefit.

> **Applicable when**: Never applicable as described — this strategy caused severe regression. The pattern of adding autotune with larger XBLOCK to improve L2 reuse is only valid when: (1) the access pattern genuinely reuses data across threads in the same block, (2) register count does not increase with block size, and (3) executed instruction count does not grow superlinearly with block size.

> **Avoid when**: Kernel has low register count (≤16 regs/thread) at baseline and the autotune search space includes configs that cause register doubling. Also avoid when L2 hit rate is low due to non-spatial access patterns (e.g., strided channel-last layouts where neighboring threads access non-contiguous memory), as larger blocks will not improve L2 reuse. Avoid when xnumel is already evenly divisible by the baseline XBLOCK (making the xmask boundary check unnecessary overhead). Avoid when the grid is already large enough (>20 waves/SM) to hide memory latency.

#### ❌ Plan D: `increase_xblock_tile_size_for_memory_latency_hiding` (0.492×)

Adding @triton.autotune with larger XBLOCK configs caused the autotuner to select XBLOCK=256 with 32 registers/thread (doubled from 16), which doubled total executed instructions (9.4M → 26.2M) and elapsed cycles (50K → 102K), resulting in a 2x slowdown despite DRAM throughput dropping from 74% to 36%. The L2 hit rate did not improve meaningfully (20.84% → 21.14%), indicating the larger tile provided no additional reuse for this strided access pattern. When a pointwise kernel has very low register pressure at the baseline block size, adding autotune with larger block configs can cause the compiler to spill or allocate more registers per thread, dramatically increasing instruction count and negating any latency-hiding benefit.

> **Applicable when**: Kernel has very low register pressure at baseline (≤16 regs/thread), pointwise access pattern with no spatial reuse opportunity, and L2 hit rate is already low due to strided/non-local access pattern that larger tiles cannot improve.

> **Avoid when**: Kernel has low register pressure (≤16 regs/thread) and strided memory access patterns where larger tiles do not improve L2 hit rate — adding autotune with larger block configs risks doubling register allocation and instruction count. Also avoid when the baseline already uses no mask (xnumel exactly divisible by XBLOCK), as adding xmask introduces unnecessary predication overhead.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_autotune_and_mask_overhead_to_restore_baseline_register_pressure | 0.995× | 0.968× | ✓ |  |
| B | remove_register_inflating_hints_and_unnecessary_mask | 0.994× | 0.997× | ✓ |  |
| C | restore_minimal_xblock_eliminate_mask_overhead | 0.995× | 1.000× | ✓ |  |
| D | restore_small_xblock_low_register_pressure_config | 1.057× | 1.003× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | remove_autotune_a... | Δ | remove_register_i... | Δ | restore_minimal_x... | Δ | restore_small_xbl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33500 ns | 34620 ns | +1120 ns (+3.3%) | 33600 ns | +100 ns (+0.3%) | 33500 ns | +0 ns (+0.0%) | 33410 ns | -90 ns (-0.3%) |
| SM Throughput (%) | 36.05 % | 35.69 % | -0.36 pp | 35.56 % | -0.49 pp | 35.80 % | -0.25 pp | 23.39 % | -12.66 pp |
| Memory Throughput (%) | 74.37 % | 72.13 % | -2.24 pp | 74.29 % | -0.08 pp | 74.43 % | +0.06 pp | 74.56 % | +0.19 pp |
| DRAM Throughput (%) | 74.37 % | 72.13 % | -2.24 pp | 74.29 % | -0.08 pp | 74.43 % | +0.06 pp | 74.56 % | +0.19 pp |
| Achieved Occupancy (%) | 63.75 % | 64.06 % | +0.31 pp | 67.01 % | +3.26 pp | 66.03 % | +2.28 pp | 44.11 % | -19.64 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 22.00 register/thread | +6 |
| Waves/SM | 23.76 | 23.76 | +0.00 | 23.76 | +0.00 | 23.76 | +0.00 | 11.88 | -11.88 |
| No Eligible (%) | 62.30 % | 62.13 % | -0.17 pp | 61.87 % | -0.43 pp | 62.12 % | -0.18 pp | 75.87 % | +13.57 pp |
| IPC Active | 1.51 inst/cycle | 1.53 inst/cycle | +0.02 | 1.50 inst/cycle | -0.01 | 1.53 inst/cycle | +0.02 | 0.96 inst/cycle | -0.55 |

---

## Iteration 3

**Best so far**: 1.0027× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads_evict_first | 1.102× | 1.003× | ✓ | **✓ best** |
| B | eviction_policies_streaming_loads_evict_first | 1.056× | 1.003× | ✓ |  |
| C | eviction_policies_streaming_loads_evict_first | 1.057× | 0.984× | ✓ |  |
| D | evict_first_cache_modifier_on_streaming_loads_and_store | 1.090× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | eviction_policies... | Δ | eviction_policies... | Δ | evict_first_cache... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33500 ns | 33410 ns | -90 ns (-0.3%) | 33410 ns | -90 ns (-0.3%) | 34050 ns | +550 ns (+1.6%) | 34240 ns | +740 ns (+2.2%) |
| SM Throughput (%) | 36.05 % | 27.72 % | -8.33 pp | 40.11 % | +4.06 pp | 40.19 % | +4.14 pp | 39.08 % | +3.03 pp |
| Memory Throughput (%) | 74.37 % | 64.98 % | -9.39 pp | 65.28 % | -9.09 pp | 64.05 % | -10.32 pp | 68.87 % | -5.50 pp |
| DRAM Throughput (%) | 74.37 % | 64.98 % | -9.39 pp | 65.28 % | -9.09 pp | 64.05 % | -10.32 pp | 68.87 % | -5.50 pp |
| Achieved Occupancy (%) | 63.75 % | 39.76 % | -23.99 pp | 53.39 % | -10.36 pp | 52.64 % | -11.11 pp | 64.75 % | +1.00 pp |
| Registers/Thread | 16.00 register/thread | 22.00 register/thread | +6 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 23.76 | 11.88 | -11.88 | 23.76 | +0.00 | 23.76 | +0.00 | 23.76 | +0.00 |
| No Eligible (%) | 62.30 % | 69.05 % | +6.75 pp | 57.40 % | -4.90 pp | 56.16 % | -6.14 pp | 58.03 % | -4.27 pp |
| IPC Active | 1.51 inst/cycle | 1.25 inst/cycle | -0.26 | 1.75 inst/cycle | +0.24 | 1.75 inst/cycle | +0.24 | 1.70 inst/cycle | +0.19 |

---

## Iteration 4

**Best so far**: 1.0027× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_tune_num_warps_for_latency_hiding | 1.121× | 1.008× | ✓ | **✓ best** |
| B | increase_num_warps_within_fixed_xblock_for_latency_hiding | 1.093× | 1.002× | ✓ |  |
| C | increase_num_warps_for_l1tex_latency_hiding | 1.099× | 1.002× | ✓ |  |
| D | increase_num_warps_to_hide_l1tex_latency_without_xblock_change | 1.096× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33500 ns | 33250 ns | -250 ns (-0.7%) | 33440 ns | -60 ns (-0.2%) | 33440 ns | -60 ns (-0.2%) | 33890 ns | +390 ns (+1.2%) |
| SM Throughput (%) | 36.05 % | 25.92 % | -10.13 pp | 40.83 % | +4.78 pp | 40.41 % | +4.36 pp | 39.60 % | +3.55 pp |
| Memory Throughput (%) | 74.37 % | 32.95 % | -41.42 pp | 65.28 % | -9.09 pp | 65.30 % | -9.07 pp | 69.74 % | -4.63 pp |
| DRAM Throughput (%) | 74.37 % | 32.95 % | -41.42 pp | 65.28 % | -9.09 pp | 65.30 % | -9.07 pp | 69.74 % | -4.63 pp |
| Achieved Occupancy (%) | 63.75 % | 28.96 % | -34.79 pp | 52.43 % | -11.32 pp | 52.08 % | -11.67 pp | 65.56 % | +1.81 pp |
| Registers/Thread | 16.00 register/thread | 17.00 register/thread | +1 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 23.76 | 11.88 | -11.88 | 23.76 | +0.00 | 23.76 | +0.00 | 23.76 | +0.00 |
| No Eligible (%) | 62.30 % | 70.79 % | +8.49 pp | 56.83 % | -5.47 pp | 56.75 % | -5.55 pp | 58.12 % | -4.18 pp |
| IPC Active | 1.51 inst/cycle | 1.16 inst/cycle | -0.35 | 1.73 inst/cycle | +0.22 | 1.73 inst/cycle | +0.22 | 1.63 inst/cycle | +0.12 |

---

## Iteration 5

**Best so far**: 1.0075× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_low_warp_configs_and_fix_autotune_grid_to_restore_occupancy | 1.122× | 1.007× | ✓ | **✓ best** |
| B | evict_last_on_store_for_downstream_l2_residency | 1.078× | 1.000× | ✓ |  |
| C | evict_last_on_output_store_for_l2_residency | 1.074× | 1.002× | ✓ |  |
| D | restructure_index_decomposition_for_l2_compression_and_spatial_locality | 0.958× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_low_warp_c... | Δ | evict_last_on_sto... | Δ | evict_last_on_out... | Δ | restructure_index... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33500 ns | 33280 ns | -220 ns (-0.7%) | 33500 ns | +0 ns (+0.0%) | 33440 ns | -60 ns (-0.2%) | 33410 ns | -90 ns (-0.3%) |
| SM Throughput (%) | 36.05 % | 49.65 % | +13.60 pp | 43.88 % | +7.83 pp | 43.23 % | +7.18 pp | 40.72 % | +4.67 pp |
| Memory Throughput (%) | 74.37 % | 32.88 % | -41.49 pp | 64.77 % | -9.60 pp | 64.98 % | -9.39 pp | 72.27 % | -2.10 pp |
| DRAM Throughput (%) | 74.37 % | 32.88 % | -41.49 pp | 64.77 % | -9.60 pp | 64.98 % | -9.39 pp | 72.27 % | -2.10 pp |
| Achieved Occupancy (%) | 63.75 % | 44.53 % | -19.22 pp | 54.61 % | -9.14 pp | 52.65 % | -11.10 pp | 61.15 % | -2.60 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 23.76 | 23.76 | +0.00 | 23.76 | +0.00 | 23.76 | +0.00 | 23.76 | +0.00 |
| No Eligible (%) | 62.30 % | 46.48 % | -15.82 pp | 54.03 % | -8.27 pp | 54.03 % | -8.27 pp | 56.84 % | -5.46 pp |
| IPC Active | 1.51 inst/cycle | 2.14 inst/cycle | +0.63 | 1.87 inst/cycle | +0.36 | 1.87 inst/cycle | +0.36 | 1.72 inst/cycle | +0.21 |

---

## Cumulative Progress

```
Baseline:  33500 ns  (1.000×)
Iter 2     33410 ns  (1.003×)  — restore_small_xblock_low_register_pressure_config
Iter 4     33250 ns  (1.008×)  — fix_autotune_grid_lambda_and_tune_num_warps_for_latency_hiding

Final best: 33280 ns  (1.0066×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,33 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 6422528
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 64)
     x1 = ((xindex // 64) % 112)
     x2 = xindex // 7168
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 28672*x2), None)
-    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 28672*x2), None)
-    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 128*x1 + 28672*x2), None)
-    tmp5 = tl.load(in_ptr0 + (14400 + x0 + 128*x1 + 28672*x2), None)
+    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 28672*x2), xmask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 28672*x2), xmask, eviction_policy='evict_first')
+    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 128*x1 + 28672*x2), xmask, eviction_policy='evict_first')
+    tmp5 = tl.load(in_ptr0 + (14400 + x0 + 128*x1 + 28672*x2), xmask, eviction_policy='evict_first')
     tmp2 = tl.maximum(tmp0, tmp1)
     tmp4 = tl.maximum(tmp2, tmp3)
     tmp6 = tl.maximum(tmp4, tmp5)
-    tl.store(out_ptr0 + (x3), tmp6, None)
-
-
+    tl.store(out_ptr0 + (x3), tmp6, xmask, eviction_policy='evict_last')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 6422528
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4[grid](arg_0, arg_1, 6422528, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4[grid](arg_0, arg_1, 6422528)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

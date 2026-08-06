# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 41060 ns — **Best latency**: 41090 ns (**0.9993×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62720,1,1), Waves/SM: 29.70
- **Registers/thread**: 17
- **Duration**: 41060 ns
- **SM Throughput**: 56.54 % | **Memory**: 24.52 % | **DRAM**: 18.60 %
- **Achieved Occupancy**: 54.40 %
- **IPC Active**: 2.39 inst/cycle | **No Eligible**: 40.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_pointwise_kernel | 0.994× | 0.213× | ✓ |  |
| B | autotune_xblock_num_warps_num_stages | 1.000× | 0.991× | ✓ |  |
| C | vectorized_loads_stores_via_larger_xblock_autotune | 0.997× | 0.213× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_latency_hiding | 0.996× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | vectorized_loads_... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41060 ns | 192960 ns | +151900 ns (+369.9%) | 41440 ns | +380 ns (+0.9%) | 192930 ns | +151870 ns (+369.9%) | 41120 ns | +60 ns (+0.1%) |
| SM Throughput (%) | 56.54 % | 89.70 % | +33.16 pp | 66.43 % | +9.89 pp | 89.69 % | +33.15 pp | 67.20 % | +10.66 pp |
| Memory Throughput (%) | 24.52 % | 19.27 % | -5.25 pp | 24.60 % | +0.08 pp | 19.26 % | -5.26 pp | 24.57 % | +0.05 pp |
| DRAM Throughput (%) | 18.60 % | 3.93 % | -14.67 pp | 18.34 % | -0.26 pp | 3.94 % | -14.66 pp | 18.52 % | -0.08 pp |
| Achieved Occupancy (%) | 54.40 % | 55.31 % | +0.91 pp | 58.45 % | +4.05 pp | 55.36 % | +0.96 pp | 58.38 % | +3.98 pp |
| Registers/Thread | 17.00 register/thread | 48.00 register/thread | +31 | 18.00 register/thread | +1 | 48.00 register/thread | +31 | 18.00 register/thread | +1 |
| Waves/SM | 29.70 | 95.03 | +65.33 | 29.70 | +0.00 | 95.03 | +65.33 | 29.70 | +0.00 |
| No Eligible (%) | 40.24 % | 9.23 % | -31.01 pp | 29.55 % | -10.69 pp | 9.54 % | -30.70 pp | 28.87 % | -11.37 pp |
| IPC Active | 2.39 inst/cycle | 3.63 inst/cycle | +1.24 | 2.80 inst/cycle | +0.41 | 3.62 inst/cycle | +1.23 | 2.83 inst/cycle | +0.44 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_pointwise_kernel` (0.213×)

Increasing XBLOCK from 128 to 2048 via autotune caused registers_per_thread to jump from 17 to 48, which reduced theoretical occupancy from 100% to 62.5% (Block_Limit_Registers dropped to 5 blocks/SM) and caused executed instructions to balloon from 18M to 137M — a 7.5x increase. Although scheduler efficiency improved (No_Eligible dropped from 40% to 9%), the massive instruction count increase and register pressure dominated, causing duration to increase from 41 µs to 193 µs (4.7x slower). The autotune selected a large tile (XBLOCK=2048, num_warps=8) that the compiler could not handle without significant register spilling and instruction expansion, negating any ILP benefit.

> **Applicable when**: Kernel has very low registers/thread (<20), zero shared memory, pure pointwise with no arithmetic complexity, and is stalled on L1TEX latency with low ILP.

> **Avoid when**: Kernel body contains multi-step arithmetic (e.g., batch_norm: sqrt, division, multiply, add) that causes the compiler to allocate many more registers at larger tile sizes. When registers_per_thread is already near 20+, increasing XBLOCK will cause register spilling, reduce theoretical occupancy via Block_Limit_Registers, and massively inflate instruction count. Also avoid when the autotune grid is fixed (not lambda) — the grid was left as cdiv(N, 128) while autotune selected larger XBLOCK, causing incorrect launch configuration.

#### ❌ Plan C: `vectorized_loads_stores_via_larger_xblock_autotune` (0.213×)

Adding a @triton.autotune decorator with large XBLOCK configs (512–2048) caused the compiler to select a configuration that tripled registers-per-thread (17→48), reducing Block Limit Registers from 21 to 5 and capping theoretical occupancy at 62.5%. Although compute SM throughput improved (57%→90%) and eligible warps per scheduler increased (1.11→5.25), the total executed instruction count exploded 7.5x (18M→137M) and kernel duration increased 4.7x (41→193 µs). The autotune also introduced a broken grid launch (grid was hardcoded to cdiv(xnumel,128) instead of using meta['XBLOCK']), meaning the autotuned XBLOCK was never reflected in the grid, causing massive redundant work. The net effect was a 4.7x regression despite better scheduler utilization.

> **Applicable when**: Never apply this exact pattern. The grid must be updated to use meta['XBLOCK'] when using autotune.

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK if the launch grid is hardcoded to a fixed XBLOCK value. This causes the grid to be computed with the wrong tile size, leading to massive over-computation and register pressure from larger tile sizes. Also avoid large XBLOCK configs when registers-per-thread is already low (≤20) and theoretical occupancy is near 100%—larger tiles may cause register spill that caps occupancy and negates any vectorization benefit.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_and_add_streaming_eviction_policy | 0.994× | 0.998× | ✓ |  |
| B | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 0.997× | 0.997× | ✓ |  |
| C | fix_hardcoded_grid_to_lambda_meta_xblock | 1.005× | 0.902× | ✓ | **✓ best** |
| D | fix_broken_autotune_grid_restore_small_xblock | 0.999× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | increase_num_warp... | Δ | fix_hardcoded_gri... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41060 ns | 41150 ns | +90 ns (+0.2%) | 41180 ns | +120 ns (+0.3%) | 45540 ns | +4480 ns (+10.9%) | 41120 ns | +60 ns (+0.1%) |
| SM Throughput (%) | 56.54 % | 70.73 % | +14.19 pp | 65.87 % | +9.33 pp | 77.72 % | +21.18 pp | 66.56 % | +10.02 pp |
| Memory Throughput (%) | 24.52 % | 24.54 % | +0.02 pp | 24.43 % | -0.09 pp | 29.54 % | +5.02 pp | 24.63 % | +0.11 pp |
| DRAM Throughput (%) | 18.60 % | 18.50 % | -0.10 pp | 18.49 % | -0.11 pp | 16.80 % | -1.80 pp | 18.55 % | -0.05 pp |
| Achieved Occupancy (%) | 54.40 % | 60.04 % | +5.64 pp | 57.90 % | +3.50 pp | 60.58 % | +6.18 pp | 58.02 % | +3.62 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 24.00 register/thread | +7 | 18.00 register/thread | +1 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 |
| No Eligible (%) | 40.24 % | 24.74 % | -15.50 pp | 29.46 % | -10.78 pp | 16.65 % | -23.59 pp | 29.53 % | -10.71 pp |
| IPC Active | 2.39 inst/cycle | 2.98 inst/cycle | +0.59 | 2.81 inst/cycle | +0.42 | 3.34 inst/cycle | +0.95 | 2.82 inst/cycle | +0.43 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_redundant_arithmetic_to_reduce_register_pressure_and_enable_fma | 1.006× | 1.000× | ✓ |  |
| B | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 0.994× | 0.998× | ✓ |  |
| C | fix_hardcoded_grid_lambda_and_add_evict_first_on_streaming_load | 0.994× | 0.999× | ✓ |  |
| D | replace_int_div_sqrt_chain_with_rsqrt_for_fma_fusion | 1.007× | 0.999× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_redunda... | Δ | increase_num_warp... | Δ | fix_hardcoded_gri... | Δ | replace_int_div_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41060 ns | 41060 ns | +0 ns (+0.0%) | 41150 ns | +90 ns (+0.2%) | 41090 ns | +30 ns (+0.1%) | 41090 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 56.54 % | 40.38 % | -16.16 pp | 70.92 % | +14.38 pp | 70.46 % | +13.92 pp | 45.63 % | -10.91 pp |
| Memory Throughput (%) | 24.52 % | 24.41 % | -0.11 pp | 24.60 % | +0.08 pp | 24.70 % | +0.18 pp | 32.83 % | +8.31 pp |
| DRAM Throughput (%) | 18.60 % | 18.61 % | +0.01 pp | 18.52 % | -0.08 pp | 18.55 % | -0.05 pp | 18.52 % | -0.08 pp |
| Achieved Occupancy (%) | 54.40 % | 40.38 % | -14.02 pp | 60.30 % | +5.90 pp | 59.93 % | +5.53 pp | 29.56 % | -24.84 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 |
| No Eligible (%) | 40.24 % | 57.26 % | +17.02 pp | 24.66 % | -15.58 pp | 25.30 % | -14.94 pp | 50.37 % | +10.13 pp |
| IPC Active | 2.39 inst/cycle | 1.73 inst/cycle | -0.66 | 2.99 inst/cycle | +0.60 | 2.98 inst/cycle | +0.59 | 1.99 inst/cycle | -0.40 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_size_to_256_for_l1tex_latency_hiding | 1.003× | 1.004× | ✓ |  |
| B | fix_grid_lambda_and_replace_int_div_with_rsqrt_for_fma_fusion | 1.000× | 0.999× | ✓ |  |
| C | fix_grid_lambda_and_add_num_stages_pipelining_to_autotune | 1.010× | 0.999× | ✓ | **✓ best** |
| D | rewrite_arithmetic_chain_for_fma_fusion_via_rsqrt | 1.001× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_si... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | rewrite_arithmeti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41060 ns | 40900 ns | -160 ns (-0.4%) | 41090 ns | +30 ns (+0.1%) | 41090 ns | +30 ns (+0.1%) | 40900 ns | -160 ns (-0.4%) |
| SM Throughput (%) | 56.54 % | 40.16 % | -16.38 pp | 44.00 % | -12.54 pp | 45.64 % | -10.90 pp | 45.31 % | -11.23 pp |
| Memory Throughput (%) | 24.52 % | 24.27 % | -0.25 pp | 24.28 % | -0.24 pp | 32.86 % | +8.34 pp | 24.52 % | +0.00 pp |
| DRAM Throughput (%) | 18.60 % | 18.67 % | +0.07 pp | 18.52 % | -0.08 pp | 18.56 % | -0.04 pp | 18.59 % | -0.01 pp |
| Achieved Occupancy (%) | 54.40 % | 40.78 % | -13.62 pp | 41.66 % | -12.74 pp | 30.31 % | -24.09 pp | 41.97 % | -12.43 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 18.00 register/thread | +1 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 |
| No Eligible (%) | 40.24 % | 57.18 % | +16.94 pp | 53.02 % | +12.78 pp | 50.06 % | +9.82 pp | 52.21 % | +11.97 pp |
| IPC Active | 2.39 inst/cycle | 1.70 inst/cycle | -0.69 | 1.88 inst/cycle | -0.51 | 1.97 inst/cycle | -0.42 | 1.91 inst/cycle | -0.48 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_size_for_warp_latency_hiding_at_fixed_xblock | 1.001× | 1.004× | ✓ |  |
| B | reorder_program_ids_for_l2_broadcast_locality | 1.002× | 1.002× | ✓ |  |
| C | fix_grid_lambda_and_add_multiple_of_hints_for_vectorized_access | 1.006× | 1.005× | ✓ |  |
| D | fix_grid_lambda_and_add_evict_first_on_large_streaming_load | 1.010× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_si... | Δ | reorder_program_i... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41060 ns | 40900 ns | -160 ns (-0.4%) | 40990 ns | -70 ns (-0.2%) | 40860 ns | -200 ns (-0.5%) | 41060 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 56.54 % | 40.29 % | -16.25 pp | 47.27 % | -9.27 pp | 44.14 % | -12.40 pp | 48.99 % | -7.55 pp |
| Memory Throughput (%) | 24.52 % | 24.38 % | -0.14 pp | 24.36 % | -0.16 pp | 24.32 % | -0.20 pp | 32.67 % | +8.15 pp |
| DRAM Throughput (%) | 18.60 % | 18.68 % | +0.08 pp | 18.62 % | +0.02 pp | 18.61 % | +0.01 pp | 18.50 % | -0.10 pp |
| Achieved Occupancy (%) | 54.40 % | 40.50 % | -13.90 pp | 42.72 % | -11.68 pp | 41.63 % | -12.77 pp | 30.78 % | -23.62 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 |
| No Eligible (%) | 40.24 % | 57.09 % | +16.85 pp | 49.84 % | +9.60 pp | 52.60 % | +12.36 pp | 46.02 % | +5.78 pp |
| IPC Active | 2.39 inst/cycle | 1.72 inst/cycle | -0.67 | 2.01 inst/cycle | -0.38 | 1.88 inst/cycle | -0.51 | 2.14 inst/cycle | -0.25 |

---

## Cumulative Progress

```
Baseline:  41060 ns  (1.000×)

Final best: 41090 ns  (0.9993×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 8028160
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 64)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    tmp5 = tmp3 + 1e-05
+    tmp6 = tl.math.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tmp17 = tl.maximum(0.0, tmp15)
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

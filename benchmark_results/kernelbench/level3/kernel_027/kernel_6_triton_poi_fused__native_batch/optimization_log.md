# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 64100 ns — **Best latency**: 63940 ns (**1.0025×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_and_add_xblock64_for_more_concurrent_ctas`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_6_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (100352,1,1), Waves/SM: 47.52
- **Registers/thread**: 19
- **Duration**: 64100 ns
- **SM Throughput**: 61.37 % | **Memory**: 28.39 % | **DRAM**: 24.13 %
- **Achieved Occupancy**: 58.36 %
- **IPC Active**: 2.56 inst/cycle | **No Eligible**: 36.21 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.640× | 0.623× | ✓ |  |
| B | increase_xblock_with_autotune_for_better_occupancy_and_ilp | 1.001× | 0.999× | ✓ | **✓ best** |
| C | increase_xblock_for_vectorized_loads_and_stores | 0.881× | 0.197× | ✓ |  |
| D | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.878× | 0.871× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_w... | Δ | increase_xblock_f... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 64100 ns | 102850 ns | +38750 ns (+60.5%) | 64160 ns | +60 ns (+0.1%) | 324640 ns | +260540 ns (+406.5%) | 73630 ns | +9530 ns (+14.9%) |
| SM Throughput (%) | 61.37 % | 86.09 % | +24.72 pp | 74.75 % | +13.38 pp | 91.02 % | +29.65 pp | 84.59 % | +23.22 pp |
| Memory Throughput (%) | 28.39 % | 27.07 % | -1.32 pp | 28.48 % | +0.09 pp | 21.41 % | -6.98 pp | 33.53 % | +5.14 pp |
| DRAM Throughput (%) | 24.13 % | 15.02 % | -9.11 pp | 24.08 % | -0.05 pp | 4.76 % | -19.37 pp | 21.02 % | -3.11 pp |
| Achieved Occupancy (%) | 58.36 % | 89.04 % | +30.68 pp | 63.39 % | +5.03 pp | 55.60 % | -2.76 pp | 76.70 % | +18.34 pp |
| Registers/Thread | 19.00 register/thread | 32.00 register/thread | +13 | 20.00 register/thread | +1 | 48.00 register/thread | +29 | 28.00 register/thread | +9 |
| Waves/SM | 47.52 | 47.52 | +0.00 | 47.52 | +0.00 | 152.05 | +104.53 | 47.52 | +0.00 |
| No Eligible (%) | 36.21 % | 12.22 % | -23.99 pp | 21.96 % | -14.25 pp | 8.36 % | -27.85 pp | 12.55 % | -23.66 pp |
| IPC Active | 2.56 inst/cycle | 3.51 inst/cycle | +0.95 | 3.11 inst/cycle | +0.55 | 3.67 inst/cycle | +1.11 | 3.48 inst/cycle | +0.92 |


### Key Learnings

#### ❌ Plan C: `increase_xblock_for_vectorized_loads_and_stores` (0.197×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) caused the Triton compiler to select a configuration that increased registers per thread from 19 to 48, reducing Block Limit Registers from 21 to 5 and theoretical occupancy from 100% to 62.5%. More critically, executed instructions exploded from 31M to 233M (7.5x), indicating the larger tile size triggered massive instruction expansion (likely due to unrolling or predicated masking overhead from the newly introduced xmask), causing duration to increase from 64 µs to 325 µs. When the original kernel had no masking (tl.full True mask), adding xmask = xindex < xnumel with autotune introduces predicated instructions across all loads/stores for every XBLOCK config, dramatically increasing instruction count even when xnumel is exactly divisible by XBLOCK.

> **Applicable when**: Never — this specific combination of autotune + masking introduction + broken grid caused severe regression.

> **Avoid when**: Do NOT add @triton.autotune with larger XBLOCK configs when: (1) the original kernel uses tl.full True mask (no predication) and the rewrite introduces xindex < xnumel masking — this dramatically increases instruction count; (2) the grid is not updated to use lambda meta: (cdiv(xnumel, meta['XBLOCK']),) — the grid remained hardcoded to cdiv(xnumel, 128) while autotune selected XBLOCK=256, causing each CTA to process 256 elements but only 128 grid slots were launched, resulting in incorrect coverage and wasted work; (3) registers per thread are already low (≤20) but the autotune configs cause the compiler to allocate significantly more registers (48 vs 19), crushing Block Limit Registers from 21 to 5 and reducing theoretical occupancy from 100% to 62.5%.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_sqrt_div_with_rsqrt_to_eliminate_non_fused_fp_instructions | 1.005× | 1.001× | ✓ | **✓ best** |
| B | simplify_arithmetic_to_enable_fused_fp32_instructions | 1.004× | 0.999× | ✓ |  |
| C | fuse_sqrt_reciprocal_to_rsqrt_for_fp_instruction_reduction | 0.900× | 0.916× | ✓ |  |
| D | reduce_register_pressure_via_rsqrt_and_smaller_block | 1.002× | 1.003× | ✓ |  |
| E | add_triton_autotune | 1.001× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_sqrt_div_... | Δ | simplify_arithmet... | Δ | fuse_sqrt_recipro... | Δ | reduce_register_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 64100 ns | 64060 ns | -40 ns (-0.1%) | 64190 ns | +90 ns (+0.1%) | 69950 ns | +5850 ns (+9.1%) | 63900 ns | -200 ns (-0.3%) | 64160 ns | +60 ns (+0.1%) |
| SM Throughput (%) | 61.37 % | 48.35 % | -13.02 pp | 55.36 % | -6.01 pp | 65.62 % | +4.25 pp | 48.34 % | -13.03 pp | 75.15 % | +13.78 pp |
| Memory Throughput (%) | 28.39 % | 28.30 % | -0.09 pp | 38.59 % | +10.20 pp | 40.69 % | +12.30 pp | 28.28 % | -0.11 pp | 28.66 % | +0.27 pp |
| DRAM Throughput (%) | 24.13 % | 24.12 % | -0.01 pp | 24.08 % | -0.05 pp | 22.24 % | -1.89 pp | 24.23 % | +0.10 pp | 24.08 % | -0.05 pp |
| Achieved Occupancy (%) | 58.36 % | 43.68 % | -14.68 pp | 32.77 % | -25.59 pp | 40.25 % | -18.11 pp | 44.30 % | -14.06 pp | 64.57 % | +6.21 pp |
| Registers/Thread | 19.00 register/thread | 20.00 register/thread | +1 | 26.00 register/thread | +7 | 32.00 register/thread | +13 | 20.00 register/thread | +1 | 20.00 register/thread | +1 |
| Waves/SM | 47.52 | 47.52 | +0.00 | 47.52 | +0.00 | 47.52 | +0.00 | 47.52 | +0.00 | 47.52 | +0.00 |
| No Eligible (%) | 36.21 % | 49.80 % | +13.59 pp | 41.36 % | +5.15 pp | 31.54 % | -4.67 pp | 49.86 % | +13.65 pp | 21.40 % | -14.81 pp |
| IPC Active | 2.56 inst/cycle | 2.01 inst/cycle | -0.55 | 2.34 inst/cycle | -0.22 | 2.69 inst/cycle | +0.13 | 2.00 inst/cycle | -0.56 | 3.11 inst/cycle | +0.55 |

---

## Iteration 3

**Best so far**: 1.0006× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency | 0.767× | 0.750× | ✓ |  |
| B | add_evict_first_to_streaming_load_to_reduce_l1tex_stalls | 1.002× | 1.002× | ✓ |  |
| C | fix_autotune_grid_lambda_and_add_evict_first_for_streaming_load | 1.005× | 1.000× | ✓ | **✓ best** |
| D | fix_autotune_grid_lambda_and_eliminate_unnecessary_xmask | 0.901× | 0.918× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | add_evict_first_t... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 64100 ns | 85440 ns | +21340 ns (+33.3%) | 63970 ns | -130 ns (-0.2%) | 64130 ns | +30 ns (+0.0%) | 69820 ns | +5720 ns (+8.9%) |
| SM Throughput (%) | 61.37 % | 74.07 % | +12.70 pp | 51.75 % | -9.62 pp | 59.27 % | -2.10 pp | 65.21 % | +3.84 pp |
| Memory Throughput (%) | 28.39 % | 40.57 % | +12.18 pp | 28.33 % | -0.06 pp | 38.76 % | +10.37 pp | 40.53 % | +12.14 pp |
| DRAM Throughput (%) | 24.13 % | 18.09 % | -6.04 pp | 24.15 % | +0.02 pp | 24.13 % | +0.00 pp | 22.26 % | -1.87 pp |
| Achieved Occupancy (%) | 58.36 % | 77.36 % | +19.00 pp | 44.63 % | -13.73 pp | 33.39 % | -24.97 pp | 40.17 % | -18.19 pp |
| Registers/Thread | 19.00 register/thread | 22.00 register/thread | +3 | 20.00 register/thread | +1 | 26.00 register/thread | +7 | 32.00 register/thread | +13 |
| Waves/SM | 47.52 | 95.03 | +47.51 | 47.52 | +0.00 | 47.52 | +0.00 | 47.52 | +0.00 |
| No Eligible (%) | 36.21 % | 24.34 % | -11.87 pp | 46.57 % | +10.36 pp | 38.09 % | +1.88 pp | 32.00 % | -4.21 pp |
| IPC Active | 2.56 inst/cycle | 3.04 inst/cycle | +0.48 | 2.14 inst/cycle | -0.42 | 2.47 inst/cycle | -0.09 | 2.68 inst/cycle | +0.12 |


### Key Learnings

#### ❌ Plan A: `increase_num_warps_to_hide_l1tex_latency` (0.750×)

Increasing num_warps from 4 to 8 (block_size 128→256) in a Triton pointwise kernel did improve achieved occupancy (44%→77%) and eligible warps per scheduler (0.77→2.85) as intended, but caused a 33% slowdown. The autotune selected XBLOCK=256 with block_size=256, which doubled the executed instruction count from 24.5M to 49.8M and increased warp cycles per issued instruction from 13.6 to 16.0 cycles — the larger block processed more work per CTA but the per-instruction latency worsened, and the total instruction volume increase outweighed the latency-hiding benefit. When the kernel already has very high waves-per-SM (47.5), doubling block size doubles waves (95) and doubles total instructions, negating any scheduler efficiency gain.

> **Applicable when**: Never — this specific strategy of doubling num_warps via block_size increase in a high-waves-per-SM pointwise kernel caused regression despite improving occupancy metrics.

> **Avoid when**: Do NOT increase num_warps by doubling block_size when: (1) the kernel already has high waves_per_sm (>40) — doubling block_size doubles waves and total instruction count proportionally; (2) the autotune may select a larger XBLOCK config that doubles per-CTA work, since the instruction count increase (2x) outweighs the latency-hiding benefit from more eligible warps; (3) registers per thread increase even modestly (20→22) when combined with larger block_size, as Block Limit Registers drops sharply (21→10), reducing the number of concurrent blocks per SM and partially negating the occupancy gain. The eligible warps per scheduler did improve (0.77→2.85) but warp cycles per issued instruction worsened (13.6→16.0), indicating the memory latency was not actually hidden.

---

## Iteration 4

**Best so far**: 1.0006× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_stages_to_pipeline_l1tex_loads_and_cap_registers_for_occupancy | 1.002× | 1.002× | ✓ |  |
| B | increase_num_stages_for_software_pipelining_to_hide_l1tex_latency | 1.002× | 1.000× | ✓ |  |
| C | fix_grid_lambda_and_add_num_stages_pipelining | 0.904× | 0.914× | ✓ |  |
| D | restore_num_warps4_configs_and_fix_autotune_selection | 1.002× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | increase_num_stag... | Δ | fix_grid_lambda_a... | Δ | restore_num_warps... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 64100 ns | 63940 ns | -160 ns (-0.2%) | 64130 ns | +30 ns (+0.0%) | 70140 ns | +6040 ns (+9.4%) | 63900 ns | -200 ns (-0.3%) |
| SM Throughput (%) | 61.37 % | 51.87 % | -9.50 pp | 58.80 % | -2.57 pp | 64.85 % | +3.48 pp | 48.69 % | -12.68 pp |
| Memory Throughput (%) | 28.39 % | 28.42 % | +0.03 pp | 38.52 % | +10.13 pp | 40.20 % | +11.81 pp | 28.47 % | +0.08 pp |
| DRAM Throughput (%) | 24.13 % | 24.18 % | +0.05 pp | 24.09 % | -0.04 pp | 22.15 % | -1.98 pp | 24.26 % | +0.13 pp |
| Achieved Occupancy (%) | 58.36 % | 44.48 % | -13.88 pp | 34.04 % | -24.32 pp | 39.46 % | -18.90 pp | 43.73 % | -14.63 pp |
| Registers/Thread | 19.00 register/thread | 20.00 register/thread | +1 | 26.00 register/thread | +7 | 32.00 register/thread | +13 | 20.00 register/thread | +1 |
| Waves/SM | 47.52 | 47.52 | +0.00 | 47.52 | +0.00 | 47.52 | +0.00 | 47.52 | +0.00 |
| No Eligible (%) | 36.21 % | 46.61 % | +10.40 pp | 37.84 % | +1.63 pp | 31.77 % | -4.44 pp | 49.46 % | +13.25 pp |
| IPC Active | 2.56 inst/cycle | 2.14 inst/cycle | -0.42 | 2.48 inst/cycle | -0.08 | 2.73 inst/cycle | +0.17 | 2.02 inst/cycle | -0.54 |

---

## Iteration 5

**Best so far**: 1.0006× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_xblock64_for_more_concurrent_ctas | 1.014× | 1.002× | ✓ | **✓ best** |
| B | precompute_channel_scale_shift_to_reduce_per_element_fp_ops | 1.002× | 1.003× | ✓ |  |
| C | fix_static_grid_to_dynamic_lambda_for_autotune_correctness_and_efficiency | 1.003× | 0.996× | ✓ |  |
| D | eliminate_redundant_xmask_to_reduce_register_pressure_and_boost_occupancy | 1.007× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | precompute_channe... | Δ | fix_static_grid_t... | Δ | eliminate_redunda... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 64100 ns | 63940 ns | -160 ns (-0.2%) | 63900 ns | -200 ns (-0.3%) | 64350 ns | +250 ns (+0.4%) | 64190 ns | +90 ns (+0.1%) |
| SM Throughput (%) | 61.37 % | 24.18 % | -37.19 pp | 51.88 % | -9.49 pp | 59.11 % | -2.26 pp | 44.35 % | -17.02 pp |
| Memory Throughput (%) | 28.39 % | 13.44 % | -14.95 pp | 28.30 % | -0.09 pp | 38.66 % | +10.27 pp | 27.98 % | -0.41 pp |
| DRAM Throughput (%) | 24.13 % | 8.56 % | -15.57 pp | 24.19 % | +0.06 pp | 24.03 % | -0.10 pp | 24.12 % | -0.01 pp |
| Achieved Occupancy (%) | 58.36 % | 26.55 % | -31.81 pp | 45.59 % | -12.77 pp | 33.60 % | -24.76 pp | 41.45 % | -16.91 pp |
| Registers/Thread | 19.00 register/thread | 20.00 register/thread | +1 | 20.00 register/thread | +1 | 26.00 register/thread | +7 | 18.00 register/thread | -1 |
| Waves/SM | 47.52 | 23.76 | -23.76 | 47.52 | +0.00 | 47.52 | +0.00 | 47.52 | +0.00 |
| No Eligible (%) | 36.21 % | 74.24 % | +38.03 pp | 46.86 % | +10.65 pp | 37.88 % | +1.67 pp | 62.64 % | +26.43 pp |
| IPC Active | 2.56 inst/cycle | 1.03 inst/cycle | -1.53 | 2.14 inst/cycle | -0.42 | 2.48 inst/cycle | -0.08 | 1.49 inst/cycle | -1.07 |

---

## Cumulative Progress

```
Baseline:  64100 ns  (1.000×)
Iter 2     64060 ns  (1.001×)  — replace_sqrt_div_with_rsqrt_to_eliminate_non_fused_fp_instructions
Iter 5     63940 ns  (1.002×)  — fix_grid_lambda_and_add_xblock64_for_more_concurrent_ctas

Final best: 63940 ns  (1.0025×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 12845056
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 128)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
     tmp6 = tl.full([1], 1e-05, tl.float32)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.sqrt_rn(tmp7)
-    tmp9 = tl.full([1], 1, tl.int32)
-    tmp10 = (tmp9 / tmp8)
-    tmp11 = tl.full([1], 1.0, tl.float32)
-    tmp12 = tmp10 * tmp11
+    tmp12 = libdevice.rsqrt(tmp7)
     tmp13 = tmp4 * tmp12
     tmp15 = tmp13 * tmp14
     tmp17 = tmp15 + tmp16
     tmp18 = tl.full([1], 0, tl.int32)
     tmp19 = tl.maximum(tmp18, tmp17)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +50,7 @@

     arg_5 = inputs[5]
     xnumel = 12845056
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 12845056, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 12845056)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2690 ns — **Best latency**: 2300 ns (**1.1696×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_and_add_single_block_full_coverage_config`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_12_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (9,1,1), Waves/SM: 0.00
- **Registers/thread**: 18
- **Duration**: 2690 ns
- **SM Throughput**: 0.15 % | **Memory**: 0.44 % | **DRAM**: 0.14 %
- **Achieved Occupancy**: 6.14 %
- **IPC Active**: 0.25 inst/cycle | **No Eligible**: 93.45 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_grid_coverage_via_autotune_xblock | 1.006× | 1.093× | ✓ | **✓ best** |
| B | autotune_xblock_and_num_warps_for_tiny_pointwise_kernel | 0.966× | 1.107× | ✓ |  |
| C | autotune_xblock_and_vectorized_loads | 0.979× | 1.011× | ✓ |  |
| D | autotune_xblock_to_eliminate_underutilization | 0.978× | 1.027× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_grid_cov... | Δ | autotune_xblock_a... | Δ | autotune_xblock_a... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2690 ns | 2460 ns | -230 ns (-8.6%) | 2430 ns | -260 ns (-9.7%) | 2660 ns | -30 ns (-1.1%) | 2620 ns | -70 ns (-2.6%) |
| SM Throughput (%) | 0.15 % | 0.04 % | -0.11 pp | 0.14 % | -0.01 pp | 0.11 % | -0.04 pp | 0.10 % | -0.05 pp |
| Memory Throughput (%) | 0.44 % | 0.41 % | -0.03 pp | 0.49 % | +0.05 pp | 0.46 % | +0.02 pp | 0.45 % | +0.01 pp |
| DRAM Throughput (%) | 0.14 % | 0.06 % | -0.08 pp | 0.15 % | +0.01 pp | 0.14 % | +0.00 pp | 0.14 % | +0.00 pp |
| Achieved Occupancy (%) | 6.14 % | 1.56 % | -4.58 pp | 6.08 % | -0.06 pp | 3.06 % | -3.08 pp | 3.11 % | -3.03 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 18.00 register/thread | +0 | 24.00 register/thread | +6 | 24.00 register/thread | +6 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.45 % | 92.99 % | -0.46 pp | 94.17 % | +0.72 pp | 93.52 % | +0.07 pp | 92.35 % | -1.10 pp |
| IPC Active | 0.25 inst/cycle | 0.06 inst/cycle | -0.19 | 0.20 inst/cycle | -0.05 | 0.14 inst/cycle | -0.11 | 0.13 inst/cycle | -0.12 |

---

## Iteration 2

**Best so far**: 1.0935× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_xblock_with_correct_grid_lambda_for_tiny_workload | 0.968× | 0.978× | ✓ |  |
| B | eliminate_redundant_arithmetic_and_fix_eviction_policy | 0.967× | 1.093× | ✓ |  |
| C | eliminate_redundant_arithmetic_and_add_small_xblock_configs | 1.009× | 1.093× | ✓ | **✓ best** |
| D | small_xblock_configs_plus_arithmetic_simplification | 0.982× | 0.944× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_xblock_wit... | Δ | eliminate_redunda... | Δ | eliminate_redunda... | Δ | small_xblock_conf... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2690 ns | 2750 ns | +60 ns (+2.2%) | 2460 ns | -230 ns (-8.6%) | 2460 ns | -230 ns (-8.6%) | 2850 ns | +160 ns (+5.9%) |
| SM Throughput (%) | 0.15 % | 0.03 % | -0.12 pp | 0.03 % | -0.12 pp | 0.04 % | -0.11 pp | 0.04 % | -0.11 pp |
| Memory Throughput (%) | 0.44 % | 0.37 % | -0.07 pp | 0.40 % | -0.04 pp | 0.42 % | -0.02 pp | 0.40 % | -0.04 pp |
| DRAM Throughput (%) | 0.14 % | 0.05 % | -0.09 pp | 0.06 % | -0.08 pp | 0.08 % | -0.06 pp | 0.05 % | -0.09 pp |
| Achieved Occupancy (%) | 6.14 % | 1.56 % | -4.58 pp | 1.56 % | -4.58 pp | 1.56 % | -4.58 pp | 1.56 % | -4.58 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 20.00 register/thread | +2 | 18.00 register/thread | +0 | 19.00 register/thread | +1 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.45 % | 93.56 % | +0.11 pp | 93.70 % | +0.25 pp | 93.48 % | +0.03 pp | 93.08 % | -0.37 pp |
| IPC Active | 0.25 inst/cycle | 0.07 inst/cycle | -0.18 | 0.05 inst/cycle | -0.20 | 0.07 inst/cycle | -0.18 | 0.07 inst/cycle | -0.18 |

---

## Iteration 3

**Best so far**: 1.0935× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_expand_xblock_configs_for_full_sm_coverage | 0.995× | 1.093× | ✓ |  |
| B | fix_grid_lambda_and_add_num_stages_pipeline | 0.979× | 0.978× | ✓ |  |
| C | fix_grid_lambda_to_adapt_with_xblock_autotune | 1.016× | 1.039× | ✓ | **✓ best** |
| D | fix_grid_lambda_and_improve_l1_cache_reuse_via_larger_xblock | 1.007× | 1.107× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2690 ns | 2460 ns | -230 ns (-8.6%) | 2750 ns | +60 ns (+2.2%) | 2590 ns | -100 ns (-3.7%) | 2430 ns | -260 ns (-9.7%) |
| SM Throughput (%) | 0.15 % | 0.04 % | -0.11 pp | 0.03 % | -0.12 pp | 0.04 % | -0.11 pp | 0.03 % | -0.12 pp |
| Memory Throughput (%) | 0.44 % | 0.43 % | -0.01 pp | 0.38 % | -0.06 pp | 0.52 % | +0.08 pp | 0.42 % | -0.02 pp |
| DRAM Throughput (%) | 0.14 % | 0.05 % | -0.09 pp | 0.07 % | -0.07 pp | 0.04 % | -0.10 pp | 0.06 % | -0.08 pp |
| Achieved Occupancy (%) | 6.14 % | 1.56 % | -4.58 pp | 1.56 % | -4.58 pp | 1.56 % | -4.58 pp | 1.56 % | -4.58 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 18.00 register/thread | +0 | 19.00 register/thread | +1 | 19.00 register/thread | +1 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.45 % | 95.48 % | +2.03 pp | 93.63 % | +0.18 pp | 93.30 % | -0.15 pp | 93.45 % | +0.00 pp |
| IPC Active | 0.25 inst/cycle | 0.05 inst/cycle | -0.20 | 0.06 inst/cycle | -0.19 | 0.06 inst/cycle | -0.19 | 0.07 inst/cycle | -0.18 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_and_expand_xblock_configs_for_full_sm_coverage` (1.093×)

When a Triton kernel uses @triton.autotune over XBLOCK but the launch site passes a hardcoded grid (e.g., cdiv(N, 128)) instead of a lambda using meta['XBLOCK'], the autotune is effectively broken — all configs run with the same grid, making smaller XBLOCK configs under-cover the workload and larger ones over-cover it. Fixing the grid to `lambda meta: (triton.cdiv(N, meta['XBLOCK']),)` and expanding the autotune config set to include larger XBLOCK values (256, 512, 1024, 2048) allows the runtime to empirically select the optimal tile size. In this case, the L2 hit rate improved from 36.93% to 49.77% and L1TEX stall contribution dropped from 57.6% to 38.8% of warp cycles, indicating better cache reuse from larger tiles, yielding a ~9% speedup despite the kernel remaining SM-starved.

> **Applicable when**: Triton kernel has @triton.autotune over XBLOCK but the launch site uses a hardcoded grid constant instead of a meta-lambda. Detectable when: grid_size << sm_count, achieved occupancy <5%, L2 hit rate <40%, and CPIStall L1TEX stall fraction >40% of warp cycles. Especially impactful for small pointwise kernels (xnumel < 4096) where larger tiles can fit broadcast/reused tensors in L1.

> **Avoid when**: Kernel already uses a correct grid lambda (lambda meta: (triton.cdiv(N, meta['XBLOCK']),)). Do not add XBLOCK configs larger than xnumel without ensuring xmask guards all loads and stores — missing masks will cause out-of-bounds memory access. Not beneficial when the kernel is already SM-saturated (waves_per_sm >= 1.0).

---

## Iteration 4

**Best so far**: 1.0935× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_instructions_and_fix_grid_lambda_for_fma_throughput | 1.043× | 1.170× | ✓ | **✓ best** |
| B | fix_grid_lambda_plus_evict_last_broadcast_loads_plus_relu_register_save | 1.003× | 1.093× | ✓ |  |
| C | single_block_full_workload_with_l1_broadcast_reuse | 0.937× | 0.546× | ✓ |  |
| D | fix_grid_lambda_with_pipelined_stages_and_larger_tiles | 1.013× | 1.093× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_instruc... | Δ | fix_grid_lambda_p... | Δ | single_block_full... | Δ | fix_grid_lambda_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2690 ns | 2300 ns | -390 ns (-14.5%) | 2460 ns | -230 ns (-8.6%) | 4930 ns | +2240 ns (+83.3%) | 2460 ns | -230 ns (-8.6%) |
| SM Throughput (%) | 0.15 % | 0.03 % | -0.12 pp | 0.04 % | -0.11 pp | 0.52 % | +0.37 pp | 0.04 % | -0.11 pp |
| Memory Throughput (%) | 0.44 % | 0.44 % | +0.00 pp | 0.43 % | -0.01 pp | 0.33 % | -0.11 pp | 0.41 % | -0.03 pp |
| DRAM Throughput (%) | 0.14 % | 0.05 % | -0.09 pp | 0.06 % | -0.08 pp | 0.11 % | -0.03 pp | 0.08 % | -0.06 pp |
| Achieved Occupancy (%) | 6.14 % | 1.56 % | -4.58 pp | 1.56 % | -4.58 pp | 6.12 % | -0.02 pp | 1.56 % | -4.58 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 19.00 register/thread | +1 | 84.00 register/thread | +66 | 18.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.01 | +0.01 | 0.00 | +0.00 |
| No Eligible (%) | 93.45 % | 96.30 % | +2.85 pp | 93.12 % | -0.33 pp | 78.77 % | -14.68 pp | 93.18 % | -0.27 pp |
| IPC Active | 0.25 inst/cycle | 0.03 inst/cycle | -0.22 | 0.07 inst/cycle | -0.18 | 0.88 inst/cycle | +0.63 | 0.06 inst/cycle | -0.19 |


### Key Learnings

#### ❌ Plan C: `single_block_full_workload_with_l1_broadcast_reuse` (0.546×)

Setting XBLOCK=2048 for a kernel with xnumel=1152 caused Triton to unroll the masked loop body across 2048 lanes, exploding register usage from 19 to 84 registers/thread. This register pressure reduced Block Limit Registers from 84 to 5 blocks/SM, cutting theoretical occupancy from 50% to 31.25% and causing executed instruction count to balloon from 756 to 19,368 — a 25x increase in instruction work. Although L1/TEX hit rate improved (16.67% → 50%) and warp cycles per issued instruction dropped (14.41 → 4.44), the massive instruction count increase more than doubled elapsed cycles (3606 → 7342), yielding a 0.55x regression. When XBLOCK greatly exceeds xnumel, Triton generates unrolled masked code for all out-of-bounds lanes, creating catastrophic register spill and instruction bloat.

> **Applicable when**: Never: this is an anti-pattern. Do not use XBLOCK values significantly larger than xnumel.

> **Avoid when**: Always avoid XBLOCK >> xnumel. When XBLOCK exceeds xnumel by more than ~2x, Triton unrolls masked dead lanes causing register explosion (registers/thread can increase 4x+), theoretical occupancy collapse (Block Limit Registers drops sharply), and instruction count bloat (25x+ increase observed). The L1 hit rate improvement from fitting data in cache does NOT compensate for the instruction and register overhead.

---

## Iteration 5

**Best so far**: 1.1696× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_single_block_full_coverage_config | 1.006× | 1.170× | ✓ |  |
| B | fix_grid_lambda_single_block_full_workload_collapse | 1.013× | 1.093× | ✓ |  |
| C | fix_grid_lambda_plus_single_block_full_coverage_config | 1.014× | 0.989× | ✓ |  |
| D | fix_grid_lambda_small_xblock_configs_relu_dtype_register_reduction | 0.965× | 1.039× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_s... | Δ | fix_grid_lambda_p... | Δ | fix_grid_lambda_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2690 ns | 2300 ns | -390 ns (-14.5%) | 2460 ns | -230 ns (-8.6%) | 2720 ns | +30 ns (+1.1%) | 2590 ns | -100 ns (-3.7%) |
| SM Throughput (%) | 0.15 % | 0.03 % | -0.12 pp | 0.04 % | -0.11 pp | 0.04 % | -0.11 pp | 0.05 % | -0.10 pp |
| Memory Throughput (%) | 0.44 % | 0.42 % | -0.02 pp | 0.41 % | -0.03 pp | 0.41 % | -0.03 pp | 0.46 % | +0.02 pp |
| DRAM Throughput (%) | 0.14 % | 0.06 % | -0.08 pp | 0.06 % | -0.08 pp | 0.08 % | -0.06 pp | 0.13 % | -0.01 pp |
| Achieved Occupancy (%) | 6.14 % | 1.56 % | -4.58 pp | 1.56 % | -4.58 pp | 1.56 % | -4.58 pp | 1.56 % | -4.58 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 19.00 register/thread | +1 | 18.00 register/thread | +0 | 24.00 register/thread | +6 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.45 % | 96.19 % | +2.74 pp | 93.23 % | -0.22 pp | 93.38 % | -0.07 pp | 91.72 % | -1.73 pp |
| IPC Active | 0.25 inst/cycle | 0.05 inst/cycle | -0.20 | 0.06 inst/cycle | -0.19 | 0.06 inst/cycle | -0.19 | 0.07 inst/cycle | -0.18 |

---

## Cumulative Progress

```
Baseline:  2690 ns  (1.000×)
Iter 1     2460 ns  (1.093×)  — increase_grid_coverage_via_autotune_xblock
Iter 4     2300 ns  (1.170×)  — fuse_fp32_instructions_and_fix_grid_lambda_for_fma_throughput

Final best: 2300 ns  (1.1696×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16},   num_warps=1),
+        triton.Config({'XBLOCK': 32},   num_warps=1),
+        triton.Config({'XBLOCK': 64},   num_warps=2),
+        triton.Config({'XBLOCK': 128},  num_warps=2),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1152
@@ -21,20 +39,13 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    tmp6 = tl.math.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
-    tmp18 = (tmp17 / tmp9)
-    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -44,7 +55,7 @@

     arg_4 = inputs[4]
     xnumel = 1152
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1152, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_12[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1152)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

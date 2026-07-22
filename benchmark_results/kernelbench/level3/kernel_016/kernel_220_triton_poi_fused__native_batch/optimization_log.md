# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_220` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6980 ns — **Best latency**: 6590 ns (**1.0592×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_and_increase_num_warps_for_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_220`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_220_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6615,1,1), Waves/SM: 3.13
- **Registers/thread**: 22
- **Duration**: 6980 ns
- **SM Throughput**: 42.30 % | **Memory**: 16.02 % | **DRAM**: 10.04 %
- **Achieved Occupancy**: 61.97 %
- **IPC Active**: 2.55 inst/cycle | **No Eligible**: 35.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.971× | 0.974× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.915× | 0.814× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads | 0.915× | 0.820× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ | add_autotune_with... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6980 ns | 7170 ns | +190 ns (+2.7%) | 8580 ns | +1600 ns (+22.9%) | 8510 ns | +1530 ns (+21.9%) |
| SM Throughput (%) | 42.30 % | 52.27 % | +9.97 pp | 63.48 % | +21.18 pp | 63.54 % | +21.24 pp |
| Memory Throughput (%) | 16.02 % | 19.38 % | +3.36 pp | 17.29 % | +1.27 pp | 17.30 % | +1.28 pp |
| DRAM Throughput (%) | 10.04 % | 9.76 % | -0.28 pp | 8.15 % | -1.89 pp | 8.20 % | -1.84 pp |
| Achieved Occupancy (%) | 61.97 % | 72.42 % | +10.45 pp | 80.15 % | +18.18 pp | 80.14 % | +18.17 pp |
| Registers/Thread | 22.00 register/thread | 24.00 register/thread | +2 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 3.13 | 3.13 | +0.00 | 3.13 | +0.00 | 3.13 | +0.00 |
| No Eligible (%) | 35.89 % | 21.08 % | -14.81 pp | 16.96 % | -18.93 pp | 16.84 % | -19.05 pp |
| IPC Active | 2.55 inst/cycle | 3.14 inst/cycle | +0.59 | 3.23 inst/cycle | +0.68 | 3.29 inst/cycle | +0.74 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy` (0.814×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) increased registers per thread from 22 to 32, nearly doubling total executed instructions (2.25M → 4.24M) and increasing kernel duration from 6.98 µs to 8.58 µs despite improving achieved occupancy (62% → 80%) and compute throughput (42% → 63%). The instruction count explosion—likely from compiler unrolling or spill/fill code at larger tile sizes—dominated any latency-hiding benefit from higher occupancy. When a kernel already has low register pressure and the bottleneck is latency rather than occupancy, increasing tile size via autotune can trigger compiler register pressure increases that negate the scheduling gains.

> **Applicable when**: Never: this specific strategy (autotuning to larger XBLOCK for pointwise elementwise kernels with low register pressure) caused regression here. Avoid when instruction count is already moderate and register count is near the block limit boundary.

> **Avoid when**: Kernel has low register pressure (≤22 regs/thread) but moderate instruction count; increasing XBLOCK causes compiler to allocate more registers (observed jump from 22→32), nearly doubling instruction count and wall time even when occupancy improves. Do not apply when the kernel is a simple pointwise op with multiple strided loads—the compiler unrolls aggressively at large tile sizes, inflating instruction count beyond the latency-hiding benefit.

#### ❌ Plan C: `add_autotune_with_larger_xblock_for_vectorized_loads` (0.820×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) for a pointwise kernel caused a 22% regression despite improving achieved occupancy (62%→80%) and compute throughput (42%→64%). The root cause was that the selected autotune config increased registers per thread from 22 to 32, which reduced Block Limit Registers from 21 to 16 blocks/SM, and nearly doubled total executed instructions (2.25M→4.24M), indicating the autotuner selected a config that unrolled more work per thread without proportional memory bandwidth benefit. When a kernel is already register-constrained near its block limit, adding autotune configs with larger XBLOCK and higher num_warps can increase register pressure and instruction count enough to negate any occupancy or vectorization gains.

> **Applicable when**: Kernel has low achieved occupancy (<70%) AND low memory throughput (<20%) AND partial-wave tail waste (>20%) AND registers per thread are well below the block limit threshold

> **Avoid when**: Kernel already uses close to the register block limit (e.g., block_limit_registers near 21 with 22 regs/thread); adding larger XBLOCK configs causes the autotuner to select configs that increase register pressure (22→32 regs/thread), reducing block limit registers (21→16) and nearly doubling instruction count without proportional throughput gain. Avoid when the kernel is compute-light and memory-bound at small block sizes — the instruction overhead from larger unrolled tiles dominates.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_reduce_num_warps_to_cut_partial_wave | 0.991× | 0.948× | ✓ |  |
| B | revert_to_small_xblock_and_use_rsqrt_for_fma_fusion | 0.988× | 1.030× | ✓ |  |
| C | fix_autotune_grid_lambda_and_reduce_register_pressure_configs | 1.001× | 1.006× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | revert_to_small_x... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6980 ns | 7360 ns | +380 ns (+5.4%) | 6780 ns | -200 ns (-2.9%) | 6940 ns | -40 ns (-0.6%) |
| SM Throughput (%) | 42.30 % | 40.06 % | -2.24 pp | 32.10 % | -10.20 pp | 30.55 % | -11.75 pp |
| Memory Throughput (%) | 16.02 % | 14.47 % | -1.55 pp | 16.20 % | +0.18 pp | 14.41 % | -1.61 pp |
| DRAM Throughput (%) | 10.04 % | 9.50 % | -0.54 pp | 10.35 % | +0.31 pp | 10.05 % | +0.01 pp |
| Achieved Occupancy (%) | 61.97 % | 66.81 % | +4.84 pp | 50.50 % | -11.47 pp | 51.67 % | -10.30 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 18.00 register/thread | -4 | 24.00 register/thread | +2 |
| Waves/SM | 3.13 | 1.57 | -1.56 | 3.13 | +0.00 | 1.57 | -1.56 |
| No Eligible (%) | 35.89 % | 21.65 % | -14.24 pp | 59.64 % | +23.75 pp | 41.87 % | +5.98 pp |
| IPC Active | 2.55 inst/cycle | 3.05 inst/cycle | +0.50 | 1.52 inst/cycle | -1.03 | 2.22 inst/cycle | -0.33 |

---

## Iteration 3

**Best so far**: 1.0058× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_without_xblock_change | 1.015× | 1.034× | ✓ |  |
| B | replace_sqrt_div_with_rsqrt_and_align_grid_to_eliminate_partial_wave | 1.019× | 1.059× | ✓ | **✓ best** |
| C | replace_sqrt_reciprocal_with_rsqrt_for_fma_fusion | 1.012× | 1.030× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | replace_sqrt_div_... | Δ | replace_sqrt_reci... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6980 ns | 6750 ns | -230 ns (-3.3%) | 6590 ns | -390 ns (-5.6%) | 6780 ns | -200 ns (-2.9%) |
| SM Throughput (%) | 42.30 % | 32.10 % | -10.20 pp | 9.69 % | -32.61 pp | 18.60 % | -23.70 pp |
| Memory Throughput (%) | 16.02 % | 16.30 % | +0.28 pp | 15.02 % | -1.00 pp | 18.24 % | +2.22 pp |
| DRAM Throughput (%) | 10.04 % | 10.36 % | +0.32 pp | 10.58 % | +0.54 pp | 10.31 % | +0.27 pp |
| Achieved Occupancy (%) | 61.97 % | 46.41 % | -15.56 pp | 18.76 % | -43.21 pp | 42.25 % | -19.72 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 3.13 | 3.13 | +0.00 | 1.57 | -1.56 | 1.57 | -1.56 |
| No Eligible (%) | 35.89 % | 60.36 % | +24.47 pp | 79.53 % | +43.64 pp | 52.87 % | +16.98 pp |
| IPC Active | 2.55 inst/cycle | 1.57 inst/cycle | -0.98 | 0.82 inst/cycle | -1.73 | 1.62 inst/cycle | -0.93 |

---

## Iteration 4

**Best so far**: 1.0592× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_increase_num_warps_for_latency_hiding | 1.010× | 1.059× | ✓ |  |
| B | fix_int32_zero_in_relu_to_enable_fma_fusion | 0.988× | 1.034× | ✓ |  |
| C | fix_relu_int32_cast_to_enable_fma_fusion | 0.985× | 1.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_int32_zero_in... | Δ | fix_relu_int32_ca... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6980 ns | 6590 ns | -390 ns (-5.6%) | 6750 ns | -230 ns (-3.3%) | 6880 ns | -100 ns (-1.4%) |
| SM Throughput (%) | 42.30 % | 9.71 % | -32.59 pp | 32.05 % | -10.25 pp | 18.69 % | -23.61 pp |
| Memory Throughput (%) | 16.02 % | 15.02 % | -1.00 pp | 16.27 % | +0.25 pp | 18.14 % | +2.12 pp |
| DRAM Throughput (%) | 10.04 % | 10.61 % | +0.57 pp | 10.33 % | +0.29 pp | 10.15 % | +0.11 pp |
| Achieved Occupancy (%) | 61.97 % | 19.27 % | -42.70 pp | 46.90 % | -15.07 pp | 45.73 % | -16.24 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 18.00 register/thread | -4 | 32.00 register/thread | +10 |
| Waves/SM | 3.13 | 1.57 | -1.56 | 3.13 | +0.00 | 1.57 | -1.56 |
| No Eligible (%) | 35.89 % | 76.20 % | +40.31 pp | 60.45 % | +24.56 pp | 52.49 % | +16.60 pp |
| IPC Active | 2.55 inst/cycle | 0.84 inst/cycle | -1.71 | 1.57 inst/cycle | -0.98 | 1.57 inst/cycle | -0.98 |

---

## Iteration 5

**Best so far**: 1.0592× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_use_xblock256_with_num_warps2_for_wave_alignment | 1.018× | 1.048× | ✓ |  |
| B | add_evict_first_on_streaming_tensor_to_reduce_l1tex_stalls | 0.990× | 1.019× | ✓ |  |
| C | add_num_stages_pipelining_and_fma_friendly_constants | 1.019× | 1.030× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | add_evict_first_o... | Δ | add_num_stages_pi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6980 ns | 6660 ns | -320 ns (-4.6%) | 6850 ns | -130 ns (-1.9%) | 6780 ns | -200 ns (-2.9%) |
| SM Throughput (%) | 42.30 % | 16.76 % | -25.54 pp | 31.80 % | -10.50 pp | 18.24 % | -24.06 pp |
| Memory Throughput (%) | 16.02 % | 16.04 % | +0.02 pp | 16.02 % | +0.00 pp | 17.66 % | +1.64 pp |
| DRAM Throughput (%) | 10.04 % | 10.48 % | +0.44 pp | 10.21 % | +0.17 pp | 10.34 % | +0.30 pp |
| Achieved Occupancy (%) | 61.97 % | 39.30 % | -22.67 pp | 48.12 % | -13.85 pp | 42.44 % | -19.53 pp |
| Registers/Thread | 22.00 register/thread | 23.00 register/thread | +1 | 18.00 register/thread | -4 | 32.00 register/thread | +10 |
| Waves/SM | 3.13 | 1.57 | -1.56 | 3.13 | +0.00 | 1.57 | -1.56 |
| No Eligible (%) | 35.89 % | 67.97 % | +32.08 pp | 55.91 % | +20.02 pp | 53.22 % | +17.33 pp |
| IPC Active | 2.55 inst/cycle | 1.23 inst/cycle | -1.32 | 1.73 inst/cycle | -0.82 | 1.60 inst/cycle | -0.95 |

---

## Cumulative Progress

```
Baseline:  6980 ns  (1.000×)
Iter 2     6940 ns  (1.006×)  — fix_autotune_grid_lambda_and_reduce_register_pressure_configs
Iter 3     6590 ns  (1.059×)  — replace_sqrt_div_with_rsqrt_and_align_grid_to_eliminate_partial_wave

Final best: 6590 ns  (1.0592×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=1),
+        triton.Config({'XBLOCK': 128}, num_warps=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_220(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 846720
@@ -21,19 +31,12 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    tmp6 = libdevice.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
+    tmp17 = tl.maximum(0.0, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -44,7 +47,7 @@

     arg_5 = inputs[5]
     xnumel = 846720
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_220[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 846720, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_220[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 846720)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

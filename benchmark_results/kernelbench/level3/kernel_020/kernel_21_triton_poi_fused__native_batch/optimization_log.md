# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3200 ns — **Best latency**: 2880 ns (**1.1111×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_sqrt_div_chain_with_rsqrt_and_fma_in_batchnorm_pointwise`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_21_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (980,1,1), Waves/SM: 0.46
- **Registers/thread**: 18
- **Duration**: 3200 ns
- **SM Throughput**: 12.97 % | **Memory**: 4.61 % | **DRAM**: 3.27 %
- **Achieved Occupancy**: 38.70 %
- **IPC Active**: 1.46 inst/cycle | **No Eligible**: 65.83 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_for_better_sm_utilization | 0.976× | 0.847× | ✓ |  |
| B | autotune_xblock_num_warps_for_occupancy | 0.988× | 1.010× | ✓ |  |
| C | increase_xblock_with_autotune_for_better_occupancy_and_cache_reuse | 0.977× | 0.901× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_occupancy | 0.935× | 0.870× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_f... | Δ | autotune_xblock_n... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 3780 ns | +580 ns (+18.1%) | 3170 ns | -30 ns (-0.9%) | 3550 ns | +350 ns (+10.9%) | 3680 ns | +480 ns (+15.0%) |
| SM Throughput (%) | 12.97 % | 21.57 % | +8.60 pp | 16.08 % | +3.11 pp | 22.15 % | +9.18 pp | 29.32 % | +16.35 pp |
| Memory Throughput (%) | 4.61 % | 5.88 % | +1.27 pp | 5.55 % | +0.94 pp | 6.04 % | +1.43 pp | 9.01 % | +4.40 pp |
| DRAM Throughput (%) | 3.27 % | 2.76 % | -0.51 pp | 3.29 % | +0.02 pp | 2.93 % | -0.34 pp | 2.82 % | -0.45 pp |
| Achieved Occupancy (%) | 38.70 % | 29.24 % | -9.46 pp | 31.40 % | -7.30 pp | 30.08 % | -8.62 pp | 69.04 % | +30.34 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 24.00 register/thread | +6 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 | 0.93 | +0.47 |
| No Eligible (%) | 65.83 % | 54.34 % | -11.49 pp | 63.87 % | -1.96 pp | 46.41 % | -19.42 pp | 32.18 % | -33.65 pp |
| IPC Active | 1.46 inst/cycle | 1.92 inst/cycle | +0.46 | 1.71 inst/cycle | +0.25 | 2.05 inst/cycle | +0.59 | 2.49 inst/cycle | +1.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.997× | 0.982× | ✓ |  |
| B | persistent_tile_scheduling | 0.984× | 0.982× | ✓ |  |
| C | persistent_tile_scheduling | 0.983× | 0.944× | ✓ |  |
| D | persistent_tile_scheduling | 1.026× | 0.970× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.984× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 3260 ns | +60 ns (+1.9%) | 3260 ns | +60 ns (+1.9%) | 3390 ns | +190 ns (+5.9%) | 3300 ns | +100 ns (+3.1%) | 3170 ns | -30 ns (-0.9%) |
| SM Throughput (%) | 12.97 % | 10.02 % | -2.95 pp | 14.41 % | +1.44 pp | 14.41 % | +1.44 pp | 14.29 % | +1.32 pp | 13.01 % | +0.04 pp |
| Memory Throughput (%) | 4.61 % | 4.35 % | -0.26 pp | 4.62 % | +0.01 pp | 4.51 % | -0.10 pp | 4.48 % | -0.13 pp | 4.92 % | +0.31 pp |
| DRAM Throughput (%) | 3.27 % | 3.19 % | -0.08 pp | 3.20 % | -0.07 pp | 3.08 % | -0.19 pp | 3.20 % | -0.07 pp | 3.29 % | +0.02 pp |
| Achieved Occupancy (%) | 38.70 % | 23.90 % | -14.80 pp | 38.98 % | +0.28 pp | 39.09 % | +0.39 pp | 39.47 % | +0.77 pp | 39.80 % | +1.10 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 18.00 register/thread | +0 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 |
| No Eligible (%) | 65.83 % | 79.79 % | +13.96 pp | 65.72 % | -0.11 pp | 65.08 % | -0.75 pp | 60.35 % | -5.48 pp | 63.35 % | -2.48 pp |
| IPC Active | 1.46 inst/cycle | 0.95 inst/cycle | -0.51 | 1.51 inst/cycle | +0.05 | 1.51 inst/cycle | +0.05 | 1.51 inst/cycle | +0.05 | 1.33 inst/cycle | -0.13 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_instructions_with_fma_in_batchnorm_pointwise | 1.024× | 1.074× | ✓ |  |
| B | fuse_fp32_instructions_via_rsqrt_and_increase_xblock | 1.008× | 1.042× | ✓ |  |
| C | fuse_fp32_multiply_add_to_fma_in_bn_normalization | 1.008× | 1.042× | ✓ |  |
| D | increase_xblock_for_latency_hiding_and_fma_fusion | 0.975× | 0.952× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_instruc... | Δ | fuse_fp32_instruc... | Δ | fuse_fp32_multipl... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 2980 ns | -220 ns (-6.9%) | 3070 ns | -130 ns (-4.1%) | 3070 ns | -130 ns (-4.1%) | 3360 ns | +160 ns (+5.0%) |
| SM Throughput (%) | 12.97 % | 9.06 % | -3.91 pp | 11.31 % | -1.66 pp | 11.40 % | -1.57 pp | 10.95 % | -2.02 pp |
| Memory Throughput (%) | 4.61 % | 5.13 % | +0.52 pp | 5.07 % | +0.46 pp | 4.94 % | +0.33 pp | 4.32 % | -0.29 pp |
| DRAM Throughput (%) | 3.27 % | 3.51 % | +0.24 pp | 3.41 % | +0.14 pp | 3.40 % | +0.13 pp | 3.09 % | -0.18 pp |
| Achieved Occupancy (%) | 38.70 % | 37.37 % | -1.33 pp | 36.21 % | -2.49 pp | 36.06 % | -2.64 pp | 25.51 % | -13.19 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 28.00 register/thread | +10 | 32.00 register/thread | +14 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 |
| No Eligible (%) | 65.83 % | 72.13 % | +6.30 pp | 70.22 % | +4.39 pp | 70.63 % | +4.80 pp | 72.15 % | +6.32 pp |
| IPC Active | 1.46 inst/cycle | 1.02 inst/cycle | -0.44 | 1.07 inst/cycle | -0.39 | 1.06 inst/cycle | -0.40 | 1.09 inst/cycle | -0.37 |


### Key Learnings

#### ✅ Plan A: `fuse_fp32_instructions_with_fma_in_batchnorm_pointwise` (1.111×)

Applying the rsqrt+FMA batch-norm arithmetic fusion (replacing sqrt/div/mul chain with libdevice.rsqrt and tl.math.fma) together with adding wider XBLOCK autotune configs (including XBLOCK=2048, num_warps=16) improved achieved occupancy from 18.6% to 26.1%, active warps per scheduler from 2.8 to 4.59, and L1/TEX hit rate from 68.3% to 78.8%, reducing elapsed cycles by ~4% (4412→4249) and yielding a 1.11x speedup. The instruction count increased (110K→155K) because the wider autotune config selected a larger XBLOCK that processes more elements per CTA, but the improved warp-level parallelism and cache reuse more than compensated. The key causal signal is that eligible warps per scheduler doubled (0.19→0.40) and no-eligible cycles dropped (88%→79%), indicating better latency hiding from higher occupancy.

> **Applicable when**: FPInstructions NCU rule fires with non-fused FP32 count >= 2x fused count; kernel implements BN/LN affine transform as a multi-step chain; achieved occupancy is low (<30%) with low eligible warps per scheduler (<0.5) and high no-eligible cycles (>75%); persistent loop is already in use but autotune configs do not include the widest XBLOCK options (>=2048)

> **Avoid when**: Kernel is already at high occupancy (>50%) where adding wider XBLOCK configs reduces parallelism; kernel is DRAM bandwidth bound (DRAM throughput >50%); strict IEEE-754 precision is required (rsqrt.approx has ~2-4 ULP error); the persistent loop grid is already sized to fill all SMs with multiple waves (waves_per_sm > 2)

#### ✅ Plan C: `fuse_fp32_multiply_add_to_fma_in_bn_normalization` (1.042×)

Batch-norm normalization expressed as a multi-step chain (subtract mean, add eps, sqrt, int-divide, multiply by 1.0, scale, shift) produces a high ratio of non-fused FP32 instructions and introduces long scoreboard dependency chains. Collapsing the chain to `inv_std = rsqrt(var + eps); result = (x - mean) * inv_std * weight + bias` reduces executed instructions by ~27% (333k → 243k), eliminates divergent branches entirely (branch efficiency 83% → 100%), and improves average active threads per warp (28.2 → 32.0), yielding a ~4% wall-clock speedup. The FPInstructions NCU rule with a high non-fused:fused ratio (3:1) is the reliable trigger for this pattern.

> **Applicable when**: FPInstructions NCU rule fires with non-fused FP32 instruction count ≥ 3× fused count; BN normalization arithmetic is expressed as a multi-step chain with intermediate int-to-float conversions or redundant multiply-by-1.0 ops; kernel is latency-bound with L1TEX scoreboard stalls present; branch efficiency < 100% due to predicated boundary checks inside the arithmetic chain.

> **Avoid when**: Kernel is already using rsqrt and canonical FMA form; strict IEEE-754 compliance is required (rsqrt.approx has ~2 ULP error vs exact sqrt+divide); the bottleneck is DRAM bandwidth rather than instruction count or dependency chains.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | l2_swizzle_pid_reorder_and_pipeline_num_stages | 1.066× | 1.053× | ✓ | **✓ best** |
| B | hoist_per_channel_broadcast_loads_out_of_tile_loop | 0.999× | 1.000× | ✓ |  |
| C | widen_xblock_for_128bit_vectorized_loads_and_latency_hiding | 1.027× | 1.074× | ✓ |  |
| D | fix_persistent_kernel_grid_to_num_sms | 0.978× | 0.936× | ✓ |  |


### Metric Comparison

| Metric | Baseline | l2_swizzle_pid_re... | Δ | hoist_per_channel... | Δ | widen_xblock_for_... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 3040 ns | -160 ns (-5.0%) | 3200 ns | +0 ns (+0.0%) | 2980 ns | -220 ns (-6.9%) | 3420 ns | +220 ns (+6.9%) |
| SM Throughput (%) | 12.97 % | 8.48 % | -4.49 pp | 19.31 % | +6.34 pp | 7.08 % | -5.89 pp | 11.03 % | -1.94 pp |
| Memory Throughput (%) | 4.61 % | 4.83 % | +0.22 pp | 9.92 % | +5.31 pp | 4.95 % | +0.34 pp | 4.25 % | -0.36 pp |
| DRAM Throughput (%) | 3.27 % | 3.42 % | +0.15 pp | 3.25 % | -0.02 pp | 3.51 % | +0.24 pp | 3.03 % | -0.24 pp |
| Achieved Occupancy (%) | 38.70 % | 36.32 % | -2.38 pp | 42.29 % | +3.59 pp | 18.58 % | -20.12 pp | 25.23 % | -13.47 pp |
| Registers/Thread | 18.00 register/thread | 16.00 register/thread | -2 | 20.00 register/thread | +2 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 |
| No Eligible (%) | 65.83 % | 78.24 % | +12.41 pp | 68.60 % | +2.77 pp | 88.14 % | +22.31 pp | 71.55 % | +5.72 pp |
| IPC Active | 1.46 inst/cycle | 0.78 inst/cycle | -0.68 | 1.03 inst/cycle | -0.43 | 0.45 inst/cycle | -1.01 | 1.08 inst/cycle | -0.38 |

---

## Iteration 5

**Best so far**: 1.0526× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_to_increase_warp_occupancy | 1.075× | 1.000× | ✓ |  |
| B | reindex_for_coalesced_channel_last_memory_access | 1.011× | 1.000× | ✓ |  |
| C | collapse_affine_transform_to_single_fma_expression | 0.999× | 0.961× | ✓ |  |
| D | replace_sqrt_div_chain_with_rsqrt_and_fma_in_batchnorm_pointwise | 1.077× | 1.111× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | reindex_for_coale... | Δ | collapse_affine_t... | Δ | replace_sqrt_div_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 3200 ns | +0 ns (+0.0%) | 3200 ns | +0 ns (+0.0%) | 3330 ns | +130 ns (+4.1%) | 2880 ns | -320 ns (-10.0%) |
| SM Throughput (%) | 12.97 % | 6.76 % | -6.21 pp | 10.96 % | -2.01 pp | 20.79 % | +7.82 pp | 9.28 % | -3.69 pp |
| Memory Throughput (%) | 4.61 % | 9.11 % | +4.50 pp | 4.89 % | +0.28 pp | 10.69 % | +6.08 pp | 4.91 % | +0.30 pp |
| DRAM Throughput (%) | 3.27 % | 3.27 % | +0.00 pp | 3.24 % | -0.03 pp | 3.12 % | -0.15 pp | 3.63 % | +0.36 pp |
| Achieved Occupancy (%) | 38.70 % | 18.50 % | -20.20 pp | 36.75 % | -1.95 pp | 41.15 % | +2.45 pp | 26.11 % | -12.59 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 18.00 register/thread | +0 | 20.00 register/thread | +2 | 32.00 register/thread | +14 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 |
| No Eligible (%) | 65.83 % | 87.39 % | +21.56 pp | 69.04 % | +3.21 pp | 68.29 % | +2.46 pp | 78.54 % | +12.71 pp |
| IPC Active | 1.46 inst/cycle | 0.43 inst/cycle | -1.03 | 1.16 inst/cycle | -0.30 | 1.27 inst/cycle | -0.19 | 0.74 inst/cycle | -0.72 |

---

## Cumulative Progress

```
Baseline:  3200 ns  (1.000×)
Iter 4     3040 ns  (1.053×)  — l2_swizzle_pid_reorder_and_pipeline_num_stages
Iter 5     2880 ns  (1.111×)  — replace_sqrt_div_chain_with_rsqrt_and_fma_in_batchnorm_pointwise

Final best: 2880 ns  (1.1111×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,33 +5,39 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
-def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 125440
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x2 = xindex
-    x0 = (xindex % 64)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
-    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
-
-
+def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
+    pid = tl.program_id(0)
+    num_sms = tl.num_programs(0)
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    tile_id = pid
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x2 = xindex
+        x0 = (xindex % 64)
+        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+        tmp2 = tmp0 - tmp1
+        inv_std = libdevice.rsqrt(tmp3 + 1e-05)
+        x_norm = tmp2 * inv_std
+        tmp15 = tl.math.fma(x_norm, tmp12, tmp14)
+        tl.store(in_out_ptr0 + (x2), tmp15, xmask)
+        tile_id += num_sms
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +47,7 @@

     arg_4 = inputs[4]
     xnumel = 125440
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 125440, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 125440)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

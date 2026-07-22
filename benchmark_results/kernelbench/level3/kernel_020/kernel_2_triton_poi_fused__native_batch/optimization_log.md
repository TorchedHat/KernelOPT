# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21890 ns — **Best latency**: 21890 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 14.85
- **Registers/thread**: 17
- **Duration**: 21890 ns
- **SM Throughput**: 54.05 % | **Memory**: 23.64 % | **DRAM**: 14.95 %
- **Achieved Occupancy**: 56.05 %
- **IPC Active**: 2.41 inst/cycle | **No Eligible**: 39.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_for_better_l1tex_reuse | 0.919× | 0.907× | ✓ |  |
| B | autotune_xblock_to_hide_l1tex_latency | 1.000× | 0.997× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads | 0.735× | 0.713× | ✓ |  |
| D | autotune_xblock_increase_occupancy_hide_l1tex_latency | 0.989× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ | add_autotune_with... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21890 ns | 24130 ns | +2240 ns (+10.2%) | 21950 ns | +60 ns (+0.3%) | 30690 ns | +8800 ns (+40.2%) | 21920 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 54.05 % | 74.61 % | +20.56 pp | 63.59 % | +9.54 pp | 82.58 % | +28.53 pp | 63.56 % | +9.51 pp |
| Memory Throughput (%) | 23.64 % | 28.72 % | +5.08 pp | 23.84 % | +0.20 pp | 24.76 % | +1.12 pp | 23.86 % | +0.22 pp |
| DRAM Throughput (%) | 14.95 % | 13.56 % | -1.39 pp | 14.92 % | -0.03 pp | 10.67 % | -4.28 pp | 14.93 % | -0.02 pp |
| Achieved Occupancy (%) | 56.05 % | 65.50 % | +9.45 pp | 60.03 % | +3.98 pp | 86.77 % | +30.72 pp | 59.81 % | +3.76 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 39.20 % | 15.51 % | -23.69 pp | 28.58 % | -10.62 pp | 11.30 % | -27.90 pp | 28.53 % | -10.67 pp |
| IPC Active | 2.41 inst/cycle | 3.37 inst/cycle | +0.96 | 2.83 inst/cycle | +0.42 | 3.52 inst/cycle | +1.11 | 2.85 inst/cycle | +0.44 |


### Key Learnings

#### ❌ Plan C: `add_autotune_with_larger_xblock_for_vectorized_loads` (0.713×)

Increasing XBLOCK from 128 to 512–2048 via autotune doubled register usage (17→32 registers/thread) and more than doubled total executed instructions (9.3M→19.9M), causing a 40% slowdown despite improving achieved occupancy (56%→87%) and compute throughput (54%→83%). The L1/TEX hit rate dropped from 83% to 66%, indicating that larger tiles caused cache thrashing on the per-channel broadcast tensors (size 32) accessed via modulo indexing. The net effect was that the extra instruction count and cache pressure outweighed the latency-hiding benefit of higher occupancy, resulting in a longer kernel duration (21.9 µs→30.7 µs). When a kernel already has high L1 hit rate on small broadcast tensors and moderate compute throughput, increasing tile size can destroy cache locality and inflate instruction count without proportional throughput gain.

> **Applicable when**: Never beneficial in this configuration; generally avoid when small broadcast tensors (e.g., per-channel BN params) are accessed via modulo indexing and already have high L1 hit rate (>80%), and when register count is near the block-limit-register boundary.

> **Avoid when**: Kernel has high L1/TEX hit rate (>75%) on small broadcast tensors accessed via modulo; kernel is already achieving moderate compute throughput (>50%); increasing XBLOCK would double register usage and push block-limit-registers lower, reducing effective occupancy ceiling; total instruction count scales super-linearly with XBLOCK due to predicated/masked operations.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 0.999× | 0.982× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_stall_latency | 0.999× | 0.999× | ✓ |  |
| C | replace_integer_reciprocal_with_fp32_rsqrt_and_fuse_multiply_add_chain | 0.994× | 1.000× | ✓ |  |
| D | replace_integer_reciprocal_with_rsqrt_to_enable_fma_fusion | 0.917× | 0.937× | ✓ |  |
| E | add_triton_autotune | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | replace_integer_r... | Δ | replace_integer_r... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21890 ns | 22300 ns | +410 ns (+1.9%) | 21920 ns | +30 ns (+0.1%) | 21890 ns | +0 ns (+0.0%) | 23360 ns | +1470 ns (+6.7%) | 21890 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 54.05 % | 46.33 % | -7.72 pp | 63.49 % | +9.44 pp | 43.69 % | -10.36 pp | 51.05 % | -3.00 pp | 63.28 % | +9.23 pp |
| Memory Throughput (%) | 23.64 % | 21.04 % | -2.60 pp | 23.80 % | +0.16 pp | 31.71 % | +8.07 pp | 34.59 % | +10.95 pp | 23.73 % | +0.09 pp |
| DRAM Throughput (%) | 14.95 % | 14.68 % | -0.27 pp | 14.94 % | -0.01 pp | 14.95 % | +0.00 pp | 14.01 % | -0.94 pp | 14.95 % | +0.00 pp |
| Achieved Occupancy (%) | 56.05 % | 38.19 % | -17.86 pp | 60.47 % | +4.42 pp | 31.95 % | -24.10 pp | 38.45 % | -17.60 pp | 59.80 % | +3.75 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 14.85 | 7.42 | -7.43 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 39.20 % | 45.07 % | +5.87 pp | 28.69 % | -10.51 pp | 47.73 % | +8.53 pp | 41.11 % | +1.91 pp | 28.46 % | -10.74 pp |
| IPC Active | 2.41 inst/cycle | 2.17 inst/cycle | -0.24 | 2.84 inst/cycle | +0.43 | 2.05 inst/cycle | -0.36 | 2.34 inst/cycle | -0.07 | 2.85 inst/cycle | +0.44 |

---

## Cumulative Progress

```
Baseline:  21890 ns  (1.000×)

Final best: 21890 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,19 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=32),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 4014080
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 32)
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
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -33,9 +43,7 @@

     tmp17 = tl.maximum(tmp15, tmp16)
     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -45,7 +53,7 @@

     arg_4 = inputs[4]
     xnumel = 4014080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4014080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4014080)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

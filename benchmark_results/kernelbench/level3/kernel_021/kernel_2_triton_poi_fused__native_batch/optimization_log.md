# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1620000 ns — **Best latency**: 1620000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2634240,1,1), Waves/SM: 1247.27
- **Registers/thread**: 17
- **Duration**: 1620000 ns
- **SM Throughput**: 60.06 % | **Memory**: 33.48 % | **DRAM**: 33.48 %
- **Achieved Occupancy**: 61.15 %
- **IPC Active**: 2.40 inst/cycle | **No Eligible**: 39.91 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_fixed_1024_for_latency_hiding_and_reduced_grid | 1.000× | 1.000× | ✓ |  |
| B | fix_xblock_grid_mismatch_and_increase_tile_size | 1.000× | 1.000× | ✓ |  |
| C | increase_xblock_with_vectorized_loads | 0.999× | 0.994× | ✓ |  |
| D | increase_xblock_static_with_corrected_grid_formula | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_f... | Δ | fix_xblock_grid_m... | Δ | increase_xblock_w... | Δ | increase_xblock_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1620000 ns | 1620000 ns | +0 ns (+0.0%) | 1620000 ns | +0 ns (+0.0%) | 1630000 ns | +10000 ns (+0.6%) | 1620000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 60.06 % | 59.96 % | -0.10 pp | 59.79 % | -0.27 pp | 70.42 % | +10.36 pp | 59.83 % | -0.23 pp |
| Memory Throughput (%) | 33.48 % | 33.48 % | +0.00 pp | 33.46 % | -0.02 pp | 33.45 % | -0.03 pp | 33.47 % | -0.01 pp |
| DRAM Throughput (%) | 33.48 % | 33.48 % | +0.00 pp | 33.46 % | -0.02 pp | 33.45 % | -0.03 pp | 33.47 % | -0.01 pp |
| Achieved Occupancy (%) | 61.15 % | 61.14 % | -0.01 pp | 60.99 % | -0.16 pp | 65.24 % | +4.09 pp | 60.96 % | -0.19 pp |
| Registers/Thread | 17.00 register/thread | 17.00 register/thread | +0 | 17.00 register/thread | +0 | 18.00 register/thread | +1 | 17.00 register/thread | +0 |
| Waves/SM | 1247.27 | 1247.27 | +0.00 | 1247.27 | +0.00 | 1247.27 | +0.00 | 1247.27 | +0.00 |
| No Eligible (%) | 39.91 % | 40.11 % | +0.20 pp | 40.12 % | +0.21 pp | 29.50 % | -10.41 pp | 40.08 % | +0.17 pp |
| IPC Active | 2.40 inst/cycle | 2.40 inst/cycle | +0.00 | 2.39 inst/cycle | -0.01 | 2.83 inst/cycle | +0.43 | 2.40 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_fixed_xblock_512_no_autotune | 1.000× | 1.000× | ✓ |  |
| B | increase_num_warps_at_launch_to_hide_l1tex_latency | 1.000× | 1.000× | ✓ |  |
| C | increase_xblock_fixed_with_corrected_num_warps_no_autotune | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1620000 ns | 1620000 ns | +0 ns (+0.0%) | 1620000 ns | +0 ns (+0.0%) | 1620000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 60.06 % | 60.02 % | -0.04 pp | 60.05 % | -0.01 pp | 60.01 % | -0.05 pp |
| Memory Throughput (%) | 33.48 % | 33.47 % | -0.01 pp | 33.46 % | -0.02 pp | 33.48 % | +0.00 pp |
| DRAM Throughput (%) | 33.48 % | 33.47 % | -0.01 pp | 33.46 % | -0.02 pp | 33.48 % | +0.00 pp |
| Achieved Occupancy (%) | 61.15 % | 60.95 % | -0.20 pp | 60.96 % | -0.19 pp | 61.00 % | -0.15 pp |
| Registers/Thread | 17.00 register/thread | 17.00 register/thread | +0 | 17.00 register/thread | +0 | 17.00 register/thread | +0 |
| Waves/SM | 1247.27 | 1247.27 | +0.00 | 1247.27 | +0.00 | 1247.27 | +0.00 |
| No Eligible (%) | 39.91 % | 39.91 % | +0.00 pp | 39.88 % | -0.03 pp | 39.90 % | -0.01 pp |
| IPC Active | 2.40 inst/cycle | 2.40 inst/cycle | +0.00 | 2.40 inst/cycle | +0.00 | 2.40 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_static_512_corrected_grid_no_autotune_no_shadow | 1.000× | 1.000× | ✓ |  |
| B | increase_xblock_static_512_corrected_num_warps_16_no_autotune | 1.000× | 1.000× | ✓ | **✓ best** |
| C | increase_xblock_fixed_512_corrected_grid_num_warps_16 | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_s... | Δ | increase_xblock_s... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1620000 ns | 1620000 ns | +0 ns (+0.0%) | 1620000 ns | +0 ns (+0.0%) | 1620000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 60.06 % | 59.81 % | -0.25 pp | 60.03 % | -0.03 pp | 71.25 % | +11.19 pp |
| Memory Throughput (%) | 33.48 % | 33.48 % | +0.00 pp | 33.47 % | -0.01 pp | 33.47 % | -0.01 pp |
| DRAM Throughput (%) | 33.48 % | 33.48 % | +0.00 pp | 33.47 % | -0.01 pp | 33.47 % | -0.01 pp |
| Achieved Occupancy (%) | 61.15 % | 61.21 % | +0.06 pp | 61.02 % | -0.13 pp | 65.54 % | +4.39 pp |
| Registers/Thread | 17.00 register/thread | 17.00 register/thread | +0 | 17.00 register/thread | +0 | 18.00 register/thread | +1 |
| Waves/SM | 1247.27 | 1247.27 | +0.00 | 1247.27 | +0.00 | 1247.27 | +0.00 |
| No Eligible (%) | 39.91 % | 40.11 % | +0.20 pp | 40.03 % | +0.12 pp | 28.66 % | -11.25 pp |
| IPC Active | 2.40 inst/cycle | 2.41 inst/cycle | +0.01 | 2.40 inst/cycle | +0.00 | 2.85 inst/cycle | +0.45 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_streaming_load_plus_rsqrt_fusion | 1.002× | 1.000× | ✓ | **✓ best** |
| B | fuse_fp32_multiply_add_chain_and_set_xblock_256 | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | fuse_fp32_multipl... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1620000 ns | 1620000 ns | +0 ns (+0.0%) | 1620000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 60.06 % | 39.41 % | -20.65 pp | 60.00 % | -0.06 pp |
| Memory Throughput (%) | 33.48 % | 33.54 % | +0.06 pp | 33.48 % | +0.00 pp |
| DRAM Throughput (%) | 33.48 % | 33.54 % | +0.06 pp | 33.48 % | +0.00 pp |
| Achieved Occupancy (%) | 61.15 % | 45.81 % | -15.34 pp | 61.14 % | -0.01 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 17.00 register/thread | +0 |
| Waves/SM | 1247.27 | 1247.27 | +0.00 | 1247.27 | +0.00 |
| No Eligible (%) | 39.91 % | 62.18 % | +22.27 pp | 39.88 % | -0.03 pp |
| IPC Active | 2.40 inst/cycle | 1.52 inst/cycle | -0.88 | 2.40 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  1620000 ns  (1.000×)

Final best: 1620000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 337182720
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x2 = xindex
     x0 = (xindex % 672)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
+    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
     tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = tl.math.rsqrt(tmp3 + 1e-05)
     tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    tmp15 = tl.math.fma(tmp11, tmp12, tmp14)
     tmp16 = tl.full([1], 0.0, tl.float32)
     tmp17 = tl.maximum(tmp15, tmp16)
     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
     tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

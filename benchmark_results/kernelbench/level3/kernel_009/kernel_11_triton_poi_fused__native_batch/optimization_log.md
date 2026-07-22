# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3520 ns — **Best latency**: 3580 ns (**0.9832×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_11_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 22
- **Duration**: 3520 ns
- **SM Throughput**: 17.36 % | **Memory**: 9.32 % | **DRAM**: 9.32 %
- **Achieved Occupancy**: 52.98 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 54.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size_for_underutilized_pointwise_kernel | 0.976× | 1.017× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_better_wave_utilization | 0.965× | 0.457× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_better_sm_utilization | 1.005× | 0.472× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | increase_xblock_f... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 3460 ns | -60 ns (-1.7%) | 7710 ns | +4190 ns (+119.0%) | 7460 ns | +3940 ns (+111.9%) |
| SM Throughput (%) | 17.36 % | 26.34 % | +8.98 pp | 59.40 % | +42.04 pp | 60.31 % | +42.95 pp |
| Memory Throughput (%) | 9.32 % | 10.10 % | +0.78 pp | 13.67 % | +4.35 pp | 14.06 % | +4.74 pp |
| DRAM Throughput (%) | 9.32 % | 9.61 % | +0.29 pp | 4.29 % | -5.03 pp | 4.42 % | -4.90 pp |
| Achieved Occupancy (%) | 52.98 % | 53.98 % | +1.00 pp | 41.63 % | -11.35 pp | 41.33 % | -11.65 pp |
| Registers/Thread | 22.00 register/thread | 28.00 register/thread | +6 | 58.00 register/thread | +36 | 58.00 register/thread | +36 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 2.97 | +2.23 | 2.97 | +2.23 |
| No Eligible (%) | 54.08 % | 35.90 % | -18.18 pp | 18.71 % | -35.37 pp | 17.46 % | -36.62 pp |
| IPC Active | 1.67 inst/cycle | 2.49 inst/cycle | +0.82 | 3.23 inst/cycle | +1.56 | 3.28 inst/cycle | +1.61 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_configs_to_safe_xblock_with_num_warps_and_num_stages_sweep | 1.000× | 0.917× | ✓ |  |
| B | reduce_register_pressure_via_maxnreg_and_fused_rsqrt | 1.048× | 0.983× | ✓ | **✓ best** |
| C | reduce_register_pressure_via_maxnreg_to_restore_occupancy | 0.979× | 0.917× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_conf... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 3840 ns | +320 ns (+9.1%) | 3580 ns | +60 ns (+1.7%) | 3840 ns | +320 ns (+9.1%) |
| SM Throughput (%) | 17.36 % | 19.35 % | +1.99 pp | 15.19 % | -2.17 pp | 19.25 % | +1.89 pp |
| Memory Throughput (%) | 9.32 % | 8.63 % | -0.69 pp | 9.24 % | -0.08 pp | 8.59 % | -0.73 pp |
| DRAM Throughput (%) | 9.32 % | 8.63 % | -0.69 pp | 9.24 % | -0.08 pp | 8.59 % | -0.73 pp |
| Achieved Occupancy (%) | 52.98 % | 53.62 % | +0.64 pp | 55.66 % | +2.68 pp | 55.23 % | +2.25 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 21.00 register/thread | -1 | 22.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 54.08 % | 52.07 % | -2.01 pp | 73.35 % | +19.27 pp | 52.96 % | -1.12 pp |
| IPC Active | 1.67 inst/cycle | 1.85 inst/cycle | +0.18 | 1.27 inst/cycle | -0.40 | 1.79 inst/cycle | +0.12 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_stages_for_l1tex_latency_hiding_without_register_pressure | 1.003× | 0.949× | ✓ |  |
| B | replace_sqrt_int_div_with_rsqrt_to_reduce_nonfused_fp_instructions | 1.005× | 0.983× | ✓ |  |
| C | fuse_fp32_instructions_via_rsqrt_and_eliminate_redundant_multiply | 0.992× | 0.957× | ✓ |  |
| D | add_triton_autotune | 1.000× | 0.972× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | replace_sqrt_int_... | Δ | fuse_fp32_instruc... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 3710 ns | +190 ns (+5.4%) | 3580 ns | +60 ns (+1.7%) | 3680 ns | +160 ns (+4.5%) | 3620 ns | +100 ns (+2.8%) |
| SM Throughput (%) | 17.36 % | 15.26 % | -2.10 pp | 15.23 % | -2.13 pp | 15.44 % | -1.92 pp | 14.51 % | -2.85 pp |
| Memory Throughput (%) | 9.32 % | 8.82 % | -0.50 pp | 9.25 % | -0.07 pp | 8.98 % | -0.34 pp | 9.24 % | -0.08 pp |
| DRAM Throughput (%) | 9.32 % | 8.82 % | -0.50 pp | 9.25 % | -0.07 pp | 8.98 % | -0.34 pp | 9.24 % | -0.08 pp |
| Achieved Occupancy (%) | 52.98 % | 55.89 % | +2.91 pp | 54.79 % | +1.81 pp | 58.40 % | +5.42 pp | 55.62 % | +2.64 pp |
| Registers/Thread | 22.00 register/thread | 21.00 register/thread | -1 | 21.00 register/thread | -1 | 21.00 register/thread | -1 | 21.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 54.08 % | 66.30 % | +12.22 pp | 67.96 % | +13.88 pp | 67.26 % | +13.18 pp | 66.89 % | +12.81 pp |
| IPC Active | 1.67 inst/cycle | 1.28 inst/cycle | -0.39 | 1.29 inst/cycle | -0.38 | 1.14 inst/cycle | -0.53 | 1.27 inst/cycle | -0.40 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | apply_evict_first_to_streaming_residual_load_to_reduce_l1tex_stalls | 0.983× | 0.924× | ✓ |  |
| B | set_evict_first_on_streaming_loads_to_protect_bn_params_in_l1 | 1.002× | 0.983× | ✓ |  |
| C | apply_evict_first_to_streaming_tensor_loads_to_reduce_l1tex_stalls | 1.007× | 0.964× | ✓ |  |
| D | evict_first_for_streaming_loads_to_improve_channel_cache_reuse | 1.020× | 1.038× | ✓ |  |


### Metric Comparison

| Metric | Baseline | apply_evict_first... | Δ | set_evict_first_o... | Δ | apply_evict_first... | Δ | evict_first_for_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 3810 ns | +290 ns (+8.2%) | 3580 ns | +60 ns (+1.7%) | 3650 ns | +130 ns (+3.7%) | 3390 ns | -130 ns (-3.7%) |
| SM Throughput (%) | 17.36 % | 15.07 % | -2.29 pp | 15.06 % | -2.30 pp | 15.01 % | -2.35 pp | 16.04 % | -1.32 pp |
| Memory Throughput (%) | 9.32 % | 8.68 % | -0.64 pp | 9.17 % | -0.15 pp | 9.02 % | -0.30 pp | 9.76 % | +0.44 pp |
| DRAM Throughput (%) | 9.32 % | 8.68 % | -0.64 pp | 9.17 % | -0.15 pp | 9.02 % | -0.30 pp | 9.76 % | +0.44 pp |
| Achieved Occupancy (%) | 52.98 % | 55.55 % | +2.57 pp | 55.51 % | +2.53 pp | 56.20 % | +3.22 pp | 51.36 % | -1.62 pp |
| Registers/Thread | 22.00 register/thread | 21.00 register/thread | -1 | 21.00 register/thread | -1 | 21.00 register/thread | -1 | 20.00 register/thread | -2 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 54.08 % | 65.50 % | +11.42 pp | 65.79 % | +11.71 pp | 64.82 % | +10.74 pp | 65.23 % | +11.15 pp |
| IPC Active | 1.67 inst/cycle | 1.39 inst/cycle | -0.28 | 1.36 inst/cycle | -0.31 | 1.32 inst/cycle | -0.35 | 1.52 inst/cycle | -0.15 |

---

## Cumulative Progress

```
Baseline:  3520 ns  (1.000×)

Final best: 3580 ns  (0.9832×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,43 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 200704
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 128)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x2), None)
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
     tmp17 = tmp15 + tmp16
     tmp18 = tl.full([1], 0, tl.int32)
     tmp19 = tl.maximum(tmp18, tmp17)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +52,7 @@

     arg_5 = inputs[5]
     xnumel = 200704
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 200704, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 200704)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

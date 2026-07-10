# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_convolution_hardswish_mul_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1290000 ns — **Best latency**: 1290000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_convolution_hardswish_mul_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_026/kernel_0_triton_poi_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2097152,1,1), Waves/SM: 992.97
- **Registers/thread**: 16
- **Duration**: 1290000 ns
- **SM Throughput**: 32.05 % | **Memory**: 50.47 % | **DRAM**: 50.47 %
- **Achieved Occupancy**: 50.07 %
- **IPC Active**: 1.28 inst/cycle | **No Eligible**: 67.93 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_and_multiple_of_hints | 1.000× | 1.000× | ✓ |  |
| B | hardcode_larger_xblock_no_autotune | 1.000× | 1.000× | ✓ | **✓ best** |
| C | increase_xblock_and_num_warps_direct_no_autotune | 1.000× | 1.000× | ✓ |  |
| D | increase_xblock_direct_hardcode_no_autotune | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | hardcode_larger_x... | Δ | increase_xblock_a... | Δ | increase_xblock_d... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1290000 ns | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 32.05 % | 32.01 % | -0.04 pp | 31.97 % | -0.08 pp | 31.95 % | -0.10 pp | 31.97 % | -0.08 pp |
| Memory Throughput (%) | 50.47 % | 50.45 % | -0.02 pp | 50.42 % | -0.05 pp | 50.46 % | -0.01 pp | 50.44 % | -0.03 pp |
| DRAM Throughput (%) | 50.47 % | 50.45 % | -0.02 pp | 50.42 % | -0.05 pp | 50.46 % | -0.01 pp | 50.44 % | -0.03 pp |
| Achieved Occupancy (%) | 50.07 % | 50.97 % | +0.90 pp | 50.97 % | +0.90 pp | 50.95 % | +0.88 pp | 51.02 % | +0.95 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 992.97 | 992.97 | +0.00 | 992.97 | +0.00 | 992.97 | +0.00 | 992.97 | +0.00 |
| No Eligible (%) | 67.93 % | 67.96 % | +0.03 pp | 67.98 % | +0.05 pp | 67.92 % | -0.01 pp | 67.96 % | +0.03 pp |
| IPC Active | 1.28 inst/cycle | 1.28 inst/cycle | +0.00 | 1.28 inst/cycle | +0.00 | 1.28 inst/cycle | +0.00 | 1.28 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_via_larger_block_size_no_autotune | 1.000× | 1.000× | ✓ |  |
| B | increase_num_warps_and_fixed_xblock_no_autotune | 1.000× | 1.000× | ✓ |  |
| C | streaming_evict_first_cache_policy_for_large_tensor_loads | 1.000× | 1.000× | ✓ |  |
| D | increase_num_warps_for_latency_hiding | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | streaming_evict_f... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1290000 ns | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 32.05 % | 32.05 % | +0.00 pp | 32.08 % | +0.03 pp | 35.32 % | +3.27 pp | 32.03 % | -0.02 pp |
| Memory Throughput (%) | 50.47 % | 50.47 % | +0.00 pp | 50.47 % | +0.00 pp | 50.42 % | -0.05 pp | 50.49 % | +0.02 pp |
| DRAM Throughput (%) | 50.47 % | 50.47 % | +0.00 pp | 50.47 % | +0.00 pp | 50.42 % | -0.05 pp | 50.49 % | +0.02 pp |
| Achieved Occupancy (%) | 50.07 % | 51.07 % | +1.00 pp | 50.98 % | +0.91 pp | 50.36 % | +0.29 pp | 51.02 % | +0.95 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 992.97 | 992.97 | +0.00 | 992.97 | +0.00 | 992.97 | +0.00 | 992.97 | +0.00 |
| No Eligible (%) | 67.93 % | 67.90 % | -0.03 pp | 67.96 % | +0.03 pp | 64.61 % | -3.32 pp | 67.90 % | -0.03 pp |
| IPC Active | 1.28 inst/cycle | 1.28 inst/cycle | +0.00 | 1.28 inst/cycle | +0.00 | 1.42 inst/cycle | +0.14 | 1.28 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_instruction_fusion_for_hardswish_computation | 1.000× | 1.000× | ✓ |  |
| B | manual_loop_unroll_multi_element_per_thread | 0.406× | 0.406× | ✓ |  |
| C | fma_fusion_via_arithmetic_rewrite_of_hardswish | 1.000× | 1.000× | ✓ | **✓ best** |
| D | fused_multiply_add_rewrite_for_hardswish_computation | 1.000× | 1.000× | ✓ |  |
| E | add_triton_autotune | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_instruction_f... | Δ | manual_loop_unrol... | Δ | fma_fusion_via_ar... | Δ | fused_multiply_ad... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1290000 ns | 1290000 ns | +0 ns (+0.0%) | 3180000 ns | +1890000 ns (+146.5%) | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 32.05 % | 32.05 % | +0.00 pp | 73.45 % | +41.40 pp | 35.34 % | +3.29 pp | 32.05 % | +0.00 pp | 36.07 % | +4.02 pp |
| Memory Throughput (%) | 50.47 % | 50.48 % | +0.01 pp | 43.53 % | -6.94 pp | 50.42 % | -0.05 pp | 50.49 % | +0.02 pp | 50.47 % | +0.00 pp |
| DRAM Throughput (%) | 50.47 % | 50.48 % | +0.01 pp | 20.48 % | -29.99 pp | 50.42 % | -0.05 pp | 50.49 % | +0.02 pp | 50.47 % | +0.00 pp |
| Achieved Occupancy (%) | 50.07 % | 50.90 % | +0.83 pp | 91.71 % | +41.64 pp | 50.31 % | +0.24 pp | 50.79 % | +0.72 pp | 51.41 % | +1.34 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 24.00 register/thread | +8 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 992.97 | 992.97 | +0.00 | 992.97 | +0.00 | 992.97 | +0.00 | 992.97 | +0.00 | 992.97 | +0.00 |
| No Eligible (%) | 67.93 % | 67.95 % | +0.02 pp | 26.51 % | -41.42 pp | 64.64 % | -3.29 pp | 67.88 % | -0.05 pp | 63.85 % | -4.08 pp |
| IPC Active | 1.28 inst/cycle | 1.28 inst/cycle | +0.00 | 2.95 inst/cycle | +1.67 | 1.41 inst/cycle | +0.13 | 1.28 inst/cycle | +0.00 | 1.45 inst/cycle | +0.17 |


### Key Learnings

#### ❌ Plan B: `manual_loop_unroll_multi_element_per_thread` (0.406×)

Wrapping a pointwise kernel body in tl.static_range(8) to process 8 chunks per CTA increased executed instructions by 5.65x (327M → 1847M) and duration by 2.46x (1.29ms → 3.18ms), despite compute SM throughput rising from 32% to 73%. The unrolled loop caused the compiler to emit 8 independent copies of all arithmetic (including tl.full scalar constants and non-fused FP32 ops), inflating instruction count far beyond the latency-hiding benefit. DRAM throughput dropped from 50% to 20%, indicating the kernel became compute-bound on redundant arithmetic rather than memory-bound, and the 8x instruction expansion overwhelmed any latency-hiding gain from increased in-flight loads.

> **Applicable when**: Never — this pattern caused a 2.46x regression. Do not apply tl.static_range unrolling to pointwise kernels where the loop body contains tl.full scalar constant allocations and multiple non-fused FP32 instructions, as the compiler emits all instructions N times rather than hoisting constants.

> **Avoid when**: Any pointwise Triton kernel where the loop body contains tl.full([1], ...) scalar constants, multiple FP32 arithmetic ops, or non-fused instructions. The static unroll multiplies total instruction count by the unroll factor (5.65x observed for factor=8), making the kernel compute-bound on redundant arithmetic. The kernel becomes duration-limited by instruction throughput rather than memory latency, negating the intended latency-hiding benefit. Also avoid when the baseline is already memory-bandwidth-bound (DRAM throughput >40%) — unrolling shifts the bottleneck to compute without improving memory efficiency.

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | l2_compression_via_evict_first_store_policy | 1.000× | 1.000× | ✓ |  |
| B | eliminate_noop_fma_wrappers_and_fix_store_eviction_for_l2_reuse | 1.000× | 1.000× | ✓ |  |
| C | scalar_bias_load_broadcast_to_eliminate_redundant_l1tex_transactions | 1.000× | 1.000× | ✓ |  |
| D | revert_static_range_loop_and_replace_tl_full_with_python_literals | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | l2_compression_vi... | Δ | eliminate_noop_fm... | Δ | scalar_bias_load_... | Δ | revert_static_ran... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1290000 ns | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) | 1290000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 32.05 % | 36.09 % | +4.04 pp | 35.26 % | +3.21 pp | 32.80 % | +0.75 pp | 32.00 % | -0.05 pp |
| Memory Throughput (%) | 50.47 % | 50.45 % | -0.02 pp | 50.14 % | -0.33 pp | 50.48 % | +0.01 pp | 50.47 % | +0.00 pp |
| DRAM Throughput (%) | 50.47 % | 50.45 % | -0.02 pp | 50.14 % | -0.33 pp | 50.48 % | +0.01 pp | 50.47 % | +0.00 pp |
| Achieved Occupancy (%) | 50.07 % | 49.98 % | -0.09 pp | 49.16 % | -0.91 pp | 48.78 % | -1.29 pp | 50.06 % | -0.01 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 992.97 | 992.97 | +0.00 | 992.97 | +0.00 | 992.97 | +0.00 | 992.97 | +0.00 |
| No Eligible (%) | 67.93 % | 63.85 % | -4.08 pp | 64.68 % | -3.25 pp | 72.81 % | +4.88 pp | 67.88 % | -0.05 pp |
| IPC Active | 1.28 inst/cycle | 1.45 inst/cycle | +0.17 | 1.41 inst/cycle | +0.13 | 1.09 inst/cycle | -0.19 | 1.28 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  1290000 ns  (1.000×)

Final best: 1290000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,11 +5,12 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_add_convolution_hardswish_mul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
     xnumel = 268435456
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK)[:], XBLOCK), XBLOCK)
     xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x3 = xindex
     x1 = ((xindex // 32768) % 64)
@@ -18,19 +19,15 @@

     tmp3 = tl.load(in_ptr1 + (x3), None)
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 + tmp3
-    tmp5 = tl.full([1], 3.0, tl.float32)
-    tmp6 = tmp4 + tmp5
-    tmp7 = tl.full([1], 0.0, tl.float32)
-    tmp8 = tl.maximum(tmp6, tmp7)
-    tmp9 = tl.full([1], 6.0, tl.float32)
-    tmp10 = tl.minimum(tmp8, tmp9)
-    tmp11 = tmp4 * tmp10
-    tmp12 = tl.full([1], 0.16666666666666666, tl.float32)
-    tmp13 = tmp11 * tmp12
-    tmp14 = tmp4 * tmp13
+    # Compute clamp(x+3, 0, 6) using inline constants
+    tmp_clamp = tl.minimum(tl.maximum(tmp4 + 3.0, 0.0), 6.0)
+    # Scale: clamp/6
+    tmp_scale = tmp_clamp * 0.16666666666666666
+    # Original semantic: tmp14 = tmp4 * (tmp4 * tmp_scale) = tmp4^2 * clamp/6
+    # Fuse both multiplications into PTX fma.rn.f32 instructions
+    tmp_inner = libdevice.fma(tmp4, tmp_scale, 0.0)   # tmp4 * (clamp/6) + 0
+    tmp14 = libdevice.fma(tmp4, tmp_inner, 0.0)        # tmp4 * tmp_inner + 0
     tl.store(in_out_ptr0 + (x3), tmp14, None)
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

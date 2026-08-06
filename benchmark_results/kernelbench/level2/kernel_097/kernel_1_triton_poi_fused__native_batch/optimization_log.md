# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_addmm_div_mul_sigmoid_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 43230 ns — **Best latency**: 42880 ns (**1.0082×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_autotune_grid_lambda_and_sweep_num_warps_for_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_addmm_div_mul_sigmoid_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_097/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 22
- **Duration**: 43230 ns
- **SM Throughput**: 71.80 % | **Memory**: 31.02 % | **DRAM**: 19.24 %
- **Achieved Occupancy**: 67.69 %
- **IPC Active**: 3.02 inst/cycle | **No Eligible**: 23.16 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_better_occupancy_and_cache_reuse | 1.005× | 0.065× | ✓ | **✓ best** |
| B | increase_xblock_tile_size_for_better_l1_reuse_and_occupancy | 0.340× | 0.254× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 43230 ns | 661060 ns | +617830 ns (+1429.2%) | 170300 ns | +127070 ns (+293.9%) |
| SM Throughput (%) | 71.80 % | 89.42 % | +17.62 pp | 89.55 % | +17.75 pp |
| Memory Throughput (%) | 31.02 % | 18.76 % | -12.26 pp | 21.92 % | -9.10 pp |
| DRAM Throughput (%) | 19.24 % | 1.25 % | -17.99 pp | 4.85 % | -14.39 pp |
| Achieved Occupancy (%) | 67.69 % | 45.44 % | -22.25 pp | 51.92 % | -15.77 pp |
| Registers/Thread | 22.00 register/thread | 64.00 register/thread | +42 | 56.00 register/thread | +34 |
| Waves/SM | 31.03 | 248.24 | +217.21 | 55.16 | +24.13 |
| No Eligible (%) | 23.16 % | 10.20 % | -12.96 pp | 9.36 % | -13.80 pp |
| IPC Active | 3.02 inst/cycle | 3.59 inst/cycle | +0.57 | 3.62 inst/cycle | +0.60 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_better_occupancy_and_cache_reuse` (0.065×)

Adding @triton.autotune with larger XBLOCK configs caused the compiler to select a 512-thread block configuration that tripled register usage per thread (22 → 64), cutting theoretical occupancy from 100% to 50% and achieved occupancy from 67.7% to 45.4%. The instruction count exploded 19x (24M → 468M executed instructions) while memory throughput collapsed from 945 GB/s to 61 GB/s, indicating the autotune selected a configuration that caused massive register pressure and likely instruction bloat from predicated/masked loads introduced by the xmask change. The kernel duration increased 15x (43 µs → 661 µs). The root cause is that introducing xmask (replacing tl.full([XBLOCK], True) with xindex < xnumel) combined with autotune's larger block sizes caused the compiler to generate far more instructions and spill registers, overwhelming any cache-reuse benefit.

> **Applicable when**: Never - this is a negative result. Do not apply autotune with larger XBLOCK when the kernel already has low register usage (≤22 regs/thread) and the xnumel is exactly divisible by the current XBLOCK (making masks unnecessary).

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs when: (1) current register count is low (≤24 regs/thread) and larger blocks would cause register pressure; (2) xnumel is exactly divisible by current XBLOCK making xmask unnecessary — adding xmask introduces predication overhead; (3) the grid is not updated to a lambda using META['XBLOCK'] (fixed grid with autotune causes incorrect element coverage). Also avoid when the kernel is already compute-bound (Compute SM Throughput >70%) rather than memory-latency-bound.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_restore_occupancy_and_enable_fma_fusion | 1.000× | 0.942× | ✓ |  |
| B | reduce_register_pressure_via_small_xblock_autotune_with_num_warps_and_num_stages_sweep | 0.995× | 0.942× | ✓ |  |
| D | revert_to_fixed_small_xblock_eliminate_register_pressure_and_instruction_bloat | 1.007× | 1.001× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.006× | 0.949× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | revert_to_fixed_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43230 ns | 45890 ns | +2660 ns (+6.2%) | 45890 ns | +2660 ns (+6.2%) | 43200 ns | -30 ns (-0.1%) | 45540 ns | +2310 ns (+5.3%) |
| SM Throughput (%) | 71.80 % | 81.08 % | +9.28 pp | 80.89 % | +9.09 pp | 73.32 % | +1.52 pp | 81.01 % | +9.21 pp |
| Memory Throughput (%) | 31.02 % | 29.21 % | -1.81 pp | 29.11 % | -1.91 pp | 31.78 % | +0.76 pp | 29.17 % | -1.85 pp |
| DRAM Throughput (%) | 19.24 % | 18.13 % | -1.11 pp | 18.34 % | -0.90 pp | 19.24 % | +0.00 pp | 18.30 % | -0.94 pp |
| Achieved Occupancy (%) | 67.69 % | 83.44 % | +15.75 pp | 83.46 % | +15.77 pp | 68.67 % | +0.98 pp | 83.76 % | +16.07 pp |
| Registers/Thread | 22.00 register/thread | 24.00 register/thread | +2 | 24.00 register/thread | +2 | 22.00 register/thread | +0 | 24.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 23.16 % | 14.54 % | -8.62 pp | 14.71 % | -8.45 pp | 24.71 % | +1.55 pp | 14.58 % | -8.58 pp |
| IPC Active | 3.02 inst/cycle | 3.41 inst/cycle | +0.39 | 3.42 inst/cycle | +0.40 | 3.02 inst/cycle | +0.00 | 3.42 inst/cycle | +0.40 |

---

## Iteration 3

**Best so far**: 1.0007× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | enable_fma_fusion_via_rsqrt_and_fix_scalar_broadcast_load | 1.010× | 1.008× | ✓ |  |
| C | replace_sqrt_reciprocal_with_rsqrt_to_enable_fma_fusion | 1.010× | 1.008× | ✓ | **✓ best** |
| D | fma_fusion_via_arithmetic_simplification | 1.006× | 0.944× | ✓ |  |
| E | add_triton_autotune | 1.006× | 0.946× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_fma_fusion... | Δ | replace_sqrt_reci... | Δ | fma_fusion_via_ar... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 43230 ns | 42910 ns | -320 ns (-0.7%) | 42880 ns | -350 ns (-0.8%) | 45820 ns | +2590 ns (+6.0%) | 45700 ns | +2470 ns (+5.7%) |
| SM Throughput (%) | 71.80 % | 61.83 % | -9.97 pp | 61.31 % | -10.49 pp | 81.51 % | +9.71 pp | 81.29 % | +9.49 pp |
| Memory Throughput (%) | 31.02 % | 31.15 % | +0.13 pp | 30.95 % | -0.07 pp | 29.40 % | -1.62 pp | 29.27 % | -1.75 pp |
| DRAM Throughput (%) | 19.24 % | 19.38 % | +0.14 pp | 19.34 % | +0.10 pp | 18.12 % | -1.12 pp | 18.22 % | -1.02 pp |
| Achieved Occupancy (%) | 67.69 % | 49.73 % | -17.96 pp | 49.55 % | -18.14 pp | 83.17 % | +15.48 pp | 83.90 % | +16.21 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 22.00 register/thread | +0 | 24.00 register/thread | +2 | 24.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 23.16 % | 34.76 % | +11.60 pp | 34.68 % | +11.52 pp | 14.86 % | -8.30 pp | 14.89 % | -8.27 pp |
| IPC Active | 3.02 inst/cycle | 2.61 inst/cycle | -0.41 | 2.61 inst/cycle | -0.41 | 3.39 inst/cycle | +0.37 | 3.42 inst/cycle | +0.40 |

---

## Iteration 4

**Best so far**: 1.0082× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fma_fusion_via_rsqrt_and_arithmetic_noop_elimination | 1.010× | 1.007× | ✓ |  |
| D | fix_scalar_broadcast_load_and_enable_fma_via_rsqrt_libdevice | 1.010× | 1.009× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_via_rs... | Δ | fix_scalar_broadc... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 43230 ns | 42940 ns | -290 ns (-0.7%) | 42850 ns | -380 ns (-0.9%) |
| SM Throughput (%) | 71.80 % | 61.79 % | -10.01 pp | 61.75 % | -10.05 pp |
| Memory Throughput (%) | 31.02 % | 31.15 % | +0.13 pp | 31.16 % | +0.14 pp |
| DRAM Throughput (%) | 19.24 % | 19.31 % | +0.07 pp | 19.46 % | +0.22 pp |
| Achieved Occupancy (%) | 67.69 % | 49.09 % | -18.60 pp | 49.16 % | -18.53 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 23.16 % | 35.16 % | +12.00 pp | 35.38 % | +12.22 pp |
| IPC Active | 3.02 inst/cycle | 2.58 inst/cycle | -0.44 | 2.59 inst/cycle | -0.43 |

---

## Iteration 5

**Best so far**: 1.0082× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_autotune_grid_lambda_and_sweep_num_warps_for_latency_hiding | 1.010× | 1.008× | ✓ | **✓ best** |
| D | fix_scalar_broadcast_load_to_unblock_autotune_xblock_sweep | 1.006× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_scalar_broadc... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 43230 ns | 42880 ns | -350 ns (-0.8%) | 42880 ns | -350 ns (-0.8%) |
| SM Throughput (%) | 71.80 % | 61.67 % | -10.13 pp | 61.63 % | -10.17 pp |
| Memory Throughput (%) | 31.02 % | 31.10 % | +0.08 pp | 31.05 % | +0.03 pp |
| DRAM Throughput (%) | 19.24 % | 19.42 % | +0.18 pp | 19.35 % | +0.11 pp |
| Achieved Occupancy (%) | 67.69 % | 49.22 % | -18.47 pp | 49.59 % | -18.10 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 23.16 % | 35.16 % | +12.00 pp | 34.67 % | +11.51 pp |
| IPC Active | 3.02 inst/cycle | 2.58 inst/cycle | -0.44 | 2.61 inst/cycle | -0.41 |

---

## Cumulative Progress

```
Baseline:  43230 ns  (1.000×)
Iter 2     43200 ns  (1.001×)  — revert_to_fixed_small_xblock_eliminate_register_pressure_and_instruction_bloat
Iter 3     42880 ns  (1.008×)  — replace_sqrt_reciprocal_with_rsqrt_to_enable_fma_fusion

Final best: 42880 ns  (1.0082×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,41 +5,49 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=1),
+    triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=2),
+    triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=3),
+    triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=1),
+    triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=2),
+    triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=3),
+    triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=1),
+    triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=2),
+    triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=3),
+    triton.Config({'XBLOCK': 256}, num_warps=8,  num_stages=1),
+    triton.Config({'XBLOCK': 256}, num_warps=8,  num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=16, num_stages=1),
+    triton.Config({'XBLOCK': 256}, num_warps=16, num_stages=2),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_addmm_div_mul_sigmoid_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
     xnumel = 8388608
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 8192)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
-    tmp18 = tl.load(in_ptr5 + (0))
-    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
+    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
+    tmp18 = tl.load(in_ptr5 + tl.zeros([XBLOCK], dtype=tl.int32), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
     tmp6 = tl.full([1], 1e-05, tl.float32)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.sqrt_rn(tmp7)
-    tmp9 = tl.full([1], 1, tl.int32)
-    tmp10 = (tmp9 / tmp8)
-    tmp11 = tl.full([1], 1.0, tl.float32)
-    tmp12 = tmp10 * tmp11
-    tmp13 = tmp4 * tmp12
+    tmp10 = libdevice.rsqrt(tmp7)
+    tmp13 = tmp4 * tmp10
     tmp15 = tmp13 * tmp14
     tmp17 = tmp15 + tmp16
-    tmp20 = tmp17 + tmp19
-    tmp21 = tmp20 * tmp11
-    tmp22 = tl.sigmoid(tmp21)
-    tmp23 = tmp21 * tmp22
-    tl.store(in_out_ptr0 + (x2), tmp23, None)
-
-
+    tmp20 = tmp17 + tmp18
+    tmp22 = tl.sigmoid(tmp20)
+    tmp23 = tmp20 * tmp22
+    tl.store(in_out_ptr0 + (x2), tmp23, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -51,7 +59,7 @@

     arg_6 = inputs[6]
     xnumel = 8388608
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_add_addmm_div_mul_sigmoid_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_addmm_div_mul_sigmoid_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

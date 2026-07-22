# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12350 ns — **Best latency**: 12350 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 17
- **Duration**: 12350 ns
- **SM Throughput**: 46.83 % | **Memory**: 20.56 % | **DRAM**: 13.28 %
- **Achieved Occupancy**: 55.70 %
- **IPC Active**: 2.32 inst/cycle | **No Eligible**: 41.68 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_latency_hiding | 0.997× | 1.005× | ✓ |  |
| B | increase_xblock_with_autotune_to_reduce_l1tex_stalls | 0.998× | 0.426× | ✓ |  |
| C | autotune_xblock_to_hide_l1tex_latency | 0.982× | 1.007× | ✓ |  |
| D | add_autotune_increase_xblock_vectorized_loads | 0.785× | 0.747× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12350 ns | 12290 ns | -60 ns (-0.5%) | 29020 ns | +16670 ns (+135.0%) | 12260 ns | -90 ns (-0.7%) | 16540 ns | +4190 ns (+33.9%) |
| SM Throughput (%) | 46.83 % | 55.45 % | +8.62 pp | 81.22 % | +34.39 pp | 55.47 % | +8.64 pp | 75.80 % | +28.97 pp |
| Memory Throughput (%) | 20.56 % | 20.47 % | -0.09 pp | 19.05 % | -1.51 pp | 20.54 % | -0.02 pp | 22.31 % | +1.75 pp |
| DRAM Throughput (%) | 13.28 % | 13.35 % | +0.07 pp | 5.64 % | -7.64 pp | 13.36 % | +0.08 pp | 9.92 % | -3.36 pp |
| Achieved Occupancy (%) | 55.70 % | 59.03 % | +3.33 pp | 85.48 % | +29.78 pp | 59.36 % | +3.66 pp | 84.91 % | +29.21 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 18.00 register/thread | +1 | 32.00 register/thread | +15 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 14.85 | +7.43 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 41.68 % | 31.30 % | -10.38 pp | 11.93 % | -29.75 pp | 31.39 % | -10.29 pp | 13.38 % | -28.30 pp |
| IPC Active | 2.32 inst/cycle | 2.72 inst/cycle | +0.40 | 3.51 inst/cycle | +1.19 | 2.73 inst/cycle | +0.41 | 3.42 inst/cycle | +1.10 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_with_autotune_to_reduce_l1tex_stalls` (0.426×)

Adding @triton.autotune with larger XBLOCK configs (256–1024) selected a 256-thread block that doubled registers per thread from 17 to 32, halving the Block Limit Registers from 21 to 8 blocks/SM. Although scheduler idle cycles dropped from 41.68% to 11.93% and compute throughput rose from 46.83% to 81.22%, the total instruction count increased 4x (4.5M to 18.6M) and kernel duration more than doubled (12.35 µs to 29.02 µs). The autotune selected a config that increased work per thread (via unrolling or vectorization triggered by the larger tile) rather than simply issuing more independent loads, causing a net regression. When the original kernel has very low register usage (≤17 registers/thread) and the bottleneck is L1TEX latency with moderate occupancy (55%), adding autotune that allows larger blocks can trigger compiler unrolling that inflates register count and total instruction volume, overwhelming any latency-hiding benefit.

> **Applicable when**: Never: this strategy caused a 2.35x slowdown. Avoid adding autotune with larger XBLOCK when the original kernel has very low register count (≤20 registers/thread), as the compiler may unroll loops at larger tile sizes, inflating register usage and total instruction count.

> **Avoid when**: Kernel has low register count (≤20 reg/thread) and moderate occupancy (50–60%); adding autotune with XBLOCK > original may trigger compiler unrolling that doubles+ registers per thread and multiplies total instruction count, causing severe regression even if scheduler idle cycles improve. Also avoid when the grid is not updated to use meta['XBLOCK'] (fixed grid with autotune is a bug that wastes work).

#### ❌ Plan D: `add_autotune_increase_xblock_vectorized_loads` (0.747×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) nearly doubled registers per thread (17→32) and more than doubled total executed instructions (4.5M→9.8M), causing a 34% increase in kernel duration despite improved occupancy and scheduler utilization. The autotuner selected a configuration that introduced significant register pressure and instruction overhead, outweighing the gains from reduced grid launch overhead and better warp eligibility. When a kernel already has very low register usage (≤20 registers/thread) and the bottleneck is L1TEX stalls rather than occupancy, blindly increasing XBLOCK via autotune can cause the compiler to spill registers or generate more instructions, negating the benefit.

> **Applicable when**: Never — this specific direction caused regression. Avoid adding autotune with large XBLOCK configs when the kernel has low register count (≤20) but the autotuner may select configs that double register usage and instruction count.

> **Avoid when**: Do not apply autotune with large XBLOCK (512–2048) to pointwise kernels where: (1) the original register count is already low (≤20 registers/thread), (2) the L1TEX stall is the bottleneck but the kernel body contains many intermediate temporaries that scale with XBLOCK, or (3) the instruction count more than doubles after autotuning — this indicates the compiler is generating extra predication/masking code that overwhelms the benefit of reduced grid launches.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.180× | 0.152× | ✓ |  |
| B | persistent_tile_scheduling | 0.984× | 0.975× | ✓ |  |
| C | persistent_tile_scheduling_with_rsqrt_fma_fusion | 0.948× | 0.972× | ✓ |  |
| D | replace_integer_division_with_rsqrt_to_enable_fma_fusion | 0.996× | 0.738× | ✓ |  |
| E | add_triton_autotune | 0.982× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | replace_integer_d... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12350 ns | 81440 ns | +69090 ns (+559.4%) | 12670 ns | +320 ns (+2.6%) | 12700 ns | +350 ns (+2.8%) | 16740 ns | +4390 ns (+35.5%) | 12320 ns | -30 ns (-0.2%) |
| SM Throughput (%) | 46.83 % | 89.40 % | +42.57 pp | 60.33 % | +13.50 pp | 16.39 % | -30.44 pp | 67.46 % | +20.63 pp | 55.30 % | +8.47 pp |
| Memory Throughput (%) | 20.56 % | 40.85 % | +20.29 pp | 19.78 % | -0.78 pp | 19.12 % | -1.44 pp | 34.49 % | +13.93 pp | 20.55 % | -0.01 pp |
| DRAM Throughput (%) | 13.28 % | 2.09 % | -11.19 pp | 12.93 % | -0.35 pp | 12.89 % | -0.39 pp | 9.77 % | -3.51 pp | 13.31 % | +0.03 pp |
| Achieved Occupancy (%) | 55.70 % | 96.51 % | +40.81 pp | 65.33 % | +9.63 pp | 36.31 % | -19.39 pp | 78.83 % | +23.13 pp | 59.20 % | +3.50 pp |
| Registers/Thread | 17.00 register/thread | 25.00 register/thread | +8 | 30.00 register/thread | +13 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 14.85 | +7.43 | 7.42 | +0.00 |
| No Eligible (%) | 41.68 % | 7.58 % | -34.10 pp | 25.61 % | -16.07 pp | 74.04 % | +32.36 pp | 23.31 % | -18.37 pp | 31.96 % | -9.72 pp |
| IPC Active | 2.32 inst/cycle | 3.69 inst/cycle | +1.37 | 2.96 inst/cycle | +0.64 | 0.89 inst/cycle | -1.43 | 3.05 inst/cycle | +0.73 | 2.73 inst/cycle | +0.41 |

---

## Cumulative Progress

```
Baseline:  12350 ns  (1.000×)

Final best: 12350 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,19 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 2007040
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 16)
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
@@ -29,9 +42,7 @@

     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tl.store(in_out_ptr0 + (x2), tmp15, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +52,7 @@

     arg_4 = inputs[4]
     xnumel = 2007040
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2007040, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2007040)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

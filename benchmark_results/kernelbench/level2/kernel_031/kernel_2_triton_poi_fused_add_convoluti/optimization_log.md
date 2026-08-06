# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_convolution_lift_fresh_minimum_mul_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 579230 ns — **Best latency**: 579010 ns (**1.0004×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`vectorized_loads_store_hints_no_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_convolution_lift_fresh_minimum_mul_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_031/kernel_2_triton_poi_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256000,1,1), Waves/SM: 121.21
- **Registers/thread**: 30
- **Duration**: 579230 ns
- **SM Throughput**: 28.98 % | **Memory**: 78.95 % | **DRAM**: 72.45 %
- **Achieved Occupancy**: 93.73 %
- **IPC Active**: 1.17 inst/cycle | **No Eligible**: 70.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_store_hints_no_autotune | 1.000× | 1.000× | ✓ | **✓ best** |
| B | shared_memory_tiled_transpose_for_coalesced_loads | 1.000× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | shared_memory_til... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 579230 ns | 579010 ns | -220 ns (-0.0%) | 578530 ns | -700 ns (-0.1%) |
| SM Throughput (%) | 28.98 % | 28.84 % | -0.14 pp | 28.76 % | -0.22 pp |
| Memory Throughput (%) | 78.95 % | 79.02 % | +0.07 pp | 79.11 % | +0.16 pp |
| DRAM Throughput (%) | 72.45 % | 72.50 % | +0.05 pp | 72.56 % | +0.11 pp |
| Achieved Occupancy (%) | 93.73 % | 93.70 % | -0.03 pp | 93.71 % | -0.02 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 121.21 | 121.21 | +0.00 | 121.21 | +0.00 |
| No Eligible (%) | 70.90 % | 71.04 % | +0.14 pp | 70.61 % | -0.29 pp |
| IPC Active | 1.17 inst/cycle | 1.18 inst/cycle | +0.01 | 1.16 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0004× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_tuning_to_reduce_l2_thrashing | 0.805× | 0.908× | ✓ |  |
| D | eviction_policy_and_l2_compression_hints | 0.806× | 0.913× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_t... | Δ | eviction_policy_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 579230 ns | 637820 ns | +58590 ns (+10.1%) | 634080 ns | +54850 ns (+9.5%) |
| SM Throughput (%) | 28.98 % | 30.18 % | +1.20 pp | 29.96 % | +0.98 pp |
| Memory Throughput (%) | 78.95 % | 71.69 % | -7.26 pp | 72.15 % | -6.80 pp |
| DRAM Throughput (%) | 72.45 % | 65.76 % | -6.69 pp | 66.15 % | -6.30 pp |
| Achieved Occupancy (%) | 93.73 % | 94.27 % | +0.54 pp | 94.29 % | +0.56 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 121.21 | 121.21 | +0.00 | 121.21 | +0.00 |
| No Eligible (%) | 70.90 % | 69.80 % | -1.10 pp | 69.97 % | -0.93 pp |
| IPC Active | 1.17 inst/cycle | 1.22 inst/cycle | +0.05 | 1.21 inst/cycle | +0.04 |

---

## Iteration 4

**Best so far**: 1.0004× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fma_fusion_and_manual_yblock_increase_for_broadcast_reuse | 0.806× | 0.908× | ✓ |  |
| C | autotune_num_stages_and_tile_sizes_for_l1tex_latency_hiding | 0.810× | 0.915× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_and_ma... | Δ | autotune_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 579230 ns | 637890 ns | +58660 ns (+10.1%) | 633220 ns | +53990 ns (+9.3%) |
| SM Throughput (%) | 28.98 % | 30.18 % | +1.20 pp | 31.33 % | +2.35 pp |
| Memory Throughput (%) | 78.95 % | 71.78 % | -7.17 pp | 72.28 % | -6.67 pp |
| DRAM Throughput (%) | 72.45 % | 65.75 % | -6.70 pp | 66.24 % | -6.21 pp |
| Achieved Occupancy (%) | 93.73 % | 94.30 % | +0.57 pp | 94.36 % | +0.63 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 32.00 register/thread | +2 |
| Waves/SM | 121.21 | 121.21 | +0.00 | 121.21 | +0.00 |
| No Eligible (%) | 70.90 % | 69.60 % | -1.30 pp | 68.43 % | -2.47 pp |
| IPC Active | 1.17 inst/cycle | 1.22 inst/cycle | +0.05 | 1.25 inst/cycle | +0.08 |

---

## Cumulative Progress

```
Baseline:  579230 ns  (1.000×)
Iter 2     579010 ns  (1.000×)  — vectorized_loads_store_hints_no_autotune

Final best: 579010 ns  (1.0004×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused_add_convolution_lift_fresh_minimum_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -19,7 +20,7 @@

     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 2032128*y1), xmask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 2032128*y1), mask=xmask, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
     tmp5 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
@@ -28,9 +29,7 @@

     tmp6 = tmp4 + tmp5
     tmp7 = tl.full([1, 1], 2.0, tl.float32)
     tmp8 = tmp6 * tmp7
-    tl.store(out_ptr0 + (x2 + 15876*y3), tmp8, xmask)
-
-
+    tl.store(out_ptr0 + (x2 + 15876*y3), tmp8, mask=xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

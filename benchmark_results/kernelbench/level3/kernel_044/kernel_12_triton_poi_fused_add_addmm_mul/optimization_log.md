# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_addmm_mul_pow_tanh_view_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 971140 ns — **Best latency**: 969660 ns (**1.0015×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_addmm_mul_pow_tanh_view_12`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_12_triton_poi_fused_add_addmm_mul/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 16
- **Duration**: 971140 ns
- **SM Throughput**: 54.87 % | **Memory**: 33.19 % | **DRAM**: 33.19 %
- **Achieved Occupancy**: 55.80 %
- **IPC Active**: 2.20 inst/cycle | **No Eligible**: 44.75 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | manual_xblock_increase_persistent_stride_loop | 1.000× | 1.001× | ✓ |  |
| B | increase_xblock_manually_without_autotune_to_hide_l1tex_latency | 1.000× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | manual_xblock_inc... | Δ | increase_xblock_m... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 971140 ns | 970370 ns | -770 ns (-0.1%) | 970340 ns | -800 ns (-0.1%) |
| SM Throughput (%) | 54.87 % | 54.86 % | -0.01 pp | 54.87 % | +0.00 pp |
| Memory Throughput (%) | 33.19 % | 33.22 % | +0.03 pp | 33.23 % | +0.04 pp |
| DRAM Throughput (%) | 33.19 % | 33.22 % | +0.03 pp | 33.23 % | +0.04 pp |
| Achieved Occupancy (%) | 55.80 % | 55.82 % | +0.02 pp | 55.73 % | -0.07 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 44.75 % | 44.97 % | +0.22 pp | 44.98 % | +0.23 pp |
| IPC Active | 2.20 inst/cycle | 2.20 inst/cycle | +0.00 | 2.20 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | static_xblock_increase_to_1024_with_corrected_grid | 1.000× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | static_xblock_inc... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 971140 ns | 970460 ns | -680 ns (-0.1%) |
| SM Throughput (%) | 54.87 % | 54.89 % | +0.02 pp |
| Memory Throughput (%) | 33.19 % | 33.22 % | +0.03 pp |
| DRAM Throughput (%) | 33.19 % | 33.22 % | +0.03 pp |
| Achieved Occupancy (%) | 55.80 % | 55.79 % | -0.01 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 |
| No Eligible (%) | 44.75 % | 44.97 % | +0.22 pp |
| IPC Active | 2.20 inst/cycle | 2.20 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | vectorized_loads_stores_via_2d_tile_reshape_fixed_grid | 1.000× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 971140 ns | 969790 ns | -1350 ns (-0.1%) |
| SM Throughput (%) | 54.87 % | 50.82 % | -4.05 pp |
| Memory Throughput (%) | 33.19 % | 33.24 % | +0.05 pp |
| DRAM Throughput (%) | 33.19 % | 33.24 % | +0.05 pp |
| Achieved Occupancy (%) | 55.80 % | 53.19 % | -2.61 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 |
| No Eligible (%) | 44.75 % | 49.05 % | +4.30 pp |
| IPC Active | 2.20 inst/cycle | 2.04 inst/cycle | -0.16 |

---

## Iteration 5

**Best so far**: 1.0014× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | flat_1d_xblock1024_autotune_num_warps_num_stages | 0.522× | 0.522× | ✓ |  |
| E | add_triton_autotune | 1.000× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | flat_1d_xblock102... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 971140 ns | 1860000 ns | +888860 ns (+91.5%) | 969660 ns | -1480 ns (-0.2%) |
| SM Throughput (%) | 54.87 % | 93.68 % | +38.81 pp | 50.83 % | -4.04 pp |
| Memory Throughput (%) | 33.19 % | 17.35 % | -15.84 pp | 33.25 % | +0.06 pp |
| DRAM Throughput (%) | 33.19 % | 17.35 % | -15.84 pp | 33.25 % | +0.06 pp |
| Achieved Occupancy (%) | 55.80 % | 82.66 % | +26.86 pp | 53.19 % | -2.61 pp |
| Registers/Thread | 16.00 register/thread | 26.00 register/thread | +10 | 16.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 44.75 % | 6.24 % | -38.51 pp | 49.05 % | +4.30 pp |
| IPC Active | 2.20 inst/cycle | 3.76 inst/cycle | +1.56 | 2.04 inst/cycle | -0.16 |


### Key Learnings

#### ❌ Plan C: `flat_1d_xblock1024_autotune_num_warps_num_stages` (0.522×)

Switching from XBLOCK=128 to XBLOCK=1024 with @triton.autotune over num_warps/num_stages successfully eliminated the L1TEX stall bottleneck (eligible warps rose from 0.77 to 8.11, No Eligible dropped from 49% to 6%), but the kernel regressed 2x in wall time. The executed instruction count exploded 3.5x (390M → 1373M), registers per thread increased from 16 to 26 (reducing block-level occupancy headroom), and DRAM throughput halved (33% → 17%), indicating the larger XBLOCK caused the compiler to unroll more aggressively and emit far more non-fused FP32 instructions per element. The compute throughput saturated at 93.7% (compute-bound), meaning the extra instruction volume — not memory latency — became the new bottleneck, and the net effect was a 1.9x slowdown despite better scheduling.

> **Applicable when**: Never — this specific direction (flat 1D XBLOCK=1024 with autotune over num_warps/num_stages only) causes regression for this kernel despite fixing the scheduling bottleneck.

> **Avoid when**: When the kernel performs transcendental operations (tanh, libdevice functions) combined with polynomial arithmetic: increasing XBLOCK causes the Triton compiler to unroll the loop body more aggressively, emitting 3-4x more non-fused FP32 instructions per element and increasing register pressure. The resulting compute-bound regime (93.7% SM throughput) is worse than the original latency-bound regime (50.8% SM throughput) because the instruction count increase outweighs the scheduling improvement. Do not apply large XBLOCK increases to kernels with high per-element arithmetic intensity (GELU, softmax, layer norm) where compiler unrolling will dominate.

---

## Cumulative Progress

```
Baseline:  971140 ns  (1.000×)
Iter 2     970340 ns  (1.001×)  — increase_xblock_manually_without_autotune_to_hide_l1tex_latency
Iter 4     969790 ns  (1.001×)  — vectorized_loads_stores_via_2d_tile_reshape_fixed_grid
Iter 5     969660 ns  (1.002×)  — add_triton_autotune

Final best: 969660 ns  (1.0015×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,41 +5,55 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused_add_addmm_mul_pow_tanh_view_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
+    VEC : tl.constexpr = 4
+    XBLOCK_V : tl.constexpr = XBLOCK // VEC  # 1024//4 = 256
     pid = tl.program_id(0)
-    num_programs = tl.num_programs(0)
-    xoffset = pid * XBLOCK
-    while xoffset < xnumel:
-        xindex = xoffset + tl.arange(0, XBLOCK)[:]
-        x0 = xindex % 3072
-        x2 = xindex
-        tmp0 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
-        tmp1 = tl.load(in_out_ptr0 + x2, None)
-        tmp2 = tmp0 + tmp1
-        tmp3 = tl.full([1], 0.5, tl.float32)
-        tmp4 = tmp2 * tmp3
-        tmp5 = tmp2 * tmp2
-        tmp6 = tmp5 * tmp2
-        tmp7 = tl.full([1], 0.044715, tl.float32)
-        tmp8 = tmp6 * tmp7
-        tmp9 = tmp2 + tmp8
-        tmp10 = tl.full([1], 0.7978845608028654, tl.float32)
-        tmp11 = tmp9 * tmp10
-        tmp12 = libdevice.tanh(tmp11)
-        tmp13 = tl.full([1], 1.0, tl.float32)
-        tmp14 = tmp12 + tmp13
-        tmp15 = tmp4 * tmp14
-        tl.store(in_out_ptr0 + x2, tmp15, None)
-        xoffset += num_programs * XBLOCK
-
+    row_idx = tl.arange(0, XBLOCK_V)[:, None]   # [256, 1]
+    col_idx = tl.arange(0, VEC)[None, :]         # [1, 4]
+    xbase = pid * XBLOCK
+    xindex = xbase + row_idx * VEC + col_idx     # [256, 4] contiguous
+    mask = xindex < xnumel
+    x0 = xindex % 3072
+    x2 = xindex
+    # 128-bit vectorized loads: inner dim=4 floats = 16 bytes per thread group
+    tmp0 = tl.load(in_ptr0 + x0, mask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + x2, mask)
+    tmp2 = tmp0 + tmp1
+    tmp3 = tl.full([1, 1], 0.5, tl.float32)
+    tmp4 = tmp2 * tmp3
+    tmp5 = tmp2 * tmp2
+    tmp6 = tmp5 * tmp2
+    tmp7 = tl.full([1, 1], 0.044715, tl.float32)
+    tmp8 = tmp6 * tmp7
+    tmp9 = tmp2 + tmp8
+    tmp10 = tl.full([1, 1], 0.7978845608028654, tl.float32)
+    tmp11 = tmp9 * tmp10
+    tmp12 = libdevice.tanh(tmp11)
+    tmp13 = tl.full([1, 1], 1.0, tl.float32)
+    tmp14 = tmp12 + tmp13
+    tmp15 = tmp4 * tmp14
+    # 128-bit vectorized store
+    tl.store(in_out_ptr0 + x2, tmp15, mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 201326592
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_add_addmm_mul_pow_tanh_view_12[grid](arg_0, arg_1, 201326592, XBLOCK=128)
+    triton_poi_fused_add_addmm_mul_pow_tanh_view_12[grid](arg_0, arg_1, 201326592)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

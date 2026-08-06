# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_max_pool2d_with_indices_76` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4900 ns — **Best latency**: 4900 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_max_pool2d_with_indices_76`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_76_triton_poi_fused_max_pool2d_wi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3185,1,1), Waves/SM: 1.51
- **Registers/thread**: 22
- **Duration**: 4900 ns
- **SM Throughput**: 37.36 % | **Memory**: 18.67 % | **DRAM**: 6.85 %
- **Achieved Occupancy**: 68.87 %
- **IPC Active**: 2.40 inst/cycle | **No Eligible**: 45.84 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave | 1.001× | 1.000× | ✓ | **✓ best** |
| B | eliminate_partial_wave_via_autotune_xblock | 0.987× | 0.994× | ✓ |  |
| C | autotune_xblock_vectorized_loads | 0.912× | 0.875× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | eliminate_partial... | Δ | autotune_xblock_v... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4900 ns | 4900 ns | +0 ns (+0.0%) | 4930 ns | +30 ns (+0.6%) | 5600 ns | +700 ns (+14.3%) |
| SM Throughput (%) | 37.36 % | 37.24 % | -0.12 pp | 41.89 % | +4.53 pp | 45.87 % | +8.51 pp |
| Memory Throughput (%) | 18.67 % | 18.35 % | -0.32 pp | 22.69 % | +4.02 pp | 21.47 % | +2.80 pp |
| DRAM Throughput (%) | 6.85 % | 6.84 % | -0.01 pp | 6.77 % | -0.08 pp | 5.97 % | -0.88 pp |
| Achieved Occupancy (%) | 68.87 % | 68.79 % | -0.08 pp | 68.83 % | -0.04 pp | 70.02 % | +1.15 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 1.51 | 1.51 | +0.00 | 1.51 | +0.00 | 1.51 | +0.00 |
| No Eligible (%) | 45.84 % | 39.45 % | -6.39 pp | 35.50 % | -10.34 pp | 25.66 % | -20.18 pp |
| IPC Active | 2.40 inst/cycle | 2.41 inst/cycle | +0.01 | 2.76 inst/cycle | +0.36 | 2.92 inst/cycle | +0.52 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_to_respect_autotuned_xblock_and_add_warp_configs | 0.969× | 0.976× | ✓ |  |
| B | fix_grid_xblock_coupling_and_add_num_stages_autotune | 0.973× | 0.994× | ✓ |  |
| C | fix_grid_to_use_autotuned_xblock_eliminating_partial_wave | 0.977× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ | fix_grid_xblock_c... | Δ | fix_grid_to_use_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4900 ns | 5020 ns | +120 ns (+2.4%) | 4930 ns | +30 ns (+0.6%) | 4930 ns | +30 ns (+0.6%) |
| SM Throughput (%) | 37.36 % | 34.08 % | -3.28 pp | 40.21 % | +2.85 pp | 37.34 % | -0.02 pp |
| Memory Throughput (%) | 18.67 % | 17.70 % | -0.97 pp | 21.85 % | +3.18 pp | 18.31 % | -0.36 pp |
| DRAM Throughput (%) | 6.85 % | 6.69 % | -0.16 pp | 6.79 % | -0.06 pp | 6.83 % | -0.02 pp |
| Achieved Occupancy (%) | 68.87 % | 67.86 % | -1.01 pp | 67.89 % | -0.98 pp | 69.27 % | +0.40 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 1.51 | 1.51 | +0.00 | 1.51 | +0.00 | 1.51 | +0.00 |
| No Eligible (%) | 45.84 % | 46.61 % | +0.77 pp | 29.35 % | -16.49 pp | 39.44 % | -6.40 pp |
| IPC Active | 2.40 inst/cycle | 2.35 inst/cycle | -0.05 | 2.77 inst/cycle | +0.37 | 2.25 inst/cycle | -0.15 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_warps2_configs_and_fix_grid_lambda | 0.994× | 1.049× | ✓ |  |
| B | add_num_stages_pipelining_to_hide_memory_latency | 0.977× | 0.968× | ✓ |  |
| C | fix_grid_lambda_and_add_num_stages_for_masked_load_pipelining | 0.961× | 0.982× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_warps2_co... | Δ | add_num_stages_pi... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4900 ns | 4670 ns | -230 ns (-4.7%) | 5060 ns | +160 ns (+3.3%) | 4990 ns | +90 ns (+1.8%) |
| SM Throughput (%) | 37.36 % | 22.08 % | -15.28 pp | 37.16 % | -0.20 pp | 36.63 % | -0.73 pp |
| Memory Throughput (%) | 18.67 % | 18.63 % | -0.04 pp | 17.72 % | -0.95 pp | 18.08 % | -0.59 pp |
| DRAM Throughput (%) | 6.85 % | 7.17 % | +0.32 pp | 6.61 % | -0.24 pp | 6.73 % | -0.12 pp |
| Achieved Occupancy (%) | 68.87 % | 46.88 % | -21.99 pp | 69.86 % | +0.99 pp | 66.98 % | -1.89 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 1.51 | 0.75 | -0.76 | 1.51 | +0.00 | 1.51 | +0.00 |
| No Eligible (%) | 45.84 % | 50.01 % | +4.17 pp | 39.58 % | -6.26 pp | 40.92 % | -4.92 pp |
| IPC Active | 2.40 inst/cycle | 2.01 inst/cycle | -0.39 | 2.32 inst/cycle | -0.08 | 2.35 inst/cycle | -0.05 |

---

## Cumulative Progress

```
Baseline:  4900 ns  (1.000×)

Final best: 4900 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_max_pool2d_with_indices_76(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 407680
@@ -25,57 +41,55 @@

     tmp8 = tmp6 < tmp3
     tmp9 = tmp7 & tmp8
     tmp10 = tmp5 & tmp9
-    tmp11 = tl.load(in_ptr0 + ((-6656) + x6), tmp10 & xmask, other=float("-inf"))
+    tmp11 = tl.load(in_ptr0 + ((-6656) + x6), mask=tmp10 & xmask, other=float("-inf"))
     tmp12 = x1
     tmp13 = tmp12 >= tmp1
     tmp14 = tmp12 < tmp3
     tmp15 = tmp13 & tmp14
     tmp16 = tmp5 & tmp15
-    tmp17 = tl.load(in_ptr0 + ((-5824) + x6), tmp16 & xmask, other=float("-inf"))
+    tmp17 = tl.load(in_ptr0 + ((-5824) + x6), mask=tmp16 & xmask, other=float("-inf"))
     tmp18 = tl.maximum(tmp11, tmp17)
     tmp19 = 1 + x1
     tmp20 = tmp19 >= tmp1
     tmp21 = tmp19 < tmp3
     tmp22 = tmp20 & tmp21
     tmp23 = tmp5 & tmp22
-    tmp24 = tl.load(in_ptr0 + ((-4992) + x6), tmp23 & xmask, other=float("-inf"))
+    tmp24 = tl.load(in_ptr0 + ((-4992) + x6), mask=tmp23 & xmask, other=float("-inf"))
     tmp25 = tl.maximum(tmp18, tmp24)
     tmp26 = x2
     tmp27 = tmp26 >= tmp1
     tmp28 = tmp26 < tmp3
     tmp29 = tmp27 & tmp28
     tmp30 = tmp29 & tmp9
-    tmp31 = tl.load(in_ptr0 + ((-832) + x6), tmp30 & xmask, other=float("-inf"))
+    tmp31 = tl.load(in_ptr0 + ((-832) + x6), mask=tmp30 & xmask, other=float("-inf"))
     tmp32 = tl.maximum(tmp25, tmp31)
     tmp33 = tmp29 & tmp15
-    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=float("-inf"))
+    tmp34 = tl.load(in_ptr0 + (x6), mask=tmp33 & xmask, other=float("-inf"))
     tmp35 = tl.maximum(tmp32, tmp34)
     tmp36 = tmp29 & tmp22
-    tmp37 = tl.load(in_ptr0 + (832 + x6), tmp36 & xmask, other=float("-inf"))
+    tmp37 = tl.load(in_ptr0 + (832 + x6), mask=tmp36 & xmask, other=float("-inf"))
     tmp38 = tl.maximum(tmp35, tmp37)
     tmp39 = 1 + x2
     tmp40 = tmp39 >= tmp1
     tmp41 = tmp39 < tmp3
     tmp42 = tmp40 & tmp41
     tmp43 = tmp42 & tmp9
-    tmp44 = tl.load(in_ptr0 + (4992 + x6), tmp43 & xmask, other=float("-inf"))
+    tmp44 = tl.load(in_ptr0 + (4992 + x6), mask=tmp43 & xmask, other=float("-inf"))
     tmp45 = tl.maximum(tmp38, tmp44)
     tmp46 = tmp42 & tmp15
-    tmp47 = tl.load(in_ptr0 + (5824 + x6), tmp46 & xmask, other=float("-inf"))
+    tmp47 = tl.load(in_ptr0 + (5824 + x6), mask=tmp46 & xmask, other=float("-inf"))
     tmp48 = tl.maximum(tmp45, tmp47)
     tmp49 = tmp42 & tmp22
-    tmp50 = tl.load(in_ptr0 + (6656 + x6), tmp49 & xmask, other=float("-inf"))
+    tmp50 = tl.load(in_ptr0 + (6656 + x6), mask=tmp49 & xmask, other=float("-inf"))
     tmp51 = tl.maximum(tmp48, tmp50)
-    tl.store(out_ptr0 + (x6), tmp51, xmask)
-
-
+    tl.store(out_ptr0 + (x6), tmp51, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 407680
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_max_pool2d_with_indices_76[grid](arg_0, arg_1, 407680, XBLOCK=128)
+    triton_poi_fused_max_pool2d_with_indices_76[grid](arg_0, arg_1, 407680)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

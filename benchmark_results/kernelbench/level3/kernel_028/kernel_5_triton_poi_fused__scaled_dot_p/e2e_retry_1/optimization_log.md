# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3390 ns — **Best latency**: 3360 ns (**1.0089×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_5_triton_poi_fused__scaled_dot_p/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1576,1,1), Waves/SM: 0.75
- **Registers/thread**: 16
- **Duration**: 3390 ns
- **SM Throughput**: 10.63 % | **Memory**: 7.19 % | **DRAM**: 4.96 %
- **Achieved Occupancy**: 55.13 %
- **IPC Active**: 1.20 inst/cycle | **No Eligible**: 67.02 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 1.036× | 1.009× | ✓ | **✓ best** |
| B | autotune_xblock_increase_tile_size | 1.023× | 1.069× | ✓ |  |
| C | add_triton_autotune | 1.023× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | autotune_xblock_i... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 3360 ns | -30 ns (-0.9%) | 3170 ns | -220 ns (-6.5%) | 3390 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 10.63 % | 10.70 % | +0.07 pp | 12.95 % | +2.32 pp | 10.72 % | +0.09 pp |
| Memory Throughput (%) | 7.19 % | 7.24 % | +0.05 pp | 7.36 % | +0.17 pp | 7.33 % | +0.14 pp |
| DRAM Throughput (%) | 4.96 % | 4.95 % | -0.01 pp | 5.30 % | +0.34 pp | 4.94 % | -0.02 pp |
| Achieved Occupancy (%) | 55.13 % | 52.54 % | -2.59 pp | 49.10 % | -6.03 pp | 52.44 % | -2.69 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 |
| No Eligible (%) | 67.02 % | 67.88 % | +0.86 pp | 56.86 % | -10.16 pp | 73.06 % | +6.04 pp |
| IPC Active | 1.20 inst/cycle | 1.14 inst/cycle | -0.06 | 1.33 inst/cycle | +0.13 | 1.13 inst/cycle | -0.07 |


### Key Learnings

#### ✅ Plan B: `autotune_xblock_increase_tile_size` (1.069×)

When a pointwise Triton kernel launches with very few waves per SM (≤1.0) and schedulers are idle >60% of cycles due to L1TEX latency stalls, replacing a hardcoded small XBLOCK with @triton.autotune over larger tile sizes (512–2048) allows the autotuner to find a configuration that packs more warps per CTA and better hides memory latency. In this case, eligible warps per scheduler rose from 1.35 to 1.78 and warp cycles per issued instruction dropped from 26.5 to 20.6, yielding a 6.9% speedup despite a slight occupancy drop (55% → 49%) from the increased register count (16 → 18 per thread). The gain is modest because the grid size (waves/SM) did not change—the autotuner selected a config that kept the same grid shape but improved per-warp scheduling efficiency.

> **Applicable when**: Triton pointwise kernel with: (1) waves_per_sm ≤ 1.0, (2) scheduler No Eligible > 50%, (3) CPIStall rule triggered for L1TEX with >10 stall cycles/warp, (4) low register count per thread (≤ 20) leaving headroom for larger tiles without spilling, (5) no shared memory usage.

> **Avoid when**: Kernel already uses large XBLOCK (≥512) or has high register pressure (>24 registers/thread) where increasing tile size would cause register spilling and reduce occupancy below the threshold needed for latency hiding. Also avoid when the grid is already large (waves_per_sm > 2) and the bottleneck is compute or memory bandwidth rather than scheduler idle cycles.

---

## Cumulative Progress

```
Baseline:  3390 ns  (1.000×)
Iter 1     3360 ns  (1.009×)  — autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel

Final best: 3360 ns  (1.0089×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 201728
@@ -20,15 +32,13 @@

     tmp2 = tmp0 + tmp1
     tl.store(out_ptr0 + (x4), tmp2, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     arg_2 = inputs[2]
     xnumel = 201728
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5[grid](arg_0, arg_1, arg_2, 201728, XBLOCK=128)
+    triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5[grid](arg_0, arg_1, arg_2, 201728)
     return arg_2
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`

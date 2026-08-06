# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5700 ns — **Best latency**: 3970 ns (**1.4358×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_and_eliminate_partial_wave`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_002/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5700 ns
- **SM Throughput**: 15.61 % | **Memory**: 43.47 % | **DRAM**: 21.63 %
- **Achieved Occupancy**: 54.88 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 72.51 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.416× | 1.380× | ✓ |  |
| B | autotune_tile_sizes_and_fix_coalescing | 1.426× | 1.696× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.258× | 1.272× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5700 ns | 4130 ns | -1570 ns (-27.5%) | 3360 ns | -2340 ns (-41.1%) | 4480 ns | -1220 ns (-21.4%) |
| SM Throughput (%) | 15.61 % | 27.54 % | +11.93 pp | 16.12 % | +0.51 pp | 27.78 % | +12.17 pp |
| Memory Throughput (%) | 43.47 % | 21.87 % | -21.60 pp | 20.41 % | -23.06 pp | 22.09 % | -21.38 pp |
| DRAM Throughput (%) | 21.63 % | 4.02 % | -17.61 pp | 4.95 % | -16.68 pp | 3.68 % | -17.95 pp |
| Achieved Occupancy (%) | 54.88 % | 69.54 % | +14.66 pp | 44.22 % | -10.66 pp | 67.42 % | +12.54 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 72.51 % | 40.42 % | -32.09 pp | 54.57 % | -17.94 pp | 37.59 % | -34.92 pp |
| IPC Active | 1.14 inst/cycle | 2.31 inst/cycle | +1.17 | 1.72 inst/cycle | +0.58 | 2.38 inst/cycle | +1.24 |

---

## Iteration 2

**Best so far**: 1.6964× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.405× | 1.404× | ✓ |  |
| B | fix_autotune_grid_lambda_and_add_yblock1_configs_for_coalesced_stores | 1.403× | 1.310× | ✓ |  |
| C | vectorized_loads_and_yblock1_store_coalescing | 1.395× | 1.319× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_autotune_grid... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5700 ns | 4060 ns | -1640 ns (-28.8%) | 4350 ns | -1350 ns (-23.7%) | 4320 ns | -1380 ns (-24.2%) |
| SM Throughput (%) | 15.61 % | 24.94 % | +9.33 pp | 25.59 % | +9.98 pp | 25.87 % | +10.26 pp |
| Memory Throughput (%) | 43.47 % | 19.80 % | -23.67 pp | 20.34 % | -23.13 pp | 20.29 % | -23.18 pp |
| DRAM Throughput (%) | 21.63 % | 4.08 % | -17.55 pp | 3.80 % | -17.83 pp | 3.85 % | -17.78 pp |
| Achieved Occupancy (%) | 54.88 % | 69.57 % | +14.69 pp | 69.73 % | +14.85 pp | 67.62 % | +12.74 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 72.51 % | 37.86 % | -34.65 pp | 37.18 % | -35.33 pp | 41.97 % | -30.54 pp |
| IPC Active | 1.14 inst/cycle | 2.41 inst/cycle | +1.27 | 2.35 inst/cycle | +1.21 | 2.39 inst/cycle | +1.25 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_contiguous_hints` (1.404×)

Adding tl.max_contiguous and tl.multiple_of hints to a naturally contiguous load index in Triton informs the compiler that consecutive threads access consecutive addresses, enabling 128-bit vectorized loads instead of scalar 32-bit loads. In this case the hints alone did not produce the speedup — the dominant effect came from the autotuner selecting a larger block size (128→256) driven by the newly added XBLOCK=1024 configs, which doubled waves/SM (0.74→1.48), raised achieved occupancy (44%→70%), and increased eligible warps/scheduler (2.08→3.98). The per-kernel duration increased (3.36→4.06 µs) but end-to-end throughput improved 1.40x, suggesting the occupancy/wave improvement dominated over any per-block overhead from the larger tile.

> **Applicable when**: Triton kernel has a naturally contiguous load index (xoffset + arange(0, XBLOCK)) but lacks tl.max_contiguous/tl.multiple_of hints, causing scalar 32-bit loads instead of 128-bit vectorized loads. Kernel also has waves/SM < 1.0 and achieved occupancy < 50% with autotune configs that do not include large XBLOCK values (e.g., 1024). NCU reports UncoalescedGlobalAccess and IssueSlotUtilization OPT rules.

> **Avoid when**: The load index is not naturally contiguous (e.g., involves modulo or non-unit stride in the fast thread dimension). Waves/SM is already >= 2.0 and occupancy >= 80% — adding larger XBLOCK configs will not help. The store-side scatter pattern is the dominant bottleneck and load-side hints cannot address it. Autotune overhead is unacceptable.

---

## Iteration 3

**Best so far**: 1.6964× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | yblock3_aligned_grid_to_eliminate_partial_wave_and_reduce_store_scatter | 1.408× | 1.370× | ✓ |  |
| B | fix_grid_lambda_and_eliminate_partial_wave | 1.452× | 1.436× | ✓ | **✓ best** |
| C | grid_aligned_1d_launch_to_eliminate_partial_wave | 1.359× | 1.319× | ✓ |  |


### Metric Comparison

| Metric | Baseline | yblock3_aligned_g... | Δ | fix_grid_lambda_a... | Δ | grid_aligned_1d_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5700 ns | 4160 ns | -1540 ns (-27.0%) | 3970 ns | -1730 ns (-30.4%) | 4320 ns | -1380 ns (-24.2%) |
| SM Throughput (%) | 15.61 % | 25.54 % | +9.93 pp | 26.81 % | +11.20 pp | 26.47 % | +10.86 pp |
| Memory Throughput (%) | 43.47 % | 20.30 % | -23.17 pp | 21.33 % | -22.14 pp | 21.03 % | -22.44 pp |
| DRAM Throughput (%) | 21.63 % | 3.98 % | -17.65 pp | 4.14 % | -17.49 pp | 3.82 % | -17.81 pp |
| Achieved Occupancy (%) | 54.88 % | 69.22 % | +14.34 pp | 68.52 % | +13.64 pp | 67.87 % | +12.99 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 72.51 % | 37.99 % | -34.52 pp | 38.28 % | -34.23 pp | 39.94 % | -32.57 pp |
| IPC Active | 1.14 inst/cycle | 2.25 inst/cycle | +1.11 | 2.42 inst/cycle | +1.28 | 2.38 inst/cycle | +1.24 |

---

## Iteration 4

**Best so far**: 1.6964× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_evict_first_streaming_load | 1.410× | 1.390× | ✓ |  |
| B | fix_hardcoded_grid_to_use_autotune_meta_and_switch_to_evict_first | 0.846× | 1.380× | ✓ |  |
| C | activate_autotune_via_grid_lambda | 1.361× | 1.370× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | activate_autotune... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5700 ns | 4100 ns | -1600 ns (-28.1%) | 4130 ns | -1570 ns (-27.5%) | 4160 ns | -1540 ns (-27.0%) |
| SM Throughput (%) | 15.61 % | 27.37 % | +11.76 pp | 27.05 % | +11.44 pp | 26.17 % | +10.56 pp |
| Memory Throughput (%) | 43.47 % | 21.72 % | -21.75 pp | 21.49 % | -21.98 pp | 20.78 % | -22.69 pp |
| DRAM Throughput (%) | 21.63 % | 4.04 % | -17.59 pp | 4.00 % | -17.63 pp | 3.96 % | -17.67 pp |
| Achieved Occupancy (%) | 54.88 % | 67.61 % | +12.73 pp | 71.68 % | +16.80 pp | 68.02 % | +13.14 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 72.51 % | 37.82 % | -34.69 pp | 37.96 % | -34.55 pp | 38.32 % | -34.19 pp |
| IPC Active | 1.14 inst/cycle | 2.41 inst/cycle | +1.27 | 2.24 inst/cycle | +1.10 | 2.43 inst/cycle | +1.29 |

---

## Iteration 5

**Best so far**: 1.6964× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_lambda_to_activate_autotune_and_eliminate_store_scatter | 1.387× | 1.329× | ✓ |  |
| C | fix_hardcoded_grid_lambda_to_activate_autotune_yblock1 | 1.385× | 1.351× | ✓ |  |
| D | add_triton_autotune | 1.415× | 1.370× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_hardcoded_gri... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5700 ns | 4290 ns | -1410 ns (-24.7%) | 4220 ns | -1480 ns (-26.0%) | 4160 ns | -1540 ns (-27.0%) |
| SM Throughput (%) | 15.61 % | 27.01 % | +11.40 pp | 26.89 % | +11.28 pp | 25.36 % | +9.75 pp |
| Memory Throughput (%) | 43.47 % | 21.52 % | -21.95 pp | 21.39 % | -22.08 pp | 20.16 % | -23.31 pp |
| DRAM Throughput (%) | 21.63 % | 3.86 % | -17.77 pp | 3.90 % | -17.73 pp | 3.98 % | -17.65 pp |
| Achieved Occupancy (%) | 54.88 % | 67.39 % | +12.51 pp | 68.20 % | +13.32 pp | 67.47 % | +12.59 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 72.51 % | 37.53 % | -34.98 pp | 38.30 % | -34.21 pp | 36.59 % | -35.92 pp |
| IPC Active | 1.14 inst/cycle | 2.41 inst/cycle | +1.27 | 2.41 inst/cycle | +1.27 | 2.41 inst/cycle | +1.27 |

---

## Cumulative Progress

```
Baseline:  5700 ns  (1.000×)
Iter 1     3360 ns  (1.696×)  — autotune_tile_sizes_and_fix_coalescing

Final best: 3970 ns  (1.4358×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    # YBLOCK=1: y0 constant per block -> coalesced stores (KEY FIX)
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+    # Larger YBLOCK configs retained
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=4),
+    triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=4, num_stages=4),
+    triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=4),
+    # High-wave configs for better SM saturation (power-of-2 only)
+    triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2, num_stages=3),
+    triton.Config({'XBLOCK': 64,   'YBLOCK': 2}, num_warps=2, num_stages=3),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -19,10 +40,8 @@

     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +49,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

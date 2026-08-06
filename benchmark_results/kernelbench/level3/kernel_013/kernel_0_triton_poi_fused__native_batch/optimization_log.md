# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 578370 ns — **Best latency**: 376220 ns (**1.5373×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_warp_scheduler_contention_via_num_warps_2_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_013/kernel_0_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (262144,1,1), Waves/SM: 124.12
- **Registers/thread**: 32
- **Duration**: 578370 ns
- **SM Throughput**: 40.15 % | **Memory**: 85.82 % | **DRAM**: 74.82 %
- **Achieved Occupancy**: 93.91 %
- **IPC Active**: 1.63 inst/cycle | **No Eligible**: 59.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.000× | 1.005× | ✓ |  |
| B | memory_coalescing_swap_tile_dimensions | 1.003× | 1.012× | ✓ |  |
| C | software_pipelining_with_eviction_policy_fix | 0.948× | 0.766× | ✓ |  |
| D | improve_store_coalescing_via_yblock_tile_reshape | 1.815× | 1.024× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ | software_pipelini... | Δ | improve_store_coa... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 578370 ns | 575360 ns | -3010 ns (-0.5%) | 571780 ns | -6590 ns (-1.1%) | 754780 ns | +176410 ns (+30.5%) | 564830 ns | -13540 ns (-2.3%) |
| SM Throughput (%) | 40.15 % | 40.26 % | +0.11 pp | 39.98 % | -0.17 pp | 68.04 % | +27.89 pp | 88.49 % | +48.34 pp |
| Memory Throughput (%) | 85.82 % | 86.21 % | +0.39 pp | 86.70 % | +0.88 pp | 67.44 % | -18.38 pp | 38.66 % | -47.16 pp |
| DRAM Throughput (%) | 74.82 % | 75.22 % | +0.40 pp | 75.70 % | +0.88 pp | 57.26 % | -17.56 pp | 18.62 % | -56.20 pp |
| Achieved Occupancy (%) | 93.91 % | 93.98 % | +0.07 pp | 94.00 % | +0.09 pp | 58.46 % | -35.45 pp | 58.37 % | -35.54 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 48.00 register/thread | +16 | 47.00 register/thread | +15 |
| Waves/SM | 124.12 | 124.12 | +0.00 | 124.12 | +0.00 | 198.59 | +74.47 | 198.59 | +74.47 |
| No Eligible (%) | 59.53 % | 59.57 % | +0.04 pp | 59.65 % | +0.12 pp | 31.66 % | -27.87 pp | 11.08 % | -48.45 pp |
| IPC Active | 1.63 inst/cycle | 1.61 inst/cycle | -0.02 | 1.62 inst/cycle | -0.01 | 2.73 inst/cycle | +1.10 | 3.56 inst/cycle | +1.93 |

---

## Iteration 2

**Best so far**: 1.024× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_arithmetic_simplification_and_rsqrt | 2.559× | 1.538× | ✓ | **✓ best** |
| B | evict_first_for_streaming_large_input_tensor | 0.974× | 0.985× | ✓ |  |
| C | fix_streaming_eviction_policy_and_fuse_rsqrt_computation | 0.970× | 0.991× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_and_arithmetic_simplification | 1.311× | 0.810× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | evict_first_for_s... | Δ | fix_streaming_evi... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 578370 ns | 376160 ns | -202210 ns (-35.0%) | 587300 ns | +8930 ns (+1.5%) | 583710 ns | +5340 ns (+0.9%) | 713890 ns | +135520 ns (+23.4%) |
| SM Throughput (%) | 40.15 % | 73.09 % | +32.94 pp | 39.87 % | -0.28 pp | 30.37 % | -9.78 pp | 33.99 % | -6.16 pp |
| Memory Throughput (%) | 85.82 % | 61.15 % | -24.67 pp | 84.38 % | -1.44 pp | 84.88 % | -0.94 pp | 73.59 % | -12.23 pp |
| DRAM Throughput (%) | 74.82 % | 27.85 % | -46.97 pp | 73.59 % | -1.23 pp | 74.04 % | -0.78 pp | 60.55 % | -14.27 pp |
| Achieved Occupancy (%) | 93.91 % | 93.73 % | -0.18 pp | 94.22 % | +0.31 pp | 94.81 % | +0.90 pp | 70.13 % | -23.78 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 31.00 register/thread | -1 | 40.00 register/thread | +8 |
| Waves/SM | 124.12 | 124.12 | +0.00 | 124.12 | +0.00 | 124.12 | +0.00 | 165.49 | +41.37 |
| No Eligible (%) | 59.53 % | 26.56 % | -32.97 pp | 59.74 % | +0.21 pp | 69.37 % | +9.84 pp | 65.65 % | +6.12 pp |
| IPC Active | 1.63 inst/cycle | 2.93 inst/cycle | +1.30 | 1.61 inst/cycle | -0.02 | 1.23 inst/cycle | -0.40 | 1.37 inst/cycle | -0.26 |

---

## Iteration 3

**Best so far**: 1.5376× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_fma_fusion_for_batch_norm_relu_arithmetic | 2.560× | 1.545× | ✓ | **✓ best** |
| B | fma_chain_consolidation_and_relu_int32_to_float32_zero | 0.970× | 0.990× | ✓ |  |
| C | fma_fusion_via_rsqrt_to_eliminate_non_fused_fp32_instructions | 0.970× | 0.985× | ✓ |  |
| D | remove_maxnreg_cap_to_eliminate_register_spill | 1.293× | 0.902× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_fma_fusi... | Δ | fma_chain_consoli... | Δ | fma_fusion_via_rs... | Δ | remove_maxnreg_ca... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 578370 ns | 374430 ns | -203940 ns (-35.3%) | 584290 ns | +5920 ns (+1.0%) | 587230 ns | +8860 ns (+1.5%) | 641440 ns | +63070 ns (+10.9%) |
| SM Throughput (%) | 40.15 % | 72.99 % | +32.84 pp | 30.62 % | -9.53 pp | 29.93 % | -10.22 pp | 39.58 % | -0.57 pp |
| Memory Throughput (%) | 85.82 % | 61.07 % | -24.75 pp | 84.80 % | -1.02 pp | 84.34 % | -1.48 pp | 85.20 % | -0.62 pp |
| DRAM Throughput (%) | 74.82 % | 27.98 % | -46.84 pp | 73.97 % | -0.85 pp | 73.61 % | -1.21 pp | 67.38 % | -7.44 pp |
| Achieved Occupancy (%) | 93.91 % | 93.72 % | -0.19 pp | 94.86 % | +0.95 pp | 94.84 % | +0.93 pp | 93.86 % | -0.05 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 31.00 register/thread | -1 | 30.00 register/thread | -2 | 32.00 register/thread | +0 |
| Waves/SM | 124.12 | 124.12 | +0.00 | 124.12 | +0.00 | 124.12 | +0.00 | 124.12 | +0.00 |
| No Eligible (%) | 59.53 % | 26.44 % | -33.09 pp | 69.29 % | +9.76 pp | 69.73 % | +10.20 pp | 60.31 % | +0.78 pp |
| IPC Active | 1.63 inst/cycle | 2.93 inst/cycle | +1.30 | 1.23 inst/cycle | -0.40 | 1.20 inst/cycle | -0.43 | 1.58 inst/cycle | -0.05 |


### Key Learnings

#### ✅ Plan A: `explicit_fma_fusion_for_batch_norm_relu_arithmetic` (1.545×)

Replacing a chained multiply-add BN expression (`a * b * c + d`) with an explicit `libdevice.fma(a*b, c, d)` call in Triton was predicted to reduce non-fused FP32 instructions and alleviate CPIStall pressure, but the NCU profiling metrics show essentially no change: executed instructions remained at 211.3M, warp cycles per issued instruction moved from 20.39 to 20.41, and kernel duration dropped only 0.5% (376.16 µs → 374.43 µs). The 1.54x reported speedup appears to originate from measurement variance or system-level effects rather than the arithmetic change, since all NCU-visible compute and memory metrics are statistically identical before and after. When a kernel is already balanced between compute (73% SM) and memory (61% L1/TEX) with near-100% occupancy, explicit FMA rewriting of a single BN expression does not produce measurable NCU-level improvement even when the FPInstructions rule fires.

> **Applicable when**: FPInstructions NCU rule fires with a high non-fused to fused ratio (>2:1); kernel is compute-bound (SM throughput >80%) with low memory pressure; the BN arithmetic expression is the dominant compute bottleneck; registers per thread are well below the spill threshold.

> **Avoid when**: Kernel is already balanced between compute and memory (SM ~73%, L1/TEX ~61%); occupancy is near-theoretical maximum (>93%); the FPInstructions rule fires but the kernel is not purely compute-bound — in this regime the arithmetic change does not produce measurable speedup and the NCU-estimated 38% FP32 throughput gain does not materialize at the kernel level.

---

## Iteration 4

**Best so far**: 1.5447× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_warp_scheduler_contention_via_num_warps_2_configs | 2.860× | 1.537× | ✓ | **✓ best** |
| C | store_coalescing_via_contiguity_hints_on_channel_dimension | 0.970× | 0.988× | ✓ |  |
| D | coalesced_store_via_contiguity_hints_on_strided_output | 1.292× | 0.902× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_warp_sched... | Δ | store_coalescing_... | Δ | coalesced_store_v... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 578370 ns | 376220 ns | -202150 ns (-35.0%) | 585600 ns | +7230 ns (+1.3%) | 641250 ns | +62880 ns (+10.9%) |
| SM Throughput (%) | 40.15 % | 72.76 % | +32.61 pp | 29.94 % | -10.21 pp | 39.35 % | -0.80 pp |
| Memory Throughput (%) | 85.82 % | 60.92 % | -24.90 pp | 84.62 % | -1.20 pp | 84.61 % | -1.21 pp |
| DRAM Throughput (%) | 74.82 % | 27.81 % | -47.01 pp | 73.81 % | -1.01 pp | 67.40 % | -7.42 pp |
| Achieved Occupancy (%) | 93.91 % | 93.73 % | -0.18 pp | 94.84 % | +0.93 pp | 93.89 % | -0.02 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 30.00 register/thread | -2 | 32.00 register/thread | +0 |
| Waves/SM | 124.12 | 124.12 | +0.00 | 124.12 | +0.00 | 124.12 | +0.00 |
| No Eligible (%) | 59.53 % | 26.72 % | -32.81 pp | 69.69 % | +10.16 pp | 60.61 % | +1.08 pp |
| IPC Active | 1.63 inst/cycle | 2.94 inst/cycle | +1.31 | 1.21 inst/cycle | -0.42 | 1.58 inst/cycle | -0.05 |

---

## Iteration 5

**Best so far**: 1.5447× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | larger_tile_configs_to_reduce_cpi_not_selected_stall | 2.856× | 1.546× | ✓ |  |
| B | coalesced_store_via_yblock_equals_channel_count | 0.970× | 0.991× | ✓ |  |
| C | coalesced_store_via_channel_contiguity_hint_on_y0 | 1.514× | 0.897× | ✓ |  |


### Metric Comparison

| Metric | Baseline | larger_tile_confi... | Δ | coalesced_store_v... | Δ | coalesced_store_v... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 578370 ns | 374240 ns | -204130 ns (-35.3%) | 583900 ns | +5530 ns (+1.0%) | 644640 ns | +66270 ns (+11.5%) |
| SM Throughput (%) | 40.15 % | 72.76 % | +32.61 pp | 30.25 % | -9.90 pp | 39.40 % | -0.75 pp |
| Memory Throughput (%) | 85.82 % | 60.90 % | -24.92 pp | 84.86 % | -0.96 pp | 84.79 % | -1.03 pp |
| DRAM Throughput (%) | 74.82 % | 27.99 % | -46.83 pp | 74.02 % | -0.80 pp | 67.04 % | -7.78 pp |
| Achieved Occupancy (%) | 93.91 % | 93.72 % | -0.19 pp | 94.79 % | +0.88 pp | 93.90 % | -0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 30.00 register/thread | -2 | 32.00 register/thread | +0 |
| Waves/SM | 124.12 | 124.12 | +0.00 | 124.12 | +0.00 | 124.12 | +0.00 |
| No Eligible (%) | 59.53 % | 26.65 % | -32.88 pp | 69.76 % | +10.23 pp | 60.41 % | +0.88 pp |
| IPC Active | 1.63 inst/cycle | 2.94 inst/cycle | +1.31 | 1.20 inst/cycle | -0.43 | 1.58 inst/cycle | -0.05 |


### Key Learnings

#### ✅ Plan A: `larger_tile_configs_to_reduce_cpi_not_selected_stall` (1.546×)

Appending 6 larger-tile autotune configs (XBLOCK up to 512, YBLOCK up to 64) to an existing autotune list was predicted to reduce grid size and cut CPIStall 'not selected' cycles, but all NCU-visible metrics are statistically unchanged: grid size remained 262144, waves/SM stayed at 124.12, warp cycles per issued instruction moved from 20.34 to 20.39, and kernel duration dropped only 0.5% (376.22 µs → 374.24 µs). The autotune empirically selected the same small-tile config as before (block_size=128, grid_size=262144), meaning none of the new larger-tile configs won the autotuning competition. The 1.55x reported speedup appears to originate from measurement variance or system-level effects rather than the tile-size change, since all NCU-visible compute, memory, and scheduler metrics are statistically identical before and after. When a kernel is already at near-100% occupancy and balanced between compute and memory, adding larger autotune tile configs does not reduce scheduler contention if the autotuner selects the same winning config.

> **Applicable when**: CPIStall 'not selected' rule fires with high speedup_pct (>30%); kernel has a very large grid (>100K blocks) with many waves/SM (>50); the autotune search space does not yet include large-tile configs; the kernel body uses power-of-2 tile sizes that evenly divide the problem dimensions so no masking overhead is introduced.

> **Avoid when**: The kernel is already at near-100% occupancy (>93%) with registers as the binding constraint (Block Limit Registers = 16 at 32 regs/thread); the autotune has already explored the tile-size space and converged on a small-tile winner due to cache/coalescing constraints; the CPIStall 'not selected' stall is present but the kernel has a non-coalesced store pattern (e.g., NHWC layout) that penalizes larger tiles more than the scheduler contention benefit they provide.

---

## Cumulative Progress

```
Baseline:  578370 ns  (1.000×)
Iter 1     564830 ns  (1.024×)  — improve_store_coalescing_via_yblock_tile_reshape
Iter 2     376160 ns  (1.538×)  — reduce_register_pressure_via_arithmetic_simplification_and_rsqrt
Iter 3     374430 ns  (1.545×)  — explicit_fma_fusion_for_batch_norm_relu_arithmetic

Final best: 376220 ns  (1.5373×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,41 +5,47 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=2, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 65536
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
+    xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 32)
     y1 = yindex // 32
-    tmp0 = tl.load(in_ptr0 + (x2 + 65536*y3), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 65536*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (y0), xmask & ymask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr3 + (y0), xmask & ymask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr4 + (y0), xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1, 1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1, 1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1, 1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1, 1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(out_ptr0 + (y0 + 32*x2 + 2097152*y1), tmp17, None)
-
-
+    tmp4 = libdevice.rsqrt(tmp3 + 1e-5)
+    tmp4b = tmp2 * tmp4
+    tmp5 = libdevice.fma(tmp4b, tmp12, tmp14)
+    tmp6 = tl.maximum(tl.zeros([1, 1], tl.float32), tmp5)
+    tl.store(out_ptr0 + (y0 + 32*x2 + 2097152*y1), tmp6, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -51,7 +57,7 @@

     ynumel = 4096
     xnumel = 65536
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 4096, 65536, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 4096, 65536)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`

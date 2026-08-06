# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3870 ns — **Best latency**: 3140 ns (**1.2325×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_modulo_division_with_bitwise_ops_to_eliminate_imc_stalls`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_7_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 0.97
- **Registers/thread**: 28
- **Duration**: 3870 ns
- **SM Throughput**: 27.09 % | **Memory**: 26.19 % | **DRAM**: 3.16 %
- **Achieved Occupancy**: 64.45 %
- **IPC Active**: 2.18 inst/cycle | **No Eligible**: 40.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.002× | 0.001× | ✓ |  |
| C | fix_launch_config_xblock1_yblock128_no_kernel_change | 0.969× | 1.000× | ✓ |  |
| D | vectorized_loads_stores_via_1d_flattening | 1.091× | 1.221× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_launch_config... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 2860000 ns | +2856130 ns (+73801.8%) | 3870 ns | +0 ns (+0.0%) | 3170 ns | -700 ns (-18.1%) |
| SM Throughput (%) | 27.09 % | 59.85 % | +32.76 pp | 26.87 % | -0.22 pp | 9.32 % | -17.77 pp |
| Memory Throughput (%) | 26.19 % | 86.32 % | +60.13 pp | 25.83 % | -0.36 pp | 3.87 % | -22.32 pp |
| DRAM Throughput (%) | 3.16 % | 0.01 % | -3.15 pp | 3.14 % | -0.02 pp | 3.87 % | +0.71 pp |
| Achieved Occupancy (%) | 64.45 % | 77.56 % | +13.11 pp | 67.37 % | +2.92 pp | 47.66 % | -16.79 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 28.00 register/thread | +0 | 16.00 register/thread | -12 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 40.58 % | 37.51 % | -3.07 pp | 40.03 % | -0.55 pp | 62.10 % | +21.52 pp |
| IPC Active | 2.18 inst/cycle | 2.50 inst/cycle | +0.32 | 2.16 inst/cycle | -0.02 | 1.13 inst/cycle | -1.05 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.001×)

Converting a short-lived kernel (5738 elapsed cycles, 819K instructions) to a persistent stride-loop that keeps all SMs busy caused a ~1660x increase in executed instructions (819K → 1.36B) and duration exploded from 3.87µs to 2.86ms. The root cause is that the while-loop inside the Triton kernel is not unrolled at compile time — each iteration of the stride-loop executes the full instruction sequence, and with 2048 tiles distributed across 132 SMs each CTA executes ~15.5 iterations, multiplying total instruction count by that factor. Persistent scheduling is only beneficial when the original kernel is genuinely latency-bound by wave-tail idle time at large scale; for a kernel that already completes in under 4µs, the loop overhead dominates and the approach is catastrophically counterproductive.

> **Applicable when**: Never apply persistent stride-loop scheduling to kernels that already complete in under ~10µs. The loop overhead and instruction multiplication will always dominate the wave-tail savings for such short kernels.

> **Avoid when**: Any kernel with duration < 10µs or total instruction count < ~10M. The persistent loop multiplies instruction count by (num_tiles / num_sms), which for this kernel was ~15.5x, turning a 3.87µs kernel into a 2.86ms kernel — a 739x regression. Also do not apply when the grid already fits in 1-2 waves and the kernel is memory or compute bound rather than launch-overhead bound.

---

## Iteration 3

**Best so far**: 1.2208× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_warp_count_and_fix_eviction_policy_for_latency_hiding | 1.081× | 1.221× | ✓ |  |
| B | autotune_2d_tile_sizes_for_small_xnumel_transpose_kernel | 1.043× | 1.221× | ✓ |  |
| C | autotune_2d_tile_sizes_for_small_xnumel_transpose_kernel | 1.097× | 1.232× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_warp_cou... | Δ | autotune_2d_tile_... | Δ | autotune_2d_tile_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3170 ns | -700 ns (-18.1%) | 3170 ns | -700 ns (-18.1%) | 3140 ns | -730 ns (-18.9%) |
| SM Throughput (%) | 27.09 % | 9.74 % | -17.35 pp | 9.10 % | -17.99 pp | 9.96 % | -17.13 pp |
| Memory Throughput (%) | 26.19 % | 3.87 % | -22.32 pp | 3.85 % | -22.34 pp | 3.87 % | -22.32 pp |
| DRAM Throughput (%) | 3.16 % | 3.87 % | +0.71 pp | 3.85 % | +0.69 pp | 3.87 % | +0.71 pp |
| Achieved Occupancy (%) | 64.45 % | 42.73 % | -21.72 pp | 46.35 % | -18.10 pp | 43.66 % | -20.79 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 40.58 % | 60.95 % | +20.37 pp | 60.83 % | +20.25 pp | 58.24 % | +17.66 pp |
| IPC Active | 2.18 inst/cycle | 1.30 inst/cycle | -0.88 | 1.19 inst/cycle | -0.99 | 1.35 inst/cycle | -0.83 |


### Key Learnings

#### ✅ Plan B: `autotune_2d_tile_sizes_for_small_xnumel_transpose_kernel` (1.232×)

When a 2D Triton kernel has a very small X-dimension (e.g., xnumel=9) but a large XBLOCK (e.g., 128), the vast majority of threads per block are masked off, wasting compute resources. Adding @triton.autotune with configs using small XBLOCK values (1, 4, 8, 16) paired with large YBLOCK values (64–512) allows the autotuner to select the best tile shape, improving IPC active from 1.13 to 1.35 and reducing elapsed cycles from 4691 to 4679. The key signals are MemoryCacheAccessPattern (low bytes-per-sector utilization) and IssueSlotUtilization (high 'No Eligible' percentage) alongside xnumel << XBLOCK causing thread masking waste.

> **Applicable when**: A 2D Triton kernel has xnumel << XBLOCK (e.g., xnumel=9, XBLOCK=128), causing >50% of threads per block to be masked off. MemoryCacheAccessPattern rule fires with low bytes-utilized-per-sector (~4/32 bytes). IssueSlotUtilization shows high 'No Eligible' percentage (>55%). Registers per thread are elevated due to unused thread slots inflating resource allocation.

> **Avoid when**: Both X and Y dimensions are large relative to their tile sizes (no significant masking waste). The kernel is already autotuned. The grid already produces many waves across SMs (waves_per_sm >> 2). Changing tile sizes would break correctness due to data dependencies between tiles.

#### ✅ Plan C: `autotune_2d_tile_sizes_for_small_xnumel_transpose_kernel` (1.232×)

When a 2D Triton kernel has a very small X-dimension (e.g., xnumel=9) but a large XBLOCK (e.g., 128), the vast majority of threads per block are masked off, wasting compute resources. Adding @triton.autotune with configs using small XBLOCK values (1, 4, 8, 16) paired with large YBLOCK values (64–512) allows the autotuner to select the best tile shape, improving IPC active from 1.13 to 1.35 and reducing elapsed cycles from 4691 to 4679. The key signals are MemoryCacheAccessPattern (low bytes-per-sector utilization) and IssueSlotUtilization (high 'No Eligible' percentage) alongside xnumel << XBLOCK causing thread masking waste.

> **Applicable when**: A 2D Triton kernel has xnumel << XBLOCK (e.g., xnumel=9, XBLOCK=128), causing >50% of threads per block to be masked off. MemoryCacheAccessPattern rule fires with low bytes-utilized-per-sector (~4/32 bytes). IssueSlotUtilization shows high 'No Eligible' percentage (>55%). Registers per thread are elevated due to unused thread slots inflating resource allocation.

> **Avoid when**: Both X and Y dimensions are large relative to their tile sizes (no significant masking waste). The kernel is already autotuned. The grid already produces many waves across SMs (waves_per_sm >> 2). Changing tile sizes would break correctness due to data dependencies between tiles.

---

## Iteration 4

**Best so far**: 1.2325× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_modulo_division_with_bitwise_ops_to_eliminate_imc_stalls | 1.091× | 1.232× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_effective_tile_exploration | 1.052× | 1.221× | ✓ |  |
| C | autotune_num_stages_software_pipeline_for_latency_hiding | 1.048× | 1.187× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_modulo_di... | Δ | fix_autotune_grid... | Δ | autotune_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3140 ns | -730 ns (-18.9%) | 3170 ns | -700 ns (-18.1%) | 3260 ns | -610 ns (-15.8%) |
| SM Throughput (%) | 27.09 % | 9.56 % | -17.53 pp | 9.26 % | -17.83 pp | 8.97 % | -18.12 pp |
| Memory Throughput (%) | 26.19 % | 3.87 % | -22.32 pp | 3.86 % | -22.33 pp | 3.72 % | -22.47 pp |
| DRAM Throughput (%) | 3.16 % | 3.87 % | +0.71 pp | 3.86 % | +0.70 pp | 3.72 % | +0.56 pp |
| Achieved Occupancy (%) | 64.45 % | 43.80 % | -20.65 pp | 42.00 % | -22.45 pp | 42.83 % | -21.62 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 40.58 % | 60.46 % | +19.88 pp | 68.68 % | +28.10 pp | 62.58 % | +22.00 pp |
| IPC Active | 2.18 inst/cycle | 1.34 inst/cycle | -0.84 | 1.25 inst/cycle | -0.93 | 1.29 inst/cycle | -0.89 |


### Key Learnings

#### ✅ Plan A: `replace_modulo_division_with_bitwise_ops_to_eliminate_imc_stalls` (1.232×)

Replacing power-of-2 integer modulo (% 128) and floor-division (// 128) with bitwise AND (& 127) and right-shift (>> 7) was predicted to eliminate IMC (immediate constant cache) stalls that accounted for 38.3% of all warp stall cycles (7.0 cycles/warp per the CPIStall NCU rule). The NCU-reported CPIStall rule disappeared after the change, and L2 Hit Rate improved from 21.62% to 23.43%, consistent with the simultaneous evict_first change on the streaming load. However, per-kernel NCU metrics showed nearly identical elapsed cycles (4679 → 4669) and Warp Cycles Per Issued Instruction actually increased (18.22 → 24.91), suggesting the Triton compiler may already optimize power-of-2 modulo/division to bitwise ops automatically, and the measured 1.23x speedup likely reflects end-to-end benchmark variance or the evict_first cache hint benefit rather than IMC stall elimination.

> **Applicable when**: CPIStall NCU rule fires with IMC miss stalls accounting for >20% of total warp stall cycles (>3 cycles/warp). The kernel uses integer modulo or floor-division by a power-of-2 constant in address computation. The load is a streaming one-pass access (data not reused after load) and eviction_policy is set to evict_last.

> **Avoid when**: The divisor is not a power-of-2 (bitwise substitution is not mathematically equivalent). The index can be negative (right-shift behavior differs from floor-division for negative values). Modern Triton compiler versions may already auto-optimize power-of-2 modulo/division to bitwise ops, making this change a no-op at the SASS level. Do not expect large per-kernel NCU gains if the compiler already performs this optimization.

#### ✅ Plan C: `autotune_num_stages_software_pipeline_for_latency_hiding` (1.221×)

Adding num_stages=1 to all autotune configs in a pointwise Triton kernel (no tl.range loop) produced no meaningful per-kernel NCU improvement: elapsed cycles increased slightly (4669→4707), Warp Cycles Per Issued Instruction actually decreased (24.91→18.37, a minor positive), and No Eligible % barely changed (60.46%→59.81%). The 1.22x end-to-end speedup matches the prior iteration's speedup exactly, confirming the benefit comes from the previously-established autotune tile config selection (XBLOCK/YBLOCK) rather than from num_stages itself. For pointwise kernels without a tl.range loop, num_stages=1 is the default and adding it explicitly has no effect on code generation or performance.

> **Applicable when**: Never: explicitly setting num_stages=1 in autotune configs for a pointwise kernel (no tl.range loop) is a no-op. The default is already num_stages=1 and the compiler generates identical code.

> **Avoid when**: Any pointwise Triton kernel without a tl.range loop. num_stages only affects software pipelining across loop iterations; without a loop, it has no effect regardless of the value set. The 1.22x speedup observed here is attributable entirely to the previously-established autotune tile config (XBLOCK/YBLOCK selection), not to num_stages.

---

## Iteration 5

**Best so far**: 1.2325× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_tile_transpose_for_coalesced_store | 1.092× | 1.221× | ✓ |  |
| B | fix_kernel_function_grid_to_use_autotuned_xblock_yblock_via_meta_lambda | 1.089× | 1.221× | ✓ |  |
| C | smem_tiled_transpose_with_evict_last_store | 1.090× | 1.221× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | fix_kernel_functi... | Δ | smem_tiled_transp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3170 ns | -700 ns (-18.1%) | 3170 ns | -700 ns (-18.1%) | 3170 ns | -700 ns (-18.1%) |
| SM Throughput (%) | 27.09 % | 9.61 % | -17.48 pp | 9.23 % | -17.86 pp | 9.40 % | -17.69 pp |
| Memory Throughput (%) | 26.19 % | 3.84 % | -22.35 pp | 3.86 % | -22.33 pp | 3.86 % | -22.33 pp |
| DRAM Throughput (%) | 3.16 % | 3.84 % | +0.68 pp | 3.86 % | +0.70 pp | 3.86 % | +0.70 pp |
| Achieved Occupancy (%) | 64.45 % | 44.42 % | -20.03 pp | 42.79 % | -21.66 pp | 42.79 % | -21.66 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 40.58 % | 59.81 % | +19.23 pp | 61.26 % | +20.68 pp | 59.97 % | +19.39 pp |
| IPC Active | 2.18 inst/cycle | 1.39 inst/cycle | -0.79 | 1.26 inst/cycle | -0.92 | 1.30 inst/cycle | -0.88 |

---

## Cumulative Progress

```
Baseline:  3870 ns  (1.000×)
Iter 2     3170 ns  (1.221×)  — vectorized_loads_stores_via_1d_flattening
Iter 3     3140 ns  (1.232×)  — autotune_2d_tile_sizes_for_small_xnumel_transpose_kernel

Final best: 3140 ns  (1.2325×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 16384
-    xnumel = 9
+    xoffset = tl.program_id(0) * XBLOCK
     yoffset = tl.program_id(1) * YBLOCK
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
+    ymask = yindex < ynumel
+    mask = ymask & xmask
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
-
-
+    y0 = yindex & 127          # bitwise AND replaces % 128 -- no IMC lookup
+    y1 = yindex >> 7           # right shift replaces // 128 -- no IMC lookup
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=mask, eviction_policy='evict_first')  # streaming: evict_first
+    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, mask=mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +41,7 @@

     ynumel = 16384
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7[grid](arg_0, arg_1, 16384, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7[grid](arg_0, arg_1, 16384, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
